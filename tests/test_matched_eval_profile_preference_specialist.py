from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.profile_preference_specialist import (
    MECHANISM_ID,
    ProfilePreferenceBudget,
    ProfilePreferenceSpecialistError,
    adapt_profile_preference_to_typed_contribution,
    select_profile_preference_evidence,
)
from tools.matched_eval.query_expansion import (
    FrozenSourceNamespace,
    load_preflighted_query_expansion_population,
)
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_adapter import (
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(
    path: Path,
    rows: list[tuple[str, str, str, datetime]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, (source_id, role, text, created_at) in enumerate(rows):
        turn = transcript.append(
            role, text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"profile-chunk-{index}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha("profile-combined-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("profile-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("profile-database"),
            source_store_receipt_sha256=store_receipt,
        )
    return cache


def test_generic_screen_question_selects_coherent_identity_cluster_and_exact_spans(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 23, 43, tzinfo=timezone.utc)
    database_path = tmp_path / "profile-screen.db"
    cache = _write_cache(
        database_path,
        [
            (
                "history-noisy::cue-frequency",
                "user",
                "Can you recommend a show or movie to watch? I want show and movie recommendations.",
                asked - timedelta(hours=2),
            ),
            (
                "history-noisy::cue-frequency",
                "user",
                "Please suggest another movie show recommendation to watch tonight.",
                asked - timedelta(hours=1),
            ),
            (
                "history-target::comedy",
                "user",
                "As an aspiring stand-up comedian, I'm looking for advice on improving my craft. "
                "Can you recommend Netflix stand-up comedy specials with strong storytelling "
                "like John Mulaney's \"Kid Gorgeous\"?",
                asked - timedelta(days=1),
            ),
            (
                "history-target::comedy",
                "user",
                "I've been thinking about recording my jokes and comedy videos for YouTube.",
                asked - timedelta(days=1),
            ),
            (
                "history-assistant::flood",
                "assistant",
                ("Netflix show movie watch comedy storytelling John Mulaney. " * 12).strip(),
                asked - timedelta(minutes=5),
            ),
            (
                "history-future::contaminant",
                "user",
                "As an aspiring filmmaker, I love Netflix movies and rare cinema.",
                asked + timedelta(days=1),
            ),
        ],
    )
    index = build_full_store_window_index(cache)
    question = (
        "[Question asked at 2026/08/27 (Thu) 23:43]\n"
        "Can you recommend a show or movie for me to watch tonight?"
    )

    first = select_profile_preference_evidence(index, question)
    replay = select_profile_preference_evidence(index, question)

    quotes = "\n".join(row.quote for row in first.candidates)
    assert first.audit.status == "selected"
    assert {row.source_id for row in first.local_bindings} == {
        "history-target::comedy"
    }
    assert "aspiring stand-up comedian" in quotes
    assert "John Mulaney" in quotes
    assert "Netflix" in quotes
    assert first.local_projection() == replay.local_projection()
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.audit.physical_sentence_windows_scanned == len(index.windows)
    assert first.audit.future_rejected_window_count >= 1
    assert first.audit.selected_source_cluster_count == 1
    assert first.audit.question_id_filter_used is False
    assert first.audit.known_source_filter_used is False
    assert first.audit.partition_routing_used is False
    assert first.audit.gold_loaded is False
    assert first.audit.new_provider_calls == 0
    assert first.audit.retained_transformer_token_state_bytes == 0
    assert first.audit.selected_evidence_tokens <= 768
    assert len(first.candidates) <= 6

    provider_json = json.dumps(first.provider_projection(), sort_keys=True)
    assert "history-target::comedy" not in provider_json
    assert '"source_id"' not in provider_json
    with Database(database_path, read_only=True) as database:
        store = DiscourseStore(database)
        assert all(
            store.hydrate_span(binding.span) == candidate.quote
            for candidate, binding in zip(
                first.candidates, first.local_bindings, strict=True
            )
        )


def test_domain_specific_generic_recommendation_keeps_relevant_profile(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "profile-laptop.db",
        [
            (
                "work::developer-profile",
                "user",
                "I'm a developer who travels every week. I prefer a lightweight laptop "
                "with long battery life and a matte screen under $1200.",
                asked - timedelta(days=2),
            ),
            (
                "noise::generic-tech",
                "user",
                "Can you recommend a laptop computer? Please suggest a laptop recommendation.",
                asked - timedelta(hours=1),
            ),
            (
                "noise::screen-profile",
                "user",
                "As an aspiring comedian, I love Netflix stand-up storytelling specials.",
                asked - timedelta(hours=1),
            ),
        ],
    )
    result = select_profile_preference_evidence(
        build_full_store_window_index(cache),
        (
            "[Question asked at 2026/08/27 (Thu) 12:00]\n"
            "Can you recommend a laptop for me?"
        ),
    )

    assert result.audit.status == "selected"
    assert {row.source_id for row in result.local_bindings} == {
        "work::developer-profile"
    }
    assert any("lightweight laptop" in row.quote for row in result.candidates)
    assert all(row.role == "user" for row in result.candidates)


def test_plural_suggestion_question_selects_one_beverage_preference_cluster(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "profile-beverage.db",
        [
            (
                "drinks::personal-cluster",
                "user",
                "Do you have any recommendations for summer drinks made with botanical gin?",
                asked - timedelta(days=2),
            ),
            (
                "drinks::personal-cluster",
                "user",
                "I like cucumber Collins cocktails with grapefruit syrup, and I learned "
                "classic cocktails in a recent mixology class.",
                asked - timedelta(days=2),
            ),
            (
                "drinks::personal-cluster",
                "user",
                "I prefer muddled cucumber and a tall Collins glass for a refreshing drink.",
                asked - timedelta(days=2),
            ),
            (
                "food::recent-dinner",
                "user",
                "I love spicy restaurant dinners and prefer elaborate recipes for parties.",
                asked - timedelta(hours=1),
            ),
            (
                "drinks::recent-request-only",
                "user",
                "Do you have any recommendations for refreshing game-day drinks?",
                asked - timedelta(hours=1),
            ),
            (
                "assistant::beverage-flood",
                "assistant",
                ("cocktail drink gin Collins suggestions. " * 20).strip(),
                asked - timedelta(minutes=5),
            ),
        ],
    )
    result = select_profile_preference_evidence(
        build_full_store_window_index(cache),
        (
            "[Question asked at 2026/08/27 (Thu) 12:00]\n"
            "I'm making a cocktail for a get-together. Any suggestions?"
        ),
    )

    quotes = "\n".join(row.quote for row in result.candidates)
    assert result.audit.status == "selected"
    assert result.audit.recognized_domain_ids == ("beverages",)
    assert {row.source_id for row in result.local_bindings} == {
        "drinks::personal-cluster"
    }
    assert "mixology class" in quotes
    assert "grapefruit syrup" in quotes
    assert "Collins glass" in quotes
    assert "spicy restaurant dinners" not in quotes
    assert result.audit.question_id_filter_used is False
    assert result.audit.known_source_filter_used is False
    assert result.audit.gold_loaded is False
    assert result.audit.new_provider_calls == 0


def test_unsupported_generic_request_fails_closed_instead_of_injecting_noise(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "profile-unsupported.db",
        [
            (
                "profile::music",
                "user",
                "I'm a musician and my favorite album is Kind of Blue.",
                asked - timedelta(days=1),
            )
        ],
    )
    result = select_profile_preference_evidence(
        build_full_store_window_index(cache),
        (
            "[Question asked at 2026/08/27 (Thu) 12:00]\n"
            "Can you recommend something for me?"
        ),
    )

    assert result.audit.status == "unsupported_query_domain"
    assert result.candidates == ()
    assert result.local_bindings == ()
    assert result.audit.selected_source_cluster_count == 0


def test_caps_audit_and_typed_adapter_preserve_one_exact_pointer_lane(
    tmp_path: Path,
) -> None:
    asked = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    rows = [
        (
            "books::reader",
            "user",
            (
                "I'm a writer and I love literary mystery novels with unusual narrators. "
                f"My favorite author constraint number {index} is distinctive."
            ),
            asked - timedelta(days=1),
        )
        for index in range(10)
    ]
    cache = _write_cache(tmp_path / "profile-cap.db", rows)
    budget = ProfilePreferenceBudget(
        max_selected_candidates=2,
        max_selected_tokens=160,
        max_windows_per_cluster=4,
    )
    result = select_profile_preference_evidence(
        build_full_store_window_index(cache),
        (
            "[Question asked at 2026/08/27 (Thu) 12:00]\n"
            "Can you recommend a book for me to read?"
        ),
        budget=budget,
    )

    assert len(result.candidates) == 2
    assert result.audit.selection_truncated is True
    assert result.audit.selected_evidence_tokens <= 160
    contribution = adapt_profile_preference_to_typed_contribution(
        result,
        handle_start=700_001,
        group_start=700_001,
    )
    assert contribution.mechanism_id == MECHANISM_ID
    assert contribution.frontier_mode is FrontierMode.BOUNDED
    assert contribution.truncated is True
    assert len(contribution.bindings) == len(result.candidates)
    assert {row.source_group_handle for row in contribution.bindings} == {
        "G700001"
    }
    assert all(
        row.origin is EvidenceOrigin.DIRECT_POINTER
        and row.provenance_grade is ProvenanceGrade.DIRECT_POINTER
        for row in contribution.bindings
    )
    assert tuple(item.summary for item in contribution.parsed.accepted_items) == tuple(
        row.quote for row in result.candidates
    )

    with pytest.raises(ProfilePreferenceSpecialistError, match="audit receipt"):
        replace(
            result.audit,
            selected_evidence_tokens=result.audit.selected_evidence_tokens + 1,
        )


@pytest.mark.skipif(
    os.environ.get("RUN_PROFILE_PREFERENCE_EXACT10_SMOKE") != "1",
    reason="opt-in read-only 1M exact-ten smoke",
)
def test_locked_q36_read_only_smoke_surfaces_decisive_profile() -> None:
    """Read the frozen offset-030 store; pass no question/source identifier."""

    repository = Path(__file__).resolve().parents[1]
    store_root = repository / (
        "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    )
    retrieval_path = store_root / "retrieval.json"
    query_root = repository / (
        "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
        "matched-eval-spine-v2/s0-plus-query-expansion-v1"
    )
    population, _ = load_preflighted_query_expansion_population(
        retrieval_path,
        output_root=query_root,
        expected_retrieval_sha256=(
            "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
        ),
        expected_question_count=100,
    )
    namespace = next(
        row
        for row in population.namespaces
        if row.namespace_id
        == "c9274e896ed9201eb961bf6e01a5358dc08733e091fa022fd6efd379552e81b9"
    )
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    shard = next(row for row in retrieval["shards"] if row["shard_offset"] == 30)
    database_path = store_root / "shards/offset-030/combined-store/memory.db"
    with Database(database_path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=shard["combined_store_receipt"][
                "target_database_sha256"
            ],
            source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
        )
    result = select_profile_preference_evidence(
        build_full_store_window_index(cache),
        (
            "[Question asked at 2023/05/30 (Tue) 23:43]\n"
            "Can you recommend a show or movie for me to watch tonight?"
        ),
    )

    quotes = "\n".join(row.quote for row in result.candidates)
    assert result.audit.status == "selected"
    assert "aspiring stand-up comedian" in quotes
    assert "Netflix" in quotes
    assert "storytelling" in quotes
    assert "John Mulaney" in quotes
    assert result.audit.question_id_filter_used is False
    assert result.audit.known_source_filter_used is False
    assert result.audit.partition_routing_used is False
    assert result.audit.new_provider_calls == 0
    assert result.audit.retained_transformer_token_state_bytes == 0
