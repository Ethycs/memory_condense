from __future__ import annotations

import hashlib
import base64
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools import run_locked_semantic_residual_construction_v4 as runner
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_residual_eligibility import (
    SemanticResidualEligibilityPolicy,
)
from tools.matched_eval.semantic_residual_search import (
    SemanticResidualPolicy,
    build_semantic_residual_index,
    compile_semantic_residual_query,
    search_semantic_residual,
    semantic_residual_query_facets,
)


BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _index(tmp_path: Path, *, payload_token_cap: int = 2_400):
    path = tmp_path / "terminal.db"
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    rows = (
        ("history-a", "I chose cobalt blue paint for the studio.", BASE),
        ("history-a", "The oak desk belongs in the studio.", BASE + timedelta(days=1)),
        ("history-b", "The studio has three walnut shelves.", BASE + timedelta(days=2)),
    )
    for ordinal, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{ordinal}",
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
    store_receipt = _sha("store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=store_receipt,
        )
    window = build_full_store_window_index(cache)
    policy = SemanticResidualPolicy(
        max_cell_tokens=12,
        payload_token_cap=payload_token_cap,
        cosine_upper_bound_floor=0.0,
        dual_gate_enabled=False,
    )
    return build_semantic_residual_index(
        window,
        {"history-a": [1.0, 0.0], "history-b": [1.0, 0.0]},
        policy=policy,
    )


def _deduped_result(tmp_path: Path, *, payload_token_cap: int = 2_400):
    index = _index(tmp_path, payload_token_cap=payload_token_cap)
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What studio paint, desk, and shelves did I mention?"
    )
    facets = semantic_residual_query_facets(question)
    query = compile_semantic_residual_query(
        index, question, query_vectors=[[1.0, 0.0] for _ in facets]
    )
    first = search_semantic_residual(index, query)
    protected = (first.local_bindings[0],)
    result = search_semantic_residual(
        index, query, protected_evidence=protected
    )
    assert result.protected_duplicates
    assert result.evidence
    return index, question, protected, result


def _policy(cap: int = 2_400) -> SemanticResidualEligibilityPolicy:
    return SemanticResidualEligibilityPolicy(
        residual_payload_token_cap=cap,
        hard_complete_chat_token_cap=8_000,
        output_token_reserve=768,
    )


class _FakeEmbedder:
    model_name = DEFAULT_MODEL_NAME
    model_revision = DEFAULT_MODEL_REVISION
    checkpoint_sha256 = BGE_M3_CHECKPOINT_SHA256

    @property
    def execution_identity(self) -> dict[str, object]:
        return {
            "backend": "sentence-transformers.encode-v1",
            "batch_size": 2,
            "device": "cpu",
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        values = np.zeros((len(texts), DEFAULT_MODEL_DIM), dtype=np.float32)
        values[:, 0] = 1.0
        return values


def _gate() -> SealedArtifact:
    row = {
        "dated_question_sha256": _sha("dated"),
        "eligibility": {"eligible": True},
        "facet_texts": ["first facet", "second facet"],
        "gate_row_receipt_sha256": _sha("gate-row"),
        "ordinal": 0,
        "question_id": "q000",
        "question_sha256": _sha("question"),
    }
    return SealedArtifact(Path("gate.json"), _sha("gate"), {"questions": [row]})


def test_terminal_reinjects_exact_owner_and_seals_lossless_unified_groups(
    tmp_path: Path,
) -> None:
    index, question, protected, result = _deduped_result(tmp_path)
    prompt, reason = runner.build_separate_terminal_prompt(
        dated_question=question,
        current_prediction="The studio details are unknown.",
        result=result,
        residual_index=index,
        protected_evidence=protected,
        policy=_policy(),
    )

    assert reason == "none"
    assert prompt is not None
    provider = prompt["provider_input"]
    residual = provider["residual_evidence"]
    owners = provider["protected_owner_evidence"]
    closure = provider["lossless_post_selection_closure"]
    assert owners
    assert owners[0]["quote_sha256"] == protected[0].quote_sha256
    assert closure["every_removed_duplicate_has_exact_provider_visible_owner"] is True
    assert closure["owner_count"] == len(owners) == len(result.protected_duplicates)
    assert provider["residual_frontier"]["packing_closed"] is True
    assert provider["residual_frontier"][
        "all_novel_survivors_protected"
    ] is False
    assert provider["residual_frontier"]["support_closure_proven"] is False
    assert "closed" not in provider["residual_frontier"]
    assert prompt["complete_chat_plus_output_tokens"] <= 8_000
    assert prompt["parent_prompt_tokens_borrowed"] == 0

    mapping = prompt["prompt_external_unified_group_mapping"]["rows"]
    groups_by_handle = {
        handle: row["source_group_handle"]
        for row in mapping
        for handle in row["evidence_handle_ids"]
    }
    rendered_groups = {
        row["evidence_handle"]: row["source_group_handle"]
        for row in (*residual, *owners)
    }
    assert groups_by_handle == rendered_groups
    assert len({row["source_group_handle"] for row in mapping}) == len(mapping)

    accounting = prompt["residual_evidence_accounting"]
    exact = canonical_json_bytes({"residual_evidence": residual})[:-1]
    assert accounting["exact_serialized_utf8_bytes"] == len(exact)
    assert accounting["exact_serialized_field_sha256"] == hashlib.sha256(
        exact
    ).hexdigest()
    assert accounting["token_proxy"] == result.packed_residual_evidence_tokens
    assert (
        accounting["exact_serialized_field_sha256"]
        == result.packed_residual_evidence_sha256
    )
    assert [row["source_group_handle"] for row in residual] == [
        row.source_group_handle for row in result.evidence
    ]


def test_terminal_exact_near_cap_keeps_search_time_groups_after_owner_dedup(
    tmp_path: Path,
) -> None:
    high_root = tmp_path / "high"
    high_root.mkdir()
    _index_high, _question_high, _protected_high, high = _deduped_result(
        high_root
    )
    exact_cap = high.packed_residual_evidence_tokens

    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    index, question, protected, result = _deduped_result(
        exact_root,
        payload_token_cap=exact_cap,
    )
    prompt, reason = runner.build_separate_terminal_prompt(
        dated_question=question,
        current_prediction="The studio details are unknown.",
        result=result,
        residual_index=index,
        protected_evidence=protected,
        policy=_policy(exact_cap),
    )

    assert reason == "none"
    assert prompt is not None
    residual_rows = prompt["provider_input"]["residual_evidence"]
    assert prompt["residual_evidence_accounting"]["token_proxy"] == exact_cap
    assert [row["source_group_handle"] for row in residual_rows] == [
        row.source_group_handle for row in result.evidence
    ]
    assert prompt["residual_evidence_accounting"][
        "exact_serialized_field_sha256"
    ] == result.packed_residual_evidence_sha256


def test_compact_commitments_omit_full_search_graph_but_keep_exact_packed_rows(
    tmp_path: Path,
) -> None:
    index, _question, _protected, result = _deduped_result(tmp_path)

    index_commitment = runner._compact_index_commitment(index)  # noqa: SLF001
    query_commitment = runner._compact_query_commitment(result.query)  # noqa: SLF001
    search_commitment = runner._compact_search_commitment(result)  # noqa: SLF001
    selected = runner._selected_provenance(index, result)  # noqa: SLF001
    compact = json.dumps(
        {
            "index": index_commitment,
            "query": query_commitment,
            "search": search_commitment,
        },
        sort_keys=True,
    )

    assert '"core_result"' not in compact
    assert '"visits"' not in compact
    assert '"decision_audits"' not in compact
    assert '"quote"' not in compact
    assert search_commitment["search_receipt_sha256"] == result.receipt_sha256
    assert search_commitment["all_novel_survivors_protected"] is False
    assert search_commitment["packed_residual_evidence_sha256"] == (
        result.packed_residual_evidence_sha256
    )
    assert selected["packed_exact_row_count"] == len(result.evidence)
    assert all(row["exact_segment"]["quote"] for row in selected["rows"])
    assert all(row["exact_local_binding"] for row in selected["rows"])


def test_terminal_owner_plane_has_independent_nonborrowable_cap(
    tmp_path: Path,
) -> None:
    index, question, protected, result = _deduped_result(tmp_path)
    prompt, reason = runner.build_separate_terminal_prompt(
        dated_question=question,
        current_prediction="The studio details are unknown.",
        result=result,
        residual_index=index,
        protected_evidence=protected,
        policy=_policy(),
        protected_owner_token_cap=1,
    )

    assert prompt is None
    assert reason == "protected_owner_reinjection_exceeds_cap"


def test_terminal_exact_residual_plane_enforces_its_own_cap() -> None:
    rows = [
        {
            "created_at": "2026-08-01T00:00:00+00:00",
            "event_dates": ["2026-08-01"],
            "evidence_handle": "R0001",
            "quote": "A separately serialized residual quote.",
            "role": "user",
            "source_group_handle": "G0001",
        }
    ]
    accounting = runner._terminal_plane_accounting(  # noqa: SLF001
        "residual_evidence", rows, token_cap=1
    )

    exact = canonical_json_bytes({"residual_evidence": rows})[:-1]
    assert accounting["within_cap"] is False
    assert accounting["token_proxy"] > accounting["token_cap"] == 1
    assert accounting["exact_serialized_utf8_bytes"] == len(exact)


def test_construct_cli_requires_the_sealed_vector_replay_boundary() -> None:
    args = runner.build_parser().parse_args(
        [
            "construct",
            "--expected-gate-sha256",
            "a" * 64,
            "--expected-vector-sha256",
            "b" * 64,
        ]
    )

    assert args.vector_replay == runner.DEFAULT_VECTOR_REPLAY
    assert args.expected_vector_sha256 == "b" * 64


def test_query_vectors_use_one_pinned_local_batch_and_round_trip(
    tmp_path: Path,
) -> None:
    gate = _gate()
    payload = runner.build_query_vector_payload(gate, _FakeEmbedder())
    artifact, _ = publish_sealed_json(tmp_path / runner.VECTOR_NAME, payload)

    loaded, vectors = runner._load_vectors(  # noqa: SLF001
        artifact.path, artifact.sha256, gate
    )

    assert loaded.sha256 == artifact.sha256
    assert payload["facet_count"] == 2
    assert payload["local_embedding_batch_calls"] == 1
    assert payload["new_provider_calls"] == 0
    assert len(vectors[0]) == 2
    assert len(vectors[0][0]) == DEFAULT_MODEL_DIM


def test_query_vector_loader_rejects_receipted_tampered_bytes(
    tmp_path: Path,
) -> None:
    gate = _gate()
    payload = runner.build_query_vector_payload(gate, _FakeEmbedder())
    tampered = copy.deepcopy(payload)
    facet = tampered["rows"][0]["facets"][0]
    raw = base64.b64decode(facet["vector_base64"])
    facet["vector_base64"] = base64.b64encode(b"x" + raw[1:]).decode("ascii")
    facet_body = dict(facet)
    facet_body.pop("facet_receipt_sha256")
    facet["facet_receipt_sha256"] = identity_sha256(facet_body)
    row_body = dict(tampered["rows"][0])
    row_body.pop("vector_row_receipt_sha256")
    tampered["rows"][0]["vector_row_receipt_sha256"] = identity_sha256(
        row_body
    )
    artifact_body = dict(tampered)
    artifact_body.pop("vector_identity_sha256")
    tampered["vector_identity_sha256"] = identity_sha256(artifact_body)
    artifact, _ = publish_sealed_json(tmp_path / "tampered.json", tampered)

    with pytest.raises(
        runner.LockedSemanticResidualConstructionError,
        match="query vector facet changed",
    ):
        runner._load_vectors(artifact.path, artifact.sha256, gate)  # noqa: SLF001


def test_source_store_embedding_identity_must_match_query_identity(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "shard" / "combined-store"
    store_dir.mkdir(parents=True)
    source_database_sha = _sha("source-db")
    target_database_sha = _sha("target-db")
    store_identity = _sha("store-identity")
    embedding = {
        "backend": "sentence-transformers.encode-v1",
        "batch_size": 32,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "device": "cuda",
        "dimension": DEFAULT_MODEL_DIM,
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    selection_body = {
        "database_sha256": source_database_sha,
        "embedding_identity": embedding,
        "embedding_identity_sha256": identity_sha256(embedding),
        "format": "synthetic-source-selection-v1",
    }
    selection = {**selection_body, "receipt_sha256": identity_sha256(selection_body)}
    (store_dir.parent / "source-current-selection.json").write_text(
        json.dumps(selection, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    combined_body = {
        "format": "synthetic-combined-store-v1",
        "source_database_sha256": source_database_sha,
        "source_store_identity_sha256": store_identity,
        "target_database_sha256": target_database_sha,
        "target_store_identity_sha256": store_identity,
    }
    combined_receipt = {
        **combined_body,
        "receipt_sha256": identity_sha256(combined_body),
    }
    (store_dir / "combined-cumulative-store.json").write_text(
        json.dumps(
            {"combined_store_receipt": combined_receipt},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    scoped = SimpleNamespace(
        store_dir=store_dir,
        database_sha256=target_database_sha,
        namespace=SimpleNamespace(
            combined_store_receipt_sha256=combined_receipt["receipt_sha256"]
        ),
    )
    source_vectors = SimpleNamespace(
        vector_dimension=DEFAULT_MODEL_DIM,
        receipt_sha256=_sha("source-vectors"),
    )
    query = runner.build_query_vector_payload(_gate(), _FakeEmbedder())["embedding"]

    binding = runner._verified_source_embedding_binding(  # noqa: SLF001
        scoped, source_vectors, query
    )

    assert binding["stored_and_query_identity_match"] is True
    assert binding["source_embedding_identity_sha256"] == identity_sha256(
        embedding
    )

    mismatched = copy.deepcopy(selection)
    mismatched["embedding_identity"]["checkpoint_sha256"] = "f" * 64
    mismatched["embedding_identity_sha256"] = identity_sha256(
        mismatched["embedding_identity"]
    )
    mismatched_body = dict(mismatched)
    mismatched_body.pop("receipt_sha256")
    mismatched["receipt_sha256"] = identity_sha256(mismatched_body)
    (store_dir.parent / "source-current-selection.json").write_text(
        json.dumps(mismatched, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(
        runner.LockedSemanticResidualConstructionError,
        match="stored and query embedding identities differ",
    ):
        runner._verified_source_embedding_binding(  # noqa: SLF001
            scoped, source_vectors, query
        )
