from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from memory_condense.associations.association_models import AssociationArtifact
from memory_condense.associations.association_store import AssociationStore
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.fast_hebbian_prompts import (
    ARM_IDS,
    FAST_HEBBIAN_ALIAS_BINDING_FORMAT,
    FAST_HEBBIAN_CATALOG_FORMAT,
    FAST_HEBBIAN_PROMPT_POPULATION_FORMAT,
    HALF_LIFE_TURNS,
    MAX_CANDIDATES,
    MAX_PROMPT_TOKEN_INCREASE,
    MAX_SEED_CONCEPTS,
    MIN_SCORE,
    S0_STAGE_ID,
    FastHebbianPromptValidationError,
    build_fast_hebbian_prompt_population,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    QUESTION_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)
from memory_condense.persistence.db import Database


_HISTORY_RECEIPT_SHA = "9" * 64
_RAW_QUESTION = "Which item was linked?"
_DATED_QUESTION = "[Question asked at 2026/08/22 (Sat) 12:00]\n" + _RAW_QUESTION


def _learning_policy() -> dict[str, object]:
    return {
        "format": "memory-condense.hebbian-learning-policy.v1",
        "learning_rate": 1.0,
        "half_life_turns": 200.0,
        "max_concepts_per_event": 12,
        "max_degree": 32,
        "min_edge_score": 0.0,
        "retain_all_event_receipts": True,
    }


def _association_artifact(
    fast_artifact: FastRetrievalArtifact,
) -> AssociationArtifact:
    policy_sha = identity_sha256(_learning_policy())
    created = AssociationArtifact.create(
        model_id="memory-condense/hebbian-rank-coaccess-v1",
        checkpoint_id="external-scalar-coaccess-v1",
        prefix_layers=1,
        head_layer=0,
        cav_layer=None,
        concept_names=(),
        head_count=1,
        metadata={
            "format": "memory-condense.hebbian-association-namespace.v1",
            "history_artifact_sha256": "f" * 64,
            "history_receipt_sha256": _HISTORY_RECEIPT_SHA,
            "source_store_receipt_sha256": (
                fast_artifact.combined_store_receipt_sha256
            ),
            "learning_policy_sha256": policy_sha,
        },
    )
    return replace(created, created_at="1970-01-01T00:00:00+00:00")


def _association_payload(artifact: AssociationArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "model_id": artifact.model_id,
        "checkpoint_id": artifact.checkpoint_id,
        "prefix_layers": artifact.prefix_layers,
        "head_layer": artifact.head_layer,
        "cav_layer": artifact.cav_layer,
        "concept_names": list(artifact.concept_names),
        "head_count": artifact.head_count,
        "created_at": artifact.created_at,
        "metadata": dict(artifact.metadata),
    }


def _stage(
    evidence: tuple[FastEvidence, ...],
    stage_id: str,
    protected_chunk_ids: tuple[str, ...],
) -> FastRetrievalStage:
    context = "legacy context"
    messages = (
        FastProviderMessage("system", "legacy system"),
        FastProviderMessage(
            "user",
            f"{context}\n\nQuestion: {_DATED_QUESTION}\nShort answer:",
        ),
    )
    return FastRetrievalStage(
        stage_id=stage_id,
        stage_receipt_sha256=identity_sha256(
            {"stage_id": stage_id, "evidence": [row.evidence_id for row in evidence]}
        ),
        matched_controls_sha256="1" * 64,
        evidence_projection_sha256=identity_sha256(
            {
                "protected_excerpts": [
                    {
                        "chunk_id": chunk_id,
                        "source_id": row.source_id,
                        "text_sha256": quote_sha256(row.text),
                    }
                    for chunk_id, row in zip(
                        protected_chunk_ids,
                        evidence,
                        strict=True,
                    )
                ],
                "admitted_atoms": [],
            }
        ),
        context_sha256=quote_sha256(context),
        prompt_messages_sha256=identity_sha256(
            [{"role": row.role, "content": row.content} for row in messages]
        ),
        context_token_proxy=10,
        max_context_token_proxy=7_500,
        prompt_token_proxy=50,
        max_prompt_token_proxy=8_000,
        responder_output_token_reserve=64,
        admission_status="added",
        added_evidence_ids=tuple(row.evidence_id for row in evidence),
        context=context,
        evidence=evidence,
        provider_messages=messages,
        feature_row_indices=tuple(range(len(evidence))),
    )


def _fast_artifact(
    *,
    evidence_texts: tuple[str, str] = (
        "Alpha is the durable anchor.",
        "Beta is the replaceable tail evidence.",
    ),
    source_ids: tuple[str, str] = ("source-alpha", "source-beta"),
    protected_chunk_ids: tuple[str, ...] = ("chunk-alpha", "chunk-beta"),
) -> FastRetrievalArtifact:
    evidence = tuple(
        FastEvidence(f"evidence-{index}", source_id, content)
        for index, (source_id, content) in enumerate(
            zip(source_ids, evidence_texts, strict=True),
            start=1,
        )
    )
    stages = tuple(
        _stage(evidence, stage_id, protected_chunk_ids)
        for stage_id in STAGE_IDS
    )
    question_sha = quote_sha256(_RAW_QUESTION)
    dated_sha = quote_sha256(_DATED_QUESTION)
    final_user = stages[-1].provider_messages[-1]
    question = FastRetrievalQuestion(
        ordinal=0,
        question_id="fixture-question",
        question_sha256=question_sha,
        dated_question_sha256=dated_sha,
        predecessor_receipt_sha256="2" * 64,
        retrieval_receipt_sha256="3" * 64,
        protected_chunk_ids=protected_chunk_ids,
        retained_request_token_state_bytes=0,
        question=_RAW_QUESTION,
        dated_question=_DATED_QUESTION,
        final_user_message=final_user,
        question_parse_receipt=FastQuestionParseReceipt(
            framing="memory-condense-qa-user-template-v1",
            source_stage_id=STAGE_IDS[-1],
            provider_message_index=1,
            provider_message_sha256=quote_sha256(final_user.content),
            question_marker_occurrences=1,
            matching_framing_candidates=1,
            dated_question_sha256=dated_sha,
            question_sha256=question_sha,
            question_form="dated_header",
        ),
        feature_rows=tuple(
            FastFeatureRow(
                question=_RAW_QUESTION,
                evidence_text=row.text,
                row_sha256=identity_sha256(
                    {"question": _RAW_QUESTION, "evidence_text": row.text}
                ),
            )
            for row in evidence
        ),
        stages=stages,
    )
    return FastRetrievalArtifact(
        source_path="fixture/retrieval.json",
        raw_sha256="a" * 64,
        format=RETRIEVAL_FORMAT,
        campaign_format=CAMPAIGN_FORMAT,
        population_identity_sha256="4" * 64,
        source_store_receipt_sha256="5" * 64,
        combined_store_receipt_sha256="6" * 64,
        retrieval_implementation_sha256="7" * 64,
        retrieval_policy_sha256="8" * 64,
        transcript_tokens=1_000_001,
        turn_count=3,
        retained_request_token_state_bytes=0,
        stage_ids=STAGE_IDS,
        questions=(question,),
    )


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _derived_store(
    tmp_path: Path,
    fast_artifact: FastRetrievalArtifact,
    *,
    reinforce: bool = True,
    chunk_texts: tuple[str, str, str] = (
        "Alpha is the durable anchor.",
        "Beta is the replaceable tail evidence with several extra words.",
        "Linked neighbor.",
    ),
    token_counts: tuple[int, int, int] = (6, 12, 3),
) -> tuple[Path, str, str]:
    root = tmp_path / "derived"
    root.mkdir(parents=True)
    database_path = root / "memory.db"
    association = _association_artifact(fast_artifact)
    chunk_ids = ("chunk-alpha", "chunk-beta", "chunk-neighbor")
    source_ids = ("source-alpha", "source-beta", "source-neighbor")
    with Database(database_path) as database:
        for ordinal, (chunk_id, source_id, content, token_count) in enumerate(
            zip(
                chunk_ids,
                source_ids,
                chunk_texts,
                token_counts,
                strict=True,
            ),
            start=1,
        ):
            turn_id = f"turn-{ordinal}"
            database.execute(
                "INSERT INTO turns "
                "(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES (?, 'user', ?, ?, '2026-08-22T00:00:00+00:00', ?)",
                (turn_id, content, source_id, ordinal),
            )
            database.execute(
                "INSERT INTO chunks "
                "(chunk_id, turn_id, text, start_char, end_char, token_count, "
                "embedding, hnsw_label) VALUES (?, ?, ?, 0, ?, ?, NULL, ?)",
                (chunk_id, turn_id, content, len(content), token_count, ordinal),
            )
        database.commit()
        store = AssociationStore(database)
        stored = store.register_artifact(association)
        if reinforce:
            store.reinforce_retrieval_coaccess(
                stored.artifact_id,
                "history-event-1",
                {"chunk-alpha": 1.0, "chunk-neighbor": 1.0},
                now_turn=3,
            )

    index_path = root / "hnsw_index.bin"
    index_path.write_bytes(b"fixture-index")
    association_sha = identity_sha256(_association_payload(association))
    policy = _learning_policy()
    policy_sha = identity_sha256(policy)
    manifest_body: dict[str, object] = {
        "format": "memory-condense.hebbian-derived-store.v1",
        "source_database_sha256": "b" * 64,
        "source_index_sha256": "c" * 64,
        "source_store_receipt_sha256": (
            fast_artifact.combined_store_receipt_sha256
        ),
        "source_turn_sequence_sha256": "d" * 64,
        "source_chunk_sequence_sha256": "e" * 64,
        "derived_database_sha256": file_sha256(database_path),
        "derived_index_sha256": file_sha256(index_path),
        "derived_turn_sequence_sha256": "d" * 64,
        "derived_chunk_sequence_sha256": "e" * 64,
        "history_artifact_sha256": "f" * 64,
        "history_receipt_sha256": _HISTORY_RECEIPT_SHA,
        "implementation_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
        "learning_policy": policy,
        "learning_policy_sha256": policy_sha,
        "association_artifact_id": association.artifact_id,
        "association_artifact_sha256": association_sha,
        "events_offered": int(reinforce),
        "events_applied": int(reinforce),
        "graph_nodes": 2 if reinforce else 0,
        "graph_edges": 1 if reinforce else 0,
        "graph_event_receipts": int(reinforce),
        "retained_request_token_state_bytes": 0,
    }
    receipt_sha = identity_sha256(manifest_body)
    manifest = {**manifest_body, "receipt_sha256": receipt_sha}
    (root / "hebbian-derived-store.json").write_bytes(_canonical_bytes(manifest))
    return root, association.artifact_id, receipt_sha


def _build(
    artifact: FastRetrievalArtifact,
    store_path: Path,
    association_artifact_id: str,
    derived_receipt_sha256: str,
):
    return build_fast_hebbian_prompt_population(
        artifact,
        store_path,
        association_artifact_id=association_artifact_id,
        history_receipt_sha256=_HISTORY_RECEIPT_SHA,
        derived_store_receipt_sha256=derived_receipt_sha256,
    )


def test_builds_matched_s0_h1_replacement_with_exact_provenance(tmp_path: Path) -> None:
    artifact = _fast_artifact()
    store_path, association_id, derived_sha = _derived_store(tmp_path, artifact)

    population = _build(artifact, store_path, association_id, derived_sha)

    assert population.format == FAST_HEBBIAN_PROMPT_POPULATION_FORMAT
    assert population.stage_id == S0_STAGE_ID
    assert population.logical_prompt_count == population.unique_prompt_count == 2
    assert tuple(row.arm_id for row in population.logical_prompts) == ARM_IDS
    base, h1 = population.logical_prompts
    assert base.chunk_ids == ("chunk-alpha", "chunk-beta")
    assert h1.chunk_ids == ("chunk-alpha", "chunk-neighbor")
    assert h1.prompt_token_proxy <= base.prompt_token_proxy <= 8_000
    assert base.alias_order[0] == h1.alias_order[0]

    receipt = population.question_receipts[0]
    assert receipt.catalog_format == FAST_HEBBIAN_CATALOG_FORMAT
    assert receipt.retrieval_artifact_sha256 == artifact.raw_sha256
    assert receipt.source_store_receipt_sha256 == (
        artifact.combined_store_receipt_sha256
    )
    assert receipt.predecessor_receipt_sha256 == "2" * 64
    assert receipt.retrieval_receipt_sha256 == "3" * 64
    assert receipt.stage_receipt_sha256 == artifact.questions[0].stage(
        S0_STAGE_ID
    ).stage_receipt_sha256
    assert receipt.s0_evidence_projection_sha256 == artifact.questions[0].stage(
        S0_STAGE_ID
    ).evidence_projection_sha256
    assert receipt.history_receipt_sha256 == _HISTORY_RECEIPT_SHA
    assert receipt.derived_store_receipt_sha256 == derived_sha
    assert receipt.association_artifact_id == association_id
    assert receipt.protected_chunk_ids == ("chunk-alpha", "chunk-beta")
    assert receipt.s0_evidence_ids == ("evidence-1", "evidence-2")
    assert receipt.effective_status == receipt.expansion_receipt.status == "replaced"
    assert receipt.expansion_receipt.hebbian_slots == 1
    assert receipt.expansion_receipt.max_seed_concepts == MAX_SEED_CONCEPTS
    assert receipt.expansion_receipt.max_candidates == MAX_CANDIDATES
    assert receipt.expansion_receipt.half_life_turns == HALF_LIFE_TURNS
    assert receipt.expansion_receipt.min_score == MIN_SCORE
    assert (
        receipt.expansion_receipt.max_prompt_token_increase
        == MAX_PROMPT_TOKEN_INCREASE
    )
    assert receipt.expansion_receipt.removed_chunk_ids == ("chunk-beta",)
    assert receipt.expansion_receipt.added_chunk_ids == ("chunk-neighbor",)
    assert receipt.retained_request_token_state_bytes == 0

    bindings = {row.chunk_id: row for row in receipt.alias_bindings}
    assert set(bindings) == {"chunk-alpha", "chunk-beta", "chunk-neighbor"}
    assert bindings["chunk-alpha"].source_id == "source-alpha"
    assert bindings["chunk-neighbor"].origin == "hebbian_candidate"
    assert all(
        row.format == FAST_HEBBIAN_ALIAS_BINDING_FORMAT
        for row in bindings.values()
    )
    base_messages, h1_messages = population.logical_message_population
    assert f"[{bindings['chunk-alpha'].alias}]" in base_messages[1]["content"]
    assert f"[{bindings['chunk-alpha'].alias}]" in h1_messages[1]["content"]
    assert "Linked neighbor." not in base_messages[1]["content"]
    assert "Linked neighbor." in h1_messages[1]["content"]

    # Candidate discovery cannot renumber or otherwise perturb the base arm.
    control_store, control_association_id, control_derived_sha = _derived_store(
        tmp_path / "control",
        artifact,
        reinforce=False,
    )
    control = _build(
        artifact,
        control_store,
        control_association_id,
        control_derived_sha,
    )
    assert population.logical_message_population[0] == (
        control.logical_message_population[0]
    )


def test_exact_render_overage_rolls_back_to_byte_identical_base(tmp_path: Path) -> None:
    artifact = _fast_artifact(evidence_texts=("A.", "B."))
    store_path, association_id, derived_sha = _derived_store(
        tmp_path,
        artifact,
        chunk_texts=(
            "A raw anchor.",
            "The raw tail is large but its sealed S0 projection is tiny.",
            "This learned neighbor has many words and therefore makes the exact "
            "rendered provider prompt materially larger than the tiny S0 tail.",
        ),
        # Core H1 admission uses these authoritative stored chunk totals and
        # therefore accepts 1 + 50 <= 1 + 100.  Exact rendered admission then
        # catches the larger provider prompt.
        token_counts=(1, 100, 50),
    )

    population = _build(artifact, store_path, association_id, derived_sha)

    receipt = population.question_receipts[0]
    assert receipt.expansion_receipt.status == "replaced"
    assert receipt.effective_status == "exact_prompt_budget_rollback"
    assert receipt.effective_h1_chunk_ids == receipt.protected_chunk_ids
    assert receipt.base_messages_sha256 == receipt.h1_messages_sha256
    assert population.unique_prompt_count == 1
    assert population.logical_prompts[0].unique_prompt_ordinal == 0
    assert population.logical_prompts[1].unique_prompt_ordinal == 0
    assert population.logical_message_population[0] == (
        population.logical_message_population[1]
    )
    assert any(
        row.chunk_id == "chunk-neighbor" for row in receipt.alias_bindings
    )


def test_no_neighbor_deduplicates_base_and_h1_without_provider_work(
    tmp_path: Path,
) -> None:
    artifact = _fast_artifact()
    store_path, association_id, derived_sha = _derived_store(
        tmp_path,
        artifact,
        reinforce=False,
    )

    population = _build(artifact, store_path, association_id, derived_sha)

    assert population.logical_prompt_count == 2
    assert population.unique_prompt_count == 1
    assert population.question_receipts[0].effective_status == "no_neighbor"
    assert tuple(row.unique_prompt_ordinal for row in population.logical_prompts) == (
        0,
        0,
    )


def test_hashes_and_exact_prompt_counts_recompute_and_seals_reject_tampering(
    tmp_path: Path,
) -> None:
    artifact = _fast_artifact()
    store_path, association_id, derived_sha = _derived_store(tmp_path, artifact)
    population = _build(artifact, store_path, association_id, derived_sha)

    for unique in population.unique_prompts:
        mappings = unique.as_mappings()
        assert unique.messages_sha256 == identity_sha256(mappings)
        assert unique.prompt_token_proxy == count_chat_prompt_token_proxy(mappings)
    for arm in population.logical_prompts:
        assert arm.arm_prompt_sha256 == identity_sha256(
            arm.identity_payload(include_receipt=False)
        )
    receipt = population.question_receipts[0]
    assert receipt.receipt_sha256 == identity_sha256(
        receipt.identity_payload(include_receipt=False)
    )
    assert population.prompt_population_sha256 == identity_sha256(
        population.identity_payload(include_receipt=False)
    )
    with pytest.raises(FrozenInstanceError):
        receipt.effective_status = "no_neighbor"  # type: ignore[misc]
    with pytest.raises(ValueError, match="does not match"):
        replace(receipt, history_receipt_sha256="8" * 64)
    changed_unique = replace(
        population.unique_prompts[0],
        context_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="logical prompt metadata"):
        replace(
            population,
            unique_prompts=(changed_unique, *population.unique_prompts[1:]),
            prompt_population_sha256="",
        )


def test_fails_closed_on_coordinate_source_and_derived_receipt_mismatches(
    tmp_path: Path,
) -> None:
    artifact = _fast_artifact()
    store_path, association_id, derived_sha = _derived_store(tmp_path, artifact)

    wrong_count_question = replace(
        artifact.questions[0],
        protected_chunk_ids=("chunk-alpha",),
    )
    wrong_count = replace(artifact, questions=(wrong_count_question,))
    with pytest.raises(ValueError, match="count must exactly match"):
        _build(wrong_count, store_path, association_id, derived_sha)

    wrong_coordinate_question = replace(
        artifact.questions[0],
        protected_chunk_ids=("chunk-neighbor", "chunk-beta"),
    )
    wrong_coordinate = replace(
        artifact,
        questions=(wrong_coordinate_question,),
    )
    with pytest.raises(ValueError, match="sealed S0 evidence projection"):
        _build(wrong_coordinate, store_path, association_id, derived_sha)

    wrong_source = _fast_artifact(source_ids=("source-wrong", "source-beta"))
    with pytest.raises(ValueError, match="changed durable source"):
        _build(wrong_source, store_path, association_id, derived_sha)

    with pytest.raises(ValueError, match="derived_store_receipt_sha256"):
        build_fast_hebbian_prompt_population(
            artifact,
            store_path,
            association_artifact_id=association_id,
            history_receipt_sha256=_HISTORY_RECEIPT_SHA,
            derived_store_receipt_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="association artifact ID"):
        build_fast_hebbian_prompt_population(
            artifact,
            store_path,
            association_artifact_id="assoc-not-the-sealed-one",
            history_receipt_sha256=_HISTORY_RECEIPT_SHA,
            derived_store_receipt_sha256=derived_sha,
        )


def test_receipts_are_text_free_and_retain_zero_transformer_state(
    tmp_path: Path,
) -> None:
    artifact = _fast_artifact()
    store_path, association_id, derived_sha = _derived_store(tmp_path, artifact)
    population = _build(artifact, store_path, association_id, derived_sha)

    receipt_types = (
        type(population.logical_prompts[0]),
        type(population.question_receipts[0]),
        type(population.question_receipts[0].alias_bindings[0]),
    )
    names = {field.name for kind in receipt_types for field in fields(kind)}
    forbidden = {
        "text",
        "content",
        "messages",
        "query",
        "question",
        "gold",
        "answer",
        "token_ids",
        "kv_cache",
        "hidden_states",
        "residual",
    }
    assert not names & forbidden
    assert population.retained_request_token_state_bytes == 0
    assert all(
        row.retained_request_token_state_bytes == 0
        for row in population.logical_prompts
    )
    assert all(
        row.retained_request_token_state_bytes == 0
        for row in population.question_receipts
    )


def test_module_import_does_not_import_torch_or_an_llm_router() -> None:
    code = r"""
import sys
import memory_condense.eval.fast_hebbian_prompts
assert "torch" not in sys.modules
assert "memory_condense.search.fusion.latent_router" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_fixture_uses_the_current_sealed_question_format_constant() -> None:
    # Guard against accidentally building this unit fixture around a stale
    # question schema while the public fast adapter evolves.
    assert QUESTION_FORMAT == "memory-condense-recall-guarded-cumulative-1m-query-v1"
