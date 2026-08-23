from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from memory_condense.eval import fast_cav_prompts as cav_prompts
from memory_condense.eval.fast_cav_prompts import (
    ARM_IDS,
    ORIGINAL_CONTROL_KIND,
    TensorFreeStageOrder,
    build_fast_cav_prompt_population,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)


_QUESTION_ID = "fixture-question"
_STAGE_ID = STAGE_IDS[1]
_RAW_QUESTION = "Which codes were selected?"
_DATED_QUESTION = (
    "[Question asked at 2026/08/22 (Sat) 12:00]\n" + _RAW_QUESTION
)
_EVIDENCE = (
    FastEvidence("e-alpha", "source-alpha", "  Alpha was selected.\n"),
    FastEvidence("e-beta", "source-beta", "Beta was selected — exactly."),
    FastEvidence("e-gamma", "source-gamma", "Gamma was ruled out."),
)


def _stage(
    stage_id: str,
    *,
    max_prompt_tokens: int,
) -> FastRetrievalStage:
    legacy_context = "VERBOSE ARTIFACT CONTEXT THAT MUST NOT BE REUSED"
    legacy_messages = (
        FastProviderMessage("system", "legacy artifact system prompt"),
        FastProviderMessage(
            "user",
            f"{legacy_context}\n\nQuestion: {_DATED_QUESTION}\nShort answer:",
        ),
    )
    return FastRetrievalStage(
        stage_id=stage_id,
        stage_receipt_sha256=identity_sha256({"stage_id": stage_id}),
        matched_controls_sha256="1" * 64,
        evidence_projection_sha256=identity_sha256(
            [
                {
                    "evidence_id": row.evidence_id,
                    "source_id": row.source_id,
                    "text": row.text,
                }
                for row in _EVIDENCE
            ]
        ),
        context_sha256=quote_sha256(legacy_context),
        prompt_messages_sha256=identity_sha256(
            [
                {"role": row.role, "content": row.content}
                for row in legacy_messages
            ]
        ),
        context_token_proxy=50,
        max_context_token_proxy=7_000,
        prompt_token_proxy=100,
        max_prompt_token_proxy=max_prompt_tokens,
        responder_output_token_reserve=32,
        admission_status="added",
        added_evidence_ids=tuple(row.evidence_id for row in _EVIDENCE),
        context=legacy_context,
        evidence=_EVIDENCE,
        provider_messages=legacy_messages,
        feature_row_indices=(0, 1, 2),
    )


def _artifact(*, max_prompt_tokens: int = 8_000) -> FastRetrievalArtifact:
    stages = tuple(
        _stage(stage_id, max_prompt_tokens=max_prompt_tokens)
        for stage_id in STAGE_IDS
    )
    question_sha = quote_sha256(_RAW_QUESTION)
    dated_sha = quote_sha256(_DATED_QUESTION)
    final_user = stages[-1].provider_messages[-1]
    question = FastRetrievalQuestion(
        ordinal=0,
        question_id=_QUESTION_ID,
        question_sha256=question_sha,
        dated_question_sha256=dated_sha,
        predecessor_receipt_sha256="2" * 64,
        retrieval_receipt_sha256="3" * 64,
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
            for row in _EVIDENCE
        ),
        stages=stages,
    )
    return FastRetrievalArtifact(
        source_path="fixture/retrieval.json",
        raw_sha256="a" * 64,
        format="memory-condense-recall-guarded-cumulative-1m-retrieval-v1",
        campaign_format="memory-condense-recall-guarded-cumulative-1m-campaign-v1",
        population_identity_sha256="4" * 64,
        source_store_receipt_sha256="5" * 64,
        combined_store_receipt_sha256="6" * 64,
        retrieval_implementation_sha256="7" * 64,
        retrieval_policy_sha256="8" * 64,
        transcript_tokens=1_000_001,
        turn_count=5_400,
        retained_request_token_state_bytes=0,
        stage_ids=STAGE_IDS,
        questions=(question,),
    )


def _order(
    *,
    original: tuple[str, ...] = ("e-alpha", "e-beta", "e-gamma"),
    base: tuple[str, ...] = ("e-beta", "e-alpha", "e-gamma"),
    treatment: tuple[str, ...] = ("e-gamma", "e-beta", "e-alpha"),
    question_id: str = _QUESTION_ID,
    stage_id: str = _STAGE_ID,
) -> TensorFreeStageOrder:
    return TensorFreeStageOrder(
        question_id=question_id,
        stage_id=stage_id,
        original_evidence_ids=original,
        base_evidence_ids=base,
        treatment_evidence_ids=treatment,
        upstream_receipt_sha256="9" * 64,
    )


def test_renders_matched_catalog_arms_with_exact_alias_and_source_bindings() -> None:
    artifact = _artifact()

    population = build_fast_cav_prompt_population(
        artifact,
        [_order()],
        stage_ids=(_STAGE_ID,),
    )

    assert population.logical_prompt_count == 3
    assert population.unique_prompt_count == 3
    assert tuple(row.arm_id for row in population.logical_prompts) == ARM_IDS
    assert population.selected_stage_ids == (_STAGE_ID,)
    receipt = population.stage_receipts[0]
    assert receipt.artifact_sha256 == artifact.raw_sha256
    assert receipt.original_control_kind == ORIGINAL_CONTROL_KIND
    assert receipt.upstream_order_receipt_sha256 == "9" * 64
    assert receipt.retained_tensor_bytes == population.retained_tensor_bytes == 0
    assert tuple(
        (row.alias, row.evidence_id, row.source_id, row.text_sha256)
        for row in receipt.alias_bindings
    ) == (
        ("E001", "e-alpha", "source-alpha", quote_sha256(_EVIDENCE[0].text)),
        ("E002", "e-beta", "source-beta", quote_sha256(_EVIDENCE[1].text)),
        ("E003", "e-gamma", "source-gamma", quote_sha256(_EVIDENCE[2].text)),
    )

    contexts = [
        population.unique_prompts[row.unique_prompt_ordinal].context
        for row in population.logical_prompts
    ]
    assert contexts[0].index("[E001]") < contexts[0].index("[E002]")
    assert contexts[1].index("[E002]") < contexts[1].index("[E001]")
    assert contexts[2].index("[E003]") < contexts[2].index("[E002]")
    for context in contexts:
        for evidence in _EVIDENCE:
            assert evidence.text in context
            assert f'source_id="{evidence.source_id}"' in context
        assert "evidence_id=" not in context

    exact_rows = {
        row.evidence_id: (row.source_id, row.text) for row in _EVIDENCE
    }
    for arm in population.logical_prompts:
        assert set(arm.evidence_ids) == set(exact_rows)
        assert arm.alias_order == tuple(
            next(
                binding.alias
                for binding in receipt.alias_bindings
                if binding.evidence_id == evidence_id
            )
            for evidence_id in arm.evidence_ids
        )


def test_uses_exact_qa_prompts_dated_question_and_fresh_catalog_control() -> None:
    artifact = _artifact()
    population = build_fast_cav_prompt_population(
        artifact,
        [_order()],
        stage_ids=(_STAGE_ID,),
    )
    original = population.unique_prompts[0]
    mappings = list(original.as_mappings())

    assert mappings[0] == {"role": "system", "content": QA_SYSTEM_PROMPT}
    assert mappings[1] == {
        "role": "user",
        "content": QA_USER_TEMPLATE.format(
            context=original.context,
            question=_DATED_QUESTION,
        ),
    }
    assert _DATED_QUESTION in mappings[1]["content"]
    assert _RAW_QUESTION in mappings[1]["content"]
    artifact_messages = artifact.questions[0].stage(_STAGE_ID).provider_messages
    assert original.messages != artifact_messages
    assert "VERBOSE ARTIFACT CONTEXT THAT MUST NOT BE REUSED" not in original.context
    assert population.stage_receipts[0].original_control_kind == (
        "canonical_evidence_catalog_original_order_not_artifact_provider_prompt"
    )


def test_hashes_and_token_counts_recompute_exactly() -> None:
    population = build_fast_cav_prompt_population(
        _artifact(),
        [_order()],
        stage_ids=(_STAGE_ID,),
    )

    for unique in population.unique_prompts:
        mappings = list(unique.as_mappings())
        assert unique.messages_sha256 == identity_sha256(mappings)
        assert unique.context_sha256 == quote_sha256(unique.context)
        assert unique.prompt_token_proxy == count_chat_prompt_token_proxy(mappings)
        assert unique.prompt_token_proxy <= 8_000
    for arm in population.logical_prompts:
        body = arm.identity_payload(include_sha256=False)
        assert arm.arm_prompt_sha256 == identity_sha256(body)
        assert arm.messages_sha256 == population.unique_prompts[
            arm.unique_prompt_ordinal
        ].messages_sha256
    for receipt in population.stage_receipts:
        assert receipt.receipt_sha256 == identity_sha256(
            receipt.identity_payload(include_sha256=False)
        )
    assert population.prompt_population_sha256 == identity_sha256(
        population.identity_payload(include_sha256=False)
    )


@pytest.mark.parametrize(
    ("artifact_cap", "observed_tokens", "expected_cap"),
    [(50, 51, 50), (9_000, 8_001, 8_000)],
)
def test_recounts_and_enforces_minimum_hard_prompt_cap(
    monkeypatch: pytest.MonkeyPatch,
    artifact_cap: int,
    observed_tokens: int,
    expected_cap: int,
) -> None:
    monkeypatch.setattr(
        cav_prompts,
        "count_chat_prompt_token_proxy",
        lambda _messages: observed_tokens,
    )

    with pytest.raises(
        ValueError,
        match=rf"{observed_tokens} > {expected_cap}",
    ):
        build_fast_cav_prompt_population(
            _artifact(max_prompt_tokens=artifact_cap),
            [_order()],
            stage_ids=(_STAGE_ID,),
        )


def test_identical_arm_messages_dedupe_without_losing_logical_mapping() -> None:
    original = tuple(row.evidence_id for row in _EVIDENCE)
    population = build_fast_cav_prompt_population(
        _artifact(),
        [_order(original=original, base=original, treatment=original)],
        stage_ids=(_STAGE_ID,),
    )

    assert population.logical_prompt_count == 3
    assert population.unique_prompt_count == 1
    assert tuple(row.arm_id for row in population.logical_prompts) == ARM_IDS
    assert tuple(
        row.unique_prompt_ordinal for row in population.logical_prompts
    ) == (0, 0, 0)
    assert len(population.logical_message_population) == 3
    assert (
        population.logical_message_population[0]
        == population.logical_message_population[1]
        == population.logical_message_population[2]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base", ("e-alpha", "e-beta"), "base_evidence_ids.*exact evidence set"),
        (
            "treatment",
            ("e-alpha", "e-beta", "e-extra"),
            "treatment_evidence_ids.*exact evidence set",
        ),
        (
            "base",
            ("e-alpha", "e-alpha", "e-gamma"),
            "base_evidence_ids.*unique",
        ),
    ],
)
def test_order_input_rejects_changed_evidence_membership(
    field: str,
    value: tuple[str, ...],
    message: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        _order(**kwargs)


def test_original_order_must_equal_exact_artifact_stage_order() -> None:
    changed_original = ("e-beta", "e-alpha", "e-gamma")
    with pytest.raises(ValueError, match="original evidence order.*exact artifact"):
        build_fast_cav_prompt_population(
            _artifact(),
            [
                _order(
                    original=changed_original,
                    base=changed_original,
                    treatment=changed_original,
                )
            ],
            stage_ids=(_STAGE_ID,),
        )


def test_stage_order_inputs_must_exactly_cover_selected_question_stages() -> None:
    with pytest.raises(ValueError, match="do not exactly cover"):
        build_fast_cav_prompt_population(
            _artifact(),
            [],
            stage_ids=(_STAGE_ID,),
        )
    with pytest.raises(ValueError, match="do not exactly cover"):
        build_fast_cav_prompt_population(
            _artifact(),
            [_order(stage_id=STAGE_IDS[2])],
            stage_ids=(_STAGE_ID,),
        )


def test_order_input_is_minimal_tensor_free_and_rejects_retained_bytes() -> None:
    assert {field.name for field in fields(TensorFreeStageOrder)} == {
        "question_id",
        "stage_id",
        "original_evidence_ids",
        "base_evidence_ids",
        "treatment_evidence_ids",
        "upstream_receipt_sha256",
        "retained_tensor_bytes",
    }
    assert not any(
        "score" in field.name or "tensor" in field.name and field.name != "retained_tensor_bytes"
        for field in fields(TensorFreeStageOrder)
    )
    with pytest.raises(ValueError, match="zero tensor bytes"):
        TensorFreeStageOrder(
            question_id=_QUESTION_ID,
            stage_id=_STAGE_ID,
            original_evidence_ids=("e-alpha",),
            base_evidence_ids=("e-alpha",),
            treatment_evidence_ids=("e-alpha",),
            upstream_receipt_sha256="9" * 64,
            retained_tensor_bytes=4,
        )


def test_module_import_does_not_import_torch_or_router() -> None:
    code = r"""
import sys
import memory_condense.eval.fast_cav_prompts
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
