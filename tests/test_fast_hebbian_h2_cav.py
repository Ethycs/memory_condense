from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_hebbian_h2_cav import (
    FAST_H2_CAV_LAYER_IDS,
    FAST_H2_CAV_SOURCE_STAGE_ID,
    bind_fast_h2_cav_input,
    build_fast_h2_cav_preflight,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import STAGE_IDS
from tests.test_fast_cav_feature_session import _FakeEncoder, _FakeRouter
from tests.test_fast_hebbian_h2 import _build


def _contains_tensor(value: object) -> bool:
    import torch

    if type(value) is torch.Tensor:
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, row.name)) for row in fields(value))
    if isinstance(value, (tuple, list, dict)):
        children = value.values() if isinstance(value, dict) else value
        return any(_contains_tensor(row) for row in children)
    return False


def test_h2_append_flows_through_genuine_cav_and_actual_prompt_preflight(
    tmp_path,
) -> None:
    retrieval, _history, _derived, h2 = _build(tmp_path)
    encoder = _FakeEncoder()
    router = _FakeRouter()

    result = build_fast_h2_cav_preflight(
        retrieval,
        h2,
        encoder=encoder,
        router=router,
        layer=2,
        batch_size=8,
    )

    bound = result.bound_input
    question = bound.artifact_view.questions[0]
    source_s3 = retrieval.questions[0].stage(STAGE_IDS[-1])
    h2_receipt = h2.question_receipts[0]
    overlay_s3 = question.stage(STAGE_IDS[-1])
    feature = result.feature_session.stage(question.question_id, STAGE_IDS[-1])
    links = feature.links
    synthesis_receipt = result.synthesis.stage_receipts[0]
    coordinates = tuple(
        (row.evidence_id, row.source_id, quote_sha256(row.text))
        for row in h2.final_evidence[0]
    )

    assert bound.source_stage_id == FAST_H2_CAV_SOURCE_STAGE_ID == STAGE_IDS[-1]
    assert bound.layer_ids == FAST_H2_CAV_LAYER_IDS
    assert bound.source_retrieval_sha256 == retrieval.raw_sha256
    assert bound.overlay_sha256 != retrieval.raw_sha256
    assert overlay_s3.evidence[: len(source_s3.evidence)] == source_s3.evidence
    assert overlay_s3.evidence == h2.final_evidence[0]
    assert overlay_s3.stage_receipt_sha256 == h2_receipt.receipt_sha256
    assert feature.source_stage_receipt_sha256 == h2_receipt.receipt_sha256
    assert synthesis_receipt.source_stage_receipt_sha256 == h2_receipt.receipt_sha256
    assert links is not None
    assert coordinates == tuple(
        zip(
            feature.evidence_ids,
            feature.source_ids,
            feature.evidence_text_sha256s,
            strict=True,
        )
    )
    assert coordinates == tuple(
        zip(
            links.evidence_ids,
            links.source_ids,
            links.evidence_text_sha256s,
            strict=True,
        )
    )
    assert tuple(row.evidence_id for row in synthesis_receipt.aliases) == tuple(
        row.evidence_id for row in h2.final_evidence[0]
    )
    assert all(
        row.prompt_token_proxy <= row.hard_prompt_token_cap == 8_000
        for row in result.synthesis.prompts
    )
    appended_text = h2.final_evidence[0][-1].text
    assert all(
        messages[1]["content"].count(appended_text) == 1
        for messages in result.synthesis.logical_message_population
    )
    assert h2.provider_calls == 0
    assert len(encoder.calls) == 1
    assert result.feature_session.persisted_token_state_bytes == 0
    assert result.synthesis.persisted_token_state_bytes == 0
    assert links.persisted_token_state_bytes == 0
    assert not _contains_tensor(result)


def test_h2_cav_bridge_rejects_another_retrieval_identity(tmp_path) -> None:
    retrieval, _history, _derived, h2 = _build(tmp_path)

    with pytest.raises(ValueError, match="retrieval binding"):
        bind_fast_h2_cav_input(replace(retrieval, raw_sha256="0" * 64), h2)
