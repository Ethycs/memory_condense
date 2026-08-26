from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_cav_feature_artifact import (
    FAST_CAV_FEATURE_ARTIFACT_FORMAT,
    LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT,
    FastCAVFeatureArtifactError,
    load_fast_cav_feature_artifact,
)
from memory_condense.eval.fast_cav_feature_session import (
    FastCAVFeatureSessionReceipt,
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_cav_links import FastCAVLinkReceipt
from memory_condense.eval.fast_cav_prompts import TensorFreeStageOrder
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
    load_fast_retrieval_artifact,
)
from memory_condense.search.fusion.fixed_cav_router import (
    FixedCAVForward,
    FixedCAVRuntimeReceipt,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _quote(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _publish(tmp_path: Path, name: str, value: dict) -> tuple[Path, str]:
    path = tmp_path / name
    raw = _canonical_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return path, digest


def _feature_row(question: str, evidence_text: str) -> FastFeatureRow:
    return FastFeatureRow(
        question=question,
        evidence_text=evidence_text,
        row_sha256=identity_sha256(
            {
                "format": "memory-condense-fast-feature-row-v1",
                "question": question,
                "evidence_text": evidence_text,
            }
        ),
    )


def _retrieval_artifact() -> FastRetrievalArtifact:
    question_text = "Which two codes were selected?"
    dated_question = "[Question asked at 2026/08/23 (Sun) 12:00]\n" + question_text
    alpha = FastEvidence("e-alpha", "source-alpha", "Alpha was selected.")
    beta = FastEvidence("e-beta", "source-beta", "Beta was selected.")
    beta_alias = FastEvidence("e-beta-alias", "source-beta-alias", beta.text)
    ladder = (
        (alpha,),
        (alpha, beta),
        (alpha, beta, beta_alias),
        (alpha, beta, beta_alias),
    )
    feature_rows = (
        _feature_row(question_text, alpha.text),
        _feature_row(question_text, beta.text),
    )
    stages: list[FastRetrievalStage] = []
    prior_count = 0
    for ordinal, (stage_id, evidence) in enumerate(
        zip(STAGE_IDS, ladder, strict=True)
    ):
        context = "\n".join(row.text for row in evidence)
        user_content = (
            "Retrieved excerpts from the conversation history:\n"
            f"{context}\n\nQuestion: {dated_question}\nShort answer:"
        )
        stages.append(
            FastRetrievalStage(
                stage_id=stage_id,
                stage_receipt_sha256=_digest(f"stage-{ordinal}"),
                matched_controls_sha256=_digest("controls"),
                evidence_projection_sha256=_digest(f"projection-{ordinal}"),
                context_sha256=_quote(context),
                prompt_messages_sha256=_digest(f"messages-{ordinal}"),
                context_token_proxy=len(context),
                max_context_token_proxy=1000,
                prompt_token_proxy=len(user_content),
                max_prompt_token_proxy=2000,
                responder_output_token_reserve=64,
                admission_status="root" if ordinal == 0 else "added",
                added_evidence_ids=tuple(
                    row.evidence_id for row in evidence[prior_count:]
                ),
                context=context,
                evidence=evidence,
                provider_messages=(
                    FastProviderMessage("system", "Use only supplied evidence."),
                    FastProviderMessage("user", user_content),
                ),
                feature_row_indices=tuple(
                    0 if row.text == alpha.text else 1 for row in evidence
                ),
            )
        )
        prior_count = len(evidence)
    final_user_message = stages[-1].provider_messages[-1]
    question = FastRetrievalQuestion(
        ordinal=0,
        question_id="question-0",
        question_sha256=_quote(question_text),
        dated_question_sha256=_quote(dated_question),
        predecessor_receipt_sha256=_digest("predecessor"),
        retrieval_receipt_sha256=_digest("retrieval-receipt"),
        protected_chunk_ids=(alpha.evidence_id,),
        retained_request_token_state_bytes=0,
        question=question_text,
        dated_question=dated_question,
        final_user_message=final_user_message,
        question_parse_receipt=FastQuestionParseReceipt(
            framing="memory-condense-qa-user-template-v1",
            source_stage_id=STAGE_IDS[-1],
            provider_message_index=1,
            provider_message_sha256=_quote(final_user_message.content),
            question_marker_occurrences=1,
            matching_framing_candidates=1,
            dated_question_sha256=_quote(dated_question),
            question_sha256=_quote(question_text),
            question_form="dated_header",
        ),
        feature_rows=feature_rows,
        stages=tuple(stages),
    )
    return FastRetrievalArtifact(
        source_path="sealed/retrieval.json",
        raw_sha256=_digest("retrieval-artifact"),
        format=RETRIEVAL_FORMAT,
        campaign_format=CAMPAIGN_FORMAT,
        population_identity_sha256=_digest("population"),
        source_store_receipt_sha256=_digest("source-store"),
        combined_store_receipt_sha256=_digest("combined-store"),
        retrieval_implementation_sha256=_digest("retrieval-implementation"),
        retrieval_policy_sha256=_digest("retrieval-policy"),
        transcript_tokens=1_000_000,
        turn_count=5000,
        retained_request_token_state_bytes=0,
        stage_ids=STAGE_IDS,
        questions=(question,),
    )


def _router_receipt() -> FixedCAVRuntimeReceipt:
    artifacts = (_digest("concept-file-a"), _digest("concept-file-b"))
    keys = ("concept_a.layer_2", "concept_b.layer_2")
    bank_sha = identity_sha256(
        {
            "format": "memory-condense-fixed-cav-source-bank-v1",
            "artifact_file_sha256s": list(artifacts),
            "ordered_tensor_keys": list(keys),
            "layer": 2,
            "num_cavs": 2,
            "hidden_dim": 4,
            "artifact_dtype": "torch.float32",
        }
    )
    return FixedCAVRuntimeReceipt(
        artifact_file_sha256s=artifacts,
        ordered_tensor_keys=keys,
        layer=2,
        num_cavs=2,
        hidden_dim=4,
        artifact_dtype="torch.float32",
        execution_dtype="torch.float32",
        device="cpu",
        extraction_temperature=1.0,
        reinjection_temperature=1.0,
        alpha=0.25,
        bank_identity_sha256=bank_sha,
        normalized_cav_bank_sha256=_digest("normalized-bank"),
    )


class _Encoder:
    checkpoint_sha256 = _digest("checkpoint")
    feature_backend_identity_sha256 = _digest("feature-backend")
    dtype_name = "torch.float32"
    device = "cpu"
    layers = 3

    def encode_layers(
        self,
        texts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        batch_size: int,
    ) -> dict[int, Any]:
        torch = pytest.importorskip("torch")
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values.append([0.25 + digest[index] / 255.0 for index in range(4)])
        return {layers[0]: torch.tensor(values, dtype=torch.float32)}


class _Router:
    max_atoms = 64

    def __init__(self, receipt: FixedCAVRuntimeReceipt) -> None:
        self.runtime_receipt = receipt
        self.layer = receipt.layer
        self.hidden_dim = receipt.hidden_dim
        self.num_cavs = receipt.num_cavs
        self.runtime_identity_sha256 = receipt.runtime_sha256
        self.bank_identity_sha256 = receipt.bank_identity_sha256

    def route_one(self, node_features: Any) -> FixedCAVForward:
        torch = pytest.importorskip("torch")
        node_count = int(node_features.shape[0])
        return FixedCAVForward(
            steered_nodes=(node_features + 0.125).detach(),
            extraction_attention=torch.full(
                (self.num_cavs, node_count),
                1.0 / node_count,
                dtype=node_features.dtype,
            ),
            reinjection_attention=torch.full(
                (node_count, self.num_cavs),
                1.0 / self.num_cavs,
                dtype=node_features.dtype,
            ),
        )


def _order_payload(order: TensorFreeStageOrder) -> dict:
    result = order.identity_payload()
    result["order_input_sha256"] = order.order_input_sha256
    return result


def _v2_manifest() -> tuple[
    FastRetrievalArtifact,
    FastCAVFeatureSessionReceipt,
    tuple[TensorFreeStageOrder, ...],
    dict,
]:
    pytest.importorskip("torch")
    artifact = _retrieval_artifact()
    router_receipt = _router_receipt()
    session = run_fast_cav_feature_session(
        artifact,
        encoder=_Encoder(),
        router=_Router(router_receipt),
        layer=2,
        batch_size=3,
    )
    orders = tuple(
        TensorFreeStageOrder(
            question_id=row.question_id,
            stage_id=row.stage_id,
            original_evidence_ids=row.readout.original_atom_order,
            base_evidence_ids=row.readout.base_order,
            treatment_evidence_ids=row.readout.treatment_order,
            upstream_receipt_sha256=row.stage_output_sha256,
        )
        for row in session.stage_receipts
    )
    manifest = {
        "format": FAST_CAV_FEATURE_ARTIFACT_FORMAT,
        "retrieval_sha256": artifact.raw_sha256,
        "transcript_tokens": artifact.transcript_tokens,
        "turn_count": artifact.turn_count,
        "question_count": artifact.question_count,
        "zero_state": {
            "contract": "tensor-free-fast-1m-cav-phase-boundary-v1",
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
        },
        "router_runtime_receipt": asdict(router_receipt),
        "feature_session": asdict(session),
        "stage_orders": [_order_payload(order) for order in orders],
    }
    return artifact, session, orders, manifest


def test_existing_v1_artifact_loads_only_when_links_are_not_required() -> None:
    root = Path(__file__).resolve().parents[1]
    retrieval_path = root / (
        "eval_results/longmemeval-1m-recall-guarded-cumulative-"
        "development-20260821/retrieval.json"
    )
    feature_path = root / (
        "eval_results/longmemeval-1m-fast-cav-development-20260822/features.json"
    )
    if not retrieval_path.is_file() or not feature_path.is_file():
        pytest.skip("sealed local legacy feature artifact is not present")
    retrieval = load_fast_retrieval_artifact(
        retrieval_path,
        expected_sha256=(
            "aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97"
        ),
    )

    artifact = load_fast_cav_feature_artifact(
        feature_path,
        retrieval_artifact=retrieval,
        expected_sha256=(
            "f57aee2ddb654989d4c35117e1fb7cf8d0e7e13ab89259f199817d198a603f89"
        ),
        require_links=False,
    )

    assert artifact.format == LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT
    assert artifact.is_legacy
    assert not artifact.has_genuine_links
    assert artifact.question_count == 10
    assert len(artifact.stage_orders) == 40
    assert all(row.links is None for row in artifact.feature_session.stage_receipts)
    first = retrieval.questions[0].stages[0]
    assert artifact.stage_order(retrieval.questions[0].question_id, first.stage_id).original_evidence_ids == first.evidence_ids
    with pytest.raises(FastCAVFeatureArtifactError, match="genuine links required"):
        load_fast_cav_feature_artifact(
            feature_path,
            retrieval_artifact=retrieval,
        )


def test_v2_loads_full_typed_links_and_sealed_bindings(tmp_path: Path) -> None:
    retrieval, session, orders, manifest = _v2_manifest()
    path, digest = _publish(tmp_path, "features-v2.json", manifest)

    artifact = load_fast_cav_feature_artifact(
        path,
        retrieval_artifact=retrieval,
        expected_sha256=digest,
    )

    assert artifact.format == FAST_CAV_FEATURE_ARTIFACT_FORMAT
    assert artifact.has_genuine_links
    assert not artifact.is_legacy
    assert artifact.feature_session == session
    assert artifact.stage_orders == orders
    assert all(
        type(row.links) is FastCAVLinkReceipt
        for row in artifact.feature_session.stage_receipts
    )
    assert artifact.stage_order("question-0", STAGE_IDS[2]) == orders[2]
    with pytest.raises(FrozenInstanceError):
        artifact.raw_sha256 = _digest("changed")  # type: ignore[misc]


def test_v2_rejects_resealed_coordinate_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    retrieval, session, orders, manifest = _v2_manifest()

    changed_stage = replace(
        session.stage_receipts[0],
        source_stage_receipt_sha256=_digest("foreign-source-stage"),
        stage_output_sha256="",
    )
    changed_session = replace(
        session,
        stage_receipts=(changed_stage, *session.stage_receipts[1:]),
        session_receipt_sha256="",
    )
    changed_order = replace(
        orders[0], upstream_receipt_sha256=changed_stage.stage_output_sha256
    )
    coordinate_tamper = copy.deepcopy(manifest)
    coordinate_tamper["feature_session"] = asdict(changed_session)
    coordinate_tamper["stage_orders"][0] = _order_payload(changed_order)
    coordinate_path, _ = _publish(
        tmp_path, "coordinate-tamper.json", coordinate_tamper
    )
    with pytest.raises(FastCAVFeatureArtifactError, match="evidence coordinates"):
        load_fast_cav_feature_artifact(
            coordinate_path,
            retrieval_artifact=retrieval,
        )

    manifest_tamper = copy.deepcopy(manifest)
    manifest_tamper["question_count"] = 2
    manifest_path, _ = _publish(tmp_path, "manifest-tamper.json", manifest_tamper)
    with pytest.raises(FastCAVFeatureArtifactError, match="supplied retrieval"):
        load_fast_cav_feature_artifact(
            manifest_path,
            retrieval_artifact=retrieval,
        )

    link_tamper = copy.deepcopy(manifest)
    del link_tamper["feature_session"]["stage_receipts"][0]["links"]
    link_path, _ = _publish(tmp_path, "missing-links.json", link_tamper)
    with pytest.raises(FastCAVFeatureArtifactError, match="links"):
        load_fast_cav_feature_artifact(
            link_path,
            retrieval_artifact=retrieval,
        )


def test_loader_requires_a_sidecar_or_explicit_digest(tmp_path: Path) -> None:
    retrieval, _session, _orders, manifest = _v2_manifest()
    path, digest = _publish(tmp_path, "features-v2.json", manifest)
    path.with_name(path.name + ".sha256").unlink()

    with pytest.raises(FastCAVFeatureArtifactError, match="sidecar"):
        load_fast_cav_feature_artifact(path, retrieval_artifact=retrieval)
    artifact = load_fast_cav_feature_artifact(
        path,
        retrieval_artifact=retrieval,
        expected_sha256=digest,
        verify_sidecar=False,
    )
    assert artifact.raw_sha256 == digest
