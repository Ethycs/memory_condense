from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_cav_feature_session import (
    FAST_CAV_FEATURE_BACKEND_FORMAT,
    FAST_CAV_ORDERING_PROXY_ROLE,
    FastCAVFeatureSessionError,
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_cav_links import FAST_CAV_LINK_COMPLEXITY
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    ORIGINAL_1M_RETRIEVAL_SHA256,
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
from memory_condense.search.fusion.fixed_cav_router import FixedCAVForward


def _quote(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


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


def _artifact() -> FastRetrievalArtifact:
    question_text = "Which two codes were selected?"
    dated_question = "[Question asked at 2026/08/22 (Sat) 12:00]\n" + question_text
    alpha = FastEvidence("e-alpha", "source-alpha", "Alpha was selected.")
    beta = FastEvidence("e-beta", "source-beta", "Beta was selected.")
    beta_alias = FastEvidence(
        "e-beta-alias",
        "source-beta-alias",
        beta.text,
    )
    evidence_ladder = (
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
    for stage_ordinal, (stage_id, evidence) in enumerate(
        zip(STAGE_IDS, evidence_ladder, strict=True)
    ):
        context = "\n".join(item.text for item in evidence)
        user_content = (
            "Retrieved excerpts from the conversation history:\n"
            f"{context}\n\nQuestion: {dated_question}\nShort answer:"
        )
        stages.append(
            FastRetrievalStage(
                stage_id=stage_id,
                stage_receipt_sha256=_digest(f"stage-{stage_ordinal}"),
                matched_controls_sha256=_digest("controls"),
                evidence_projection_sha256=_digest(
                    "projection-" + "-".join(item.evidence_id for item in evidence)
                ),
                context_sha256=_quote(context),
                prompt_messages_sha256=_digest(f"messages-{stage_ordinal}"),
                context_token_proxy=len(context),
                max_context_token_proxy=1000,
                prompt_token_proxy=len(user_content),
                max_prompt_token_proxy=2000,
                responder_output_token_reserve=64,
                admission_status="root" if stage_ordinal == 0 else "added",
                added_evidence_ids=tuple(
                    item.evidence_id for item in evidence[prior_count:]
                ),
                context=context,
                evidence=evidence,
                provider_messages=(
                    FastProviderMessage("system", "Use only supplied evidence."),
                    FastProviderMessage("user", user_content),
                ),
                feature_row_indices=tuple(
                    0 if item.text == alpha.text else 1 for item in evidence
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
        retrieval_receipt_sha256=_digest("retrieval"),
        protected_chunk_ids=("chunk-alpha",),
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
        raw_sha256=_digest("artifact"),
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


class _FakeEncoder:
    checkpoint_sha256 = _digest("checkpoint")
    feature_backend_identity_sha256 = _digest("fake-feature-backend")
    dtype_name = "float32"
    device = "cpu"
    layers = 3

    def __init__(self, *, mode: str = "valid") -> None:
        self.mode = mode
        self.calls: list[tuple[tuple[str, ...], tuple[int, ...], int]] = []

    def encode_layers(
        self,
        texts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        batch_size: int,
    ) -> dict[int, torch.Tensor]:
        self.calls.append((tuple(texts), tuple(layers), batch_size))
        vectors = []
        for text in texts:
            value = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([0.25 + value[index] / 255.0 for index in range(4)])
        tensor = torch.tensor(vectors, dtype=torch.float32)
        if self.mode == "nan":
            tensor[0, 0] = float("nan")
        elif self.mode == "grad":
            tensor.requires_grad_(True)
        elif self.mode == "view":
            backing = torch.cat((tensor, tensor), dim=0)
            tensor = backing[: len(texts)]
        if self.mode == "extra_layer":
            return {layers[0]: tensor, layers[0] + 1: tensor.clone()}
        return {layers[0]: tensor}


class _FakeRouter:
    layer = 2
    hidden_dim = 4
    max_atoms = 64
    num_cavs = 2
    runtime_identity_sha256 = _digest("router-runtime")
    bank_identity_sha256 = _digest("router-bank")
    concept_artifact_file_sha256s = (
        _digest("concept-artifact-0"),
        _digest("concept-artifact-1"),
    )
    concept_tensor_keys = ("concept_a.layer_2", "concept_b.layer_2")

    def __init__(self, *, tamper_identity: bool = False) -> None:
        self.calls: list[tuple[tuple[float, ...], ...]] = []
        self.tamper_identity = tamper_identity

    def route_one(self, node_features: torch.Tensor) -> FixedCAVForward:
        self.calls.append(
            tuple(tuple(float(value) for value in row) for row in node_features)
        )
        node_count = int(node_features.shape[0])
        result = FixedCAVForward(
            steered_nodes=(node_features + 0.125).detach(),
            extraction_attention=torch.full(
                (self.num_cavs, node_count),
                1.0 / node_count,
                dtype=node_features.dtype,
                device=node_features.device,
            ),
            reinjection_attention=torch.full(
                (node_count, self.num_cavs),
                1.0 / self.num_cavs,
                dtype=node_features.dtype,
                device=node_features.device,
            ),
        )
        if self.tamper_identity:
            self.runtime_identity_sha256 = _digest("changed-router-runtime")
        return result


def _contains_tensor(value: object) -> bool:
    if type(value) is torch.Tensor:
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    return False


def test_one_encoder_call_router_packet_reuse_and_tensor_free_receipts():
    artifact = _artifact()
    encoder = _FakeEncoder()
    router = _FakeRouter()

    receipt = run_fast_cav_feature_session(
        artifact,
        encoder=encoder,
        router=router,
        layer=2,
        batch_size=3,
    )

    assert receipt.feature_backend_format == FAST_CAV_FEATURE_BACKEND_FORMAT
    assert receipt.artifact_sha256 == artifact.raw_sha256
    assert receipt.encoder_api_call_count == 1
    assert len(encoder.calls) == 1
    encoded_texts, encoded_layers, encoded_batch_size = encoder.calls[0]
    assert encoded_texts == tuple(sorted(encoded_texts, key=lambda item: (len(item), item)))
    assert encoded_layers == (2,)
    assert encoded_batch_size == 3
    assert receipt.question_count == 1
    assert receipt.stage_placement_count == 4
    assert receipt.logical_evidence_placement_count == 9
    assert receipt.per_question_unique_feature_row_count == 2
    assert receipt.global_unique_evidence_text_count == 2
    assert receipt.global_unique_question_text_count == 1
    assert receipt.global_unique_text_count == 3
    assert receipt.unique_router_call_count == 3
    assert len(router.calls) == 3
    assert [item.reused_router_result for item in receipt.stage_receipts] == [
        False,
        False,
        False,
        True,
    ]
    assert [item.router_call_ordinal for item in receipt.stage_receipts] == [0, 1, 2, 2]
    assert (
        receipt.stage_receipts[2].readout.readout_sha256
        == receipt.stage_receipts[3].readout.readout_sha256
    )
    assert (
        receipt.stage_receipts[2].links.link_receipt_sha256
        == receipt.stage_receipts[3].links.link_receipt_sha256
    )
    for stage in receipt.stage_receipts:
        links = stage.links
        assert links is not None
        assert links.complexity_contract == FAST_CAV_LINK_COMPLEXITY
        assert links.extraction_shape == (router.num_cavs, len(stage.evidence_ids))
        assert links.reinjection_shape == (len(stage.evidence_ids), router.num_cavs)
        assert links.evidence_ids == stage.evidence_ids
        assert links.source_ids == stage.source_ids
        assert links.evidence_pair_matrix_constructed is False
        assert links.evidence_pair_matrix_cell_count == 0
        assert links.retained_tensor_bytes == 0
        assert stage.readout_role == FAST_CAV_ORDERING_PROXY_ROLE
    assert receipt.stage("question-0", STAGE_IDS[1]) is receipt.stage_receipts[1]
    assert receipt.question_stages("question-0") == receipt.stage_receipts
    assert receipt.result_retained_tensor_bytes == 0
    assert receipt.retained_token_id_count == 0
    assert receipt.persisted_token_state_bytes == 0
    assert not _contains_tensor(receipt)
    with pytest.raises(FrozenInstanceError):
        receipt.encoder_api_call_count = 2  # type: ignore[misc]


def test_duplicate_text_keeps_distinct_evidence_provenance():
    router = _FakeRouter()
    receipt = run_fast_cav_feature_session(
        _artifact(),
        encoder=_FakeEncoder(),
        router=router,
        layer=2,
    )

    third = receipt.stage_receipts[2]
    assert third.evidence_ids == ("e-alpha", "e-beta", "e-beta-alias")
    assert third.source_ids == (
        "source-alpha",
        "source-beta",
        "source-beta-alias",
    )
    assert third.evidence_text_sha256s[1] == third.evidence_text_sha256s[2]
    assert third.evidence_feature_row_indices == (0, 1, 1)
    assert third.links is not None
    assert third.links.evidence_ids == third.evidence_ids
    assert third.links.source_ids == third.source_ids
    assert third.links.evidence_text_sha256s == third.evidence_text_sha256s
    assert third.readout.original_atom_order == third.evidence_ids
    assert router.calls[2][1] == router.calls[2][2]


@pytest.mark.parametrize("mode", ["nan", "grad", "view", "extra_layer"])
def test_rejects_encoder_tensor_or_layer_contract_violations(mode: str):
    encoder = _FakeEncoder(mode=mode)

    with pytest.raises((FastCAVFeatureSessionError, RuntimeError)):
        run_fast_cav_feature_session(
            _artifact(),
            encoder=encoder,
            router=_FakeRouter(),
            layer=2,
        )

    assert len(encoder.calls) == 1


def test_rejects_artifact_projection_tampering_before_encoder_call():
    artifact = _artifact()
    question = artifact.questions[0]
    second = question.stages[1]
    tampered_stage = replace(second, added_evidence_ids=("wrong-coordinate",))
    tampered_question = replace(
        question,
        stages=(question.stages[0], tampered_stage, *question.stages[2:]),
    )
    tampered = replace(artifact, questions=(tampered_question,))
    encoder = _FakeEncoder()

    with pytest.raises(FastCAVFeatureSessionError, match="added-evidence suffix"):
        run_fast_cav_feature_session(
            tampered,
            encoder=encoder,
            router=_FakeRouter(),
            layer=2,
        )

    assert encoder.calls == []


def test_rejects_router_identity_change_after_route():
    with pytest.raises(RuntimeError, match="router identity changed"):
        run_fast_cav_feature_session(
            _artifact(),
            encoder=_FakeEncoder(),
            router=_FakeRouter(tamper_identity=True),
            layer=2,
        )


def test_real_sealed_artifact_has_expected_single_pass_scaling():
    path = Path(
        "eval_results/longmemeval-1m-recall-guarded-cumulative-"
        "development-20260821/retrieval.json"
    )
    if not path.is_file():
        pytest.skip("sealed local 1M retrieval artifact is not present")
    artifact = load_fast_retrieval_artifact(
        path,
        expected_sha256=ORIGINAL_1M_RETRIEVAL_SHA256,
    )
    encoder = _FakeEncoder()
    router = _FakeRouter()

    receipt = run_fast_cav_feature_session(
        artifact,
        encoder=encoder,
        router=router,
        layer=2,
        batch_size=64,
    )

    assert receipt.artifact_sha256 == ORIGINAL_1M_RETRIEVAL_SHA256
    assert receipt.question_count == 10
    assert receipt.stage_placement_count == 40
    assert receipt.logical_evidence_placement_count == 1_939
    assert receipt.per_question_unique_feature_row_count == 530
    assert receipt.global_unique_evidence_text_count == 526
    assert receipt.global_unique_question_text_count == 10
    assert receipt.global_unique_text_count == 536
    assert receipt.encoder_api_call_count == 1
    assert len(encoder.calls) == 1
    assert receipt.unique_router_call_count == 22
    assert len(router.calls) == 22
    assert not _contains_tensor(receipt)
