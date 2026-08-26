"""Small provider-free bridge from append-only Hebbian H2 to genuine CAV."""

from __future__ import annotations

from dataclasses import dataclass, replace

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._fast_hebbian_h2_scaffold import (
    _catalog as _h2_catalog,
    build_fast_hebbian_h2_scaffold,
)
from memory_condense.eval.fast_cav_feature_session import (
    FastCAVFeatureSessionReceipt,
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    FastCAVLinkSynthesisPopulation,
    _GUIDE_SLOT_SENTINEL,
    _messages,
    build_fast_cav_link_synthesis_population,
)
from memory_condense.eval.fast_hebbian_h2 import (
    FastHebbianH2Population,
    FastHebbianH2QuestionReceipt,
    FastHebbianH2ValidationError,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)


FAST_H2_CAV_INPUT_FORMAT = "memory-condense-fast-h2-cav-input-v1"
FAST_H2_CAV_LAYER_IDS = (
    "hebbian_h2",
    "cav_links",
    "cav_link_synthesis_preflight",
)
FAST_H2_CAV_SOURCE_STAGE_ID = STAGE_IDS[-1]


def _coordinates(evidence: tuple[FastEvidence, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (row.evidence_id, row.source_id, quote_sha256(row.text)) for row in evidence
    )


def _projection(evidence: tuple[FastEvidence, ...]) -> str:
    return identity_sha256(
        [
            {"evidence_id": evidence_id, "source_id": source_id, "text_sha256": text_sha}
            for evidence_id, source_id, text_sha in _coordinates(evidence)
        ]
    )


def _feature_table(
    question: FastRetrievalQuestion,
    final_evidence: tuple[FastEvidence, ...],
) -> tuple[tuple[FastFeatureRow, ...], dict[str, int]]:
    ordered = (row for stage in question.stages[:-1] for row in stage.evidence)
    rows: list[FastFeatureRow] = []
    by_text: dict[str, int] = {}
    for evidence in (*ordered, *final_evidence):
        if evidence.text in by_text:
            continue
        by_text[evidence.text] = len(rows)
        rows.append(
            FastFeatureRow(
                question.question,
                evidence.text,
                identity_sha256(
                    {
                        "format": "memory-condense-fast-feature-row-v1",
                        "question": question.question,
                        "evidence_text": evidence.text,
                    }
                ),
            )
        )
    return tuple(rows), by_text


def _overlay_question(
    question: FastRetrievalQuestion,
    receipt: FastHebbianH2QuestionReceipt,
    evidence: tuple[FastEvidence, ...],
) -> FastRetrievalQuestion:
    catalog = _h2_catalog(evidence)
    scaffold = build_fast_hebbian_h2_scaffold(evidence, question.dated_question)
    if scaffold != (
        receipt.final_evidence_catalog_sha256,
        receipt.final_scaffold_sha256,
        receipt.final_prompt_token_proxy,
    ):
        raise FastHebbianH2ValidationError("H2 final scaffold projection changed")
    messages = tuple(
        FastProviderMessage(row["role"], row["content"])
        for row in _messages(
            dated_question=question.dated_question,
            catalog=catalog,
            guide=_GUIDE_SLOT_SENTINEL,
        )
    )
    feature_rows, feature_index = _feature_table(question, evidence)
    prefix: list[FastRetrievalStage] = []
    prior_count = 0
    for stage in question.stages[:-1]:
        prefix.append(
            replace(
                stage,
                added_evidence_ids=tuple(
                    row.evidence_id for row in stage.evidence[prior_count:]
                ),
                feature_row_indices=tuple(feature_index[row.text] for row in stage.evidence),
            )
        )
        prior_count = len(stage.evidence)
    source = question.stages[-1]
    final = FastRetrievalStage(
        stage_id=FAST_H2_CAV_SOURCE_STAGE_ID,
        stage_receipt_sha256=receipt.receipt_sha256,
        matched_controls_sha256=identity_sha256(
            {"source": source.matched_controls_sha256, "h2_policy": receipt.h2_policy_sha256}
        ),
        evidence_projection_sha256=_projection(evidence),
        context_sha256=scaffold[0],
        prompt_messages_sha256=scaffold[1],
        context_token_proxy=count_tokens(catalog),
        max_context_token_proxy=FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        prompt_token_proxy=scaffold[2],
        max_prompt_token_proxy=FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
        responder_output_token_reserve=FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
        admission_status="hebbian_h2_overlay",
        added_evidence_ids=tuple(row.evidence_id for row in evidence[prior_count:]),
        context=catalog,
        evidence=evidence,
        provider_messages=messages,
        feature_row_indices=tuple(feature_index[row.text] for row in evidence),
    )
    user_index = next(i for i, row in enumerate(messages) if row.role == "user")
    user = messages[user_index]
    return replace(
        question,
        final_user_message=user,
        question_parse_receipt=replace(
            question.question_parse_receipt,
            framing="memory-condense-cav-link-synthesis-scaffold-v1",
            provider_message_index=user_index,
            provider_message_sha256=quote_sha256(user.content),
        ),
        feature_rows=feature_rows,
        stages=(*prefix, final),
    )


def _overlay_sha256(
    source_sha256: str,
    h2_sha256: str,
    questions: tuple[FastRetrievalQuestion, ...],
) -> str:
    return identity_sha256(
        {
            "format": FAST_H2_CAV_INPUT_FORMAT,
            "source_retrieval_sha256": source_sha256,
            "h2_population_sha256": h2_sha256,
            "question_projections": [
                {
                    "question_id": row.question_id,
                    "h2_receipt_sha256": row.stages[-1].stage_receipt_sha256,
                    "evidence_projection_sha256": row.stages[-1].evidence_projection_sha256,
                    "prompt_messages_sha256": row.stages[-1].prompt_messages_sha256,
                }
                for row in questions
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class FastH2CAVInput:
    """Digest-bound transient compatibility view; never persisted as retrieval JSON."""

    artifact_view: FastRetrievalArtifact
    source_retrieval_sha256: str
    h2_population_sha256: str
    overlay_sha256: str
    source_stage_id: str = FAST_H2_CAV_SOURCE_STAGE_ID
    layer_ids: tuple[str, ...] = FAST_H2_CAV_LAYER_IDS
    retained_request_token_state_bytes: int = 0
    format: str = FAST_H2_CAV_INPUT_FORMAT

    def __post_init__(self) -> None:
        expected = _overlay_sha256(
            self.source_retrieval_sha256,
            self.h2_population_sha256,
            self.artifact_view.questions,
        )
        if (
            type(self.artifact_view) is not FastRetrievalArtifact
            or self.overlay_sha256 != expected
            or self.artifact_view.raw_sha256 != expected
            or self.source_stage_id != FAST_H2_CAV_SOURCE_STAGE_ID
            or self.layer_ids != FAST_H2_CAV_LAYER_IDS
            or self.retained_request_token_state_bytes != 0
            or self.format != FAST_H2_CAV_INPUT_FORMAT
        ):
            raise FastHebbianH2ValidationError("invalid H2-to-CAV overlay binding")


@dataclass(frozen=True, slots=True)
class FastH2CAVPreflight:
    bound_input: FastH2CAVInput
    feature_session: FastCAVFeatureSessionReceipt
    synthesis: FastCAVLinkSynthesisPopulation


def bind_fast_h2_cav_input(
    retrieval: FastRetrievalArtifact,
    h2: FastHebbianH2Population,
) -> FastH2CAVInput:
    """Overlay exact H2 evidence onto S3 while retaining the sealed parent chain."""

    if type(retrieval) is not FastRetrievalArtifact:
        raise TypeError("retrieval must be an exact FastRetrievalArtifact")
    if type(h2) is not FastHebbianH2Population:
        raise TypeError("h2 must be an exact FastHebbianH2Population")
    if (
        h2.retrieval_artifact_sha256 != retrieval.raw_sha256
        or h2.question_count != retrieval.question_count
    ):
        raise FastHebbianH2ValidationError("H2 population changed retrieval binding")
    questions: list[FastRetrievalQuestion] = []
    for question, receipt, evidence in zip(
        retrieval.questions, h2.question_receipts, h2.final_evidence, strict=True
    ):
        source = question.stages[-1]
        base_coordinates = tuple(
            (row.evidence_id, row.source_id, row.evidence_text_sha256)
            for row in receipt.base_s3_coordinates
        )
        if (
            (receipt.question_ordinal, receipt.question_id)
            != (question.ordinal, question.question_id)
            or (receipt.question_sha256, receipt.dated_question_sha256)
            != (question.question_sha256, question.dated_question_sha256)
            or receipt.retrieval_question_receipt_sha256
            != question.retrieval_receipt_sha256
            or receipt.s3_stage_receipt_sha256 != source.stage_receipt_sha256
            or receipt.s3_evidence_projection_sha256 != source.evidence_projection_sha256
            or base_coordinates != _coordinates(source.evidence)
            or evidence[: len(source.evidence)] != source.evidence
        ):
            raise FastHebbianH2ValidationError("H2 question changed exact S3 provenance")
        questions.append(_overlay_question(question, receipt, evidence))
    question_tuple = tuple(questions)
    overlay_sha = _overlay_sha256(
        retrieval.raw_sha256, h2.population_sha256, question_tuple
    )
    view = replace(
        retrieval,
        source_path=f"transient:h2-cav-overlay:{overlay_sha}",
        raw_sha256=overlay_sha,
        questions=question_tuple,
    )
    return FastH2CAVInput(
        view, retrieval.raw_sha256, h2.population_sha256, overlay_sha
    )


def build_fast_h2_cav_preflight(
    retrieval: FastRetrievalArtifact,
    h2: FastHebbianH2Population,
    *,
    encoder: object,
    router: object,
    layer: int,
    batch_size: int = 8,
) -> FastH2CAVPreflight:
    """Run genuine CAV linking and the authoritative final 8k prompt preflight."""

    bound = bind_fast_h2_cav_input(retrieval, h2)
    features = run_fast_cav_feature_session(
        bound.artifact_view,
        encoder=encoder,
        router=router,
        layer=layer,
        batch_size=batch_size,
    )
    synthesis = build_fast_cav_link_synthesis_population(
        bound.artifact_view, features
    )
    return FastH2CAVPreflight(bound, features, synthesis)


__all__ = [
    "FAST_H2_CAV_INPUT_FORMAT",
    "FAST_H2_CAV_LAYER_IDS",
    "FAST_H2_CAV_SOURCE_STAGE_ID",
    "FastH2CAVInput",
    "FastH2CAVPreflight",
    "bind_fast_h2_cav_input",
    "build_fast_h2_cav_preflight",
]
