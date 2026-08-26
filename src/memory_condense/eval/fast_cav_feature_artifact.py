"""Typed, read-only loader for sealed fast-CAV feature manifests.

Version 1 manifests retain the historical X/X1 ordering readout but contain
no genuine concept/evidence links.  Version 2 manifests contain the complete
``FastCAVFeatureSessionReceipt`` with bounded two-pass link receipts.  The
loader defaults to ``require_links=True`` so a legacy ordering proxy cannot be
silently consumed as the linking technique.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval.fast_cav_feature_session import (
    FAST_CAV_ORDERING_PROXY_ROLE,
    FAST_CAV_SESSION_RECEIPT_FORMAT,
    LEGACY_FAST_CAV_SESSION_RECEIPT_FORMAT,
    FastCAVFeatureSessionReceipt,
    FastCAVStageReceipt,
)
from memory_condense.eval.fast_cav_links import (
    FastCAVConceptProvenance,
    FastCAVExtractionLink,
    FastCAVLinkReceipt,
    FastCAVReinjectionLink,
    build_fast_cav_concepts,
)
from memory_condense.eval.fast_cav_prompts import (
    FAST_CAV_ORDER_INPUT_FORMAT,
    TensorFreeStageOrder,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastRetrievalArtifact,
)
from memory_condense.search.fusion.fixed_cav_router import FixedCAVRuntimeReceipt
from memory_condense.search.fusion.steered_readout import MatchedSteeredReadout


LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT = (
    "memory-condense-fast-1m-cav-features-v1"
)
FAST_CAV_FEATURE_ARTIFACT_FORMAT = "memory-condense-fast-1m-cav-features-v2"
FAST_CAV_ZERO_STATE_CONTRACT = "tensor-free-fast-1m-cav-phase-boundary-v1"

_FORMATS = {
    LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT,
    FAST_CAV_FEATURE_ARTIFACT_FORMAT,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ZERO_STATE = {
    "contract": FAST_CAV_ZERO_STATE_CONTRACT,
    "persisted_transformer_token_state": False,
    "retained_transformer_token_state_bytes": 0,
}


class FastCAVFeatureArtifactError(ValueError):
    """Raised when a feature manifest cannot prove its sealed provenance."""


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise FastCAVFeatureArtifactError(f"{label} must be an exact object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise FastCAVFeatureArtifactError(f"{label} must be an exact array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise FastCAVFeatureArtifactError(f"{label} must be exact non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FastCAVFeatureArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FastCAVFeatureArtifactError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _construct(factory: Any, body: Mapping[str, Any], label: str) -> Any:
    try:
        return factory(**body)
    except (TypeError, ValueError) as exc:
        raise FastCAVFeatureArtifactError(f"{label} did not validate: {exc}") from exc


def _validate_sidecar(path: Path, sidecar: Path, digest: str) -> None:
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise FastCAVFeatureArtifactError(
            f"feature artifact digest sidecar is missing or invalid: {sidecar}"
        )


def _read_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None,
    verify_sidecar: bool,
    sidecar_path: str | Path | None,
) -> tuple[dict[str, Any], str, Path]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if type(verify_sidecar) is not bool:
        raise TypeError("verify_sidecar must be an exact bool")
    if expected_sha256 is not None:
        expected_sha256 = _digest(expected_sha256, "expected_sha256")
    if not verify_sidecar and expected_sha256 is None:
        raise FastCAVFeatureArtifactError(
            "an expected SHA-256 is required when sidecar verification is disabled"
        )
    if not verify_sidecar and sidecar_path is not None:
        raise ValueError("sidecar_path requires verify_sidecar=True")

    raw = artifact_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise FastCAVFeatureArtifactError(
            f"feature artifact SHA-256 mismatch ({digest} != {expected_sha256})"
        )
    if verify_sidecar:
        sidecar = (
            Path(sidecar_path)
            if sidecar_path is not None
            else artifact_path.with_name(artifact_path.name + ".sha256")
        )
        _validate_sidecar(artifact_path, sidecar, digest)
    try:
        payload = json.loads(raw, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FastCAVFeatureArtifactError(
            "feature artifact is not finite UTF-8 JSON"
        ) from exc
    root = _mapping(payload, "feature artifact")
    if raw != _canonical_json_bytes(root):
        raise FastCAVFeatureArtifactError("feature artifact is not canonical JSON")
    return root, digest, artifact_path


def _parse_order(raw: object, index: int) -> TensorFreeStageOrder:
    row = _mapping(raw, f"stage_orders[{index}]")
    if row.get("format") != FAST_CAV_ORDER_INPUT_FORMAT:
        raise FastCAVFeatureArtifactError("feature stage-order format changed")
    body = {
        "question_id": row.get("question_id"),
        "stage_id": row.get("stage_id"),
        "original_evidence_ids": tuple(row.get("original_evidence_ids", ())),
        "base_evidence_ids": tuple(row.get("base_evidence_ids", ())),
        "treatment_evidence_ids": tuple(row.get("treatment_evidence_ids", ())),
        "upstream_receipt_sha256": row.get("upstream_receipt_sha256"),
        "retained_tensor_bytes": row.get("retained_tensor_bytes", -1),
    }
    order = _construct(TensorFreeStageOrder, body, f"stage_orders[{index}]")
    projection = order.identity_payload()
    projection["order_input_sha256"] = order.order_input_sha256
    if row != projection:
        raise FastCAVFeatureArtifactError(
            f"stage_orders[{index}] typed projection changed"
        )
    return order


def _parse_readout(raw: object, label: str) -> MatchedSteeredReadout:
    body = dict(_mapping(raw, label))
    for key in (
        "original_atom_order",
        "base_scores",
        "treatment_scores",
        "base_order",
        "treatment_order",
    ):
        body[key] = tuple(body.get(key, ()))
    return _construct(MatchedSteeredReadout, body, label)


def _parse_links(raw: object, label: str) -> FastCAVLinkReceipt:
    body = dict(_mapping(raw, label))
    body["concepts"] = tuple(
        _construct(FastCAVConceptProvenance, _mapping(row, label), label)
        for row in _list(body.get("concepts"), f"{label}.concepts")
    )
    body["extraction_links"] = tuple(
        _construct(FastCAVExtractionLink, _mapping(row, label), label)
        for row in _list(
            body.get("extraction_links"), f"{label}.extraction_links"
        )
    )
    body["reinjection_links"] = tuple(
        _construct(FastCAVReinjectionLink, _mapping(row, label), label)
        for row in _list(
            body.get("reinjection_links"), f"{label}.reinjection_links"
        )
    )
    for key in (
        "evidence_ids",
        "source_ids",
        "evidence_text_sha256s",
        "extraction_shape",
        "reinjection_shape",
    ):
        body[key] = tuple(body.get(key, ()))
    return _construct(FastCAVLinkReceipt, body, label)


def _parse_stage(raw: object, *, legacy: bool, index: int) -> FastCAVStageReceipt:
    label = f"feature_session.stage_receipts[{index}]"
    body = dict(_mapping(raw, label))
    body["readout"] = _parse_readout(body.get("readout"), f"{label}.readout")
    if not legacy:
        body["links"] = _parse_links(body.get("links"), f"{label}.links")
    for key in (
        "evidence_feature_row_indices",
        "evidence_ids",
        "source_ids",
        "evidence_text_sha256s",
    ):
        body[key] = tuple(body.get(key, ()))
    return _construct(FastCAVStageReceipt, body, label)


def _parse_session(raw: object, *, legacy: bool) -> FastCAVFeatureSessionReceipt:
    root = _mapping(raw, "feature_session")
    expected_format = (
        LEGACY_FAST_CAV_SESSION_RECEIPT_FORMAT
        if legacy
        else FAST_CAV_SESSION_RECEIPT_FORMAT
    )
    if root.get("format") != expected_format:
        raise FastCAVFeatureArtifactError(
            "feature manifest mixed session receipt generations"
        )
    stages = tuple(
        _parse_stage(row, legacy=legacy, index=index)
        for index, row in enumerate(
            _list(root.get("stage_receipts"), "feature_session.stage_receipts")
        )
    )
    body = dict(root)
    body["stage_ids"] = tuple(body.get("stage_ids", ()))
    body["stage_receipts"] = stages
    session = _construct(FastCAVFeatureSessionReceipt, body, "feature_session")
    projection = asdict(session)
    if legacy:
        for stage in projection["stage_receipts"]:
            stage.pop("links")
            stage.pop("readout_role")
    if _canonical_json_bytes(projection) != _canonical_json_bytes(root):
        raise FastCAVFeatureArtifactError(
            "feature session typed receipt projection changed"
        )
    return session


def _parse_router(raw: object) -> FixedCAVRuntimeReceipt:
    root = _mapping(raw, "router_runtime_receipt")
    body = dict(root)
    body["artifact_file_sha256s"] = tuple(
        body.get("artifact_file_sha256s", ())
    )
    body["ordered_tensor_keys"] = tuple(body.get("ordered_tensor_keys", ()))
    receipt = _construct(FixedCAVRuntimeReceipt, body, "router_runtime_receipt")
    if _canonical_json_bytes(asdict(receipt)) != _canonical_json_bytes(root):
        raise FastCAVFeatureArtifactError(
            "router runtime typed receipt projection changed"
        )
    return receipt


def _quote_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_session_dimensions(
    session: FastCAVFeatureSessionReceipt,
    retrieval: FastRetrievalArtifact,
) -> None:
    evidence_texts = {
        row.evidence_text
        for question in retrieval.questions
        for row in question.feature_rows
    }
    question_texts = {question.question for question in retrieval.questions}
    ordered_texts = tuple(
        sorted(evidence_texts | question_texts, key=lambda value: (len(value), value))
    )
    input_projection = identity_sha256(
        {
            "format": "memory-condense-fast-cav-encoder-input-projection-v1",
            "ordered_text_sha256s": [_quote_sha256(value) for value in ordered_texts],
        }
    )
    expected = (
        session.artifact_sha256,
        session.stage_ids,
        session.question_count,
        session.stage_placement_count,
        session.logical_evidence_placement_count,
        session.per_question_unique_feature_row_count,
        session.global_unique_evidence_text_count,
        session.global_unique_question_text_count,
        session.global_unique_text_count,
        session.encoder_input_projection_sha256,
    )
    observed = (
        retrieval.raw_sha256,
        retrieval.stage_ids,
        retrieval.question_count,
        retrieval.question_count * len(retrieval.stage_ids),
        retrieval.logical_feature_row_count,
        retrieval.unique_feature_row_count,
        len(evidence_texts),
        len(question_texts),
        len(ordered_texts),
        input_projection,
    )
    if expected != observed:
        raise FastCAVFeatureArtifactError(
            "feature session dimensions do not bind the retrieval artifact"
        )


def _validate_router_binding(
    session: FastCAVFeatureSessionReceipt,
    router: FixedCAVRuntimeReceipt,
) -> tuple[FastCAVConceptProvenance, ...]:
    expected = (
        session.router_runtime_identity_sha256,
        session.router_bank_identity_sha256,
        session.router_num_cavs,
        session.feature_layer,
        session.feature_hidden_dim,
    )
    observed = (
        router.runtime_sha256,
        router.bank_identity_sha256,
        router.num_cavs,
        router.layer,
        router.hidden_dim,
    )
    if expected != observed:
        raise FastCAVFeatureArtifactError(
            "feature session does not bind the router runtime receipt"
        )
    try:
        return build_fast_cav_concepts(
            bank_identity_sha256=router.bank_identity_sha256,
            artifact_file_sha256s=router.artifact_file_sha256s,
            tensor_keys=router.ordered_tensor_keys,
        )
    except (TypeError, ValueError) as exc:
        raise FastCAVFeatureArtifactError(
            f"router concept provenance did not validate: {exc}"
        ) from exc


def _expected_placements(retrieval: FastRetrievalArtifact) -> tuple[Any, ...]:
    return tuple(
        (question, stage_ordinal, stage)
        for question in retrieval.questions
        for stage_ordinal, stage in enumerate(question.stages)
    )


def _validate_coordinates(
    *,
    retrieval: FastRetrievalArtifact,
    orders: tuple[TensorFreeStageOrder, ...],
    session: FastCAVFeatureSessionReceipt,
    concepts: tuple[FastCAVConceptProvenance, ...],
    has_links: bool,
) -> None:
    placements = _expected_placements(retrieval)
    if len(orders) != len(placements) or len(session.stage_receipts) != len(
        placements
    ):
        raise FastCAVFeatureArtifactError(
            "feature stage population does not match retrieval placements"
        )
    for placement_ordinal, (expected, order, receipt) in enumerate(
        zip(placements, orders, session.stage_receipts, strict=True)
    ):
        question, stage_ordinal, stage = expected
        evidence_text_sha256s = tuple(_quote_sha256(row.text) for row in stage.evidence)
        receipt_coordinates = (
            receipt.artifact_sha256,
            receipt.placement_ordinal,
            receipt.question_ordinal,
            receipt.question_id,
            receipt.question_sha256,
            receipt.dated_question_sha256,
            receipt.stage_ordinal,
            receipt.stage_id,
            receipt.source_stage_receipt_sha256,
            receipt.evidence_projection_sha256,
            receipt.evidence_feature_row_indices,
            receipt.evidence_ids,
            receipt.source_ids,
            receipt.evidence_text_sha256s,
        )
        expected_coordinates = (
            retrieval.raw_sha256,
            placement_ordinal,
            question.ordinal,
            question.question_id,
            question.question_sha256,
            question.dated_question_sha256,
            stage_ordinal,
            stage.stage_id,
            stage.stage_receipt_sha256,
            stage.evidence_projection_sha256,
            stage.feature_row_indices,
            stage.evidence_ids,
            stage.source_ids,
            evidence_text_sha256s,
        )
        if receipt_coordinates != expected_coordinates:
            raise FastCAVFeatureArtifactError(
                "feature stage changed exact retrieval evidence coordinates"
            )
        order_coordinates = (
            order.question_id,
            order.stage_id,
            order.original_evidence_ids,
            order.base_evidence_ids,
            order.treatment_evidence_ids,
            order.upstream_receipt_sha256,
        )
        readout = receipt.readout
        expected_order_coordinates = (
            question.question_id,
            stage.stage_id,
            readout.original_atom_order,
            readout.base_order,
            readout.treatment_order,
            receipt.stage_output_sha256,
        )
        if order_coordinates != expected_order_coordinates or (
            order.original_evidence_ids != stage.evidence_ids
        ):
            raise FastCAVFeatureArtifactError(
                "feature stage order does not bind its session readout"
            )
        if has_links:
            links = receipt.links
            if (
                type(links) is not FastCAVLinkReceipt
                or links.concepts != concepts
                or links.evidence_ids != stage.evidence_ids
                or links.source_ids != stage.source_ids
                or links.evidence_text_sha256s != evidence_text_sha256s
                or receipt.readout_role != FAST_CAV_ORDERING_PROXY_ROLE
            ):
                raise FastCAVFeatureArtifactError(
                    "linked feature stage changed genuine link provenance"
                )
        elif receipt.links is not None or receipt.readout_role:
            raise FastCAVFeatureArtifactError(
                "legacy feature stage cannot claim genuine links"
            )


@dataclass(frozen=True, slots=True)
class FastCAVFeatureArtifact:
    """Immutable typed projection of one sealed feature manifest.

    For legacy v1, ``stage_orders`` is the supported experimental output and
    ``has_genuine_links`` is false.  ``feature_session`` remains available only
    as typed provenance for validating those historical orders.
    """

    source_path: str
    raw_sha256: str
    format: str
    retrieval_sha256: str
    transcript_tokens: int
    turn_count: int
    question_count: int
    zero_state_contract: str
    stage_orders: tuple[TensorFreeStageOrder, ...]
    feature_session: FastCAVFeatureSessionReceipt
    router_runtime_receipt: FixedCAVRuntimeReceipt
    has_genuine_links: bool

    def __post_init__(self) -> None:
        _text(self.source_path, "source_path")
        _digest(self.raw_sha256, "raw_sha256")
        _digest(self.retrieval_sha256, "retrieval_sha256")
        if self.format not in _FORMATS:
            raise FastCAVFeatureArtifactError("unsupported feature artifact format")
        _integer(self.transcript_tokens, "transcript_tokens", minimum=1)
        _integer(self.turn_count, "turn_count", minimum=1)
        _integer(self.question_count, "question_count", minimum=1)
        if self.zero_state_contract != FAST_CAV_ZERO_STATE_CONTRACT:
            raise FastCAVFeatureArtifactError("zero-state contract changed")
        if type(self.stage_orders) is not tuple or any(
            type(row) is not TensorFreeStageOrder for row in self.stage_orders
        ):
            raise FastCAVFeatureArtifactError(
                "stage_orders must contain exact TensorFreeStageOrder values"
            )
        if type(self.feature_session) is not FastCAVFeatureSessionReceipt:
            raise FastCAVFeatureArtifactError(
                "feature_session must be an exact typed receipt"
            )
        if type(self.router_runtime_receipt) is not FixedCAVRuntimeReceipt:
            raise FastCAVFeatureArtifactError(
                "router_runtime_receipt must be an exact typed receipt"
            )
        expected_links = self.format == FAST_CAV_FEATURE_ARTIFACT_FORMAT
        if type(self.has_genuine_links) is not bool or (
            self.has_genuine_links != expected_links
        ):
            raise FastCAVFeatureArtifactError(
                "feature capability does not match its manifest generation"
            )

    @property
    def is_legacy(self) -> bool:
        return self.format == LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT

    def stage_order(self, question_id: str, stage_id: str) -> TensorFreeStageOrder:
        for order in self.stage_orders:
            if order.question_id == question_id and order.stage_id == stage_id:
                return order
        raise KeyError((question_id, stage_id))


def load_fast_cav_feature_artifact(
    path: str | Path,
    *,
    retrieval_artifact: FastRetrievalArtifact,
    expected_sha256: str | None = None,
    verify_sidecar: bool = True,
    sidecar_path: str | Path | None = None,
    require_links: bool = True,
) -> FastCAVFeatureArtifact:
    """Load a sealed v1/v2 feature manifest and bind it to retrieval.

    ``require_links=True`` is intentionally the default.  Pass false only for
    historical order-only analysis of a v1 artifact.
    """

    if type(retrieval_artifact) is not FastRetrievalArtifact:
        raise TypeError("retrieval_artifact must be an exact FastRetrievalArtifact")
    if type(require_links) is not bool:
        raise TypeError("require_links must be an exact bool")
    root, raw_sha256, artifact_path = _read_manifest(
        path,
        expected_sha256=expected_sha256,
        verify_sidecar=verify_sidecar,
        sidecar_path=sidecar_path,
    )
    manifest_format = root.get("format")
    if manifest_format not in _FORMATS:
        raise FastCAVFeatureArtifactError(
            "feature artifact has an unsupported format"
        )
    legacy = manifest_format == LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT
    if legacy and require_links:
        raise FastCAVFeatureArtifactError(
            "legacy v1 feature artifact has ordering proxies only; genuine links required"
        )
    retrieval_sha256 = _digest(root.get("retrieval_sha256"), "retrieval_sha256")
    transcript_tokens = _integer(
        root.get("transcript_tokens"), "transcript_tokens", minimum=1
    )
    turn_count = _integer(root.get("turn_count"), "turn_count", minimum=1)
    question_count = _integer(
        root.get("question_count"), "question_count", minimum=1
    )
    if (
        retrieval_sha256 != retrieval_artifact.raw_sha256
        or transcript_tokens != retrieval_artifact.transcript_tokens
        or turn_count != retrieval_artifact.turn_count
        or question_count != retrieval_artifact.question_count
    ):
        raise FastCAVFeatureArtifactError(
            "feature manifest does not bind the supplied retrieval artifact"
        )
    if _mapping(root.get("zero_state"), "zero_state") != _ZERO_STATE:
        raise FastCAVFeatureArtifactError(
            "feature manifest changed the zero-state boundary"
        )

    orders = tuple(
        _parse_order(row, index)
        for index, row in enumerate(
            _list(root.get("stage_orders"), "stage_orders")
        )
    )
    session = _parse_session(root.get("feature_session"), legacy=legacy)
    router = _parse_router(root.get("router_runtime_receipt"))
    _validate_session_dimensions(session, retrieval_artifact)
    concepts = _validate_router_binding(session, router)
    _validate_coordinates(
        retrieval=retrieval_artifact,
        orders=orders,
        session=session,
        concepts=concepts,
        has_links=not legacy,
    )
    return FastCAVFeatureArtifact(
        source_path=str(artifact_path),
        raw_sha256=raw_sha256,
        format=manifest_format,
        retrieval_sha256=retrieval_sha256,
        transcript_tokens=transcript_tokens,
        turn_count=turn_count,
        question_count=question_count,
        zero_state_contract=FAST_CAV_ZERO_STATE_CONTRACT,
        stage_orders=orders,
        feature_session=session,
        router_runtime_receipt=router,
        has_genuine_links=not legacy,
    )


__all__ = [
    "FAST_CAV_FEATURE_ARTIFACT_FORMAT",
    "FAST_CAV_ZERO_STATE_CONTRACT",
    "LEGACY_FAST_CAV_FEATURE_ARTIFACT_FORMAT",
    "FastCAVFeatureArtifact",
    "FastCAVFeatureArtifactError",
    "load_fast_cav_feature_artifact",
]
