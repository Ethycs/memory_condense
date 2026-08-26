"""Tensor-free links from the two rectangular fixed-CAV attention passes.

The fixed router produces extraction weights ``E[K,N]`` and reinjection
weights ``R[N,K]``.  This module consumes those transient matrices once and
retains only their canonical hashes plus bounded top links.  It never forms or
accepts an ``N x N`` evidence-pair matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from memory_condense.domain._discourse_identity import _sha256, identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.fusion.fixed_cav_router import (
    FIXED_CAV_ALGORITHM,
    FixedCAVRuntimeReceipt,
)
from memory_condense.search.fusion.tensor_identity import (
    CANONICAL_TENSOR_DTYPE,
    canonical_float32_tensor,
)


FAST_CAV_CONCEPT_FORMAT = "memory-condense-fast-cav-concept-provenance-v1"
FAST_CAV_EXTRACTION_LINK_FORMAT = "memory-condense-fast-cav-extraction-link-v1"
FAST_CAV_REINJECTION_LINK_FORMAT = "memory-condense-fast-cav-reinjection-link-v1"
FAST_CAV_LINK_RECEIPT_FORMAT = "memory-condense-fast-cav-two-pass-links-v1"
FAST_CAV_LINK_COMPLEXITY = "two-rectangular-passes-o-k-n-no-n-by-n-v1"
FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT = 4
FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE = 4
FAST_CAV_MAX_CONCEPTS = 16
FAST_CAV_MAX_EVIDENCE = 64
FAST_CAV_MAX_RECTANGULAR_ROUTE_CELLS = 2 * (
    FAST_CAV_MAX_CONCEPTS * FAST_CAV_MAX_EVIDENCE
)


class FastCAVLinkError(ValueError):
    """Raised when two-pass CAV link provenance cannot be sealed."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise FastCAVLinkError(f"{label} must be an exact non-empty string")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise FastCAVLinkError(f"{label} must be an exact integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise FastCAVLinkError(f"{label} exceeds {maximum}")
    return value


def _zero(value: object, label: str) -> int:
    if type(value) is not int or value != 0:
        raise FastCAVLinkError(f"{label} must remain exactly zero")
    return 0


def _strings(
    values: object,
    label: str,
    *,
    unique: bool,
) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or not value.strip() for value in values
    ):
        raise FastCAVLinkError(
            f"{label} must be an exact tuple of non-empty strings"
        )
    if unique and len(values) != len(set(values)):
        raise FastCAVLinkError(f"{label} must contain unique values")
    return values


def _weight(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FastCAVLinkError(
            f"{label} must be one canonical finite float inside [0, 1]"
        )
    return value


def _weight_sha256(value: float) -> str:
    return canonical_float32_tensor(
        (value,),
        label="CAV link scalar weight",
        retain_values=False,
    ).tensor_sha256


def _concept_id(
    *,
    bank_identity_sha256: str,
    ordinal: int,
    artifact_file_sha256: str,
    tensor_key: str,
) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-fast-cav-concept-id-v1",
            "bank_identity_sha256": bank_identity_sha256,
            "concept_ordinal": ordinal,
            "artifact_file_sha256": artifact_file_sha256,
            "tensor_key": tensor_key,
        }
    )


@dataclass(frozen=True, slots=True)
class FastCAVConceptProvenance(SealedIdentity):
    """One ordered CAV coordinate bound to its source tensor artifact."""

    _SEAL_FIELD = "concept_sha256"
    _SEAL_MISMATCH = "fast CAV concept seal does not match its contents"

    format: str
    bank_identity_sha256: str
    concept_ordinal: int
    concept_id: str
    artifact_file_sha256: str
    tensor_key: str
    concept_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_CONCEPT_FORMAT:
            raise FastCAVLinkError("unsupported fast CAV concept format")
        bank = _sha256(self.bank_identity_sha256, "bank_identity_sha256")
        ordinal = _exact_int(
            self.concept_ordinal,
            "concept_ordinal",
            maximum=FAST_CAV_MAX_CONCEPTS - 1,
        )
        artifact = _sha256(
            self.artifact_file_sha256,
            "artifact_file_sha256",
        )
        key = _text(self.tensor_key, "tensor_key")
        if _sha256(self.concept_id, "concept_id") != _concept_id(
            bank_identity_sha256=bank,
            ordinal=ordinal,
            artifact_file_sha256=artifact,
            tensor_key=key,
        ):
            raise FastCAVLinkError("concept_id changed its exact bank provenance")
        self._seal()


def build_fast_cav_concepts(
    *,
    bank_identity_sha256: str,
    artifact_file_sha256s: Sequence[str],
    tensor_keys: Sequence[str],
) -> tuple[FastCAVConceptProvenance, ...]:
    """Build the ordered concept coordinates exposed by a fixed CAV bank."""

    bank = _sha256(bank_identity_sha256, "bank_identity_sha256")
    try:
        artifacts = tuple(artifact_file_sha256s)
        keys = tuple(tensor_keys)
    except TypeError as exc:
        raise FastCAVLinkError(
            "CAV artifact hashes and tensor keys must be sequences"
        ) from exc
    if not artifacts or len(artifacts) != len(keys):
        raise FastCAVLinkError("CAV artifact hashes and tensor keys must align")
    if len(artifacts) > FAST_CAV_MAX_CONCEPTS:
        raise FastCAVLinkError("CAV concept population exceeds the K ceiling")
    if any(type(value) is not str for value in artifacts) or any(
        type(value) is not str for value in keys
    ):
        raise FastCAVLinkError("CAV provenance values must be exact strings")
    rows = tuple(
        FastCAVConceptProvenance(
            format=FAST_CAV_CONCEPT_FORMAT,
            bank_identity_sha256=bank,
            concept_ordinal=ordinal,
            concept_id=_concept_id(
                bank_identity_sha256=bank,
                ordinal=ordinal,
                artifact_file_sha256=_sha256(
                    artifact_sha,
                    f"artifact_file_sha256s[{ordinal}]",
                ),
                tensor_key=_text(key, f"tensor_keys[{ordinal}]"),
            ),
            artifact_file_sha256=artifact_sha,
            tensor_key=key,
        )
        for ordinal, (artifact_sha, key) in enumerate(
            zip(artifacts, keys, strict=True)
        )
    )
    if len({row.concept_id for row in rows}) != len(rows):
        raise FastCAVLinkError("CAV concept provenance IDs must be unique")
    return rows


def build_fast_cav_concepts_from_router(
    router: Any,
    *,
    runtime_identity_sha256: str,
    bank_identity_sha256: str,
    layer: int,
    hidden_dim: int,
    num_cavs: int,
) -> tuple[FastCAVConceptProvenance, ...]:
    """Read exact ordered concept provenance from a fixed router receipt."""

    receipt = getattr(router, "runtime_receipt", None)
    if type(receipt) is FixedCAVRuntimeReceipt:
        receipt._seal()
        if (
            receipt.runtime_sha256 != runtime_identity_sha256
            or receipt.bank_identity_sha256 != bank_identity_sha256
            or receipt.layer != layer
            or receipt.hidden_dim != hidden_dim
            or receipt.num_cavs != num_cavs
        ):
            raise FastCAVLinkError("router receipt changed fixed-bank provenance")
        artifacts = receipt.artifact_file_sha256s
        keys = receipt.ordered_tensor_keys
    else:
        artifacts = getattr(router, "concept_artifact_file_sha256s", None)
        keys = getattr(router, "concept_tensor_keys", None)
    concepts = build_fast_cav_concepts(
        bank_identity_sha256=bank_identity_sha256,
        artifact_file_sha256s=artifacts,
        tensor_keys=keys,
    )
    if len(concepts) != num_cavs:
        raise FastCAVLinkError("concept provenance disagrees with num_cavs")
    return concepts


@dataclass(frozen=True, slots=True)
class FastCAVExtractionLink(SealedIdentity):
    """One CAV concept's ranked contribution from a source evidence node."""

    _SEAL_FIELD = "link_sha256"
    _SEAL_MISMATCH = "fast CAV extraction link seal does not match its contents"

    format: str
    concept_ordinal: int
    concept_id: str
    concept_sha256: str
    evidence_ordinal: int
    evidence_id: str
    source_id: str
    evidence_text_sha256: str
    rank: int
    weight: float
    weight_sha256: str
    link_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_EXTRACTION_LINK_FORMAT:
            raise FastCAVLinkError("unsupported fast CAV extraction-link format")
        _exact_int(
            self.concept_ordinal,
            "concept_ordinal",
            maximum=FAST_CAV_MAX_CONCEPTS - 1,
        )
        _sha256(self.concept_id, "concept_id")
        _sha256(self.concept_sha256, "concept_sha256")
        _exact_int(
            self.evidence_ordinal,
            "evidence_ordinal",
            maximum=FAST_CAV_MAX_EVIDENCE - 1,
        )
        _text(self.evidence_id, "evidence_id")
        _text(self.source_id, "source_id")
        _sha256(self.evidence_text_sha256, "evidence_text_sha256")
        _exact_int(
            self.rank,
            "rank",
            minimum=1,
            maximum=FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT,
        )
        weight = _weight(self.weight, "weight")
        if _sha256(self.weight_sha256, "weight_sha256") != _weight_sha256(weight):
            raise FastCAVLinkError("extraction-link weight changed canonical bytes")
        self._seal()


@dataclass(frozen=True, slots=True)
class FastCAVReinjectionLink(SealedIdentity):
    """One evidence node's ranked assignment from an updated CAV concept."""

    _SEAL_FIELD = "link_sha256"
    _SEAL_MISMATCH = "fast CAV reinjection link seal does not match its contents"

    format: str
    evidence_ordinal: int
    evidence_id: str
    source_id: str
    evidence_text_sha256: str
    concept_ordinal: int
    concept_id: str
    concept_sha256: str
    rank: int
    weight: float
    weight_sha256: str
    link_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_REINJECTION_LINK_FORMAT:
            raise FastCAVLinkError("unsupported fast CAV reinjection-link format")
        _exact_int(
            self.evidence_ordinal,
            "evidence_ordinal",
            maximum=FAST_CAV_MAX_EVIDENCE - 1,
        )
        _text(self.evidence_id, "evidence_id")
        _text(self.source_id, "source_id")
        _sha256(self.evidence_text_sha256, "evidence_text_sha256")
        _exact_int(
            self.concept_ordinal,
            "concept_ordinal",
            maximum=FAST_CAV_MAX_CONCEPTS - 1,
        )
        _sha256(self.concept_id, "concept_id")
        _sha256(self.concept_sha256, "concept_sha256")
        _exact_int(
            self.rank,
            "rank",
            minimum=1,
            maximum=FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE,
        )
        weight = _weight(self.weight, "weight")
        if _sha256(self.weight_sha256, "weight_sha256") != _weight_sha256(weight):
            raise FastCAVLinkError("reinjection-link weight changed canonical bytes")
        self._seal()


def _stable_top_indices(values: tuple[float, ...], limit: int) -> tuple[int, ...]:
    """Return stable top indices in O(width * fixed_limit) time."""

    selected: list[int] = []
    for index, value in enumerate(values):
        insertion = len(selected)
        for position, prior in enumerate(selected):
            if (-value, index) < (-values[prior], prior):
                insertion = position
                break
        selected.insert(insertion, index)
        if len(selected) > limit:
            selected.pop()
    return tuple(selected)


def _row(
    values: tuple[float, ...],
    *,
    row: int,
    width: int,
) -> tuple[float, ...]:
    start = row * width
    return values[start : start + width]


def _validate_attention_rows(
    values: tuple[float, ...],
    *,
    height: int,
    width: int,
    label: str,
) -> None:
    if len(values) != height * width:
        raise FastCAVLinkError(f"{label} scalar population changed")
    for row_index in range(height):
        row = _row(values, row=row_index, width=width)
        if any(not 0.0 <= value <= 1.0 for value in row):
            raise FastCAVLinkError(f"{label} contains a non-probability weight")
        if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=2e-5):
            raise FastCAVLinkError(f"{label} row is not softmax-normalized")


def _validate_ranked_links(rows: Sequence[Any], coordinate_name: str) -> None:
    if tuple(row.rank for row in rows) != tuple(range(1, len(rows) + 1)):
        raise FastCAVLinkError("CAV link ranks are not contiguous")
    coordinates = tuple(getattr(row, coordinate_name) for row in rows)
    observed = tuple(
        (-row.weight, coordinate)
        for row, coordinate in zip(rows, coordinates, strict=True)
    )
    if len(coordinates) != len(set(coordinates)) or observed != tuple(sorted(observed)):
        raise FastCAVLinkError("CAV links changed stable descending top order")


@dataclass(frozen=True, slots=True)
class FastCAVLinkReceipt(SealedIdentity):
    """Bounded scalar projection of genuine concept/evidence attention links."""

    _SEAL_FIELD = "link_receipt_sha256"
    _SEAL_MISMATCH = "fast CAV link receipt seal does not match its contents"

    format: str
    algorithm: str
    complexity_contract: str
    packet_identity_sha256: str
    router_runtime_identity_sha256: str
    router_bank_identity_sha256: str
    concepts: tuple[FastCAVConceptProvenance, ...]
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_text_sha256s: tuple[str, ...]
    extraction_shape: tuple[int, int]
    reinjection_shape: tuple[int, int]
    extraction_matrix_sha256: str
    reinjection_matrix_sha256: str
    canonical_dtype: str
    max_evidence_links_per_concept: int
    max_concept_links_per_evidence: int
    extraction_links: tuple[FastCAVExtractionLink, ...]
    reinjection_links: tuple[FastCAVReinjectionLink, ...]
    extraction_links_sha256: str
    reinjection_links_sha256: str
    rectangular_route_cell_count: int
    evidence_pair_matrix_constructed: bool
    evidence_pair_matrix_cell_count: int
    retained_token_id_count: int
    retained_tensor_bytes: int
    persisted_token_state_bytes: int
    link_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_CAV_LINK_RECEIPT_FORMAT:
            raise FastCAVLinkError("unsupported fast CAV link-receipt format")
        if self.algorithm != FIXED_CAV_ALGORITHM:
            raise FastCAVLinkError("CAV link receipt changed router algorithm")
        if self.complexity_contract != FAST_CAV_LINK_COMPLEXITY:
            raise FastCAVLinkError("CAV link receipt changed O(KN) complexity")
        for name in (
            "packet_identity_sha256",
            "router_runtime_identity_sha256",
            "router_bank_identity_sha256",
            "extraction_matrix_sha256",
            "reinjection_matrix_sha256",
            "extraction_links_sha256",
            "reinjection_links_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.concepts) is not tuple or not self.concepts or any(
            type(row) is not FastCAVConceptProvenance for row in self.concepts
        ):
            raise FastCAVLinkError("concepts must be a non-empty exact tuple")
        concept_count = len(self.concepts)
        if concept_count > FAST_CAV_MAX_CONCEPTS or tuple(
            row.concept_ordinal for row in self.concepts
        ) != tuple(range(concept_count)):
            raise FastCAVLinkError("concept coordinates are not bounded and contiguous")
        if any(
            row.bank_identity_sha256 != self.router_bank_identity_sha256
            for row in self.concepts
        ):
            raise FastCAVLinkError("concept changed router-bank provenance")
        evidence_ids = _strings(self.evidence_ids, "evidence_ids", unique=True)
        source_ids = _strings(self.source_ids, "source_ids", unique=False)
        text_hashes = _strings(
            self.evidence_text_sha256s,
            "evidence_text_sha256s",
            unique=False,
        )
        for index, value in enumerate(text_hashes):
            _sha256(value, f"evidence_text_sha256s[{index}]")
        evidence_count = len(evidence_ids)
        if not 1 <= evidence_count <= FAST_CAV_MAX_EVIDENCE or not (
            len(source_ids) == len(text_hashes) == evidence_count
        ):
            raise FastCAVLinkError("evidence provenance coordinates disagree")
        if self.extraction_shape != (concept_count, evidence_count) or (
            self.reinjection_shape != (evidence_count, concept_count)
        ):
            raise FastCAVLinkError("CAV links changed the exact KxN/NxK shapes")
        if self.canonical_dtype != CANONICAL_TENSOR_DTYPE:
            raise FastCAVLinkError("CAV link matrices must use canonical float32-le")
        extraction_cap = _exact_int(
            self.max_evidence_links_per_concept,
            "max_evidence_links_per_concept",
            minimum=1,
            maximum=FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT,
        )
        reinjection_cap = _exact_int(
            self.max_concept_links_per_evidence,
            "max_concept_links_per_evidence",
            minimum=1,
            maximum=FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE,
        )
        if type(self.extraction_links) is not tuple or any(
            type(row) is not FastCAVExtractionLink for row in self.extraction_links
        ):
            raise FastCAVLinkError("extraction_links must be an exact tuple")
        if type(self.reinjection_links) is not tuple or any(
            type(row) is not FastCAVReinjectionLink for row in self.reinjection_links
        ):
            raise FastCAVLinkError("reinjection_links must be an exact tuple")
        expected_extraction_count = concept_count * min(
            extraction_cap,
            evidence_count,
        )
        expected_reinjection_count = evidence_count * min(
            reinjection_cap,
            concept_count,
        )
        if len(self.extraction_links) != expected_extraction_count or len(
            self.reinjection_links
        ) != expected_reinjection_count:
            raise FastCAVLinkError("bounded CAV link population count changed")
        extraction_width = min(extraction_cap, evidence_count)
        reinjection_width = min(reinjection_cap, concept_count)
        if tuple(row.concept_ordinal for row in self.extraction_links) != tuple(
            concept for concept in range(concept_count) for _ in range(extraction_width)
        ) or tuple(row.evidence_ordinal for row in self.reinjection_links) != tuple(
            evidence for evidence in range(evidence_count) for _ in range(reinjection_width)
        ):
            raise FastCAVLinkError("CAV link groups changed canonical coordinate order")
        concepts = {row.concept_ordinal: row for row in self.concepts}
        evidence = {
            ordinal: (evidence_id, source_ids[ordinal], text_hashes[ordinal])
            for ordinal, evidence_id in enumerate(evidence_ids)
        }
        extraction_groups: dict[int, list[FastCAVExtractionLink]] = {}
        for link in self.extraction_links:
            concept = concepts.get(link.concept_ordinal)
            coordinate = evidence.get(link.evidence_ordinal)
            if concept is None or coordinate is None or (
                link.concept_id,
                link.concept_sha256,
            ) != (concept.concept_id, concept.concept_sha256) or (
                link.evidence_id,
                link.source_id,
                link.evidence_text_sha256,
            ) != coordinate:
                raise FastCAVLinkError("extraction link changed exact provenance")
            extraction_groups.setdefault(link.concept_ordinal, []).append(link)
        reinjection_groups: dict[int, list[FastCAVReinjectionLink]] = {}
        for link in self.reinjection_links:
            concept = concepts.get(link.concept_ordinal)
            coordinate = evidence.get(link.evidence_ordinal)
            if concept is None or coordinate is None or (
                link.concept_id,
                link.concept_sha256,
            ) != (concept.concept_id, concept.concept_sha256) or (
                link.evidence_id,
                link.source_id,
                link.evidence_text_sha256,
            ) != coordinate:
                raise FastCAVLinkError("reinjection link changed exact provenance")
            reinjection_groups.setdefault(link.evidence_ordinal, []).append(link)
        for rows in extraction_groups.values():
            _validate_ranked_links(rows, "evidence_ordinal")
        for rows in reinjection_groups.values():
            _validate_ranked_links(rows, "concept_ordinal")
        expected_extraction_sha = identity_sha256(
            [row.identity_payload() for row in self.extraction_links]
        )
        expected_reinjection_sha = identity_sha256(
            [row.identity_payload() for row in self.reinjection_links]
        )
        if self.extraction_links_sha256 != expected_extraction_sha or (
            self.reinjection_links_sha256 != expected_reinjection_sha
        ):
            raise FastCAVLinkError("CAV link population hashes changed")
        expected_cells = 2 * concept_count * evidence_count
        if (
            self.rectangular_route_cell_count != expected_cells
            or expected_cells > FAST_CAV_MAX_RECTANGULAR_ROUTE_CELLS
        ):
            raise FastCAVLinkError("CAV rectangular route-cell bound changed")
        if self.evidence_pair_matrix_constructed is not False:
            raise FastCAVLinkError("CAV link receipt cannot claim an N x N matrix")
        _zero(self.evidence_pair_matrix_cell_count, "evidence_pair_matrix_cell_count")
        _zero(self.retained_token_id_count, "retained_token_id_count")
        _zero(self.retained_tensor_bytes, "retained_tensor_bytes")
        _zero(self.persisted_token_state_bytes, "persisted_token_state_bytes")
        self._seal()


def build_fast_cav_link_receipt(
    *,
    packet_identity_sha256: str,
    router_runtime_identity_sha256: str,
    router_bank_identity_sha256: str,
    concepts: Sequence[FastCAVConceptProvenance],
    evidence_ids: Sequence[str],
    source_ids: Sequence[str],
    evidence_text_sha256s: Sequence[str],
    extraction_attention: Any,
    reinjection_attention: Any,
    max_evidence_links_per_concept: int = (
        FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT
    ),
    max_concept_links_per_evidence: int = (
        FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE
    ),
) -> FastCAVLinkReceipt:
    """Consume transient E[K,N]/R[N,K] and retain bounded scalar links."""

    packet_sha = _sha256(packet_identity_sha256, "packet_identity_sha256")
    runtime_sha = _sha256(
        router_runtime_identity_sha256,
        "router_runtime_identity_sha256",
    )
    bank_sha = _sha256(router_bank_identity_sha256, "router_bank_identity_sha256")
    concept_rows = tuple(concepts)
    if not concept_rows or any(
        type(row) is not FastCAVConceptProvenance for row in concept_rows
    ):
        raise FastCAVLinkError("concepts must contain exact provenance rows")
    evidence = _strings(tuple(evidence_ids), "evidence_ids", unique=True)
    sources = _strings(tuple(source_ids), "source_ids", unique=False)
    text_hashes = _strings(
        tuple(evidence_text_sha256s),
        "evidence_text_sha256s",
        unique=False,
    )
    concept_count = len(concept_rows)
    evidence_count = len(evidence)
    if not 1 <= concept_count <= FAST_CAV_MAX_CONCEPTS or not (
        1 <= evidence_count <= FAST_CAV_MAX_EVIDENCE
    ):
        raise FastCAVLinkError("CAV K/N dimensions exceed their hard bounds")
    if len(sources) != evidence_count or len(text_hashes) != evidence_count:
        raise FastCAVLinkError("evidence provenance coordinates disagree")
    for index, value in enumerate(text_hashes):
        _sha256(value, f"evidence_text_sha256s[{index}]")
    extraction = canonical_float32_tensor(
        extraction_attention,
        label="fixed CAV extraction attention E[K,N]",
    )
    reinjection = canonical_float32_tensor(
        reinjection_attention,
        label="fixed CAV reinjection attention R[N,K]",
    )
    if extraction.shape != (concept_count, evidence_count) or (
        reinjection.shape != (evidence_count, concept_count)
    ):
        raise FastCAVLinkError("transient CAV matrices changed KxN/NxK structure")
    _validate_attention_rows(
        extraction.flat_values,
        height=concept_count,
        width=evidence_count,
        label="extraction attention",
    )
    _validate_attention_rows(
        reinjection.flat_values,
        height=evidence_count,
        width=concept_count,
        label="reinjection attention",
    )
    extraction_cap = _exact_int(
        max_evidence_links_per_concept,
        "max_evidence_links_per_concept",
        minimum=1,
        maximum=FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT,
    )
    reinjection_cap = _exact_int(
        max_concept_links_per_evidence,
        "max_concept_links_per_evidence",
        minimum=1,
        maximum=FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE,
    )
    extraction_links: list[FastCAVExtractionLink] = []
    for concept_ordinal, concept in enumerate(concept_rows):
        row = _row(
            extraction.flat_values,
            row=concept_ordinal,
            width=evidence_count,
        )
        for rank, evidence_ordinal in enumerate(
            _stable_top_indices(row, min(extraction_cap, evidence_count)),
            start=1,
        ):
            weight = row[evidence_ordinal]
            extraction_links.append(
                FastCAVExtractionLink(
                    format=FAST_CAV_EXTRACTION_LINK_FORMAT,
                    concept_ordinal=concept_ordinal,
                    concept_id=concept.concept_id,
                    concept_sha256=concept.concept_sha256,
                    evidence_ordinal=evidence_ordinal,
                    evidence_id=evidence[evidence_ordinal],
                    source_id=sources[evidence_ordinal],
                    evidence_text_sha256=text_hashes[evidence_ordinal],
                    rank=rank,
                    weight=weight,
                    weight_sha256=_weight_sha256(weight),
                )
            )
    reinjection_links: list[FastCAVReinjectionLink] = []
    for evidence_ordinal in range(evidence_count):
        row = _row(
            reinjection.flat_values,
            row=evidence_ordinal,
            width=concept_count,
        )
        for rank, concept_ordinal in enumerate(
            _stable_top_indices(row, min(reinjection_cap, concept_count)),
            start=1,
        ):
            concept = concept_rows[concept_ordinal]
            weight = row[concept_ordinal]
            reinjection_links.append(
                FastCAVReinjectionLink(
                    format=FAST_CAV_REINJECTION_LINK_FORMAT,
                    evidence_ordinal=evidence_ordinal,
                    evidence_id=evidence[evidence_ordinal],
                    source_id=sources[evidence_ordinal],
                    evidence_text_sha256=text_hashes[evidence_ordinal],
                    concept_ordinal=concept_ordinal,
                    concept_id=concept.concept_id,
                    concept_sha256=concept.concept_sha256,
                    rank=rank,
                    weight=weight,
                    weight_sha256=_weight_sha256(weight),
                )
            )
    extraction_tuple = tuple(extraction_links)
    reinjection_tuple = tuple(reinjection_links)
    return FastCAVLinkReceipt(
        format=FAST_CAV_LINK_RECEIPT_FORMAT,
        algorithm=FIXED_CAV_ALGORITHM,
        complexity_contract=FAST_CAV_LINK_COMPLEXITY,
        packet_identity_sha256=packet_sha,
        router_runtime_identity_sha256=runtime_sha,
        router_bank_identity_sha256=bank_sha,
        concepts=concept_rows,
        evidence_ids=evidence,
        source_ids=sources,
        evidence_text_sha256s=text_hashes,
        extraction_shape=(concept_count, evidence_count),
        reinjection_shape=(evidence_count, concept_count),
        extraction_matrix_sha256=extraction.tensor_sha256,
        reinjection_matrix_sha256=reinjection.tensor_sha256,
        canonical_dtype=CANONICAL_TENSOR_DTYPE,
        max_evidence_links_per_concept=extraction_cap,
        max_concept_links_per_evidence=reinjection_cap,
        extraction_links=extraction_tuple,
        reinjection_links=reinjection_tuple,
        extraction_links_sha256=identity_sha256(
            [row.identity_payload() for row in extraction_tuple]
        ),
        reinjection_links_sha256=identity_sha256(
            [row.identity_payload() for row in reinjection_tuple]
        ),
        rectangular_route_cell_count=2 * concept_count * evidence_count,
        evidence_pair_matrix_constructed=False,
        evidence_pair_matrix_cell_count=0,
        retained_token_id_count=0,
        retained_tensor_bytes=0,
        persisted_token_state_bytes=0,
    )


__all__ = [
    "FAST_CAV_CONCEPT_FORMAT",
    "FAST_CAV_EXTRACTION_LINK_FORMAT",
    "FAST_CAV_LINK_COMPLEXITY",
    "FAST_CAV_LINK_RECEIPT_FORMAT",
    "FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE",
    "FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT",
    "FAST_CAV_REINJECTION_LINK_FORMAT",
    "FastCAVConceptProvenance",
    "FastCAVExtractionLink",
    "FastCAVLinkError",
    "FastCAVLinkReceipt",
    "FastCAVReinjectionLink",
    "build_fast_cav_concepts",
    "build_fast_cav_concepts_from_router",
    "build_fast_cav_link_receipt",
]
