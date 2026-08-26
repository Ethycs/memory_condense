"""Provider-free append-only Hebbian H2 proposals over exact S3 evidence.

H2 is deliberately separate from the older matched S0/H1 replacement arm.
It starts from every sealed S3 evidence row in its published order and may
only append a robustly supported source chunk.  The history/derived-store
producer identity is recorded as provenance; it is never required to equal
the current H2 consumer implementation identity.

No answer, gold label, provider, CAV tensor, or CAV link is consumed here.
Candidate admission is preflighted through the canonical downstream
CAV-synthesis scaffold with its immutable guide-slot sentinel and the same
8,000-token hard proxy cap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.coaccess_graph import rank_discount
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._fast_hebbian_h2_consumer import (
    FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES,
    FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM,
    FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE,
    FastHebbianH2ConsumerSourceManifest,
    build_fast_hebbian_h2_consumer_source_manifest,
)
from memory_condense.eval._fast_hebbian_h2_io import (
    FastHebbianH2ValidationError,
    read_canonical_json as _read_canonical_json,
    verify_digest_anchor as _verify_digest_anchor,
)
from memory_condense.eval._fast_hebbian_h2_scaffold import (
    build_fast_hebbian_h2_scaffold as _scaffold,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_POLICY_SHA256,
)
from memory_condense.eval.hebbian_derived_store import (
    MANIFEST_NAME as HEBBIAN_DERIVED_STORE_MANIFEST,
    HebbianDerivedStoreReceipt,
    load_hebbian_derived_store_receipt,
    verify_hebbian_derived_store,
)
from memory_condense.eval.hebbian_history import (
    HebbianHistoryArtifact,
    load_hebbian_history_artifact,
    verify_hebbian_history_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
    FastRetrievalArtifact,
)
from memory_condense.eval.reproducibility import environment_lock_sha256
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result


FAST_HEBBIAN_H2_POLICY_FORMAT = "memory-condense-fast-hebbian-h2-policy-static-local-closure-v3"
FAST_HEBBIAN_H2_S3_SOURCE_FORMAT = "memory-condense-fast-hebbian-h2-s3-source-coordinates-v1"
FAST_HEBBIAN_H2_EVIDENCE_FORMAT = "memory-condense-fast-hebbian-h2-evidence-coordinate-v1"
FAST_HEBBIAN_H2_CANDIDATE_FORMAT = "memory-condense-fast-hebbian-h2-candidate-receipt-v1"
FAST_HEBBIAN_H2_QUESTION_FORMAT = "memory-condense-fast-hebbian-h2-question-receipt-static-local-closure-v3"
FAST_HEBBIAN_H2_POPULATION_FORMAT = "memory-condense-fast-hebbian-h2-population-receipt-static-local-closure-v3"
FAST_HEBBIAN_H2_STAGE_ID = STAGE_IDS[-1]
FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS = FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS
FAST_HEBBIAN_H2_DEFAULT_MIN_SUPPORT = 2
FAST_HEBBIAN_H2_DEFAULT_MIN_COACCESS_COUNT = 2
FAST_HEBBIAN_H2_DEFAULT_MAX_ADDITIONS = 1
FAST_HEBBIAN_H2_DEFAULT_MAX_SEED_CHUNKS = 64
FAST_HEBBIAN_H2_DEFAULT_MAX_NEIGHBORS = 256
FAST_HEBBIAN_H2_DEFAULT_HALF_LIFE_TURNS = 200.0
FAST_HEBBIAN_H2_DEFAULT_MIN_SCORE = 0.05

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_STATUSES = frozenset({"appended", "budget_rejected", "addition_cap_rejected"})
_OUTCOMES = frozenset({"appended", "no_robust_candidate", "no_budget_admissible_candidate"})


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FastHebbianH2ValidationError(
            f"{label} must be an exact non-empty string"
        )
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise FastHebbianH2ValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        suffix = f" in [{minimum}, {maximum}]" if maximum is not None else ""
        raise FastHebbianH2ValidationError(f"{label} must be an integer{suffix}")
    return value


def _finite(value: object, label: str, *, minimum: float) -> float:
    if type(value) not in (int, float):
        raise FastHebbianH2ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise FastHebbianH2ValidationError(
            f"{label} must be finite and at least {minimum}"
        )
    return result


def _ids(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise FastHebbianH2ValidationError(f"{label} must be an exact tuple")
    rows = tuple(_text(row, f"{label} item") for row in value)
    if not allow_empty and not rows:
        raise FastHebbianH2ValidationError(f"{label} must not be empty")
    if len(rows) != len(set(rows)):
        raise FastHebbianH2ValidationError(f"{label} must contain unique IDs")
    return rows


@dataclass(frozen=True, slots=True)
class FastHebbianH2Policy(SealedIdentity):
    """Frozen H2 graph/admission policy, separate from negative S0/H1."""

    _SEAL_FIELD = "policy_sha256"
    _SEAL_MISMATCH = "H2 policy seal does not match its contents"

    format: str = FAST_HEBBIAN_H2_POLICY_FORMAT
    stage_id: str = FAST_HEBBIAN_H2_STAGE_ID
    intervention: str = "append-only-no-replacement-v1"
    ranking: str = "graph-score-desc-support-desc-chunk-id-asc-v1"
    max_seed_chunks: int = FAST_HEBBIAN_H2_DEFAULT_MAX_SEED_CHUNKS
    max_neighbor_candidates: int = FAST_HEBBIAN_H2_DEFAULT_MAX_NEIGHBORS
    max_additions: int = FAST_HEBBIAN_H2_DEFAULT_MAX_ADDITIONS
    min_support: int = FAST_HEBBIAN_H2_DEFAULT_MIN_SUPPORT
    min_coaccess_count: int = FAST_HEBBIAN_H2_DEFAULT_MIN_COACCESS_COUNT
    half_life_turns: float = FAST_HEBBIAN_H2_DEFAULT_HALF_LIFE_TURNS
    min_score: float = FAST_HEBBIAN_H2_DEFAULT_MIN_SCORE
    hard_prompt_token_cap: int = FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS
    downstream_synthesis_policy_sha256: str = (
        FAST_CAV_LINK_SYNTHESIS_POLICY_SHA256
    )
    downstream_guide_state: str = "immutable-guide-slot-sentinel-before-cav-v1"
    consumer_identity_contract: str = FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM
    consumer_identity_scope: str = FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE
    consumer_environment_contract: str = "pixi-lock-file-sha256-v1"
    gold_fields_consumed: bool = False
    cav_links_computed: bool = False
    retained_request_token_state_bytes: int = 0
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_POLICY_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 policy format")
        if self.stage_id != FAST_HEBBIAN_H2_STAGE_ID:
            raise FastHebbianH2ValidationError("H2 policy must start at exact S3")
        if self.intervention != "append-only-no-replacement-v1":
            raise FastHebbianH2ValidationError("H2 policy changed append-only mode")
        if self.ranking != "graph-score-desc-support-desc-chunk-id-asc-v1":
            raise FastHebbianH2ValidationError("H2 policy changed ranking")
        _integer(self.max_seed_chunks, "max_seed_chunks", minimum=1, maximum=512)
        _integer(
            self.max_neighbor_candidates,
            "max_neighbor_candidates",
            minimum=1,
            maximum=4096,
        )
        _integer(
            self.max_additions,
            "max_additions",
            minimum=1,
            maximum=self.max_neighbor_candidates,
        )
        _integer(self.min_support, "min_support", minimum=2)
        _integer(self.min_coaccess_count, "min_coaccess_count", minimum=2)
        half_life = _finite(self.half_life_turns, "half_life_turns", minimum=0.0)
        score = _finite(self.min_score, "min_score", minimum=0.0)
        if half_life <= 0.0 or score > 1.0:
            raise FastHebbianH2ValidationError("invalid H2 graph scoring policy")
        object.__setattr__(self, "half_life_turns", half_life)
        object.__setattr__(self, "min_score", score)
        if self.hard_prompt_token_cap != FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS:
            raise FastHebbianH2ValidationError("H2 hard prompt cap must be 8000")
        if (
            self.downstream_synthesis_policy_sha256
            != FAST_CAV_LINK_SYNTHESIS_POLICY_SHA256
            or self.downstream_guide_state
            != "immutable-guide-slot-sentinel-before-cav-v1"
        ):
            raise FastHebbianH2ValidationError(
                "H2 changed the downstream canonical scaffold binding"
            )
        if self.consumer_identity_contract != FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM:
            raise FastHebbianH2ValidationError(
                "H2 changed the scoped consumer identity contract"
            )
        if self.consumer_identity_scope != FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE:
            raise FastHebbianH2ValidationError(
                "H2 changed the static local-import closure scope"
            )
        if self.consumer_environment_contract != "pixi-lock-file-sha256-v1":
            raise FastHebbianH2ValidationError(
                "H2 changed the consumer environment-lock contract"
            )
        if self.gold_fields_consumed is not False or self.cav_links_computed is not False:
            raise FastHebbianH2ValidationError("H2 policy claimed gold or CAV work")
        if self.retained_request_token_state_bytes != 0:
            raise FastHebbianH2ValidationError("H2 policy retained token state")
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianH2S3QuestionSource(SealedIdentity):
    """Exact S3 evidence/source-chunk coordinates retained by retrieval JSON."""

    _SEAL_FIELD = "coordinate_sha256"
    _SEAL_MISMATCH = "H2 S3 coordinate seal does not match its contents"

    question_ordinal: int
    question_id: str
    retrieval_question_receipt_sha256: str
    s3_stage_receipt_sha256: str
    s3_evidence_projection_sha256: str
    s3_evidence_ids: tuple[str, ...]
    s3_source_chunk_ids: tuple[str, ...]
    coordinate_sha256: str = ""

    def __post_init__(self) -> None:
        _integer(self.question_ordinal, "question_ordinal")
        _text(self.question_id, "question_id")
        for name in (
            "retrieval_question_receipt_sha256",
            "s3_stage_receipt_sha256",
            "s3_evidence_projection_sha256",
        ):
            _digest(getattr(self, name), name)
        _ids(self.s3_evidence_ids, "s3_evidence_ids")
        _ids(self.s3_source_chunk_ids, "s3_source_chunk_ids")
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianH2RetrievalSource(SealedIdentity):
    """Provider-free projection of the raw receipt fields absent from the view."""

    _SEAL_FIELD = "source_sha256"
    _SEAL_MISMATCH = "H2 retrieval source seal does not match its contents"
    _PAYLOAD_EXCLUDE = frozenset({"source_path"})

    format: str
    source_path: str
    retrieval_artifact_sha256: str
    questions: tuple[FastHebbianH2S3QuestionSource, ...]
    source_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_S3_SOURCE_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 S3 source format")
        _text(self.source_path, "source_path")
        _digest(self.retrieval_artifact_sha256, "retrieval_artifact_sha256")
        if type(self.questions) is not tuple or not self.questions or any(
            type(row) is not FastHebbianH2S3QuestionSource for row in self.questions
        ):
            raise FastHebbianH2ValidationError(
                "H2 retrieval source requires exact question coordinates"
            )
        if tuple(row.question_ordinal for row in self.questions) != tuple(
            range(len(self.questions))
        ) or len({row.question_id for row in self.questions}) != len(self.questions):
            raise FastHebbianH2ValidationError(
                "H2 retrieval source changed question order or identity"
            )
        self._seal()

    def question(self, question_id: str) -> FastHebbianH2S3QuestionSource:
        for row in self.questions:
            if row.question_id == question_id:
                return row
        raise KeyError(question_id)


@dataclass(frozen=True, slots=True)
class FastHebbianH2HistorySource:
    """Canonical history file identity plus its fully verified typed artifact."""

    source_path: str
    raw_sha256: str
    artifact: HebbianHistoryArtifact

    def __post_init__(self) -> None:
        _text(self.source_path, "source_path")
        _digest(self.raw_sha256, "raw_sha256")
        if type(self.artifact) is not HebbianHistoryArtifact:
            raise TypeError("artifact must be an exact HebbianHistoryArtifact")
        verify_hebbian_history_artifact(self.artifact)


@dataclass(frozen=True, slots=True)
class FastHebbianH2EvidenceCoordinate:
    """Text-free final evidence coordinate; text is bound by SHA-256."""

    format: str
    evidence_ordinal: int
    evidence_id: str
    source_id: str
    evidence_text_sha256: str
    source_chunk_id: str | None
    origin: Literal["s3", "hebbian_h2"]

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_EVIDENCE_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 evidence format")
        _integer(self.evidence_ordinal, "evidence_ordinal")
        _text(self.evidence_id, "evidence_id")
        _text(self.source_id, "source_id")
        _digest(self.evidence_text_sha256, "evidence_text_sha256")
        if self.origin == "s3":
            if self.source_chunk_id is not None:
                raise FastHebbianH2ValidationError(
                    "S3 atom coordinates must use the separately sealed source map"
                )
        elif self.origin == "hebbian_h2":
            _text(self.source_chunk_id, "source_chunk_id")
        else:
            raise FastHebbianH2ValidationError("unsupported H2 evidence origin")

    def identity_payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "evidence_ordinal": self.evidence_ordinal,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "evidence_text_sha256": self.evidence_text_sha256,
            "source_chunk_id": self.source_chunk_id,
            "origin": self.origin,
        }


FastHebbianH2CandidateStatus = Literal[
    "appended", "budget_rejected", "addition_cap_rejected"
]


@dataclass(frozen=True, slots=True)
class FastHebbianH2CandidateReceipt(SealedIdentity):
    """One robust graph neighbor and its append-only admission outcome."""

    _SEAL_FIELD = "candidate_receipt_sha256"
    _SEAL_MISMATCH = "H2 candidate receipt does not match its contents"

    format: str
    rank: int
    source_chunk_id: str
    evidence_id: str
    source_id: str
    evidence_text_sha256: str
    score: float
    support: int
    anchor_chunk_id: str
    coaccess_count: int
    last_reinforced_turn: int
    admission_status: FastHebbianH2CandidateStatus
    proposed_prompt_token_proxy: int
    final_evidence_ordinal: int | None
    candidate_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_CANDIDATE_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 candidate format")
        _integer(self.rank, "rank", minimum=1)
        for name in ("source_chunk_id", "evidence_id", "source_id", "anchor_chunk_id"):
            _text(getattr(self, name), name)
        _digest(self.evidence_text_sha256, "evidence_text_sha256")
        score = _finite(self.score, "score", minimum=0.0)
        if score > 1.0:
            raise FastHebbianH2ValidationError("candidate score exceeds one")
        object.__setattr__(self, "score", score)
        _integer(self.support, "support", minimum=1)
        _integer(self.coaccess_count, "coaccess_count", minimum=1)
        _integer(self.last_reinforced_turn, "last_reinforced_turn")
        if self.admission_status not in _CANDIDATE_STATUSES:
            raise FastHebbianH2ValidationError("unsupported H2 admission status")
        _integer(
            self.proposed_prompt_token_proxy,
            "proposed_prompt_token_proxy",
            minimum=1,
        )
        if self.admission_status == "appended":
            _integer(self.final_evidence_ordinal, "final_evidence_ordinal")
        elif self.final_evidence_ordinal is not None:
            raise FastHebbianH2ValidationError(
                "rejected H2 candidate gained a final evidence coordinate"
            )
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianH2QuestionReceipt(SealedIdentity):
    """Sealed append-only H2 proposal for one exact S3 question."""

    _SEAL_FIELD = "receipt_sha256"
    _SEAL_MISMATCH = "H2 question receipt does not match its contents"

    format: str
    question_ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    stage_id: str
    retrieval_artifact_sha256: str
    retrieval_question_receipt_sha256: str
    s3_stage_receipt_sha256: str
    s3_evidence_projection_sha256: str
    s3_source_coordinate_sha256: str
    history_receipt_sha256: str
    derived_store_receipt_sha256: str
    h2_policy_sha256: str
    h2_consumer_source_sha256: str
    h2_consumer_source_manifest_sha256: str
    h2_consumer_environment_lock_sha256: str
    seed_source_chunk_ids: tuple[str, ...]
    base_s3_coordinates: tuple[FastHebbianH2EvidenceCoordinate, ...]
    final_coordinates: tuple[FastHebbianH2EvidenceCoordinate, ...]
    ranked_candidates: tuple[FastHebbianH2CandidateReceipt, ...]
    appended_source_chunk_ids: tuple[str, ...]
    appended_evidence_ids: tuple[str, ...]
    outcome: str
    base_evidence_catalog_sha256: str
    final_evidence_catalog_sha256: str
    base_scaffold_sha256: str
    final_scaffold_sha256: str
    base_prompt_token_proxy: int
    final_prompt_token_proxy: int
    hard_prompt_token_cap: int
    gold_fields_consumed: bool
    cav_links_computed: bool
    retained_request_token_state_bytes: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_QUESTION_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 question format")
        _integer(self.question_ordinal, "question_ordinal")
        _text(self.question_id, "question_id")
        if self.stage_id != FAST_HEBBIAN_H2_STAGE_ID:
            raise FastHebbianH2ValidationError("H2 question must remain S3-based")
        for name in (
            "question_sha256",
            "dated_question_sha256",
            "retrieval_artifact_sha256",
            "retrieval_question_receipt_sha256",
            "s3_stage_receipt_sha256",
            "s3_evidence_projection_sha256",
            "s3_source_coordinate_sha256",
            "history_receipt_sha256",
            "derived_store_receipt_sha256",
            "h2_policy_sha256",
            "h2_consumer_source_sha256",
            "h2_consumer_source_manifest_sha256",
            "h2_consumer_environment_lock_sha256",
            "base_evidence_catalog_sha256",
            "final_evidence_catalog_sha256",
            "base_scaffold_sha256",
            "final_scaffold_sha256",
        ):
            _digest(getattr(self, name), name)
        seeds = _ids(self.seed_source_chunk_ids, "seed_source_chunk_ids")
        appended_chunks = _ids(
            self.appended_source_chunk_ids,
            "appended_source_chunk_ids",
            allow_empty=True,
        )
        appended_evidence = _ids(
            self.appended_evidence_ids,
            "appended_evidence_ids",
            allow_empty=True,
        )
        if len(appended_chunks) != len(appended_evidence):
            raise FastHebbianH2ValidationError(
                "appended source/evidence coordinates changed cardinality"
            )
        if type(self.base_s3_coordinates) is not tuple or not self.base_s3_coordinates:
            raise FastHebbianH2ValidationError("H2 base S3 coordinates are empty")
        if type(self.final_coordinates) is not tuple or any(
            type(row) is not FastHebbianH2EvidenceCoordinate
            for row in self.final_coordinates
        ):
            raise FastHebbianH2ValidationError("H2 final coordinates changed type")
        base = self.base_s3_coordinates
        if any(
            type(row) is not FastHebbianH2EvidenceCoordinate or row.origin != "s3"
            for row in base
        ):
            raise FastHebbianH2ValidationError("H2 base contains non-S3 evidence")
        if self.final_coordinates[: len(base)] != base:
            raise FastHebbianH2ValidationError(
                "H2 final evidence is not an exact S3 prefix"
            )
        tail = self.final_coordinates[len(base) :]
        if (
            tuple(row.source_chunk_id for row in tail) != appended_chunks
            or tuple(row.evidence_id for row in tail) != appended_evidence
            or any(row.origin != "hebbian_h2" for row in tail)
        ):
            raise FastHebbianH2ValidationError("H2 appended tail changed membership")
        if tuple(row.evidence_ordinal for row in self.final_coordinates) != tuple(
            range(len(self.final_coordinates))
        ):
            raise FastHebbianH2ValidationError("H2 evidence ordinals changed")
        if len({row.evidence_id for row in self.final_coordinates}) != len(
            self.final_coordinates
        ):
            raise FastHebbianH2ValidationError("H2 final evidence IDs are not unique")
        if set(appended_chunks).intersection(seeds):
            raise FastHebbianH2ValidationError("H2 appended an existing S3 source")
        if type(self.ranked_candidates) is not tuple or any(
            type(row) is not FastHebbianH2CandidateReceipt
            for row in self.ranked_candidates
        ):
            raise FastHebbianH2ValidationError("H2 candidates changed type")
        if tuple(row.rank for row in self.ranked_candidates) != tuple(
            range(1, len(self.ranked_candidates) + 1)
        ):
            raise FastHebbianH2ValidationError("H2 candidate ranking changed")
        admitted = tuple(
            row for row in self.ranked_candidates if row.admission_status == "appended"
        )
        if (
            tuple(row.source_chunk_id for row in admitted) != appended_chunks
            or tuple(row.evidence_id for row in admitted) != appended_evidence
            or tuple(row.final_evidence_ordinal for row in admitted)
            != tuple(row.evidence_ordinal for row in tail)
        ):
            raise FastHebbianH2ValidationError(
                "H2 candidate decisions changed the appended tail"
            )
        if self.outcome not in _OUTCOMES:
            raise FastHebbianH2ValidationError("unsupported H2 question outcome")
        expected_outcome = (
            "appended"
            if admitted
            else (
                "no_robust_candidate"
                if not self.ranked_candidates
                else "no_budget_admissible_candidate"
            )
        )
        if self.outcome != expected_outcome:
            raise FastHebbianH2ValidationError("H2 question outcome changed")
        base_tokens = _integer(
            self.base_prompt_token_proxy, "base_prompt_token_proxy", minimum=1
        )
        final_tokens = _integer(
            self.final_prompt_token_proxy, "final_prompt_token_proxy", minimum=1
        )
        if (
            self.hard_prompt_token_cap != FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS
            or base_tokens > self.hard_prompt_token_cap
            or final_tokens > self.hard_prompt_token_cap
        ):
            raise FastHebbianH2ValidationError("H2 prompt exceeded hard 8k cap")
        if not appended_chunks and (
            self.base_s3_coordinates != self.final_coordinates
            or self.base_evidence_catalog_sha256
            != self.final_evidence_catalog_sha256
            or self.base_scaffold_sha256 != self.final_scaffold_sha256
            or base_tokens != final_tokens
        ):
            raise FastHebbianH2ValidationError(
                "no-op H2 did not preserve exact S3 evidence/scaffold"
            )
        if self.gold_fields_consumed is not False or self.cav_links_computed is not False:
            raise FastHebbianH2ValidationError("H2 receipt claimed gold or CAV work")
        if self.retained_request_token_state_bytes != 0:
            raise FastHebbianH2ValidationError("H2 receipt retained token state")
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianH2Population:
    """Sealed text-free receipt plus transient final evidence for downstream CAV."""

    format: str
    retrieval_artifact_sha256: str
    retrieval_source_sha256: str
    history_file_sha256: str
    history_artifact_sha256: str
    history_receipt_sha256: str
    history_event_population_sha256: str
    history_producer_implementation_sha256: str
    history_producer_environment_sha256: str
    derived_store_receipt_sha256: str
    derived_database_sha256: str
    derived_association_artifact_id: str
    derived_association_artifact_sha256: str
    derived_store_producer_implementation_sha256: str
    h2_consumer_source_manifest: FastHebbianH2ConsumerSourceManifest
    h2_consumer_source_sha256: str
    h2_consumer_environment_lock_sha256: str
    policy: FastHebbianH2Policy
    question_receipts: tuple[FastHebbianH2QuestionReceipt, ...]
    final_evidence: tuple[tuple[FastEvidence, ...], ...]
    gold_fields_consumed: bool = False
    provider_calls: int = 0
    cav_links_computed: bool = False
    retained_request_token_state_bytes: int = 0
    population_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_H2_POPULATION_FORMAT:
            raise FastHebbianH2ValidationError("unsupported H2 population format")
        for name in (
            "retrieval_artifact_sha256",
            "retrieval_source_sha256",
            "history_file_sha256",
            "history_artifact_sha256",
            "history_receipt_sha256",
            "history_event_population_sha256",
            "history_producer_implementation_sha256",
            "history_producer_environment_sha256",
            "derived_store_receipt_sha256",
            "derived_database_sha256",
            "derived_association_artifact_sha256",
            "derived_store_producer_implementation_sha256",
            "h2_consumer_source_sha256",
            "h2_consumer_environment_lock_sha256",
        ):
            _digest(getattr(self, name), name)
        _text(self.derived_association_artifact_id, "derived_association_artifact_id")
        if type(self.h2_consumer_source_manifest) is not FastHebbianH2ConsumerSourceManifest or (
            self.h2_consumer_source_sha256
            != self.h2_consumer_source_manifest.source_sha256
        ):
            raise FastHebbianH2ValidationError(
                "H2 population changed scoped consumer source identity"
            )
        if type(self.policy) is not FastHebbianH2Policy:
            raise TypeError("policy must be an exact FastHebbianH2Policy")
        if type(self.question_receipts) is not tuple or not self.question_receipts:
            raise FastHebbianH2ValidationError("H2 question population is empty")
        if type(self.final_evidence) is not tuple or len(self.final_evidence) != len(
            self.question_receipts
        ):
            raise FastHebbianH2ValidationError("H2 final evidence population changed")
        if tuple(row.question_ordinal for row in self.question_receipts) != tuple(
            range(len(self.question_receipts))
        ):
            raise FastHebbianH2ValidationError("H2 question order changed")
        for receipt, evidence in zip(
            self.question_receipts, self.final_evidence, strict=True
        ):
            if type(receipt) is not FastHebbianH2QuestionReceipt or type(evidence) is not tuple:
                raise FastHebbianH2ValidationError("H2 proposal types changed")
            if len(evidence) != len(receipt.final_coordinates):
                raise FastHebbianH2ValidationError("H2 proposal cardinality changed")
            for row, coordinate in zip(evidence, receipt.final_coordinates, strict=True):
                if type(row) is not FastEvidence or (
                    row.evidence_id != coordinate.evidence_id
                    or row.source_id != coordinate.source_id
                    or quote_sha256(row.text) != coordinate.evidence_text_sha256
                ):
                    raise FastHebbianH2ValidationError(
                        "H2 transient evidence changed its sealed coordinate"
                    )
            if (
                receipt.retrieval_artifact_sha256 != self.retrieval_artifact_sha256
                or receipt.history_receipt_sha256 != self.history_receipt_sha256
                or receipt.derived_store_receipt_sha256
                != self.derived_store_receipt_sha256
                or receipt.h2_policy_sha256 != self.policy.policy_sha256
                or receipt.h2_consumer_source_sha256
                != self.h2_consumer_source_sha256
                or receipt.h2_consumer_source_manifest_sha256
                != self.h2_consumer_source_manifest.manifest_sha256
                or receipt.h2_consumer_environment_lock_sha256
                != self.h2_consumer_environment_lock_sha256
                or len(receipt.seed_source_chunk_ids) > self.policy.max_seed_chunks
                or len(receipt.ranked_candidates)
                > self.policy.max_neighbor_candidates
                or len(receipt.appended_evidence_ids) > self.policy.max_additions
                or any(
                    row.support < self.policy.min_support
                    or row.coaccess_count < self.policy.min_coaccess_count
                    or row.score < self.policy.min_score
                    for row in receipt.ranked_candidates
                )
            ):
                raise FastHebbianH2ValidationError(
                    "H2 question changed population policy/provenance"
                )
        if (
            self.gold_fields_consumed is not False
            or self.provider_calls != 0
            or self.cav_links_computed is not False
            or self.retained_request_token_state_bytes != 0
        ):
            raise FastHebbianH2ValidationError(
                "H2 population crossed a forbidden runtime boundary"
            )
        expected = identity_sha256(self.identity_payload(include_sha256=False))
        if self.population_sha256:
            if _digest(self.population_sha256, "population_sha256") != expected:
                raise FastHebbianH2ValidationError("H2 population seal changed")
        else:
            object.__setattr__(self, "population_sha256", expected)

    @property
    def question_count(self) -> int:
        return len(self.question_receipts)

    @property
    def appended_question_count(self) -> int:
        return sum(bool(row.appended_evidence_ids) for row in self.question_receipts)

    @property
    def appended_evidence_count(self) -> int:
        return sum(len(row.appended_evidence_ids) for row in self.question_receipts)

    def evidence_for(self, question_id: str) -> tuple[FastEvidence, ...]:
        for receipt, evidence in zip(
            self.question_receipts, self.final_evidence, strict=True
        ):
            if receipt.question_id == question_id:
                return evidence
        raise KeyError(question_id)

    def identity_payload(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "retrieval_artifact_sha256": self.retrieval_artifact_sha256,
            "retrieval_source_sha256": self.retrieval_source_sha256,
            "history_file_sha256": self.history_file_sha256,
            "history_artifact_sha256": self.history_artifact_sha256,
            "history_receipt_sha256": self.history_receipt_sha256,
            "history_event_population_sha256": self.history_event_population_sha256,
            "history_producer_implementation_sha256": (
                self.history_producer_implementation_sha256
            ),
            "history_producer_environment_sha256": (
                self.history_producer_environment_sha256
            ),
            "derived_store_receipt_sha256": self.derived_store_receipt_sha256,
            "derived_database_sha256": self.derived_database_sha256,
            "derived_association_artifact_id": self.derived_association_artifact_id,
            "derived_association_artifact_sha256": (
                self.derived_association_artifact_sha256
            ),
            "derived_store_producer_implementation_sha256": (
                self.derived_store_producer_implementation_sha256
            ),
            "h2_consumer_source_manifest": (
                self.h2_consumer_source_manifest.identity_payload()
            ),
            "h2_consumer_source_sha256": self.h2_consumer_source_sha256,
            "h2_consumer_environment_lock_sha256": (
                self.h2_consumer_environment_lock_sha256
            ),
            "policy": self.policy.identity_payload(),
            "question_receipts": [
                row.identity_payload() for row in self.question_receipts
            ],
            "gold_fields_consumed": self.gold_fields_consumed,
            "provider_calls": self.provider_calls,
            "cav_links_computed": self.cav_links_computed,
            "retained_request_token_state_bytes": (
                self.retained_request_token_state_bytes
            ),
        }
        if include_sha256:
            payload["population_sha256"] = self.population_sha256
        return payload


def load_fast_hebbian_h2_retrieval_source(
    path: str | Path,
    *,
    artifact: FastRetrievalArtifact,
) -> FastHebbianH2RetrievalSource:
    """Recover sealed S3 source-chunk coordinates from the exact retrieval JSON."""

    if type(artifact) is not FastRetrievalArtifact:
        raise TypeError("artifact must be an exact FastRetrievalArtifact")
    payload, raw_sha, resolved = _read_canonical_json(path)
    if raw_sha != artifact.raw_sha256:
        raise FastHebbianH2ValidationError(
            "S3 coordinate source changed retrieval artifact bytes"
        )
    raw_questions = payload.get("questions")
    if type(raw_questions) is not list or len(raw_questions) != artifact.question_count:
        raise FastHebbianH2ValidationError("retrieval question population changed")
    rows: list[FastHebbianH2S3QuestionSource] = []
    for ordinal, (raw, question) in enumerate(
        zip(raw_questions, artifact.questions, strict=True)
    ):
        if type(raw) is not dict or raw.get("question_id") != question.question_id:
            raise FastHebbianH2ValidationError("retrieval question order changed")
        receipt = raw.get("retrieval_receipt")
        if type(receipt) is not dict:
            raise FastHebbianH2ValidationError("retrieval receipt is absent")
        evidence_ids = receipt.get("final_evidence_ids")
        source_chunks = receipt.get("final_chunk_ids")
        if type(evidence_ids) is not list or type(source_chunks) is not list:
            raise FastHebbianH2ValidationError(
                "retrieval receipt omitted final S3 coordinates"
            )
        evidence_tuple = tuple(evidence_ids)
        chunk_tuple = tuple(source_chunks)
        stage = question.stage(FAST_HEBBIAN_H2_STAGE_ID)
        if (
            evidence_tuple != stage.evidence_ids
            or tuple(chunk_tuple[: len(question.protected_chunk_ids)])
            != question.protected_chunk_ids
            or receipt.get("receipt_sha256")
            != question.retrieval_receipt_sha256
        ):
            raise FastHebbianH2ValidationError(
                "retrieval receipt disagrees with exact typed S3 provenance"
            )
        rows.append(
            FastHebbianH2S3QuestionSource(
                question_ordinal=ordinal,
                question_id=question.question_id,
                retrieval_question_receipt_sha256=(
                    question.retrieval_receipt_sha256
                ),
                s3_stage_receipt_sha256=stage.stage_receipt_sha256,
                s3_evidence_projection_sha256=stage.evidence_projection_sha256,
                s3_evidence_ids=_ids(evidence_tuple, "final_evidence_ids"),
                s3_source_chunk_ids=_ids(chunk_tuple, "final_chunk_ids"),
            )
        )
    return FastHebbianH2RetrievalSource(
        format=FAST_HEBBIAN_H2_S3_SOURCE_FORMAT,
        source_path=str(resolved),
        retrieval_artifact_sha256=raw_sha,
        questions=tuple(rows),
    )


def load_fast_hebbian_h2_history(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_sidecar: bool = True,
) -> FastHebbianH2HistorySource:
    """Load canonical sealed history without current-implementation equality."""

    payload, raw_sha, resolved = _read_canonical_json(path)
    _verify_digest_anchor(
        resolved,
        raw_sha,
        expected_sha256=expected_sha256,
        verify_sidecar=verify_sidecar,
    )
    return FastHebbianH2HistorySource(
        source_path=str(resolved),
        raw_sha256=raw_sha,
        artifact=load_hebbian_history_artifact(payload),
    )


def _verified_derived_inputs(
    derived_store_path: str | Path,
    *,
    artifact: FastRetrievalArtifact,
    history: HebbianHistoryArtifact,
) -> tuple[Path, HebbianDerivedStoreReceipt]:
    candidate = Path(derived_store_path)
    if candidate.is_symlink():
        raise FastHebbianH2ValidationError("derived store must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FastHebbianH2ValidationError("derived store does not exist") from exc
    if not root.is_dir() or root.is_symlink():
        raise FastHebbianH2ValidationError("derived store must be a directory")
    manifest_payload, _manifest_sha, manifest_path = _read_canonical_json(
        root / HEBBIAN_DERIVED_STORE_MANIFEST
    )
    receipt = load_hebbian_derived_store_receipt(manifest_payload)
    verify_hebbian_derived_store(root, expected=receipt)
    database_path = (root / "memory.db").resolve(strict=True)
    if (
        manifest_path.parent != root
        or receipt.source_store_receipt_sha256
        != artifact.combined_store_receipt_sha256
        or history.receipt.source_store_receipt_sha256
        != artifact.combined_store_receipt_sha256
        or receipt.source_database_sha256
        != history.receipt.source_database_sha256
        or receipt.history_artifact_sha256 != history.artifact_sha256
        or receipt.history_receipt_sha256 != history.receipt.receipt_sha256
        or receipt.implementation_sha256 != history.receipt.implementation_sha256
        or receipt.environment_lock_sha256
        != history.receipt.environment_lock_sha256
        or receipt.retained_request_token_state_bytes != 0
        or receipt.derived_database_sha256 != file_sha256(database_path)
    ):
        raise FastHebbianH2ValidationError(
            "derived store changed its sealed history/source provenance"
        )
    return database_path, receipt


def _coordinates(
    evidence: Sequence[FastEvidence],
    *,
    base_count: int,
    candidate_chunks: Mapping[str, str],
) -> tuple[FastHebbianH2EvidenceCoordinate, ...]:
    rows: list[FastHebbianH2EvidenceCoordinate] = []
    for ordinal, row in enumerate(evidence):
        if type(row) is not FastEvidence:
            raise FastHebbianH2ValidationError("evidence must retain exact type")
        rows.append(
            FastHebbianH2EvidenceCoordinate(
                format=FAST_HEBBIAN_H2_EVIDENCE_FORMAT,
                evidence_ordinal=ordinal,
                evidence_id=_text(row.evidence_id, "evidence_id"),
                source_id=_text(row.source_id, "source_id"),
                evidence_text_sha256=quote_sha256(row.text),
                source_chunk_id=(
                    None
                    if ordinal < base_count
                    else candidate_chunks[row.evidence_id]
                ),
                origin="s3" if ordinal < base_count else "hebbian_h2",
            )
        )
    return tuple(rows)


def _candidate_evidence_id(
    *, source_chunk_id: str, source_id: str, evidence_text_sha256: str
) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-fast-hebbian-h2-evidence-id-v1",
            "source_chunk_id": source_chunk_id,
            "source_id": source_id,
            "evidence_text_sha256": evidence_text_sha256,
        }
    )


def _source_id(result: Any) -> str:
    source_id = result.durable_source_id.strip()
    if not source_id or len(result.source_hints) > 1:
        raise FastHebbianH2ValidationError(
            f"chunk {result.chunk.chunk_id!r} has ambiguous source provenance"
        )
    return source_id


def build_fast_hebbian_h2_population(
    artifact: FastRetrievalArtifact,
    retrieval_source: FastHebbianH2RetrievalSource,
    history_source: FastHebbianH2HistorySource,
    derived_store_path: str | Path,
    *,
    policy: FastHebbianH2Policy | None = None,
) -> FastHebbianH2Population:
    """Build exact append-only S3/H2 proposals without gold or a provider."""

    if type(artifact) is not FastRetrievalArtifact:
        raise TypeError("artifact must be an exact FastRetrievalArtifact")
    if type(retrieval_source) is not FastHebbianH2RetrievalSource:
        raise TypeError("retrieval_source must be exact H2 coordinates")
    if type(history_source) is not FastHebbianH2HistorySource:
        raise TypeError("history_source must be an exact H2 history source")
    active_policy = policy or FastHebbianH2Policy()
    if type(active_policy) is not FastHebbianH2Policy:
        raise TypeError("policy must be an exact FastHebbianH2Policy")
    consumer_source = build_fast_hebbian_h2_consumer_source_manifest()
    consumer_environment_sha = environment_lock_sha256()
    if (
        artifact.raw_sha256 != retrieval_source.retrieval_artifact_sha256
        or artifact.question_count != len(retrieval_source.questions)
        or artifact.stage_ids != STAGE_IDS
        or artifact.retained_request_token_state_bytes != 0
    ):
        raise FastHebbianH2ValidationError(
            "H2 inputs changed the exact cumulative retrieval population"
        )
    history = verify_hebbian_history_artifact(history_source.artifact)
    database_path, derived = _verified_derived_inputs(
        derived_store_path,
        artifact=artifact,
        history=history,
    )
    receipts: list[FastHebbianH2QuestionReceipt] = []
    evidence_population: list[tuple[FastEvidence, ...]] = []
    with Database(database_path, read_only=True) as database:
        if database.current_turn() != artifact.turn_count:
            raise FastHebbianH2ValidationError(
                "derived store changed the terminal turn coordinate"
            )
        associations = AssociationStore(database)
        if associations.get_artifact(derived.association_artifact_id) is None:
            raise FastHebbianH2ValidationError(
                "derived association artifact is absent"
            )
        for question, source in zip(
            artifact.questions, retrieval_source.questions, strict=True
        ):
            stage = question.stage(FAST_HEBBIAN_H2_STAGE_ID)
            if (
                question.ordinal != source.question_ordinal
                or question.question_id != source.question_id
                or stage.stage_receipt_sha256 != source.s3_stage_receipt_sha256
                or stage.evidence_projection_sha256
                != source.s3_evidence_projection_sha256
                or stage.evidence_ids != source.s3_evidence_ids
                or len(source.s3_source_chunk_ids) > active_policy.max_seed_chunks
            ):
                raise FastHebbianH2ValidationError(
                    "H2 source coordinates changed exact S3 provenance"
                )
            base_evidence = tuple(stage.evidence)
            for chunk_id in source.s3_source_chunk_ids:
                if hydrate_chunk_result(database, chunk_id, score=0.0) is None:
                    raise FastHebbianH2ValidationError(
                        f"S3 source chunk is absent from derived store: {chunk_id}"
                    )
            base_catalog_sha, base_scaffold_sha, base_tokens = _scaffold(
                base_evidence, question.dated_question
            )
            if base_tokens > active_policy.hard_prompt_token_cap:
                raise FastHebbianH2ValidationError(
                    "exact S3 evidence already exceeds downstream hard 8k cap: "
                    f"{question.question_id} {base_tokens}"
                )
            activations = {
                chunk_id: rank_discount(rank)
                for rank, chunk_id in enumerate(
                    source.s3_source_chunk_ids,
                    start=1,
                )
            }
            neighbors = associations.hebbian_neighbors(
                activations,
                derived.association_artifact_id,
                top_k=active_policy.max_neighbor_candidates,
                exclude=source.s3_source_chunk_ids,
                now_turn=artifact.turn_count,
                half_life_turns=active_policy.half_life_turns,
                min_score=active_policy.min_score,
            )
            robust = tuple(
                row
                for row in neighbors
                if row.support >= active_policy.min_support
                and row.coaccess_count >= active_policy.min_coaccess_count
            )
            if tuple(
                (row.score, row.support, row.chunk_id) for row in robust
            ) != tuple(
                sorted(
                    (
                        (row.score, row.support, row.chunk_id)
                        for row in robust
                    ),
                    key=lambda item: (-item[0], -item[1], item[2]),
                )
            ):
                raise FastHebbianH2ValidationError(
                    "derived graph returned nondeterministic H2 ranking"
                )

            final = list(base_evidence)
            base_ids = {row.evidence_id for row in base_evidence}
            candidate_chunks: dict[str, str] = {}
            candidate_receipts: list[FastHebbianH2CandidateReceipt] = []
            for rank, neighbor in enumerate(robust, start=1):
                hydrated = hydrate_chunk_result(
                    database,
                    neighbor.chunk_id,
                    score=neighbor.score,
                )
                if hydrated is None:
                    raise FastHebbianH2ValidationError(
                        f"robust H2 candidate is absent: {neighbor.chunk_id}"
                    )
                source_id = _source_id(hydrated)
                content = hydrated.chunk.text
                content_sha = quote_sha256(content)
                evidence_id = _candidate_evidence_id(
                    source_chunk_id=neighbor.chunk_id,
                    source_id=source_id,
                    evidence_text_sha256=content_sha,
                )
                if evidence_id in base_ids or evidence_id in candidate_chunks:
                    raise FastHebbianH2ValidationError(
                        "H2 candidate evidence identity collided"
                    )
                proposed = FastEvidence(evidence_id, source_id, content)
                _catalog_sha, _scaffold_sha, proposed_tokens = _scaffold(
                    (*final, proposed), question.dated_question
                )
                if len(final) - len(base_evidence) >= active_policy.max_additions:
                    status: FastHebbianH2CandidateStatus = "addition_cap_rejected"
                    final_ordinal = None
                elif proposed_tokens > active_policy.hard_prompt_token_cap:
                    status = "budget_rejected"
                    final_ordinal = None
                else:
                    status = "appended"
                    final_ordinal = len(final)
                    final.append(proposed)
                    candidate_chunks[evidence_id] = neighbor.chunk_id
                candidate_receipts.append(
                    FastHebbianH2CandidateReceipt(
                        format=FAST_HEBBIAN_H2_CANDIDATE_FORMAT,
                        rank=rank,
                        source_chunk_id=neighbor.chunk_id,
                        evidence_id=evidence_id,
                        source_id=source_id,
                        evidence_text_sha256=content_sha,
                        score=neighbor.score,
                        support=neighbor.support,
                        anchor_chunk_id=neighbor.anchor_chunk_id,
                        coaccess_count=neighbor.coaccess_count,
                        last_reinforced_turn=neighbor.last_reinforced_turn,
                        admission_status=status,
                        proposed_prompt_token_proxy=proposed_tokens,
                        final_evidence_ordinal=final_ordinal,
                    )
                )
            final_evidence = tuple(final)
            final_catalog_sha, final_scaffold_sha, final_tokens = _scaffold(
                final_evidence, question.dated_question
            )
            base_coordinates = _coordinates(
                base_evidence,
                base_count=len(base_evidence),
                candidate_chunks={},
            )
            final_coordinates = _coordinates(
                final_evidence,
                base_count=len(base_evidence),
                candidate_chunks=candidate_chunks,
            )
            appended_tail = final_coordinates[len(base_coordinates) :]
            outcome = (
                "appended"
                if appended_tail
                else (
                    "no_robust_candidate"
                    if not candidate_receipts
                    else "no_budget_admissible_candidate"
                )
            )
            receipts.append(
                FastHebbianH2QuestionReceipt(
                    format=FAST_HEBBIAN_H2_QUESTION_FORMAT,
                    question_ordinal=question.ordinal,
                    question_id=question.question_id,
                    question_sha256=question.question_sha256,
                    dated_question_sha256=question.dated_question_sha256,
                    stage_id=FAST_HEBBIAN_H2_STAGE_ID,
                    retrieval_artifact_sha256=artifact.raw_sha256,
                    retrieval_question_receipt_sha256=(
                        question.retrieval_receipt_sha256
                    ),
                    s3_stage_receipt_sha256=stage.stage_receipt_sha256,
                    s3_evidence_projection_sha256=(
                        stage.evidence_projection_sha256
                    ),
                    s3_source_coordinate_sha256=source.coordinate_sha256,
                    history_receipt_sha256=history.receipt.receipt_sha256,
                    derived_store_receipt_sha256=derived.receipt_sha256,
                    h2_policy_sha256=active_policy.policy_sha256,
                    h2_consumer_source_sha256=consumer_source.source_sha256,
                    h2_consumer_source_manifest_sha256=consumer_source.manifest_sha256,
                    h2_consumer_environment_lock_sha256=(
                        consumer_environment_sha
                    ),
                    seed_source_chunk_ids=source.s3_source_chunk_ids,
                    base_s3_coordinates=base_coordinates,
                    final_coordinates=final_coordinates,
                    ranked_candidates=tuple(candidate_receipts),
                    appended_source_chunk_ids=tuple(
                        row.source_chunk_id for row in appended_tail
                    ),
                    appended_evidence_ids=tuple(
                        row.evidence_id for row in appended_tail
                    ),
                    outcome=outcome,
                    base_evidence_catalog_sha256=base_catalog_sha,
                    final_evidence_catalog_sha256=final_catalog_sha,
                    base_scaffold_sha256=base_scaffold_sha,
                    final_scaffold_sha256=final_scaffold_sha,
                    base_prompt_token_proxy=base_tokens,
                    final_prompt_token_proxy=final_tokens,
                    hard_prompt_token_cap=active_policy.hard_prompt_token_cap,
                    gold_fields_consumed=False,
                    cav_links_computed=False,
                    retained_request_token_state_bytes=0,
                )
            )
            evidence_population.append(final_evidence)

    return FastHebbianH2Population(
        format=FAST_HEBBIAN_H2_POPULATION_FORMAT,
        retrieval_artifact_sha256=artifact.raw_sha256,
        retrieval_source_sha256=retrieval_source.source_sha256,
        history_file_sha256=history_source.raw_sha256,
        history_artifact_sha256=history.artifact_sha256,
        history_receipt_sha256=history.receipt.receipt_sha256,
        history_event_population_sha256=history.receipt.event_population_sha256,
        history_producer_implementation_sha256=(
            history.receipt.implementation_sha256
        ),
        history_producer_environment_sha256=(
            history.receipt.environment_lock_sha256
        ),
        derived_store_receipt_sha256=derived.receipt_sha256,
        derived_database_sha256=derived.derived_database_sha256,
        derived_association_artifact_id=derived.association_artifact_id,
        derived_association_artifact_sha256=derived.association_artifact_sha256,
        derived_store_producer_implementation_sha256=(
            derived.implementation_sha256
        ),
        h2_consumer_source_manifest=consumer_source,
        h2_consumer_source_sha256=consumer_source.source_sha256,
        h2_consumer_environment_lock_sha256=consumer_environment_sha,
        policy=active_policy,
        question_receipts=tuple(receipts),
        final_evidence=tuple(evidence_population),
        gold_fields_consumed=False,
        provider_calls=0,
        cav_links_computed=False,
        retained_request_token_state_bytes=0,
    )


__all__ = [
    "FAST_HEBBIAN_H2_CANDIDATE_FORMAT",
    "FAST_HEBBIAN_H2_CONSUMER_ROOT_MODULES",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_ALGORITHM",
    "FAST_HEBBIAN_H2_CONSUMER_SOURCE_SCOPE",
    "FAST_HEBBIAN_H2_DEFAULT_MAX_ADDITIONS",
    "FAST_HEBBIAN_H2_DEFAULT_MAX_NEIGHBORS",
    "FAST_HEBBIAN_H2_DEFAULT_MAX_SEED_CHUNKS",
    "FAST_HEBBIAN_H2_DEFAULT_MIN_COACCESS_COUNT",
    "FAST_HEBBIAN_H2_DEFAULT_MIN_SUPPORT",
    "FAST_HEBBIAN_H2_EVIDENCE_FORMAT",
    "FAST_HEBBIAN_H2_MAX_PROMPT_TOKENS",
    "FAST_HEBBIAN_H2_POLICY_FORMAT",
    "FAST_HEBBIAN_H2_POPULATION_FORMAT",
    "FAST_HEBBIAN_H2_QUESTION_FORMAT",
    "FAST_HEBBIAN_H2_S3_SOURCE_FORMAT",
    "FAST_HEBBIAN_H2_STAGE_ID",
    "FastHebbianH2CandidateReceipt",
    "FastHebbianH2CandidateStatus",
    "FastHebbianH2ConsumerSourceManifest",
    "FastHebbianH2EvidenceCoordinate",
    "FastHebbianH2HistorySource",
    "FastHebbianH2Policy",
    "FastHebbianH2Population",
    "FastHebbianH2QuestionReceipt",
    "FastHebbianH2RetrievalSource",
    "FastHebbianH2S3QuestionSource",
    "FastHebbianH2ValidationError",
    "build_fast_hebbian_h2_population",
    "build_fast_hebbian_h2_consumer_source_manifest",
    "load_fast_hebbian_h2_history",
    "load_fast_hebbian_h2_retrieval_source",
]
