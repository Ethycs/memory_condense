"""Provider-free source-group and episode-neighbor evidence reinjection.

The residual search plane deliberately ranks globally.  This module performs
the complementary local read: an already selected opaque source-group handle
is authenticated against its exact :class:`LocalCitationBinding`, reversed to
the immutable residual source history, and expanded to nearby answer-bearing
segments.  Selection happens before protected-evidence deduplication.  Novel
rows are then packed under an independent, non-borrowable L-plane cap while
every duplicate, overflow, and still-unmet typed obligation remains explicit.

No model is called and no embedding, K/V cache, activation, or request-token
state is created or retained here.  Optional episode expansion delegates only
the bounded topology read to the existing episode retrieval primitive; all
returned episode spans are rejoined to exact residual-index segments before
they can become evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import Episode, EvidenceSpan, quote_sha256
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.search.episodes.retrieval import (
    EpisodeLookup,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
    expand_episode_seeds,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import LocalCitationBinding, indexed_surface_terms
from .semantic_residual_search import (
    ExactCellSegment,
    SemanticResidualCell,
    SemanticResidualIndex,
    SemanticResidualQuery,
    semantic_residual_source_group_map,
    semantic_residual_source_identity_receipt,
)
from .typed_action_semantics import canonical_action_concepts
from .typed_operator_spec import AnswerShape, TemporalMode


MECHANISM_ID = "source_group_episode_neighbor_reinjection_v1_1"
POLICY_FORMAT = "memory-condense-source-group-reinjection-policy-v1"
GROUP_FORMAT = "memory-condense-authenticated-source-group-selection-v1"
GROUP_ROW_FORMAT = f"{GROUP_FORMAT}-group-row"
HANDLE_ROW_FORMAT = f"{GROUP_FORMAT}-handle-row"
OBLIGATION_FORMAT = "memory-condense-local-reinjection-obligation-v1"
ATTEMPT_FORMAT = "memory-condense-local-reinjection-attempt-v1"
EVIDENCE_FORMAT = "memory-condense-local-reinjection-evidence-v1"
DUPLICATE_FORMAT = "memory-condense-local-reinjection-protected-duplicate-v1"
FRONTIER_FORMAT = "memory-condense-local-reinjection-frontier-v1"
RESULT_FORMAT = "memory-condense-source-group-reinjection-result-v1"
PACKING_ALGORITHM = (
    "stratified-direct-assertion-lanes-post-dedup-skip-continue-v3"
)

DEFAULT_LOCAL_PAYLOAD_TOKEN_CAP = 1_200
_HANDLE_RE = re.compile(r"^[A-Z][A-Z0-9_-]{0,31}$")
_DATED_RE = re.compile(r"^\[Question asked at (?P<asked_at>.+?)\]\s*", re.I | re.S)
_CAPITALIZED_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b"
)
_FIRST_PERSON_ASSERTION_RE = re.compile(
    r"\b(?:I|I'm|I've|I'd|my|mine|we|we're|we've|our)\b", re.I
)
_GENERIC_ADVICE_RE = re.compile(
    r"\b(?:you (?:can|could|might|may|should)|consider|some options|"
    r"it depends|generally|typically|usually)\b",
    re.I,
)
_QUESTION_FUNCTION_WORDS = frozenset(
    {
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "how",
        "i",
        "is",
        "my",
        "the",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "would",
    }
)


class SourceGroupReinjectionError(MatchedEvalContractError):
    """An authenticated group, immutable history, budget, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SourceGroupReinjectionError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _receipt(value: Mapping[str, object], declared: str, label: str) -> str:
    expected = identity_sha256(value)
    if declared:
        _require(require_sha256(declared, label) == expected, f"{label} changed")
    return expected


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be ordered unique exact text",
    )
    return values


def _span_identity(span: EvidenceSpan) -> str:
    return identity_sha256(span.identity_payload())


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


@dataclass(frozen=True, slots=True)
class SourceGroupReinjectionPolicy:
    """Independent caps for the local linking plane."""

    local_payload_token_cap: int = DEFAULT_LOCAL_PAYLOAD_TOKEN_CAP
    max_selected_segments: int = 64
    base_segments_per_group: int = 3
    max_query_term_obligations: int = 6
    source_neighbor_radius: int = 1
    max_source_neighbors_per_anchor: int = 2
    max_episode_segments_per_seed: int = 4
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "local_payload_token_cap",
            "max_selected_segments",
            "base_segments_per_group",
            "max_query_term_obligations",
            "max_episode_segments_per_seed",
        ):
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        for name in ("source_neighbor_radius", "max_source_neighbors_per_anchor"):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, f"{name} must be nonnegative")
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "source-group reinjection policy receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="source_group_reinjection_policy")

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "base_segments_per_group": self.base_segments_per_group,
            "format": POLICY_FORMAT,
            "gold_loaded": False,
            "local_payload_token_cap": self.local_payload_token_cap,
            "max_episode_segments_per_seed": self.max_episode_segments_per_seed,
            "max_query_term_obligations": self.max_query_term_obligations,
            "max_selected_segments": self.max_selected_segments,
            "max_source_neighbors_per_anchor": self.max_source_neighbors_per_anchor,
            "mechanism_id": MECHANISM_ID,
            "new_provider_calls": 0,
            "packing_algorithm": PACKING_ALGORITHM,
            "retained_transformer_token_state_bytes": 0,
            "source_neighbor_radius": self.source_neighbor_radius,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceGroupRow:
    source_id: str
    source_group_handle: str
    source_identity_receipt_sha256: str
    source_history_receipt_sha256: str
    cell_receipt_sha256s: tuple[str, ...]
    segment_receipt_sha256s: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.source_id, "authenticated group source")
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
            "authenticated source group handle changed",
        )
        for value, label in (
            (self.source_identity_receipt_sha256, "source identity receipt"),
            (self.source_history_receipt_sha256, "source history receipt"),
        ):
            require_sha256(value, label)
        for values, label in (
            (self.cell_receipt_sha256s, "source group cells"),
            (self.segment_receipt_sha256s, "source group segments"),
        ):
            _ordered_unique(values, label)
            _require(bool(values), f"{label} cannot be empty")
            for value in values:
                require_sha256(value, label)
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "authenticated group-row receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "cell_receipt_sha256s": list(self.cell_receipt_sha256s),
            "format": GROUP_ROW_FORMAT,
            "segment_receipt_sha256s": list(self.segment_receipt_sha256s),
            "source_group_handle": self.source_group_handle,
            "source_history_receipt_sha256": self.source_history_receipt_sha256,
            "source_id": self.source_id,
            "source_identity_receipt_sha256": self.source_identity_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AuthenticatedSelectedHandle:
    evidence_handle: str
    source_group_handle: str
    local_binding: LocalCitationBinding
    source_group_row_receipt_sha256: str
    anchor_cell_id: str
    anchor_cell_receipt_sha256: str
    anchor_segment_receipt_sha256: str
    anchor_span_identity_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.evidence_handle) is str
            and _HANDLE_RE.fullmatch(self.evidence_handle) is not None,
            "selected evidence handle changed",
        )
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None
            and type(self.local_binding) is LocalCitationBinding,
            "selected handle lost its exact opaque group binding",
        )
        require_text(self.anchor_cell_id, "selected anchor cell")
        for value, label in (
            (self.source_group_row_receipt_sha256, "selected group-row receipt"),
            (self.anchor_cell_receipt_sha256, "selected anchor cell receipt"),
            (self.anchor_segment_receipt_sha256, "selected anchor segment receipt"),
            (self.anchor_span_identity_sha256, "selected anchor span identity"),
        ):
            require_sha256(value, label)
        _require(
            self.anchor_span_identity_sha256 == _span_identity(self.local_binding.span),
            "selected handle changed anchor span",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "authenticated selected-handle receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "anchor_cell_id": self.anchor_cell_id,
            "anchor_cell_receipt_sha256": self.anchor_cell_receipt_sha256,
            "anchor_segment_receipt_sha256": self.anchor_segment_receipt_sha256,
            "anchor_span_identity_sha256": self.anchor_span_identity_sha256,
            "evidence_handle": self.evidence_handle,
            "format": HANDLE_ROW_FORMAT,
            "local_binding": self.local_binding.projection(),
            "source_group_handle": self.source_group_handle,
            "source_group_row_receipt_sha256": self.source_group_row_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class AuthenticatedSourceGroupSelection:
    residual_index_receipt_sha256: str
    group_universe_source_ids: tuple[str, ...]
    group_rows: tuple[AuthenticatedSourceGroupRow, ...]
    selected_handles: tuple[AuthenticatedSelectedHandle, ...]
    group_mapping_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.residual_index_receipt_sha256, "selected-group index")
        _ordered_unique(self.group_universe_source_ids, "group universe sources")
        _require(
            self.group_universe_source_ids
            == tuple(sorted(self.group_universe_source_ids)),
            "group universe sources must be sorted",
        )
        _require(
            type(self.group_rows) is tuple
            and self.group_rows
            and all(type(row) is AuthenticatedSourceGroupRow for row in self.group_rows)
            and tuple(row.source_id for row in self.group_rows)
            == self.group_universe_source_ids,
            "authenticated group rows changed their exact universe",
        )
        _require(
            len({row.source_group_handle for row in self.group_rows})
            == len(self.group_rows),
            "authenticated group handles collided",
        )
        _require(
            type(self.selected_handles) is tuple
            and self.selected_handles
            and all(
                type(row) is AuthenticatedSelectedHandle
                for row in self.selected_handles
            )
            and tuple(row.evidence_handle for row in self.selected_handles)
            == tuple(sorted(row.evidence_handle for row in self.selected_handles))
            and len({row.evidence_handle for row in self.selected_handles})
            == len(self.selected_handles),
            "selected handle population changed order or uniqueness",
        )
        by_group = {row.source_group_handle: row for row in self.group_rows}
        _require(
            all(
                row.source_group_handle in by_group
                and row.source_group_row_receipt_sha256
                == by_group[row.source_group_handle].receipt_sha256
                and row.local_binding.source_id
                == by_group[row.source_group_handle].source_id
                for row in self.selected_handles
            ),
            "selected handle escaped the authenticated group map",
        )
        require_sha256(self.group_mapping_receipt_sha256, "group mapping receipt")
        _require(
            self.group_mapping_receipt_sha256
            == identity_sha256(
                {
                    "allocation_algorithm": "memory-condense-semantic-residual-source-group-allocation-v1",
                    "group_row_receipt_sha256s": [
                        row.receipt_sha256 for row in self.group_rows
                    ],
                    "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
                }
            ),
            "authenticated group mapping receipt changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "authenticated source-group selection receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="authenticated_source_group_selection")

    @property
    def selected_source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({row.local_binding.source_id for row in self.selected_handles})
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "allocation_algorithm": "memory-condense-semantic-residual-source-group-allocation-v1",
            "format": GROUP_FORMAT,
            "gold_loaded": False,
            "group_mapping_receipt_sha256": self.group_mapping_receipt_sha256,
            "group_rows": [row.projection() for row in self.group_rows],
            "group_universe_source_ids": list(self.group_universe_source_ids),
            "new_provider_calls": 0,
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selected_handles": [row.projection() for row in self.selected_handles],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LocalReinjectionObligation:
    obligation_id: str
    kind: Literal[
        "selected_group",
        "typed_slot",
        "entity_term",
        "action",
        "numeric",
        "temporal",
        "required_role",
    ]
    key: str
    source_group_handle: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.obligation_id, "local obligation ID")
        _require(
            self.kind
            in {
                "selected_group",
                "typed_slot",
                "entity_term",
                "action",
                "numeric",
                "temporal",
                "required_role",
            },
            "local obligation kind changed",
        )
        require_text(self.key, "local obligation key")
        if self.source_group_handle is not None:
            _require(
                re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
                "local obligation source group changed",
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "local obligation receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": OBLIGATION_FORMAT,
            "key": self.key,
            "kind": self.kind,
            "obligation_id": self.obligation_id,
            "source_group_handle": self.source_group_handle,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LocalReinjectionEvidence:
    candidate_id: str
    evidence_handle: str
    source_group_handle: str
    source_history_receipt_sha256: str
    cell_id: str
    cell_receipt_sha256: str
    segment_receipt_sha256: str
    quote: str
    quote_sha256: str
    token_count: int
    role: str
    created_at: str
    event_dates: tuple[str, ...]
    supported_obligation_ids: tuple[str, ...]
    selection_routes: tuple[str, ...]
    citation_binding_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "local reinjection candidate")
        _require(
            re.fullmatch(r"L[0-9]{4}", self.evidence_handle) is not None,
            "local evidence handle changed",
        )
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
            "local evidence group changed",
        )
        require_text(self.cell_id, "local evidence cell")
        for value, label in (
            (self.source_history_receipt_sha256, "local source history"),
            (self.cell_receipt_sha256, "local cell"),
            (self.segment_receipt_sha256, "local segment"),
            (self.quote_sha256, "local quote"),
            (self.citation_binding_receipt_sha256, "local citation binding"),
        ):
            require_sha256(value, label)
        require_text(self.quote, "local evidence quote")
        _require(
            self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote),
            "local evidence changed exact quote bytes",
        )
        require_text(self.role, "local evidence role")
        require_text(self.created_at, "local evidence created-at")
        for values, label in (
            (self.event_dates, "local evidence event dates"),
            (self.supported_obligation_ids, "local evidence obligations"),
            (self.selection_routes, "local evidence routes"),
        ):
            _ordered_unique(values, label)
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "local reinjection evidence receipt",
            ),
        )

    def provider_row(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "evidence_handle": self.evidence_handle,
            "quote": self.quote,
            "role": self.role,
            "source_group_handle": self.source_group_handle,
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "cell_id": self.cell_id,
            "cell_receipt_sha256": self.cell_receipt_sha256,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "created_at": self.created_at,
            "event_dates": list(self.event_dates),
            "evidence_handle": self.evidence_handle,
            "format": EVIDENCE_FORMAT,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
            "role": self.role,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selection_routes": list(self.selection_routes),
            "source_group_handle": self.source_group_handle,
            "source_history_receipt_sha256": self.source_history_receipt_sha256,
            "supported_obligation_ids": list(self.supported_obligation_ids),
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ProtectedLocalDuplicate:
    candidate_id: str
    segment_receipt_sha256: str
    span_identity_sha256: str
    protected_evidence_handle: str
    protected_candidate_id: str
    protected_binding_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "duplicate candidate"),
            (self.segment_receipt_sha256, "duplicate segment"),
            (self.span_identity_sha256, "duplicate span"),
            (self.protected_candidate_id, "duplicate owner candidate"),
            (self.protected_binding_receipt_sha256, "duplicate owner binding"),
        ):
            require_sha256(value, label)
        _require(
            _HANDLE_RE.fullmatch(self.protected_evidence_handle) is not None,
            "duplicate owner handle changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "protected local duplicate receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "format": DUPLICATE_FORMAT,
            "protected_binding_receipt_sha256": self.protected_binding_receipt_sha256,
            "protected_candidate_id": self.protected_candidate_id,
            "protected_evidence_handle": self.protected_evidence_handle,
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "span_identity_sha256": self.span_identity_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LocalSelectionAttempt:
    candidate_id: str
    selection_rank: int
    source_group_handle: str
    source_id: str
    cell_id: str
    segment_receipt_sha256: str
    span_identity_sha256: str
    supported_obligation_ids: tuple[str, ...]
    selection_routes: tuple[str, ...]
    score_components: tuple[tuple[str, float], ...]
    disposition: Literal["protected_exact_duplicate", "packed_novel", "budget_unpacked"]
    protected_duplicate_receipt_sha256: str | None
    evidence_receipt_sha256: str | None
    citation_binding_receipt_sha256: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.candidate_id, "attempt candidate")
        _require(
            type(self.selection_rank) is int and self.selection_rank >= 0,
            "attempt selection rank changed",
        )
        _require(
            re.fullmatch(r"G[0-9]{6}", self.source_group_handle) is not None,
            "attempt group changed",
        )
        require_text(self.source_id, "attempt source")
        require_text(self.cell_id, "attempt cell")
        for value, label in (
            (self.segment_receipt_sha256, "attempt segment"),
            (self.span_identity_sha256, "attempt span"),
        ):
            require_sha256(value, label)
        _ordered_unique(self.supported_obligation_ids, "attempt obligations")
        _ordered_unique(self.selection_routes, "attempt routes")
        _require(
            type(self.score_components) is tuple
            and self.score_components
            and tuple(key for key, _value in self.score_components)
            == tuple(sorted(key for key, _value in self.score_components))
            and all(
                type(key) is str
                and key
                and type(value) is float
                and math.isfinite(value)
                for key, value in self.score_components
            ),
            "attempt score components changed",
        )
        _require(
            self.disposition
            in {"protected_exact_duplicate", "packed_novel", "budget_unpacked"},
            "attempt disposition changed",
        )
        for value, label in (
            (self.protected_duplicate_receipt_sha256, "attempt duplicate"),
            (self.evidence_receipt_sha256, "attempt evidence"),
            (self.citation_binding_receipt_sha256, "attempt citation"),
        ):
            if value is not None:
                require_sha256(value, label)
        expected_presence = {
            "protected_exact_duplicate": (True, False, False),
            "packed_novel": (False, True, True),
            "budget_unpacked": (False, False, True),
        }[self.disposition]
        _require(
            (
                self.protected_duplicate_receipt_sha256 is not None,
                self.evidence_receipt_sha256 is not None,
                self.citation_binding_receipt_sha256 is not None,
            )
            == expected_presence,
            "attempt disposition lost exact ownership",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "local selection-attempt receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "cell_id": self.cell_id,
            "citation_binding_receipt_sha256": self.citation_binding_receipt_sha256,
            "disposition": self.disposition,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "format": ATTEMPT_FORMAT,
            "protected_duplicate_receipt_sha256": self.protected_duplicate_receipt_sha256,
            "score_components": [
                {"name": key, "value": value}
                for key, value in self.score_components
            ],
            "segment_receipt_sha256": self.segment_receipt_sha256,
            "selection_rank": self.selection_rank,
            "selection_routes": list(self.selection_routes),
            "source_group_handle": self.source_group_handle,
            "source_id": self.source_id,
            "span_identity_sha256": self.span_identity_sha256,
            "supported_obligation_ids": list(self.supported_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class LocalReinjectionFrontier:
    selected_segment_receipt_sha256s: tuple[str, ...]
    packed_segment_receipt_sha256s: tuple[str, ...]
    protected_duplicate_segment_receipt_sha256s: tuple[str, ...]
    budget_unpacked_segment_receipt_sha256s: tuple[str, ...]
    required_obligation_ids: tuple[str, ...]
    covered_obligation_ids: tuple[str, ...]
    unresolved_obligation_ids: tuple[str, ...]
    selection_truncated: bool
    packing_closed: bool
    local_obligations_satisfied: bool
    needs_global_search: bool
    support_closure_proven: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for values, label in (
            (self.selected_segment_receipt_sha256s, "frontier selected segments"),
            (self.packed_segment_receipt_sha256s, "frontier packed segments"),
            (
                self.protected_duplicate_segment_receipt_sha256s,
                "frontier duplicate segments",
            ),
            (
                self.budget_unpacked_segment_receipt_sha256s,
                "frontier budget-unpacked segments",
            ),
            (self.required_obligation_ids, "frontier required obligations"),
            (self.covered_obligation_ids, "frontier covered obligations"),
            (self.unresolved_obligation_ids, "frontier unresolved obligations"),
        ):
            _ordered_unique(values, label)
        selected = set(self.selected_segment_receipt_sha256s)
        packed = set(self.packed_segment_receipt_sha256s)
        duplicates = set(self.protected_duplicate_segment_receipt_sha256s)
        unpacked = set(self.budget_unpacked_segment_receipt_sha256s)
        _require(
            packed.isdisjoint(duplicates)
            and packed.isdisjoint(unpacked)
            and duplicates.isdisjoint(unpacked)
            and packed | duplicates | unpacked == selected,
            "local frontier lost its post-selection partition",
        )
        required = set(self.required_obligation_ids)
        covered = set(self.covered_obligation_ids)
        unresolved = set(self.unresolved_obligation_ids)
        _require(
            covered.isdisjoint(unresolved)
            and covered | unresolved == required,
            "local frontier lost its obligation partition",
        )
        _require(
            type(self.selection_truncated) is bool
            and self.packing_closed == (not unpacked)
            and self.local_obligations_satisfied == (not unresolved)
            and type(self.needs_global_search) is bool
            and self.support_closure_proven is False,
            "local frontier closure flags changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "local reinjection frontier receipt",
            ),
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "budget_unpacked_segment_receipt_sha256s": list(
                self.budget_unpacked_segment_receipt_sha256s
            ),
            "covered_obligation_ids": list(self.covered_obligation_ids),
            "format": FRONTIER_FORMAT,
            "local_obligations_satisfied": self.local_obligations_satisfied,
            "needs_global_search": self.needs_global_search,
            "packed_segment_receipt_sha256s": list(
                self.packed_segment_receipt_sha256s
            ),
            "packing_closed": self.packing_closed,
            "protected_duplicate_segment_receipt_sha256s": list(
                self.protected_duplicate_segment_receipt_sha256s
            ),
            "required_obligation_ids": list(self.required_obligation_ids),
            "selected_segment_receipt_sha256s": list(
                self.selected_segment_receipt_sha256s
            ),
            "selection_truncated": self.selection_truncated,
            "support_closure_proven": False,
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SourceGroupReinjectionResult:
    residual_index_receipt_sha256: str
    query_receipt_sha256: str
    selection_receipt_sha256: str
    policy: SourceGroupReinjectionPolicy
    unresolved_input_slot_ids: tuple[str, ...]
    obligations: tuple[LocalReinjectionObligation, ...]
    protected_evidence_population_receipt_sha256: str
    episode_plan: EpisodeRetrievalPlan | None
    episode_population_receipt_sha256: str | None
    attempted_selection: tuple[LocalSelectionAttempt, ...]
    protected_duplicates: tuple[ProtectedLocalDuplicate, ...]
    evidence: tuple[LocalReinjectionEvidence, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    frontier: LocalReinjectionFrontier
    attempted_local_evidence_tokens: int
    packed_local_evidence_tokens: int
    packed_local_evidence_sha256: str
    provider_payload_tokens: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.residual_index_receipt_sha256, "reinjection index"),
            (self.query_receipt_sha256, "reinjection query"),
            (self.selection_receipt_sha256, "reinjection selection"),
            (
                self.protected_evidence_population_receipt_sha256,
                "reinjection protected population",
            ),
            (self.packed_local_evidence_sha256, "reinjection evidence plane"),
        ):
            require_sha256(value, label)
        _require(
            type(self.policy) is SourceGroupReinjectionPolicy,
            "reinjection result policy changed",
        )
        _ordered_unique(self.unresolved_input_slot_ids, "unresolved input slots")
        for values, expected, label in (
            (self.obligations, LocalReinjectionObligation, "local obligations"),
            (self.attempted_selection, LocalSelectionAttempt, "local attempts"),
            (
                self.protected_duplicates,
                ProtectedLocalDuplicate,
                "local protected duplicates",
            ),
            (self.evidence, LocalReinjectionEvidence, "local evidence"),
            (self.local_bindings, LocalCitationBinding, "local bindings"),
        ):
            _require(
                type(values) is tuple and all(type(row) is expected for row in values),
                f"{label} changed type",
            )
        _require(
            tuple(row.obligation_id for row in self.obligations)
            == self.frontier.required_obligation_ids,
            "local obligations escaped the frontier",
        )
        _require(
            tuple(row.selection_rank for row in self.attempted_selection)
            == tuple(range(len(self.attempted_selection))),
            "local attempts changed deterministic rank",
        )
        _require(
            tuple(row.segment_receipt_sha256 for row in self.attempted_selection)
            == self.frontier.selected_segment_receipt_sha256s,
            "local attempts lost selected-before-dedup order",
        )
        _require(
            tuple(row.segment_receipt_sha256 for row in self.evidence)
            == self.frontier.packed_segment_receipt_sha256s
            and tuple(row.segment_receipt_sha256 for row in self.protected_duplicates)
            == self.frontier.protected_duplicate_segment_receipt_sha256s,
            "local result lost packed/duplicate frontier order",
        )
        _require(
            tuple(row.candidate_id for row in self.evidence)
            == tuple(row.candidate_id for row in self.local_bindings)
            and all(
                evidence.citation_binding_receipt_sha256 == binding.receipt_sha256
                and evidence.quote_sha256 == binding.quote_sha256
                and evidence.source_group_handle == binding.source_group_handle
                for evidence, binding in zip(
                    self.evidence, self.local_bindings, strict=True
                )
            ),
            "local evidence lost exact citation ownership",
        )
        if self.episode_plan is None:
            _require(
                self.episode_population_receipt_sha256 is None,
                "absent episode plan retained an episode population",
            )
        else:
            _require(
                type(self.episode_plan) is EpisodeRetrievalPlan
                and self.episode_population_receipt_sha256 is not None,
                "episode plan lost exact population receipt",
            )
            require_sha256(
                self.episode_population_receipt_sha256,
                "episode population receipt",
            )
        for value, label in (
            (self.attempted_local_evidence_tokens, "attempted local tokens"),
            (self.packed_local_evidence_tokens, "packed local tokens"),
            (self.provider_payload_tokens, "provider payload tokens"),
        ):
            _require(type(value) is int and value >= 0, f"{label} changed")
        _require(
            self.packed_local_evidence_tokens
            == _local_evidence_tokens(self.evidence)
            <= self.policy.local_payload_token_cap
            and self.packed_local_evidence_sha256
            == identity_sha256([row.provider_row() for row in self.evidence]),
            "local non-borrowable evidence plane changed",
        )
        _require(
            self.provider_payload_tokens
            == count_tokens(_canonical_json(self.provider_projection())),
            "local provider payload token accounting changed",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _receipt(
                self.projection(include_receipt=False),
                self.receipt_sha256,
                "source-group reinjection result receipt",
            ),
        )
        assert_gold_blind(self.projection(), path="source_group_reinjection_result")
        assert_gold_blind(
            self.provider_projection(), path="source_group_reinjection_provider"
        )

    def provider_projection(self) -> dict[str, object]:
        return _provider_payload(self.evidence, self.frontier)

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "attempted_local_evidence_tokens": self.attempted_local_evidence_tokens,
            "attempted_selection": [row.projection() for row in self.attempted_selection],
            "dedup_after_local_selection": True,
            "episode_plan": (
                None if self.episode_plan is None else self.episode_plan.identity_payload()
            ),
            "episode_population_receipt_sha256": self.episode_population_receipt_sha256,
            "evidence": [row.projection() for row in self.evidence],
            "format": RESULT_FORMAT,
            "frontier": self.frontier.projection(),
            "gold_loaded": False,
            "local_bindings": [row.projection() for row in self.local_bindings],
            "local_payload_budget_non_borrowable": True,
            "new_provider_calls": 0,
            "obligations": [row.projection() for row in self.obligations],
            "packed_local_evidence_sha256": self.packed_local_evidence_sha256,
            "packed_local_evidence_tokens": self.packed_local_evidence_tokens,
            "policy": self.policy.projection(),
            "protected_duplicates": [
                row.projection() for row in self.protected_duplicates
            ],
            "protected_evidence_population_receipt_sha256": (
                self.protected_evidence_population_receipt_sha256
            ),
            "provider_payload_tokens": self.provider_payload_tokens,
            "query_receipt_sha256": self.query_receipt_sha256,
            "residual_index_receipt_sha256": self.residual_index_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selected_before_protected_dedup": True,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "unresolved_input_slot_ids": list(self.unresolved_input_slot_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class _HistoryInventory:
    source_id: str
    source_history_receipt_sha256: str
    cells: tuple[SemanticResidualCell, ...]
    pairs: tuple[tuple[SemanticResidualCell, ExactCellSegment], ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    cell: SemanticResidualCell
    segment: ExactCellSegment
    source_group_handle: str
    candidate_id: str
    supported_obligation_ids: tuple[str, ...]
    score_components: tuple[tuple[str, float], ...]
    score: float
    selection_routes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LaneCandidate:
    candidate: _Candidate
    family: Literal[
        "mandatory",
        "source_group",
        "source_neighbor",
        "episode_direct",
        "episode_adjacent",
    ]
    lane_key: str


def _history_inventory(
    residual_index: SemanticResidualIndex,
) -> tuple[
    Mapping[str, _HistoryInventory],
    Mapping[str, tuple[SemanticResidualCell, ExactCellSegment]],
]:
    by_source: dict[str, list[SemanticResidualCell]] = defaultdict(list)
    for cell in residual_index.cells:
        by_source[cell.source_id].append(cell)
    histories: dict[str, _HistoryInventory] = {}
    by_span: dict[str, tuple[SemanticResidualCell, ExactCellSegment]] = {}
    for source_id, cells in sorted(by_source.items()):
        ordered_cells = tuple(sorted(cells, key=lambda row: row.source_cell_ordinal))
        _require(
            tuple(row.source_cell_ordinal for row in ordered_cells)
            == tuple(range(len(ordered_cells)))
            and all(row.source_cell_count == len(ordered_cells) for row in ordered_cells)
            and len({row.source_history_receipt_sha256 for row in ordered_cells}) == 1,
            "immutable residual source history changed cell adjacency",
        )
        pairs = tuple(
            sorted(
                (
                    (cell, segment)
                    for cell in ordered_cells
                    for segment in cell.segments
                ),
                key=lambda pair: (
                    pair[1].span.ordinal,
                    pair[1].span.turn_start_char or 0,
                    pair[1].span.start_char,
                    pair[1].span.end_char,
                    pair[1].receipt_sha256,
                ),
            )
        )
        _require(
            len({segment.receipt_sha256 for _cell, segment in pairs}) == len(pairs),
            "immutable source history repeated a segment",
        )
        histories[source_id] = _HistoryInventory(
            source_id,
            ordered_cells[0].source_history_receipt_sha256,
            ordered_cells,
            pairs,
        )
        for cell, segment in pairs:
            span_sha = _span_identity(segment.span)
            _require(span_sha not in by_span, "residual index repeated an exact span")
            by_span[span_sha] = (cell, segment)
    return MappingProxyType(histories), MappingProxyType(by_span)


def _assert_binding_lineage(
    residual_index: SemanticResidualIndex,
    binding: LocalCitationBinding,
) -> None:
    _require(
        type(binding) is LocalCitationBinding
        and binding.namespace_id == residual_index.namespace_id
        and binding.cache_receipt_sha256 == residual_index.cache_receipt_sha256
        and binding.source_database_sha256 == residual_index.source_database_sha256
        and binding.source_store_receipt_sha256
        == residual_index.source_store_receipt_sha256,
        "local citation binding escaped immutable residual lineage",
    )


def authenticate_source_group_selection(
    residual_index: SemanticResidualIndex,
    selected_handle_bindings: Mapping[str, LocalCitationBinding],
    /,
    *,
    group_universe_source_ids: Sequence[str],
    selected_handle_groups: Mapping[str, str] | None = None,
) -> AuthenticatedSourceGroupSelection:
    """Seal G->source->cell->segment reverse maps for exact selected handles."""

    _require(
        type(residual_index) is SemanticResidualIndex,
        "source-group authentication requires an exact residual index",
    )
    _require(
        isinstance(selected_handle_bindings, Mapping)
        and bool(selected_handle_bindings),
        "source-group authentication requires selected handles",
    )
    universe = tuple(group_universe_source_ids)
    _require(
        universe == tuple(sorted(set(universe)))
        and all(type(value) is str and value for value in universe),
        "group universe must be sorted unique exact source IDs",
    )
    histories, by_span = _history_inventory(residual_index)
    _require(
        set(universe) <= set(histories),
        "group universe escaped the immutable residual index",
    )
    group_by_source = semantic_residual_source_group_map(universe)
    declared_groups = (
        {handle: binding.source_group_handle for handle, binding in selected_handle_bindings.items()}
        if selected_handle_groups is None
        else dict(selected_handle_groups)
    )
    _require(
        set(declared_groups) == set(selected_handle_bindings),
        "selected handle group projection changed population",
    )
    group_rows: list[AuthenticatedSourceGroupRow] = []
    for source_id in universe:
        history = histories[source_id]
        group_rows.append(
            AuthenticatedSourceGroupRow(
                source_id=source_id,
                source_group_handle=group_by_source[source_id],
                source_identity_receipt_sha256=(
                    semantic_residual_source_identity_receipt(source_id)
                ),
                source_history_receipt_sha256=(
                    history.source_history_receipt_sha256
                ),
                cell_receipt_sha256s=tuple(
                    row.receipt_sha256 for row in history.cells
                ),
                segment_receipt_sha256s=tuple(
                    segment.receipt_sha256 for _cell, segment in history.pairs
                ),
            )
        )
    by_source = {row.source_id: row for row in group_rows}
    selected_rows: list[AuthenticatedSelectedHandle] = []
    for handle, binding in sorted(selected_handle_bindings.items()):
        _require(
            type(handle) is str and _HANDLE_RE.fullmatch(handle) is not None,
            "selected handle changed schema",
        )
        _assert_binding_lineage(residual_index, binding)
        _require(
            binding.source_id in by_source
            and declared_groups[handle] == group_by_source[binding.source_id],
            "selected handle G mapping does not match its authenticated universe",
        )
        pair = by_span.get(_span_identity(binding.span))
        _require(pair is not None, "selected handle span is absent from residual history")
        assert pair is not None
        cell, segment = pair
        _require(
            segment.source_id == binding.source_id
            and segment.partition_id == binding.partition_id
            and segment.quote_sha256 == binding.quote_sha256,
            "selected handle quote/source/partition binding changed",
        )
        group_row = by_source[binding.source_id]
        selected_rows.append(
            AuthenticatedSelectedHandle(
                evidence_handle=handle,
                source_group_handle=declared_groups[handle],
                local_binding=binding,
                source_group_row_receipt_sha256=group_row.receipt_sha256,
                anchor_cell_id=cell.cell_id,
                anchor_cell_receipt_sha256=cell.receipt_sha256,
                anchor_segment_receipt_sha256=segment.receipt_sha256,
                anchor_span_identity_sha256=_span_identity(segment.span),
            )
        )
    mapping_receipt = identity_sha256(
        {
            "allocation_algorithm": "memory-condense-semantic-residual-source-group-allocation-v1",
            "group_row_receipt_sha256s": [row.receipt_sha256 for row in group_rows],
            "residual_index_receipt_sha256": residual_index.receipt_sha256,
        }
    )
    return AuthenticatedSourceGroupSelection(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        group_universe_source_ids=universe,
        group_rows=tuple(group_rows),
        selected_handles=tuple(selected_rows),
        group_mapping_receipt_sha256=mapping_receipt,
    )


def validate_source_group_selection(
    residual_index: SemanticResidualIndex,
    selection: AuthenticatedSourceGroupSelection,
) -> AuthenticatedSourceGroupSelection:
    _require(
        type(selection) is AuthenticatedSourceGroupSelection
        and selection.residual_index_receipt_sha256 == residual_index.receipt_sha256,
        "authenticated selection escaped its residual index",
    )
    replayed = authenticate_source_group_selection(
        residual_index,
        {
            row.evidence_handle: row.local_binding
            for row in selection.selected_handles
        },
        group_universe_source_ids=selection.group_universe_source_ids,
        selected_handle_groups={
            row.evidence_handle: row.source_group_handle
            for row in selection.selected_handles
        },
    )
    _require(
        replayed.receipt_sha256 == selection.receipt_sha256
        and replayed.projection() == selection.projection(),
        "authenticated source-group selection replay changed",
    )
    return selection


def _obligation(
    query: SemanticResidualQuery,
    kind: str,
    key: str,
    *,
    source_group_handle: str | None = None,
) -> LocalReinjectionObligation:
    body = {
        "format": OBLIGATION_FORMAT,
        "key": key,
        "kind": kind,
        "query_receipt_sha256": query.receipt_sha256,
        "source_group_handle": source_group_handle,
    }
    return LocalReinjectionObligation(
        obligation_id=identity_sha256(body),
        kind=kind,  # type: ignore[arg-type]
        key=key,
        source_group_handle=source_group_handle,
    )


def _entity_terms(
    query: SemanticResidualQuery,
    pairs: Sequence[tuple[SemanticResidualCell, ExactCellSegment]],
    *,
    limit: int,
) -> tuple[str, ...]:
    body = _DATED_RE.sub("", query.dated_question).strip()
    explicit: set[str] = set()
    for match in _CAPITALIZED_RE.finditer(body):
        explicit.update(
            term
            for term in indexed_surface_terms(match.group(0))
            if term not in _QUESTION_FUNCTION_WORDS
        )
    population = tuple(pairs)
    document_frequency: dict[str, int] = defaultdict(int)
    for _cell, segment in population:
        for term in set(segment.surface_terms) & set(query.query_terms):
            document_frequency[term] += 1
    candidates = set(explicit) | set(query.slot_terms)
    if not candidates:
        candidates = set(query.query_terms)
    ranked = sorted(
        (term for term in candidates if term in document_frequency),
        key=lambda term: (
            document_frequency[term],
            -len(term),
            term,
        ),
    )
    return tuple(ranked[:limit])


def _build_obligations(
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    pairs: Sequence[tuple[SemanticResidualCell, ExactCellSegment]],
    *,
    unresolved_slot_ids: tuple[str, ...],
    policy: SourceGroupReinjectionPolicy,
) -> tuple[LocalReinjectionObligation, ...]:
    by_slot = {slot.slot_id: slot for slot in query.operator_spec.required_slots}
    _require(
        set(unresolved_slot_ids) <= set(by_slot),
        "unresolved local slot escaped the question-only typed specification",
    )
    output: list[LocalReinjectionObligation] = []
    group_by_source = {
        row.source_id: row.source_group_handle for row in selection.group_rows
    }
    for source_id in selection.selected_source_ids:
        group = group_by_source[source_id]
        output.append(
            _obligation(query, "selected_group", group, source_group_handle=group)
        )
    for slot in query.operator_spec.required_slots:
        if slot.slot_id in unresolved_slot_ids:
            output.append(_obligation(query, "typed_slot", slot.slot_id))
    for term in _entity_terms(
        query, pairs, limit=policy.max_query_term_obligations
    ):
        output.append(_obligation(query, "entity_term", term))
    for action in query.action_concepts:
        output.append(_obligation(query, "action", action))
    numeric_needed = bool(
        query.operator_spec.answer_shape
        in {AnswerShape.NUMBER, AnswerShape.DURATION}
        or any(
            by_slot[slot_id].requires_numeric
            for slot_id in unresolved_slot_ids
        )
    )
    if numeric_needed:
        output.append(_obligation(query, "numeric", "numeric_operand"))
    if query.operator_spec.temporal_mode is not TemporalMode.NONE:
        output.append(_obligation(query, "temporal", "event_or_source_time"))
    if query.operator_spec.required_evidence_role is not None:
        output.append(
            _obligation(
                query,
                "required_role",
                query.operator_spec.required_evidence_role,
            )
        )
    # The construction loops above are in semantic priority order.  Dedup by
    # identity without sorting so that coverage packing remains deterministic.
    return tuple({row.obligation_id: row for row in output}.values())


def _supported_obligations(
    segment: ExactCellSegment,
    group: str,
    obligations: Sequence[LocalReinjectionObligation],
    query: SemanticResidualQuery,
) -> tuple[str, ...]:
    terms = set(segment.surface_terms)
    actions = set(segment.action_concepts)
    slots = {slot.slot_id: slot for slot in query.operator_spec.required_slots}
    supported: list[str] = []
    for obligation in obligations:
        ok = False
        if obligation.kind == "selected_group":
            ok = obligation.source_group_handle == group
        elif obligation.kind == "typed_slot":
            slot = slots[obligation.key]
            ok = (
                len(set(slot.match_terms) & terms)
                >= slot.minimum_match_term_count
            )
        elif obligation.kind == "entity_term":
            ok = obligation.key in terms
        elif obligation.kind == "action":
            ok = obligation.key in actions
        elif obligation.kind == "numeric":
            ok = segment.contains_numeric_value
        elif obligation.kind == "temporal":
            ok = bool(segment.event_dates or segment.created_at)
        elif obligation.kind == "required_role":
            ok = segment.role == obligation.key
        if ok:
            supported.append(obligation.obligation_id)
    return tuple(supported)


def _has_declarative_first_person_assertion(quote: str) -> bool:
    """Recognize a user fact even when a later sentence asks a question."""

    clauses = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", quote)
        if part.strip()
    )
    return any(
        "?" not in clause and _FIRST_PERSON_ASSERTION_RE.search(clause) is not None
        for clause in clauses
    )


def _candidate_score_components(
    segment: ExactCellSegment,
    query: SemanticResidualQuery,
    supported_obligation_count: int,
    term_specificity: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    query_terms = set(query.query_terms) | set(query.slot_terms)
    terms = set(segment.surface_terms)
    action_hits = len(set(query.action_concepts) & set(segment.action_concepts))
    lexical_specificity = math.fsum(
        term_specificity.get(term, 0.0) for term in query_terms & terms
    )
    role_match = float(
        query.operator_spec.required_evidence_role is None
        or segment.role == query.operator_spec.required_evidence_role
    )
    factual_assertion = float(
        segment.role == "user"
        and bool(
            _has_declarative_first_person_assertion(segment.quote)
            or (
                "?" not in segment.quote
                and bool(
                    segment.action_concepts
                    or segment.event_dates
                    or segment.contains_numeric_value
                )
            )
        )
    )
    generic_advice = float(bool(_GENERIC_ADVICE_RE.search(segment.quote)))
    user_role_affinity = float(segment.role == "user")
    numeric_affinity = float(segment.contains_numeric_value)
    temporal_affinity = float(bool(segment.event_dates))
    components = {
        "action_overlap": float(action_hits),
        "factual_assertion": factual_assertion,
        "generic_advice_penalty": -generic_advice,
        "lexical_specificity": float(lexical_specificity),
        "numeric_affinity": numeric_affinity,
        "required_role_match": role_match,
        "supported_obligation_count": float(supported_obligation_count),
        "temporal_affinity": temporal_affinity,
        "user_role_affinity": user_role_affinity,
    }
    return tuple(sorted((key, float(value)) for key, value in components.items()))


def _score(components: tuple[tuple[str, float], ...]) -> float:
    values = dict(components)
    return math.fsum(
        (
            5.0 * values["required_role_match"],
            3.0 * values["supported_obligation_count"],
            2.5 * values["action_overlap"],
            2.0 * values["factual_assertion"],
            3.0 * values["user_role_affinity"],
            1.5 * values["lexical_specificity"],
            0.75 * values["numeric_affinity"],
            0.75 * values["temporal_affinity"],
            1.5 * values["generic_advice_penalty"],
        )
    )


def _candidate_rank_key(candidate: _Candidate) -> tuple[object, ...]:
    segment = candidate.segment
    role = dict(candidate.score_components)["required_role_match"]
    factual = dict(candidate.score_components)["factual_assertion"]
    return (
        -round(candidate.score, 12),
        -role,
        -factual,
        -_timestamp(segment.created_at),
        segment.token_count,
        segment.receipt_sha256,
    )


def _direct_episode_rank_key(candidate: _Candidate) -> tuple[object, ...]:
    """Prefer exact assertion-bearing spans inside an authenticated episode."""

    components = dict(candidate.score_components)
    return (
        -components["factual_assertion"],
        -float(components["action_overlap"] > 0.0),
        -components["temporal_affinity"],
        -components["numeric_affinity"],
        -components["user_role_affinity"],
        *_candidate_rank_key(candidate),
    )


def _candidate_population(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    obligations: tuple[LocalReinjectionObligation, ...],
) -> tuple[_Candidate, ...]:
    histories, _by_span = _history_inventory(residual_index)
    group_by_source = {
        row.source_id: row.source_group_handle for row in selection.group_rows
    }
    pairs = tuple(
        pair
        for source_id in selection.selected_source_ids
        for pair in histories[source_id].pairs
    )
    document_frequency: dict[str, int] = defaultdict(int)
    for _cell, segment in pairs:
        for term in set(segment.surface_terms):
            document_frequency[term] += 1
    population_size = max(1, len(pairs))
    specificity = MappingProxyType(
        {
            term: math.log((population_size + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
    )
    output: list[_Candidate] = []
    for cell, segment in pairs:
        group = group_by_source[cell.source_id]
        supported = _supported_obligations(segment, group, obligations, query)
        components = _candidate_score_components(
            segment, query, len(supported), specificity
        )
        candidate_id = identity_sha256(
            {
                "format": f"{EVIDENCE_FORMAT}-candidate",
                "query_receipt_sha256": query.receipt_sha256,
                "segment_receipt_sha256": segment.receipt_sha256,
                "selection_receipt_sha256": selection.receipt_sha256,
            }
        )
        output.append(
            _Candidate(
                cell,
                segment,
                group,
                candidate_id,
                supported,
                components,
                _score(components),
                (),
            )
        )
    return tuple(sorted(output, key=_candidate_rank_key))


def _merge_route(
    selected: dict[str, _Candidate],
    candidate: _Candidate,
    route: str,
) -> None:
    prior = selected.get(candidate.segment.receipt_sha256)
    routes = tuple(sorted({*(prior.selection_routes if prior else ()), route}))
    selected[candidate.segment.receipt_sha256] = _Candidate(
        candidate.cell,
        candidate.segment,
        candidate.source_group_handle,
        candidate.candidate_id,
        candidate.supported_obligation_ids,
        candidate.score_components,
        candidate.score,
        routes,
    )


def _candidate_with_route(candidate: _Candidate, route: str) -> _Candidate:
    routed: dict[str, _Candidate] = {}
    _merge_route(routed, candidate, route)
    return routed[candidate.segment.receipt_sha256]


def _merge_candidate_routes(
    selected: dict[str, _Candidate],
    candidate: _Candidate,
) -> None:
    _require(candidate.selection_routes, "lane candidate lost its selection route")
    for route in candidate.selection_routes:
        _merge_route(selected, candidate, route)


def _lane_heads_and_tails(
    rows: Sequence[_LaneCandidate],
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]]:
    """Return one seed-ordered head per lane before any lane contributes a tail."""

    by_lane: dict[str, dict[str, _Candidate]] = defaultdict(dict)
    for row in rows:
        _merge_candidate_routes(by_lane[row.lane_key], row.candidate)
    # All producers emit a deterministic priority order.  Retaining both lane
    # insertion order and row insertion order preserves authenticated upstream
    # seed priority instead of silently re-ranking one lane ahead of another.
    ordered_lanes = {
        key: tuple(values.values()) for key, values in by_lane.items()
    }
    heads = tuple(
        candidates[0]
        for candidates in ordered_lanes.values()
        if candidates
    )
    tails: list[_Candidate] = []
    maximum_depth = max((len(values) for values in ordered_lanes.values()), default=0)
    for depth in range(1, maximum_depth):
        tails.extend(
            candidates[depth]
            for candidates in ordered_lanes.values()
            if depth < len(candidates)
        )
    return heads, tuple(tails)


def _interleave_sequences(
    sequences: Sequence[Sequence[_Candidate]],
) -> tuple[_Candidate, ...]:
    output: list[_Candidate] = []
    positions = [0 for _sequence in sequences]
    while True:
        progressed = False
        for index, sequence in enumerate(sequences):
            if positions[index] >= len(sequence):
                continue
            output.append(sequence[positions[index]])
            positions[index] += 1
            progressed = True
        if not progressed:
            return tuple(output)


def _stratified_lane_order(
    rows: Sequence[_LaneCandidate],
) -> tuple[_Candidate, ...]:
    """Fairly stage direct episodes and local lanes before protected dedup."""

    by_family = {
        family: tuple(row for row in rows if row.family == family)
        for family in (
            "mandatory",
            "source_group",
            "source_neighbor",
            "episode_direct",
            "episode_adjacent",
        )
    }
    heads_and_tails = {
        family: _lane_heads_and_tails(family_rows)
        for family, family_rows in by_family.items()
    }
    direct_assertion_rows = tuple(
        _LaneCandidate(
            candidate=row.candidate,
            family=row.family,
            lane_key=(
                f"{row.lane_key}:assertion:{row.candidate.segment.receipt_sha256}"
            ),
        )
        for row in by_family["episode_direct"]
        if dict(row.candidate.score_components)["factual_assertion"] == 1.0
    )
    direct_context_rows = tuple(
        row
        for row in by_family["episode_direct"]
        if dict(row.candidate.score_components)["factual_assertion"] != 1.0
    )
    direct_assertion_heads, direct_assertion_tails = _lane_heads_and_tails(
        direct_assertion_rows
    )
    direct_context_heads, direct_context_tails = _lane_heads_and_tails(
        direct_context_rows
    )
    mandatory_heads, mandatory_tails = heads_and_tails["mandatory"]
    group_heads, group_tails = heads_and_tails["source_group"]
    source_heads, source_tails = heads_and_tails["source_neighbor"]
    adjacent_heads, adjacent_tails = heads_and_tails["episode_adjacent"]

    # Every directly anchored user assertion and every mandatory
    # anchor/coverage lane contributes before contextual episode/source tails.
    # Assertion candidates remain selected before exact protected dedup; giving
    # each one a lane prevents a protected episode anchor from hiding a novel
    # assertion behind it.
    staged = _interleave_sequences((direct_assertion_heads, mandatory_heads)) + (
        _interleave_sequences(
            (
                source_heads,
                group_heads,
                adjacent_heads,
                direct_context_heads,
                direct_assertion_tails,
                mandatory_tails,
                source_tails,
                group_tails,
                adjacent_tails,
                direct_context_tails,
            )
        )
    )
    merged: dict[str, _Candidate] = {}
    for row in rows:
        _merge_candidate_routes(merged, row.candidate)
    ordered: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in staged:
        receipt = candidate.segment.receipt_sha256
        if receipt in seen:
            continue
        seen.add(receipt)
        ordered.append(merged[receipt])
    _require(
        seen == set(merged),
        "stratified lane interleave changed the selected candidate population",
    )
    return tuple(ordered)


def _source_neighbor_candidates(
    residual_index: SemanticResidualIndex,
    selection: AuthenticatedSourceGroupSelection,
    population_by_segment: Mapping[str, _Candidate],
    *,
    policy: SourceGroupReinjectionPolicy,
) -> tuple[_LaneCandidate, ...]:
    histories, _by_span = _history_inventory(residual_index)
    output: list[_LaneCandidate] = []
    for anchor in selection.selected_handles:
        pairs = histories[anchor.local_binding.source_id].pairs
        positions = [
            index
            for index, (_cell, segment) in enumerate(pairs)
            if segment.receipt_sha256 == anchor.anchor_segment_receipt_sha256
        ]
        _require(len(positions) == 1, "selected anchor changed source-history position")
        center = positions[0]
        candidates: list[tuple[int, int, _Candidate, str]] = []
        for distance in range(1, policy.source_neighbor_radius + 1):
            for direction_order, position, direction in (
                (0, center - distance, "previous"),
                (1, center + distance, "next"),
            ):
                if 0 <= position < len(pairs):
                    segment = pairs[position][1]
                    candidates.append(
                        (
                            distance,
                            direction_order,
                            population_by_segment[segment.receipt_sha256],
                            f"source_local_{direction}:distance={distance}:anchor={anchor.evidence_handle}",
                        )
                    )
        candidates.sort(
            key=lambda row: (row[0], row[1], *_candidate_rank_key(row[2]))
        )
        output.extend(
            _LaneCandidate(
                candidate=_candidate_with_route(candidate, route),
                family="source_neighbor",
                lane_key=anchor.evidence_handle,
            )
            for _distance, _direction, candidate, route in candidates[
                : policy.max_source_neighbors_per_anchor
            ]
        )
    return tuple(output)


def _episode_expansion(
    residual_index: SemanticResidualIndex,
    selection: AuthenticatedSourceGroupSelection,
    population_by_segment: Mapping[str, _Candidate],
    *,
    episode_lookup: EpisodeLookup | None,
    episode_policy: EpisodeRetrievalPolicy | None,
    local_policy: SourceGroupReinjectionPolicy,
) -> tuple[
    EpisodeRetrievalPlan | None,
    str | None,
    tuple[_LaneCandidate, ...],
]:
    if episode_lookup is None:
        _require(episode_policy is None, "episode policy supplied without episode lookup")
        return None, None, ()
    _require(
        episode_policy is not None and episode_policy.artifact_id is not None,
        "episode reinjection requires an exact artifact-bound episode policy",
    )
    histories, _by_span = _history_inventory(residual_index)
    segment_by_receipt = {
        segment.receipt_sha256: segment
        for history in histories.values()
        for _cell, segment in history.pairs
    }
    direct: list[RetrievalResult] = []
    for anchor in selection.selected_handles:
        segment = segment_by_receipt[anchor.anchor_segment_receipt_sha256]
        direct.append(
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=segment.span.chunk_id,
                    turn_id=segment.span.turn_id or segment.partition_id,
                    text=segment.quote,
                    start_char=0,
                    end_char=len(segment.quote),
                    token_count=segment.token_count,
                ),
                score=1.0,
                route="source_group_selected_handle",
                memory_source_id=segment.source_id,
            )
        )
    plan = expand_episode_seeds(tuple(direct), episode_lookup, policy=episode_policy)
    selected_source_ids = set(selection.selected_source_ids)
    episode_rows: list[dict[str, object]] = []
    output: list[_LaneCandidate] = []
    for seed in plan.seeds:
        try:
            episode = episode_lookup.get_episode(seed.episode_id)
        except Exception as exc:  # pragma: no cover - defensive store seam
            raise SourceGroupReinjectionError(
                "episode lookup changed after bounded expansion"
            ) from exc
        _require(
            type(episode) is Episode
            and episode.artifact_id == episode_policy.artifact_id
            and episode.source_id in selected_source_ids,
            "episode expansion escaped selected immutable source groups",
        )
        assert episode is not None
        matches: dict[str, _Candidate] = {}
        history = histories[episode.source_id]
        for span in episode.evidence:
            for _cell, segment in history.pairs:
                overlaps = (
                    segment.span.chunk_id == span.chunk_id
                    and segment.span.start_char < span.end_char
                    and span.start_char < segment.span.end_char
                )
                if overlaps:
                    matches.setdefault(
                        segment.receipt_sha256,
                        population_by_segment[segment.receipt_sha256],
                    )
        ranked = sorted(
            matches.values(),
            key=(
                _direct_episode_rank_key
                if seed.route == "episode_direct"
                else _candidate_rank_key
            ),
        )
        admitted = ranked[: local_policy.max_episode_segments_per_seed]
        episode_rows.append(
            {
                "episode_receipt_sha256": episode.receipt_sha256,
                "matched_segment_receipt_sha256s": [
                    row.segment.receipt_sha256 for row in ranked
                ],
                "packed_candidate_segment_receipt_sha256s": [
                    row.segment.receipt_sha256 for row in admitted
                ],
                "seed": seed.identity_payload(),
                "source_identity_receipt_sha256": (
                    semantic_residual_source_identity_receipt(episode.source_id)
                ),
            }
        )
        family: Literal["episode_direct", "episode_adjacent"] = (
            "episode_direct"
            if seed.route == "episode_direct"
            else "episode_adjacent"
        )
        output.extend(
            _LaneCandidate(
                candidate=_candidate_with_route(
                    candidate,
                    f"{seed.route}:episode={seed.episode_id}",
                ),
                family=family,
                lane_key=seed.episode_id,
            )
            for candidate in admitted
        )
    population_receipt = identity_sha256(
        {
            "episode_plan_receipt_sha256": plan.receipt_sha256,
            "episode_rows": episode_rows,
            "residual_index_receipt_sha256": residual_index.receipt_sha256,
            "selection_receipt_sha256": selection.receipt_sha256,
        }
    )
    return plan, population_receipt, tuple(output)


def _select_candidates(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    obligations: tuple[LocalReinjectionObligation, ...],
    population: tuple[_Candidate, ...],
    *,
    policy: SourceGroupReinjectionPolicy,
    episode_lookup: EpisodeLookup | None,
    episode_policy: EpisodeRetrievalPolicy | None,
) -> tuple[
    tuple[_Candidate, ...],
    bool,
    EpisodeRetrievalPlan | None,
    str | None,
]:
    by_segment = {row.segment.receipt_sha256: row for row in population}
    lane_rows: list[_LaneCandidate] = []
    anchor_receipts: list[str] = []
    # Exact incoming handles are always selected first.  They usually become
    # protected duplicates later, but remain visible in the selection audit.
    for anchor in selection.selected_handles:
        candidate = _candidate_with_route(
            by_segment[anchor.anchor_segment_receipt_sha256],
            f"selected_handle_anchor:{anchor.evidence_handle}",
        )
        anchor_receipts.append(candidate.segment.receipt_sha256)
        lane_rows.append(
            _LaneCandidate(
                candidate=candidate,
                family="mandatory",
                lane_key=f"anchor:{anchor.evidence_handle}",
            )
        )

    # Protect one best witness for each unresolved typed/entity/date/numeric
    # obligation in an independent mandatory lane.
    for obligation in obligations:
        matches = [
            candidate
            for candidate in population
            if obligation.obligation_id in candidate.supported_obligation_ids
        ]
        if matches:
            lane_rows.append(
                _LaneCandidate(
                    candidate=_candidate_with_route(
                        matches[0],
                        f"coverage:{obligation.kind}:{obligation.obligation_id}",
                    ),
                    family="mandatory",
                    lane_key=f"coverage:{obligation.obligation_id}",
                )
            )

    per_group: dict[str, int] = defaultdict(int)
    for candidate in population:
        if per_group[candidate.source_group_handle] >= policy.base_segments_per_group:
            continue
        lane_rows.append(
            _LaneCandidate(
                candidate=_candidate_with_route(
                    candidate, "source_group_ranked_fact"
                ),
                family="source_group",
                lane_key=candidate.source_group_handle,
            )
        )
        per_group[candidate.source_group_handle] += 1

    lane_rows.extend(
        _source_neighbor_candidates(
            residual_index,
            selection,
            by_segment,
            policy=policy,
        )
    )

    episode_plan, episode_population_receipt, episode_candidates = _episode_expansion(
        residual_index,
        selection,
        by_segment,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
        local_policy=policy,
    )
    lane_rows.extend(episode_candidates)

    stratified = _stratified_lane_order(lane_rows)
    by_selected_segment = {
        row.segment.receipt_sha256: row for row in stratified
    }
    # Incoming handles remain audit-visible even when their score is low.
    ranked = tuple(
        by_selected_segment[receipt]
        for receipt in dict.fromkeys(
            (*anchor_receipts, *(row.segment.receipt_sha256 for row in stratified))
        )
    )
    truncated = len(ranked) > policy.max_selected_segments
    return (
        ranked[: policy.max_selected_segments],
        truncated,
        episode_plan,
        episode_population_receipt,
    )


def _protected_inventory(
    residual_index: SemanticResidualIndex,
    protected_handle_bindings: Mapping[str, LocalCitationBinding],
) -> tuple[Mapping[str, tuple[str, LocalCitationBinding]], str]:
    _require(
        isinstance(protected_handle_bindings, Mapping),
        "protected local evidence must be an exact handle mapping",
    )
    _histories, by_span = _history_inventory(residual_index)
    ordered: list[tuple[str, LocalCitationBinding]] = []
    by_span_owner: dict[str, tuple[str, LocalCitationBinding]] = {}
    for handle, binding in sorted(protected_handle_bindings.items()):
        _require(
            type(handle) is str and _HANDLE_RE.fullmatch(handle) is not None,
            "protected evidence handle changed",
        )
        _assert_binding_lineage(residual_index, binding)
        pair = by_span.get(_span_identity(binding.span))
        _require(pair is not None, "protected evidence escaped residual source history")
        assert pair is not None
        _cell, segment = pair
        _require(
            segment.source_id == binding.source_id
            and segment.partition_id == binding.partition_id
            and segment.quote_sha256 == binding.quote_sha256,
            "protected evidence changed exact source owner",
        )
        ordered.append((handle, binding))
        by_span_owner.setdefault(_span_identity(binding.span), (handle, binding))
    receipt = identity_sha256(
        {
            "format": f"{DUPLICATE_FORMAT}-protected-population",
            "rows": [
                {
                    "evidence_handle": handle,
                    "local_binding_receipt_sha256": binding.receipt_sha256,
                    "span_identity_sha256": _span_identity(binding.span),
                }
                for handle, binding in ordered
            ],
            "residual_index_receipt_sha256": residual_index.receipt_sha256,
        }
    )
    return MappingProxyType(by_span_owner), receipt


def _binding_for_candidate(
    residual_index: SemanticResidualIndex,
    candidate: _Candidate,
) -> LocalCitationBinding:
    segment = candidate.segment
    return LocalCitationBinding(
        candidate_id=candidate.candidate_id,
        source_group_handle=candidate.source_group_handle,
        namespace_id=residual_index.namespace_id,
        cache_receipt_sha256=residual_index.cache_receipt_sha256,
        source_database_sha256=residual_index.source_database_sha256,
        source_store_receipt_sha256=residual_index.source_store_receipt_sha256,
        source_id=segment.source_id,
        partition_id=segment.partition_id,
        span=segment.span,
        quote_sha256=segment.quote_sha256,
    )


def _evidence_for_candidate(
    candidate: _Candidate,
    binding: LocalCitationBinding,
    *,
    handle: str,
) -> LocalReinjectionEvidence:
    segment = candidate.segment
    return LocalReinjectionEvidence(
        candidate_id=candidate.candidate_id,
        evidence_handle=handle,
        source_group_handle=candidate.source_group_handle,
        source_history_receipt_sha256=candidate.cell.source_history_receipt_sha256,
        cell_id=candidate.cell.cell_id,
        cell_receipt_sha256=candidate.cell.receipt_sha256,
        segment_receipt_sha256=segment.receipt_sha256,
        quote=segment.quote,
        quote_sha256=segment.quote_sha256,
        token_count=segment.token_count,
        role=segment.role,
        created_at=segment.created_at,
        event_dates=segment.event_dates,
        supported_obligation_ids=candidate.supported_obligation_ids,
        selection_routes=candidate.selection_routes,
        citation_binding_receipt_sha256=binding.receipt_sha256,
    )


def _local_evidence_tokens(evidence: Sequence[LocalReinjectionEvidence]) -> int:
    return count_tokens(_canonical_json([row.provider_row() for row in evidence]))


def _provider_payload(
    evidence: Sequence[LocalReinjectionEvidence],
    frontier: LocalReinjectionFrontier,
) -> dict[str, object]:
    return {
        "evidence": [row.provider_row() for row in evidence],
        "format": f"{RESULT_FORMAT}-provider-payload",
        "local_frontier": {
            "local_obligations_satisfied": frontier.local_obligations_satisfied,
            "needs_global_search": frontier.needs_global_search,
            "packing_closed": frontier.packing_closed,
            "receipt_sha256": frontier.receipt_sha256,
            "support_closure_proven": False,
            "unresolved_obligation_ids": list(frontier.unresolved_obligation_ids),
        },
    }


def search_source_group_reinjection(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    /,
    *,
    protected_handle_bindings: Mapping[str, LocalCitationBinding] | None = None,
    unresolved_slot_ids: Sequence[str] | None = None,
    policy: SourceGroupReinjectionPolicy = SourceGroupReinjectionPolicy(),
    episode_lookup: EpisodeLookup | None = None,
    episode_policy: EpisodeRetrievalPolicy | None = None,
) -> SourceGroupReinjectionResult:
    """Select local evidence, then exact-dedup and independently pack L."""

    _require(
        type(residual_index) is SemanticResidualIndex,
        "local reinjection requires an exact residual index",
    )
    _require(
        type(query) is SemanticResidualQuery
        and query.residual_index_receipt_sha256 == residual_index.receipt_sha256,
        "local reinjection query escaped its residual index",
    )
    validate_source_group_selection(residual_index, selection)
    _require(
        type(policy) is SourceGroupReinjectionPolicy,
        "local reinjection policy changed",
    )
    if unresolved_slot_ids is None:
        unresolved = tuple(slot.slot_id for slot in query.operator_spec.required_slots)
    else:
        unresolved = tuple(unresolved_slot_ids)
        _ordered_unique(unresolved, "unresolved local slots")
    selected_bindings = {
        row.evidence_handle: row.local_binding for row in selection.selected_handles
    }
    protected = (
        selected_bindings
        if protected_handle_bindings is None
        else dict(protected_handle_bindings)
    )
    _require(
        all(
            handle in protected
            and protected[handle].receipt_sha256 == binding.receipt_sha256
            for handle, binding in selected_bindings.items()
        ),
        "selected local anchors must retain exact provider-visible owners",
    )
    protected_by_span, protected_population_receipt = _protected_inventory(
        residual_index, protected
    )
    histories, _by_span = _history_inventory(residual_index)
    selected_pairs = tuple(
        pair
        for source_id in selection.selected_source_ids
        for pair in histories[source_id].pairs
    )
    obligations = _build_obligations(
        query,
        selection,
        selected_pairs,
        unresolved_slot_ids=unresolved,
        policy=policy,
    )
    population = _candidate_population(
        residual_index, query, selection, obligations
    )
    selected, selection_truncated, episode_plan, episode_population_receipt = (
        _select_candidates(
            residual_index,
            query,
            selection,
            obligations,
            population,
            policy=policy,
            episode_lookup=episode_lookup,
            episode_policy=episode_policy,
        )
    )
    _require(bool(selected), "local reinjection selected no exact segment")

    duplicates: list[ProtectedLocalDuplicate] = []
    packed: list[LocalReinjectionEvidence] = []
    packed_bindings: list[LocalCitationBinding] = []
    attempts: list[LocalSelectionAttempt] = []
    unpacked_segments: list[str] = []
    attempted_novel_rows: list[LocalReinjectionEvidence] = []
    for rank, candidate in enumerate(selected):
        segment = candidate.segment
        span_sha = _span_identity(segment.span)
        owner = protected_by_span.get(span_sha)
        if owner is not None:
            owner_handle, owner_binding = owner
            duplicate = ProtectedLocalDuplicate(
                candidate_id=candidate.candidate_id,
                segment_receipt_sha256=segment.receipt_sha256,
                span_identity_sha256=span_sha,
                protected_evidence_handle=owner_handle,
                protected_candidate_id=owner_binding.candidate_id,
                protected_binding_receipt_sha256=owner_binding.receipt_sha256,
            )
            duplicates.append(duplicate)
            attempts.append(
                LocalSelectionAttempt(
                    candidate_id=candidate.candidate_id,
                    selection_rank=rank,
                    source_group_handle=candidate.source_group_handle,
                    source_id=segment.source_id,
                    cell_id=candidate.cell.cell_id,
                    segment_receipt_sha256=segment.receipt_sha256,
                    span_identity_sha256=span_sha,
                    supported_obligation_ids=candidate.supported_obligation_ids,
                    selection_routes=candidate.selection_routes,
                    score_components=candidate.score_components,
                    disposition="protected_exact_duplicate",
                    protected_duplicate_receipt_sha256=duplicate.receipt_sha256,
                    evidence_receipt_sha256=None,
                    citation_binding_receipt_sha256=None,
                )
            )
            continue
        binding = _binding_for_candidate(residual_index, candidate)
        evidence = _evidence_for_candidate(candidate, binding, handle=f"L{rank + 1:04d}")
        attempted_novel_rows.append(evidence)
        proposed = (*packed, evidence)
        if _local_evidence_tokens(proposed) <= policy.local_payload_token_cap:
            packed.append(evidence)
            packed_bindings.append(binding)
            disposition = "packed_novel"
            evidence_receipt: str | None = evidence.receipt_sha256
        else:
            # Skip and continue.  A shorter lower-ranked local assertion can
            # still fit; the skipped exact segment remains on the frontier.
            unpacked_segments.append(segment.receipt_sha256)
            disposition = "budget_unpacked"
            evidence_receipt = None
        attempts.append(
            LocalSelectionAttempt(
                candidate_id=candidate.candidate_id,
                selection_rank=rank,
                source_group_handle=candidate.source_group_handle,
                source_id=segment.source_id,
                cell_id=candidate.cell.cell_id,
                segment_receipt_sha256=segment.receipt_sha256,
                span_identity_sha256=span_sha,
                supported_obligation_ids=candidate.supported_obligation_ids,
                selection_routes=candidate.selection_routes,
                score_components=candidate.score_components,
                disposition=disposition,  # type: ignore[arg-type]
                protected_duplicate_receipt_sha256=None,
                evidence_receipt_sha256=evidence_receipt,
                citation_binding_receipt_sha256=binding.receipt_sha256,
            )
        )

    visible_obligations = {
        obligation_id
        for attempt in attempts
        if attempt.disposition != "budget_unpacked"
        for obligation_id in attempt.supported_obligation_ids
    }
    required_obligation_ids = tuple(row.obligation_id for row in obligations)
    covered_obligation_ids = tuple(
        obligation_id
        for obligation_id in required_obligation_ids
        if obligation_id in visible_obligations
    )
    unresolved_obligation_ids = tuple(
        obligation_id
        for obligation_id in required_obligation_ids
        if obligation_id not in visible_obligations
    )
    global_closure_required = bool(
        query.operator_spec.requires_complete_frontier
        or query.operator_spec.answer_shape is AnswerShape.SET_LIST
        or query.operator_spec.operation == "count_or_aggregate"
    )
    needs_global = bool(
        global_closure_required
        or selection_truncated
        or unpacked_segments
        or unresolved_obligation_ids
    )
    frozen_packed = tuple(packed)
    frozen_duplicates = tuple(duplicates)
    frontier = LocalReinjectionFrontier(
        selected_segment_receipt_sha256s=tuple(
            row.segment.receipt_sha256 for row in selected
        ),
        packed_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in frozen_packed
        ),
        protected_duplicate_segment_receipt_sha256s=tuple(
            row.segment_receipt_sha256 for row in frozen_duplicates
        ),
        budget_unpacked_segment_receipt_sha256s=tuple(unpacked_segments),
        required_obligation_ids=required_obligation_ids,
        covered_obligation_ids=covered_obligation_ids,
        unresolved_obligation_ids=unresolved_obligation_ids,
        selection_truncated=selection_truncated,
        packing_closed=not unpacked_segments,
        local_obligations_satisfied=not unresolved_obligation_ids,
        needs_global_search=needs_global,
    )
    packed_tokens = _local_evidence_tokens(frozen_packed)
    attempted_tokens = _local_evidence_tokens(tuple(attempted_novel_rows))
    packed_sha = identity_sha256([row.provider_row() for row in frozen_packed])
    provider_tokens = count_tokens(
        _canonical_json(_provider_payload(frozen_packed, frontier))
    )
    return SourceGroupReinjectionResult(
        residual_index_receipt_sha256=residual_index.receipt_sha256,
        query_receipt_sha256=query.receipt_sha256,
        selection_receipt_sha256=selection.receipt_sha256,
        policy=policy,
        unresolved_input_slot_ids=unresolved,
        obligations=obligations,
        protected_evidence_population_receipt_sha256=protected_population_receipt,
        episode_plan=episode_plan,
        episode_population_receipt_sha256=episode_population_receipt,
        attempted_selection=tuple(attempts),
        protected_duplicates=frozen_duplicates,
        evidence=frozen_packed,
        local_bindings=tuple(packed_bindings),
        frontier=frontier,
        attempted_local_evidence_tokens=attempted_tokens,
        packed_local_evidence_tokens=packed_tokens,
        packed_local_evidence_sha256=packed_sha,
        provider_payload_tokens=provider_tokens,
    )


def validate_source_group_reinjection(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    result: SourceGroupReinjectionResult,
    /,
    *,
    protected_handle_bindings: Mapping[str, LocalCitationBinding] | None = None,
    unresolved_slot_ids: Sequence[str] | None = None,
    policy: SourceGroupReinjectionPolicy | None = None,
    episode_lookup: EpisodeLookup | None = None,
    episode_policy: EpisodeRetrievalPolicy | None = None,
) -> SourceGroupReinjectionResult:
    """Return one fresh replay after requiring byte-identical sealed output."""

    _require(
        type(result) is SourceGroupReinjectionResult,
        "sealed local reinjection result changed type",
    )
    active_policy = result.policy if policy is None else policy
    active_unresolved = (
        result.unresolved_input_slot_ids
        if unresolved_slot_ids is None
        else unresolved_slot_ids
    )
    replayed = search_source_group_reinjection(
        residual_index,
        query,
        selection,
        protected_handle_bindings=protected_handle_bindings,
        unresolved_slot_ids=active_unresolved,
        policy=active_policy,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
    )
    _require(
        replayed.receipt_sha256 == result.receipt_sha256
        and replayed.projection() == result.projection()
        and replayed.provider_projection() == result.provider_projection(),
        "source-group reinjection replay differs from sealed result",
    )
    return replayed


def replay_source_group_reinjection(
    residual_index: SemanticResidualIndex,
    query: SemanticResidualQuery,
    selection: AuthenticatedSourceGroupSelection,
    sealed_result: SourceGroupReinjectionResult,
    /,
    *,
    protected_handle_bindings: Mapping[str, LocalCitationBinding] | None = None,
    episode_lookup: EpisodeLookup | None = None,
    episode_policy: EpisodeRetrievalPolicy | None = None,
) -> SourceGroupReinjectionResult:
    """Return a fresh deterministic replay after strict result validation."""

    return validate_source_group_reinjection(
        residual_index,
        query,
        selection,
        sealed_result,
        protected_handle_bindings=protected_handle_bindings,
        unresolved_slot_ids=sealed_result.unresolved_input_slot_ids,
        policy=sealed_result.policy,
        episode_lookup=episode_lookup,
        episode_policy=episode_policy,
    )


__all__ = [
    "AuthenticatedSelectedHandle",
    "AuthenticatedSourceGroupRow",
    "AuthenticatedSourceGroupSelection",
    "LocalReinjectionEvidence",
    "LocalReinjectionFrontier",
    "LocalReinjectionObligation",
    "LocalSelectionAttempt",
    "ProtectedLocalDuplicate",
    "SourceGroupReinjectionError",
    "SourceGroupReinjectionPolicy",
    "SourceGroupReinjectionResult",
    "authenticate_source_group_selection",
    "replay_source_group_reinjection",
    "search_source_group_reinjection",
    "validate_source_group_reinjection",
    "validate_source_group_selection",
]
