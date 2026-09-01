"""Provider-free, gold-blind active reconstruction over a frozen full-store index.

The first full-store pass owns the question-derived operator and temporal target.
This module never recompiles either object.  It derives a small semantic cue set
from already admitted typed items and selected provider-safe candidates, then
hands that cue set to an injected local scanner.  The scanner is deliberately an
interface: the public full-store API does not expose a safe way to substitute
active cues without recompiling the question.

At most two hops are allowed.  Every admitted child is linked through the exact
parent item/candidate receipt, cue receipt, scan-match receipt, candidate
projection receipt, and local citation receipt.  Candidate/span deduplication is
performed only after the callback has selected its rows.  Physical index
coverage never becomes a semantic-exhaustiveness or absence claim.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from fractions import Fraction
import re
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    FullStoreSlotCandidate,
    FullStoreSlotClosureResult,
    FullStoreWindowIndex,
    IndexedContentWindow,
    LocalCitationBinding,
    MECHANISM_ID as FULL_STORE_MECHANISM_ID,
    QuestionTemporalTarget,
    TemporalTargetMode,
    indexed_surface_terms,
)
from .typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedEvidenceItem,
    conservative_numeric_value,
    parse_typed_items,
)
from .typed_operator_spec import (
    AnswerShape,
    SlotKind,
    TypedOperatorSpec,
    normalized_terms,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    canonical_action_proof_terms,
)
from .typed_story_affinity import (
    derive_evidence_story_affinity,
    evidence_history_story_key_sha256,
    evidence_source_story_key_sha256,
)


MECHANISM_ID = "typed_active_reconstruction_v1"
BUDGET_FORMAT = "memory-condense-typed-active-reconstruction-budget-v1"
CUE_FORMAT = "memory-condense-typed-active-reconstruction-cue-v1"
AFFINITY_FORMAT = "memory-condense-typed-active-reconstruction-affinity-v1"
SCAN_REQUEST_FORMAT = "memory-condense-typed-active-reconstruction-scan-request-v1"
SCAN_MATCH_FORMAT = "memory-condense-typed-active-reconstruction-scan-match-v1"
SCAN_BATCH_FORMAT = "memory-condense-typed-active-reconstruction-scan-batch-v1"
DECISION_FORMAT = "memory-condense-typed-active-reconstruction-decision-v1"
LINEAGE_FORMAT = "memory-condense-typed-active-reconstruction-lineage-v1"
HOP_FORMAT = "memory-condense-typed-active-reconstruction-hop-v1"
RESULT_FORMAT = "memory-condense-typed-active-reconstruction-result-v1"
ACTIVE_CANDIDATE_ID_FORMAT = (
    "memory-condense-typed-active-reconstruction-index-candidate-v1"
)
ACTIVE_INDEX_LOOKUP_FORMAT = (
    "memory-condense-typed-active-reconstruction-index-lookup-v1"
)
ACTIVE_CUE_POSTING_FANOUT_DIVISOR = 100
ACTIVE_CUE_POSTING_FANOUT_FLOOR = 64
ACTIVE_CUE_POSTING_FANOUT_CEILING = 1_024
# Locked questions are namespace-contiguous (ten ticks per 1M namespace).  Four
# entries preserve current/adjacent reuse without retaining a second lookup for
# all ten already resident full indexes.
_ACTIVE_INDEX_LOOKUP_CACHE_CAP = 4

class TypedActiveReconstructionError(MatchedEvalContractError):
    """Raised when active reconstruction loses a boundedness or lineage invariant."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedActiveReconstructionError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(set(values)) == len(values),
        f"{label} must be an ordered unique exact tuple",
    )
    return values


def active_candidate_id_for_window(
    index: FullStoreWindowIndex, window_index: int, /
) -> str:
    """Bind an active candidate ID to one exact immutable index window."""

    _require(type(index) is FullStoreWindowIndex, "active candidate index changed")
    _require(
        type(window_index) is int and 0 <= window_index < len(index.windows),
        "active candidate window index changed",
    )
    window = index.windows[window_index]
    return active_candidate_id_for_index_span(
        index,
        window_index,
        start_char=window.start_char,
        end_char=window.end_char,
        text_sha256=window.text_sha256,
    )


def active_candidate_id_for_index_span(
    index: FullStoreWindowIndex,
    window_index: int,
    /,
    *,
    start_char: int,
    end_char: int,
    text_sha256: str,
) -> str:
    """Bind an active candidate ID to an exact excerpt inside one index window."""

    _require(type(index) is FullStoreWindowIndex, "active candidate index changed")
    _require(
        type(window_index) is int and 0 <= window_index < len(index.windows),
        "active candidate window index changed",
    )
    window = index.windows[window_index]
    _require(
        type(start_char) is int
        and type(end_char) is int
        and window.start_char <= start_char < end_char <= window.end_char,
        "active candidate excerpt escaped its index window",
    )
    quote = window.row.text[start_char:end_char]
    require_sha256(text_sha256, "active candidate excerpt text")
    _require(quote_sha256(quote) == text_sha256, "active candidate excerpt changed")
    return identity_sha256(
        {
            "end_char": end_char,
            "format": ACTIVE_CANDIDATE_ID_FORMAT,
            "index_receipt_sha256": index.receipt_sha256,
            "row_receipt_sha256": identity_sha256(
                window.row.receipt_projection()
            ),
            "start_char": start_char,
            "text_sha256": text_sha256,
        }
    )


def candidate_projection_receipt_sha256(candidate: FullStoreSlotCandidate) -> str:
    """Return the sealed identity of the complete provider-safe candidate row."""

    _require(
        type(candidate) is FullStoreSlotCandidate,
        "candidate receipt requires an exact full-store candidate",
    )
    return identity_sha256(candidate.projection())


def citation_span_receipt_sha256(binding: LocalCitationBinding) -> str:
    """Return the exact local evidence-span identity without exposing it as a cue."""

    _require(
        type(binding) is LocalCitationBinding,
        "span receipt requires an exact local citation binding",
    )
    return identity_sha256(binding.span.identity_payload())


def local_component_key_sha256(namespace_id: str, partition_id: str) -> str:
    """Legacy helper for callers that already hold an exact history component.

    Active reconstruction itself derives history identity from the selected
    source stream with :func:`local_history_key_sha256`; it never substitutes a
    physical routing bucket for that evidence-derived identity.
    """

    return evidence_history_story_key_sha256(
        namespace_id,
        partition_id,
        partition_separator="::",
    )


def local_source_key_sha256(namespace_id: str, source_id: str) -> str:
    """Return the canonical story-source key for selected provenance."""

    return evidence_source_story_key_sha256(namespace_id, source_id)


def local_history_key_sha256(namespace_id: str, source_id: str) -> str:
    """Return the shared enclosing-history key for one exact source stream."""

    return derive_evidence_story_affinity(
        namespace_id,
        source_id,
        partition_separator="::",
    ).history_story_key_sha256


@dataclass(frozen=True, slots=True)
class ActiveIndexLookup:
    """Process-local symbolic postings derived from one immutable index.

    The lookup contains no embeddings or transformer token state.  It is keyed
    by the already sealed full-store index receipt and only caches deterministic
    coordinates, opaque story keys, and the small sealed action vocabulary.
    Existing provider/result receipts intentionally do not depend on whether
    this acceleration structure was cold-built or reused.
    """

    index_receipt_sha256: str
    cache_receipt_sha256: str
    window_count: int
    window_indices_by_chunk_id: Mapping[str, tuple[int, ...]]
    window_indices_by_source_key: Mapping[str, tuple[int, ...]]
    window_indices_by_history_key: Mapping[str, tuple[int, ...]]
    window_indices_by_action_concept: Mapping[str, tuple[int, ...]]
    source_key_by_window: tuple[str, ...]
    history_key_by_window: tuple[str, ...]
    action_concepts_by_window: tuple[tuple[str, ...], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.index_receipt_sha256, "active lookup index")
        require_sha256(self.cache_receipt_sha256, "active lookup cache")
        _require(
            type(self.window_count) is int
            and self.window_count >= 0
            and len(self.source_key_by_window) == self.window_count
            and len(self.history_key_by_window) == self.window_count
            and len(self.action_concepts_by_window) == self.window_count,
            "active lookup window inventory changed",
        )
        for value in (*self.source_key_by_window, *self.history_key_by_window):
            require_sha256(value, "active lookup story key")
        for concepts in self.action_concepts_by_window:
            _require(
                type(concepts) is tuple
                and tuple(sorted(set(concepts))) == concepts,
                "active lookup action concepts changed",
            )
        for postings, label in (
            (self.window_indices_by_chunk_id, "active chunk postings"),
            (self.window_indices_by_source_key, "active source postings"),
            (self.window_indices_by_history_key, "active history postings"),
            (self.window_indices_by_action_concept, "active action postings"),
        ):
            _require(isinstance(postings, Mapping), f"{label} changed type")
            for values in postings.values():
                _require(
                    type(values) is tuple
                    and all(
                        0 <= value < self.window_count for value in values
                    )
                    and all(
                        left < right
                        for left, right in zip(values, values[1:], strict=False)
                    ),
                    f"{label} contains invalid coordinates",
                )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "active index lookup receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="active_index_lookup")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "action_bucket_count": len(self.window_indices_by_action_concept),
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "chunk_bucket_count": len(self.window_indices_by_chunk_id),
            "derivation": "exact_immutable_index_content",
            "format": ACTIVE_INDEX_LOOKUP_FORMAT,
            "history_bucket_count": len(self.window_indices_by_history_key),
            "index_receipt_sha256": self.index_receipt_sha256,
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_bucket_count": len(self.window_indices_by_source_key),
            "window_count": self.window_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


_ACTIVE_INDEX_LOOKUPS: OrderedDict[str, ActiveIndexLookup] = OrderedDict()
_ACTIVE_INDEX_LOOKUP_LOCK = RLock()
_ACTIVE_INDEX_LOOKUP_BUILD_COUNT = 0
_ACTIVE_INDEX_LOOKUP_HIT_COUNT = 0


def _frozen_postings(
    values: Mapping[str, list[int]],
) -> Mapping[str, tuple[int, ...]]:
    return MappingProxyType(
        {key: tuple(indices) for key, indices in sorted(values.items())}
    )


def _build_active_index_lookup(index: FullStoreWindowIndex) -> ActiveIndexLookup:
    chunks: dict[str, list[int]] = {}
    sources: dict[str, list[int]] = {}
    histories: dict[str, list[int]] = {}
    actions: dict[str, list[int]] = {}
    source_keys: list[str] = []
    history_keys: list[str] = []
    action_rows: list[tuple[str, ...]] = []
    story_by_source: dict[str, tuple[str, str]] = {}
    interned_actions: dict[tuple[str, ...], tuple[str, ...]] = {}
    for window_index, window in enumerate(index.windows):
        row = window.row
        story = story_by_source.get(row.source_id)
        if story is None:
            affinity = derive_evidence_story_affinity(
                row.namespace_id,
                row.source_id,
                partition_separator="::",
            )
            story = (
                affinity.source_story_key_sha256,
                affinity.history_story_key_sha256,
            )
            story_by_source[row.source_id] = story
        source_key, history_key = story
        quote = row.text[window.start_char : window.end_char]
        concepts = canonical_action_concepts(quote)
        concepts = interned_actions.setdefault(concepts, concepts)
        chunks.setdefault(row.chunk_id, []).append(window_index)
        sources.setdefault(source_key, []).append(window_index)
        histories.setdefault(history_key, []).append(window_index)
        for concept in concepts:
            actions.setdefault(concept, []).append(window_index)
        source_keys.append(source_key)
        history_keys.append(history_key)
        action_rows.append(concepts)
    return ActiveIndexLookup(
        index_receipt_sha256=index.receipt_sha256,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        window_count=len(index.windows),
        window_indices_by_chunk_id=_frozen_postings(chunks),
        window_indices_by_source_key=_frozen_postings(sources),
        window_indices_by_history_key=_frozen_postings(histories),
        window_indices_by_action_concept=_frozen_postings(actions),
        source_key_by_window=tuple(source_keys),
        history_key_by_window=tuple(history_keys),
        action_concepts_by_window=tuple(action_rows),
    )


def active_index_lookup(index: FullStoreWindowIndex, /) -> ActiveIndexLookup:
    """Return one bounded process-local lookup for an immutable index receipt."""

    global _ACTIVE_INDEX_LOOKUP_BUILD_COUNT, _ACTIVE_INDEX_LOOKUP_HIT_COUNT
    _require(type(index) is FullStoreWindowIndex, "active lookup index changed")
    key = index.receipt_sha256
    with _ACTIVE_INDEX_LOOKUP_LOCK:
        cached = _ACTIVE_INDEX_LOOKUPS.get(key)
        if cached is not None:
            _ACTIVE_INDEX_LOOKUPS.move_to_end(key)
            _ACTIVE_INDEX_LOOKUP_HIT_COUNT += 1
            return cached
    built = _build_active_index_lookup(index)
    with _ACTIVE_INDEX_LOOKUP_LOCK:
        cached = _ACTIVE_INDEX_LOOKUPS.get(key)
        if cached is not None:
            _ACTIVE_INDEX_LOOKUPS.move_to_end(key)
            _ACTIVE_INDEX_LOOKUP_HIT_COUNT += 1
            return cached
        _ACTIVE_INDEX_LOOKUPS[key] = built
        _ACTIVE_INDEX_LOOKUP_BUILD_COUNT += 1
        while len(_ACTIVE_INDEX_LOOKUPS) > _ACTIVE_INDEX_LOOKUP_CACHE_CAP:
            _ACTIVE_INDEX_LOOKUPS.popitem(last=False)
        return built


def active_index_lookup_cache_audit() -> dict[str, Any]:
    """Return prompt-external cache/reuse counters; never part of result identity."""

    with _ACTIVE_INDEX_LOOKUP_LOCK:
        value = {
            "build_count": _ACTIVE_INDEX_LOOKUP_BUILD_COUNT,
            "cache_entry_cap": _ACTIVE_INDEX_LOOKUP_CACHE_CAP,
            "cached_entry_count": len(_ACTIVE_INDEX_LOOKUPS),
            "format": f"{ACTIVE_INDEX_LOOKUP_FORMAT}-process-cache-audit",
            "hit_count": _ACTIVE_INDEX_LOOKUP_HIT_COUNT,
            "index_receipt_sha256s": list(_ACTIVE_INDEX_LOOKUPS),
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    assert_gold_blind(value, path="active_index_lookup_cache_audit")
    return value


def _reset_active_index_lookup_cache_for_tests() -> None:
    global _ACTIVE_INDEX_LOOKUP_BUILD_COUNT, _ACTIVE_INDEX_LOOKUP_HIT_COUNT
    with _ACTIVE_INDEX_LOOKUP_LOCK:
        _ACTIVE_INDEX_LOOKUPS.clear()
        _ACTIVE_INDEX_LOOKUP_BUILD_COUNT = 0
        _ACTIVE_INDEX_LOOKUP_HIT_COUNT = 0


@dataclass(frozen=True, slots=True)
class ActiveReconstructionBudget:
    """Hard cue, hop, callback-output, and aggregate admission bounds."""

    max_hops: int = 2
    max_cues_per_hop: int = 16
    max_terms_per_cue: int = 16
    max_cue_terms_per_hop: int = 128
    max_selected_candidates_per_hop: int = 8
    max_selected_tokens_per_hop: int = 1_024
    max_admitted_candidates: int = 12
    max_admitted_tokens: int = 1_536
    use_selected_provenance_affinity: bool = False
    use_index_aware_cue_ranking: bool = False
    use_fixed_scan_subchannels: bool = False
    use_coverage_aware_callback_selection: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {
                "use_selected_provenance_affinity",
                "use_index_aware_cue_ranking",
                "use_fixed_scan_subchannels",
                "use_coverage_aware_callback_selection",
            }:
                _require(type(value) is bool, "affinity policy must be exact")
                continue
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(self.max_hops <= 2, "active reconstruction is capped at two hops")

    def projection(self) -> dict[str, Any]:
        value = {
            "format": BUDGET_FORMAT,
            "max_admitted_candidates": self.max_admitted_candidates,
            "max_admitted_tokens": self.max_admitted_tokens,
            "max_cue_terms_per_hop": self.max_cue_terms_per_hop,
            "max_cues_per_hop": self.max_cues_per_hop,
            "max_hops": self.max_hops,
            "max_selected_candidates_per_hop": (
                self.max_selected_candidates_per_hop
            ),
            "max_selected_tokens_per_hop": self.max_selected_tokens_per_hop,
            "max_terms_per_cue": self.max_terms_per_cue,
            "use_selected_provenance_affinity": (
                self.use_selected_provenance_affinity
            ),
        }
        # Keep the legacy/default budget identity byte-for-byte stable.  The
        # repair is an explicit experiment until a locked treatment promotes
        # it, so only enabled policies enter the sealed budget projection.
        if self.use_index_aware_cue_ranking:
            value["use_index_aware_cue_ranking"] = True
        if self.use_fixed_scan_subchannels:
            value["use_fixed_scan_subchannels"] = True
        if self.use_coverage_aware_callback_selection:
            value["use_coverage_aware_callback_selection"] = True
        return value

    @property
    def budget_id(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class SelectedEvidenceAffinity:
    """Opaque local-only component affinity derived from selected provenance.

    The raw namespace, source, and partition never enter this object.  A local
    scanner can derive the same domain-separated hashes for index rows, but the
    values are not part of any answer-provider projection.
    """

    parent_candidate_receipt_sha256: str
    parent_local_binding_receipt_sha256: str
    component_key_sha256: str
    source_key_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.parent_candidate_receipt_sha256, "affinity parent candidate"),
            (self.parent_local_binding_receipt_sha256, "affinity parent binding"),
            (self.component_key_sha256, "affinity component key"),
            (self.source_key_sha256, "affinity source key"),
        ):
            require_sha256(value, label)
        expected = identity_sha256(self.audit_projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "affinity receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def scanner_projection(self) -> dict[str, Any]:
        return {
            "component_key_sha256": self.component_key_sha256,
            "format": AFFINITY_FORMAT,
            "receipt_sha256": self.receipt_sha256,
            "source_key_sha256": self.source_key_sha256,
        }

    def audit_projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "component_key_sha256": self.component_key_sha256,
            "format": AFFINITY_FORMAT,
            "parent_candidate_receipt_sha256": (
                self.parent_candidate_receipt_sha256
            ),
            "parent_local_binding_receipt_sha256": (
                self.parent_local_binding_receipt_sha256
            ),
            "source_key_sha256": self.source_key_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def selected_evidence_affinity(
    candidate: FullStoreSlotCandidate,
    binding: LocalCitationBinding,
) -> SelectedEvidenceAffinity:
    """Derive opaque component/source affinity from one already selected row."""

    _require(
        type(candidate) is FullStoreSlotCandidate
        and type(binding) is LocalCitationBinding
        and candidate.candidate_id == binding.candidate_id
        and candidate.citation_binding_receipt_sha256 == binding.receipt_sha256,
        "selected affinity requires an exact candidate/local-binding pair",
    )
    story = derive_evidence_story_affinity(
        binding.namespace_id,
        binding.source_id,
        partition_separator="::",
    )
    return SelectedEvidenceAffinity(
        parent_candidate_receipt_sha256=candidate_projection_receipt_sha256(
            candidate
        ),
        parent_local_binding_receipt_sha256=binding.receipt_sha256,
        component_key_sha256=story.history_story_key_sha256,
        source_key_sha256=story.source_story_key_sha256,
    )


@dataclass(frozen=True, slots=True)
class ActiveReconstructionCue:
    """One bounded semantic cue with prompt-external parent lineage."""

    hop: int
    parent_kind: Literal["typed_item", "candidate"]
    parent_receipt_sha256: str
    semantic_projection_sha256: str
    terms: tuple[str, ...]
    action_concepts: tuple[str, ...] = ()
    selected_evidence_affinity: SelectedEvidenceAffinity | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.hop) is int and 1 <= self.hop <= 2, "cue hop changed")
        _require(
            self.parent_kind in {"typed_item", "candidate"},
            "cue parent kind changed",
        )
        require_sha256(self.parent_receipt_sha256, "cue parent")
        require_sha256(self.semantic_projection_sha256, "cue semantic projection")
        _ordered_unique(self.terms, "cue terms")
        _require(
            all(
                term == term.casefold()
                and len(tuple(normalized_terms(term))) == 1
                for term in self.terms
            ),
            "cue terms must be normalized single semantic terms",
        )
        _ordered_unique(self.action_concepts, "cue action concepts")
        _require(
            all(
                canonical_action_concepts(concept) == (concept,)
                for concept in self.action_concepts
            ),
            "cue action concepts must be canonical",
        )
        if self.selected_evidence_affinity is not None:
            _require(
                type(self.selected_evidence_affinity) is SelectedEvidenceAffinity
                and self.parent_kind == "candidate"
                and self.selected_evidence_affinity.parent_candidate_receipt_sha256
                == self.parent_receipt_sha256,
                "cue affinity escaped its selected candidate",
            )
        expected = identity_sha256(self.audit_projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "cue receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.audit_projection(), path="active_reconstruction_cue")

    def scanner_projection(self) -> dict[str, Any]:
        """Only this ID/locator-free surface is supplied to the scan callback."""

        value: dict[str, Any] = {
            "cue_receipt_sha256": self.receipt_sha256,
            "format": CUE_FORMAT,
            "semantic_source": self.parent_kind,
            "terms": list(self.terms),
            "action_concepts": list(self.action_concepts),
        }
        if self.selected_evidence_affinity is not None:
            value["selected_evidence_affinity"] = (
                self.selected_evidence_affinity.scanner_projection()
            )
        return value

    def audit_projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "format": CUE_FORMAT,
            "hop": self.hop,
            "parent_kind": self.parent_kind,
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "semantic_projection_sha256": self.semantic_projection_sha256,
            "action_concepts": list(self.action_concepts),
            "selected_evidence_affinity": (
                None
                if self.selected_evidence_affinity is None
                else self.selected_evidence_affinity.audit_projection()
            ),
            "terms": list(self.terms),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActiveReconstructionScanRequest:
    """Local scan request; the immutable index is never serialized to a provider."""

    index: FullStoreWindowIndex
    operator_spec: TypedOperatorSpec
    temporal_target: QuestionTemporalTarget
    hop: int
    lineage_parent_receipt_sha256: str
    cues: tuple[ActiveReconstructionCue, ...]
    max_selected_candidates: int
    max_selected_tokens: int
    use_fixed_scan_subchannels: bool = False
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""
    use_coverage_aware_callback_selection: bool = False

    def __post_init__(self) -> None:
        _require(type(self.index) is FullStoreWindowIndex, "scan index changed")
        _require(
            type(self.operator_spec) is TypedOperatorSpec,
            "scan operator spec changed",
        )
        _require(
            type(self.temporal_target) is QuestionTemporalTarget,
            "scan temporal target changed",
        )
        _require(type(self.hop) is int and 1 <= self.hop <= 2, "scan hop changed")
        require_sha256(self.lineage_parent_receipt_sha256, "scan lineage parent")
        _require(
            type(self.cues) is tuple
            and self.cues
            and all(
                type(cue) is ActiveReconstructionCue and cue.hop == self.hop
                for cue in self.cues
            ),
            "scan cues changed",
        )
        _require(
            len({cue.receipt_sha256 for cue in self.cues}) == len(self.cues),
            "scan cues repeat",
        )
        for value, label in (
            (self.max_selected_candidates, "scan candidate cap"),
            (self.max_selected_tokens, "scan token cap"),
        ):
            _require(type(value) is int and value > 0, f"{label} changed")
        _require(
            type(self.use_fixed_scan_subchannels) is bool
            and type(self.use_coverage_aware_callback_selection) is bool,
            "scan selection policy changed",
        )
        _require(
            self.retained_transformer_token_state_bytes == 0,
            "scan request retained transformer state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "scan request receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(
            self.projection(), path="active_reconstruction_scan_request"
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "cues": [cue.scanner_projection() for cue in self.cues],
            "format": SCAN_REQUEST_FORMAT,
            "hop": self.hop,
            "index_receipt_sha256": self.index.receipt_sha256,
            "lineage_parent_receipt_sha256": (
                self.lineage_parent_receipt_sha256
            ),
            "max_selected_candidates": self.max_selected_candidates,
            "max_selected_tokens": self.max_selected_tokens,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "temporal_target_receipt_sha256": self.temporal_target.receipt_sha256,
        }
        # Preserve the sealed legacy request when the repair is not selected.
        if self.use_fixed_scan_subchannels:
            value["use_fixed_scan_subchannels"] = True
        if self.use_coverage_aware_callback_selection:
            value["use_coverage_aware_callback_selection"] = True
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class ActiveReconstructionSupportKind(str, Enum):
    DIRECT_LEXICAL = "direct_lexical"
    SEALED_ACTION_EQUIVALENCE = "sealed_action_equivalence"
    SELECTED_SOURCE_AFFINITY = "selected_source_affinity"
    SELECTED_HISTORY_AFFINITY = "selected_history_affinity"


@dataclass(frozen=True, slots=True)
class ActiveReconstructionCandidateMatch:
    """One callback-selected exact candidate with cue attribution."""

    candidate: FullStoreSlotCandidate
    local_binding: LocalCitationBinding
    support_kind: ActiveReconstructionSupportKind
    supporting_cue_receipt_sha256s: tuple[str, ...]
    matched_cue_terms: tuple[str, ...]
    matched_child_terms: tuple[str, ...]
    action_concept: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.candidate) is FullStoreSlotCandidate,
            "scan match candidate changed",
        )
        _require(
            type(self.local_binding) is LocalCitationBinding,
            "scan match local binding changed",
        )
        _require(
            type(self.support_kind) is ActiveReconstructionSupportKind,
            "scan match support kind changed",
        )
        _require(
            self.candidate.candidate_id == self.local_binding.candidate_id
            and self.candidate.source_group_handle
            == self.local_binding.source_group_handle
            and self.candidate.quote_sha256 == self.local_binding.quote_sha256
            and self.candidate.citation_binding_receipt_sha256
            == self.local_binding.receipt_sha256,
            "scan candidate lost its exact local citation",
        )
        _ordered_unique(self.supporting_cue_receipt_sha256s, "supporting cues")
        _require(self.supporting_cue_receipt_sha256s, "scan match needs a cue")
        for value in self.supporting_cue_receipt_sha256s:
            require_sha256(value, "supporting cue")
        _ordered_unique(self.matched_cue_terms, "matched cue terms")
        _ordered_unique(self.matched_child_terms, "matched child terms")
        if self.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL:
            _require(
                bool(self.matched_cue_terms)
                and self.matched_cue_terms == self.matched_child_terms
                and self.action_concept is None,
                "direct lexical proof shape changed",
            )
        elif (
            self.support_kind
            is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
        ):
            _require(
                not self.matched_cue_terms
                and bool(self.matched_child_terms)
                and type(self.action_concept) is str
                and canonical_action_concepts(self.action_concept)
                == (self.action_concept,),
                "sealed action proof shape changed",
            )
        else:
            _require(
                not self.matched_cue_terms
                and not self.matched_child_terms
                and self.action_concept is None,
                "affinity proof carried false lexical attribution",
            )
        _require(
            all(
                type(term) is str
                and term
                and term == term.casefold()
                and len(tuple(normalized_terms(term))) == 1
                for term in (*self.matched_cue_terms, *self.matched_child_terms)
            ),
            "matched proof terms must be lowercase single semantic terms",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "scan match receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="active_reconstruction_scan_match")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "action_concept": self.action_concept,
            "candidate": self.candidate.projection(),
            "candidate_projection_receipt_sha256": (
                candidate_projection_receipt_sha256(self.candidate)
            ),
            "format": SCAN_MATCH_FORMAT,
            "local_binding_receipt_sha256": self.local_binding.receipt_sha256,
            "matched_child_terms": list(self.matched_child_terms),
            "matched_cue_terms": list(self.matched_cue_terms),
            "span_receipt_sha256": citation_span_receipt_sha256(
                self.local_binding
            ),
            "supporting_cue_receipt_sha256s": list(
                self.supporting_cue_receipt_sha256s
            ),
            "support_kind": self.support_kind.value,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActiveReconstructionScanBatch:
    """Sealed callback output before wrapper-side deduplication/admission."""

    request_receipt_sha256: str
    matches: tuple[ActiveReconstructionCandidateMatch, ...]
    candidate_population_count: int
    selection_truncated: bool
    new_provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.request_receipt_sha256, "scan batch request")
        _require(
            type(self.matches) is tuple
            and all(
                type(match) is ActiveReconstructionCandidateMatch
                for match in self.matches
            ),
            "scan batch matches changed",
        )
        _require(
            type(self.candidate_population_count) is int
            and self.candidate_population_count >= len(self.matches),
            "scan candidate population changed",
        )
        _require(
            type(self.selection_truncated) is bool,
            "scan truncation flag changed",
        )
        _require(
            self.new_provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0,
            "scan batch must remain provider-free and zero-state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "scan batch receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="active_reconstruction_scan_batch")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "candidate_population_count": self.candidate_population_count,
            "format": SCAN_BATCH_FORMAT,
            "match_receipt_sha256s": [row.receipt_sha256 for row in self.matches],
            "new_provider_calls": 0,
            "request_receipt_sha256": self.request_receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selected_candidate_count": len(self.matches),
            "selected_candidate_tokens": sum(
                row.candidate.token_count for row in self.matches
            ),
            "selection_truncated": self.selection_truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class ActiveReconstructionCandidateScanner(Protocol):
    """Injected provider-free scanner over the exact supplied prebuilt index."""

    def __call__(
        self, request: ActiveReconstructionScanRequest, /
    ) -> ActiveReconstructionScanBatch: ...


DecisionStatus = Literal[
    "admitted",
    "duplicate_exact_candidate_or_span",
    "aggregate_budget_excluded",
]


@dataclass(frozen=True, slots=True)
class ActiveReconstructionDecision:
    match_receipt_sha256: str
    candidate_projection_receipt_sha256: str
    span_receipt_sha256: str
    status: DecisionStatus
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.match_receipt_sha256, "decision match"),
            (self.candidate_projection_receipt_sha256, "decision candidate"),
            (self.span_receipt_sha256, "decision span"),
        ):
            require_sha256(value, label)
        _require(
            self.status
            in {
                "admitted",
                "duplicate_exact_candidate_or_span",
                "aggregate_budget_excluded",
            },
            "active reconstruction decision changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "decision receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "candidate_projection_receipt_sha256": (
                self.candidate_projection_receipt_sha256
            ),
            "format": DECISION_FORMAT,
            "match_receipt_sha256": self.match_receipt_sha256,
            "span_receipt_sha256": self.span_receipt_sha256,
            "status": self.status,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActiveReconstructionLineage:
    """Exact parent -> cue -> selected child chain for one admitted row."""

    hop: int
    parent_receipt_sha256s: tuple[str, ...]
    cue_receipt_sha256s: tuple[str, ...]
    selected_affinity_receipt_sha256s: tuple[str, ...]
    scan_match_receipt_sha256: str
    child_candidate_projection_receipt_sha256: str
    child_local_binding_receipt_sha256: str
    child_span_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.hop) is int and 1 <= self.hop <= 2, "lineage hop changed")
        for values, label in (
            (self.parent_receipt_sha256s, "lineage parents"),
            (self.cue_receipt_sha256s, "lineage cues"),
        ):
            _ordered_unique(values, label)
            _require(values, f"{label} cannot be empty")
            for value in values:
                require_sha256(value, label)
        _ordered_unique(
            self.selected_affinity_receipt_sha256s, "lineage selected affinities"
        )
        for value in self.selected_affinity_receipt_sha256s:
            require_sha256(value, "lineage selected affinity")
        for value, label in (
            (self.scan_match_receipt_sha256, "lineage scan match"),
            (
                self.child_candidate_projection_receipt_sha256,
                "lineage child candidate",
            ),
            (self.child_local_binding_receipt_sha256, "lineage child binding"),
            (self.child_span_receipt_sha256, "lineage child span"),
        ):
            require_sha256(value, label)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "lineage receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "child_candidate_projection_receipt_sha256": (
                self.child_candidate_projection_receipt_sha256
            ),
            "child_local_binding_receipt_sha256": (
                self.child_local_binding_receipt_sha256
            ),
            "child_span_receipt_sha256": self.child_span_receipt_sha256,
            "cue_receipt_sha256s": list(self.cue_receipt_sha256s),
            "format": LINEAGE_FORMAT,
            "hop": self.hop,
            "parent_receipt_sha256s": list(self.parent_receipt_sha256s),
            "scan_match_receipt_sha256": self.scan_match_receipt_sha256,
            "selected_affinity_receipt_sha256s": list(
                self.selected_affinity_receipt_sha256s
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class ActiveReconstructionHopResult:
    request: ActiveReconstructionScanRequest
    batch: ActiveReconstructionScanBatch
    decisions: tuple[ActiveReconstructionDecision, ...]
    admitted_matches: tuple[ActiveReconstructionCandidateMatch, ...]
    lineages: tuple[ActiveReconstructionLineage, ...]
    cue_derivation_truncated: bool
    admission_truncated: bool
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            type(self.request) is ActiveReconstructionScanRequest
            and type(self.batch) is ActiveReconstructionScanBatch,
            "hop request or batch changed",
        )
        _require(
            self.batch.request_receipt_sha256 == self.request.receipt_sha256,
            "hop batch escaped its request",
        )
        _require(
            type(self.decisions) is tuple
            and all(type(row) is ActiveReconstructionDecision for row in self.decisions)
            and len(self.decisions) == len(self.batch.matches),
            "hop decisions changed",
        )
        _require(
            type(self.admitted_matches) is tuple
            and all(
                type(row) is ActiveReconstructionCandidateMatch
                for row in self.admitted_matches
            ),
            "hop admitted matches changed",
        )
        admitted_receipts = tuple(
            decision.match_receipt_sha256
            for decision in self.decisions
            if decision.status == "admitted"
        )
        _require(
            admitted_receipts
            == tuple(row.receipt_sha256 for row in self.admitted_matches),
            "hop admission order changed",
        )
        _require(
            type(self.lineages) is tuple
            and len(self.lineages) == len(self.admitted_matches)
            and all(
                type(row) is ActiveReconstructionLineage
                and row.hop == self.request.hop
                for row in self.lineages
            ),
            "hop lineages changed",
        )
        for match, lineage in zip(
            self.admitted_matches, self.lineages, strict=True
        ):
            _require(
                lineage.scan_match_receipt_sha256 == match.receipt_sha256
                and lineage.child_candidate_projection_receipt_sha256
                == candidate_projection_receipt_sha256(match.candidate)
                and lineage.child_local_binding_receipt_sha256
                == match.local_binding.receipt_sha256
                and lineage.child_span_receipt_sha256
                == citation_span_receipt_sha256(match.local_binding),
                "hop child lineage changed",
            )
        _require(
            type(self.cue_derivation_truncated) is bool
            and type(self.admission_truncated) is bool,
            "hop truncation flags changed",
        )
        _require(
            self.retained_transformer_token_state_bytes == 0,
            "hop retained transformer state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "hop receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="active_reconstruction_hop")

    @property
    def hop(self) -> int:
        return self.request.hop

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "admission_truncated": self.admission_truncated,
            "admitted_candidate_count": len(self.admitted_matches),
            "admitted_candidate_tokens": sum(
                row.candidate.token_count for row in self.admitted_matches
            ),
            "admitted_lineage_receipt_sha256s": [
                row.receipt_sha256 for row in self.lineages
            ],
            "batch_receipt_sha256": self.batch.receipt_sha256,
            "cue_derivation_truncated": self.cue_derivation_truncated,
            "decision_receipt_sha256s": [
                row.receipt_sha256 for row in self.decisions
            ],
            "format": HOP_FORMAT,
            "hop": self.request.hop,
            "request_receipt_sha256": self.request.receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "admitted_local_bindings": [
                row.local_binding.projection() for row in self.admitted_matches
            ],
            "batch": self.batch.projection(),
            "cues": [cue.audit_projection() for cue in self.request.cues],
            "decisions": [row.projection() for row in self.decisions],
            "lineages": [row.projection() for row in self.lineages],
            "receipt": self.projection(),
            "request": self.request.projection(),
        }


@dataclass(frozen=True, slots=True)
class TypedActiveReconstructionResult:
    index: FullStoreWindowIndex
    parent_result: FullStoreSlotClosureResult
    parent_contribution: TypedEvidenceContribution | None
    operator_spec: TypedOperatorSpec
    temporal_target: QuestionTemporalTarget
    hops: tuple[ActiveReconstructionHopResult, ...]
    budget: ActiveReconstructionBudget
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(type(self.index) is FullStoreWindowIndex, "result index changed")
        _require(
            type(self.parent_result) is FullStoreSlotClosureResult,
            "active reconstruction parent changed",
        )
        _require(
            self.parent_result.receipt.window_index_receipt_sha256
            == self.index.receipt_sha256
            and self.parent_result.receipt.cache_receipt_sha256
            == self.index.cache.cache_receipt_sha256,
            "active reconstruction escaped its prebuilt parent index",
        )
        _require(
            self.operator_spec is self.parent_result.operator_spec
            and self.temporal_target is self.parent_result.temporal_target,
            "active reconstruction replaced the compiled operator or temporal target",
        )
        _validated_parent_seed_items(
            self.parent_result, self.parent_contribution
        )
        _require(
            type(self.hops) is tuple
            and len(self.hops) <= self.budget.max_hops <= 2
            and tuple(row.hop for row in self.hops)
            == tuple(range(1, len(self.hops) + 1)),
            "active reconstruction hop sequence changed",
        )
        expected_lineage_parent = self.parent_result.receipt.receipt_sha256
        previous_parent_receipts = {
            row.receipt_sha256 for row in self.admitted_seed_items
        } | {
            candidate_projection_receipt_sha256(row)
            for row in self.parent_result.candidates
        }
        for hop in self.hops:
            _require(
                hop.request.index is self.index
                and hop.request.operator_spec is self.operator_spec
                and hop.request.temporal_target is self.temporal_target,
                "active reconstruction hop recompiled or changed its scan scope",
            )
            _require(
                hop.request.lineage_parent_receipt_sha256
                == expected_lineage_parent,
                "active reconstruction hop changed its lineage parent",
            )
            _require(
                all(
                    cue.parent_receipt_sha256 in previous_parent_receipts
                    for cue in hop.request.cues
                ),
                "active reconstruction cue escaped the immediately prior hop",
            )
            expected_lineage_parent = hop.receipt_sha256
            previous_parent_receipts = {
                candidate_projection_receipt_sha256(row.candidate)
                for row in hop.admitted_matches
            }
        _require(
            self.candidate_count <= self.budget.max_admitted_candidates
            and self.candidate_tokens <= self.budget.max_admitted_tokens,
            "active reconstruction aggregate admission exceeded budget",
        )
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        span_receipts = tuple(
            citation_span_receipt_sha256(row) for row in self.local_bindings
        )
        _require(
            len(set(candidate_ids)) == len(candidate_ids)
            and len(set(span_receipts)) == len(span_receipts),
            "active reconstruction did not deduplicate exact candidates/spans",
        )
        _require(
            self.retained_transformer_token_state_bytes == 0,
            "active reconstruction retained transformer state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "reconstruction receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="typed_active_reconstruction")
        assert_gold_blind(
            self.provider_projection(), path="typed_active_reconstruction_provider"
        )

    @property
    def admitted_matches(self) -> tuple[ActiveReconstructionCandidateMatch, ...]:
        return tuple(match for hop in self.hops for match in hop.admitted_matches)

    @property
    def admitted_seed_items(self) -> tuple[TypedEvidenceItem, ...]:
        return (
            ()
            if self.parent_contribution is None
            else self.parent_contribution.parsed.accepted_items
        )

    @property
    def candidates(self) -> tuple[FullStoreSlotCandidate, ...]:
        return tuple(row.candidate for row in self.admitted_matches)

    @property
    def local_bindings(self) -> tuple[LocalCitationBinding, ...]:
        return tuple(row.local_binding for row in self.admitted_matches)

    @property
    def lineages(self) -> tuple[ActiveReconstructionLineage, ...]:
        return tuple(lineage for hop in self.hops for lineage in hop.lineages)

    @property
    def candidate_count(self) -> int:
        return len(self.admitted_matches)

    @property
    def candidate_tokens(self) -> int:
        return sum(row.token_count for row in self.candidates)

    @property
    def truncated(self) -> bool:
        return any(
            hop.cue_derivation_truncated
            or hop.admission_truncated
            or hop.batch.selection_truncated
            for hop in self.hops
        )

    def provider_projection(self) -> dict[str, Any]:
        """Bounded child evidence only; no source locator or absence claim."""

        return {
            "candidates": [row.projection() for row in self.candidates],
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "new_provider_calls": 0,
            "operator_spec": self.operator_spec.projection(),
            "retained_transformer_token_state_bytes": 0,
            "semantic_completeness_status": "not_claimed",
            "temporal_target": self.temporal_target.projection(),
            "truncated": self.truncated,
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "admitted_candidate_count": self.candidate_count,
            "admitted_candidate_tokens": self.candidate_tokens,
            "admitted_lineage_receipt_sha256s": [
                row.receipt_sha256 for row in self.lineages
            ],
            "admitted_seed_item_receipt_sha256s": [
                row.receipt_sha256 for row in self.admitted_seed_items
            ],
            "budget_id": self.budget.budget_id,
            "format": RESULT_FORMAT,
            "frontier_mode": FrontierMode.BOUNDED.value,
            "hop_receipt_sha256s": [row.receipt_sha256 for row in self.hops],
            "index_receipt_sha256": self.index.receipt_sha256,
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec.receipt_sha256,
            "parent_result_receipt_sha256": self.parent_result.receipt.receipt_sha256,
            "parent_contribution_receipt_sha256": (
                None
                if self.parent_contribution is None
                else self.parent_contribution.receipt_sha256
            ),
            "retained_transformer_token_state_bytes": 0,
            "semantic_completeness_status": "not_claimed",
            "temporal_target_receipt_sha256": self.temporal_target.receipt_sha256,
            "truncated": self.truncated,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def local_audit_projection(self) -> dict[str, Any]:
        return {
            "hops": [row.local_audit_projection() for row in self.hops],
            "local_bindings": [row.projection() for row in self.local_bindings],
            "provider_projection_sha256": identity_sha256(
                self.provider_projection()
            ),
            "receipt": self.projection(),
        }


def _semantic_terms(parts: Sequence[object]) -> tuple[str, ...]:
    terms: list[str] = []
    for part in parts:
        if part is None or type(part) is bool:
            continue
        values: Sequence[object]
        if isinstance(part, (tuple, list)):
            values = part
        else:
            values = (part,)
        for value in values:
            if value is None or type(value) is bool:
                continue
            for term in normalized_terms(str(value)):
                # The shared light stemmer deliberately preserves forms such as
                # ``it's`` as ``it'`` on the first pass; a second pass strips the
                # apostrophe and reveals a stop word.  Such non-stable tokens
                # cannot satisfy the sealed single-semantic-term invariant, so
                # drop them rather than weakening validation or aborting a tick.
                if len(normalized_terms(term)) == 1 and term not in terms:
                    terms.append(term)
    return tuple(terms)


def _typed_item_semantic_projection(item: TypedEvidenceItem) -> dict[str, Any]:
    """Whitelist semantic fields; never traverse item/handle/receipt identifiers."""

    return {
        "content_coherence": item.content_coherence.value,
        "date": item.date,
        "entity_key": item.entity_key,
        "group_key": item.group_key,
        "included": item.included,
        "kind": item.kind.value,
        "numeric_role": item.numeric_role.value,
        "numeric_value": item.numeric_value,
        "participant_count": item.participant_count,
        "personalization_anchors": list(item.personalization_anchors),
        "relation": item.relation,
        "specificity_terms": list(item.specificity_terms),
        "status": item.status.value,
        "summary": item.summary,
        "unit": item.unit,
        "value_authority": item.value_authority.value,
    }


def _candidate_semantic_projection(
    candidate: FullStoreSlotCandidate,
) -> dict[str, Any]:
    """Whitelist provider fields; omit candidate/group/slot/citation identifiers."""

    return {
        "contains_numeric_value": candidate.contains_numeric_value,
        "created_at": candidate.created_at,
        "event_date": candidate.event_date,
        "event_date_basis": candidate.event_date_basis,
        "matched_query_terms": list(candidate.matched_query_terms),
        "quote": candidate.quote,
        "role": candidate.role,
        "selection_axes": list(candidate.selection_axes),
        "temporal_distance_days": candidate.temporal_distance_days,
    }


def _terms_from_typed_item(item: TypedEvidenceItem) -> tuple[str, ...]:
    return _semantic_terms(
        (
            item.summary,
            item.entity_key,
            item.group_key,
            item.numeric_value,
            item.numeric_role.value,
            item.unit,
            item.date,
            item.status.value,
            item.relation,
            item.participant_count,
            item.value_authority.value,
            item.kind.value,
            item.specificity_terms,
            item.personalization_anchors,
        )
    )


def _terms_from_candidate(candidate: FullStoreSlotCandidate) -> tuple[str, ...]:
    return _semantic_terms(
        (
            candidate.quote,
            candidate.role,
            candidate.created_at,
            candidate.event_date,
            candidate.event_date_basis,
            candidate.matched_query_terms,
            candidate.temporal_distance_days,
            candidate.selection_axes,
        )
    )


_GENERIC_CUE_TERMS = frozenset(
    {
        "assistant",
        "authored",
        "complet",
        "direct",
        "explicit",
        "none",
        "unknown",
        "user",
    }
)


@dataclass(frozen=True, slots=True)
class _CueOffer:
    parent_kind: Literal["typed_item", "candidate"]
    parent_receipt_sha256: str
    semantic: dict[str, Any]
    terms: tuple[str, ...]
    action_concepts: tuple[str, ...]
    affinity: SelectedEvidenceAffinity | None
    stable_key: str


def _rank_cue_terms(
    terms: tuple[str, ...], operator_spec: TypedOperatorSpec
) -> tuple[str, ...]:
    slot_terms = {
        term for slot in operator_spec.required_slots for term in slot.match_terms
    }
    return tuple(
        sorted(
            terms,
            key=lambda term: (
                0 if term in slot_terms else 2
                if term in _GENERIC_CUE_TERMS or len(term) <= 1
                else 1,
                terms.index(term),
                term,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class _IndexAwareCueTerm:
    term: str
    unresolved_slot_support: bool
    action_support: bool
    temporal_support: bool
    personalization_support: bool
    specificity_support: bool
    posting_fanout: int
    expansion_posting_fanout: int
    field_order: int
    surface_order: int

    @property
    def priority_key(self) -> tuple[object, ...]:
        return (
            -int(self.unresolved_slot_support),
            -int(self.expansion_posting_fanout > 0),
            -int(self.action_support),
            -int(self.temporal_support),
            -int(self.personalization_support),
            -int(self.specificity_support),
            self.posting_fanout,
            self.field_order,
            self.surface_order,
            self.term,
        )


@dataclass(frozen=True, slots=True)
class _IndexAwareSemanticField:
    value: object
    temporal: bool = False
    personalization: bool = False
    specificity: bool = False
    action: bool = False


@dataclass(frozen=True, slots=True)
class _IndexAwareCueOffer:
    parent_kind: Literal["typed_item", "candidate"]
    parent_receipt_sha256: str
    semantic: dict[str, Any]
    ranked_terms: tuple[_IndexAwareCueTerm, ...]
    action_concepts: tuple[str, ...]
    affinity: SelectedEvidenceAffinity | None
    stable_key: str


_INDEX_AWARE_SURFACE_WORD_RE = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE
)


def _index_aware_typed_item_semantic_projection(
    item: TypedEvidenceItem,
) -> dict[str, Any]:
    """Hash only evidence semantics that can become a reconstruction edge.

    Parser protocol fields (kind/coherence/authority/inclusion/numeric role)
    are deliberately absent.  The exact typed-item receipt remains the parent
    provenance proof; this projection only seals the semantic material used to
    derive cues.
    """

    return {
        "date": item.date,
        "entity_key": item.entity_key,
        "group_key": item.group_key,
        "numeric_value": item.numeric_value,
        "participant_count": item.participant_count,
        "personalization_anchors": list(item.personalization_anchors),
        "relation": item.relation,
        "specificity_terms": list(item.specificity_terms),
        "status": item.status.value,
        "summary": item.summary,
        "unit": item.unit,
    }


def _index_aware_candidate_semantic_projection(
    candidate: FullStoreSlotCandidate,
) -> dict[str, Any]:
    """Hash exact fact fields without role, timestamp, or selection metadata."""

    return {
        "event_date": candidate.event_date,
        "matched_query_terms": list(candidate.matched_query_terms),
        "quote": candidate.quote,
    }


def _typed_item_index_aware_fields(
    item: TypedEvidenceItem,
) -> tuple[_IndexAwareSemanticField, ...]:
    fields: list[_IndexAwareSemanticField] = [
        _IndexAwareSemanticField(item.summary),
    ]
    for value in (item.entity_key, item.group_key):
        if value is not None:
            fields.append(_IndexAwareSemanticField(value, specificity=True))
    for value in item.specificity_terms:
        fields.append(_IndexAwareSemanticField(value, specificity=True))
    for value in item.personalization_anchors:
        fields.append(
            _IndexAwareSemanticField(
                value,
                personalization=True,
                specificity=True,
            )
        )
    if item.relation is not None:
        fields.append(_IndexAwareSemanticField(item.relation, action=True))
    if item.date is not None:
        fields.append(_IndexAwareSemanticField(item.date, temporal=True))
    # Status is semantic state, unlike parser kind/coherence/authority.  It is
    # useful only to temporal/state operators and receives no generic boost.
    fields.append(_IndexAwareSemanticField(item.status.value, temporal=True))
    for value in (
        item.numeric_value,
        item.participant_count,
        item.unit,
    ):
        if value is not None:
            fields.append(_IndexAwareSemanticField(value))
    return tuple(fields)


def _candidate_index_aware_fields(
    candidate: FullStoreSlotCandidate,
) -> tuple[_IndexAwareSemanticField, ...]:
    fields = [_IndexAwareSemanticField(candidate.quote)]
    if candidate.matched_query_terms:
        fields.append(_IndexAwareSemanticField(candidate.matched_query_terms))
    if candidate.event_date is not None:
        fields.append(
            _IndexAwareSemanticField(candidate.event_date, temporal=True)
        )
    return tuple(fields)


def _rank_index_aware_semantic_terms(
    index: FullStoreWindowIndex,
    fields: tuple[_IndexAwareSemanticField, ...],
    /,
    *,
    unresolved_slot_terms: frozenset[str],
    temporal_required: bool,
    personalization_required: bool,
    parent_window_indices: frozenset[int],
) -> tuple[_IndexAwareCueTerm, ...]:
    """Rank searchable semantic fields by obligations then inverse fanout."""

    cap = active_cue_posting_fanout_cap(index)
    merged: dict[str, _IndexAwareCueTerm] = {}
    for field_order, field in enumerate(fields):
        raw_values: Sequence[object]
        if isinstance(field.value, (tuple, list)):
            raw_values = field.value
        else:
            raw_values = (field.value,)
        surface_order = 0
        for raw_value in raw_values:
            if raw_value is None or type(raw_value) is bool:
                continue
            value = str(raw_value)
            action_terms = {
                normalized
                for surface in _INDEX_AWARE_SURFACE_WORD_RE.findall(value)
                if canonical_action_concepts(surface)
                for normalized in normalized_terms(surface)
            }
            for term in _semantic_terms((value,)):
                if (
                    term in _GENERIC_CUE_TERMS
                    or len(term) <= 1
                    or not (
                        0
                        < (fanout := len(index.term_postings.get(term, ())))
                        <= cap
                    )
                ):
                    surface_order += 1
                    continue
                offered = _IndexAwareCueTerm(
                    term=term,
                    unresolved_slot_support=term in unresolved_slot_terms,
                    action_support=bool(field.action or term in action_terms),
                    temporal_support=bool(field.temporal and temporal_required),
                    personalization_support=bool(
                        field.personalization and personalization_required
                    ),
                    specificity_support=field.specificity,
                    posting_fanout=fanout,
                    expansion_posting_fanout=sum(
                        window_index not in parent_window_indices
                        for window_index in index.term_postings.get(term, ())
                    ),
                    field_order=field_order,
                    surface_order=surface_order,
                )
                previous = merged.get(term)
                if previous is None:
                    merged[term] = offered
                else:
                    merged[term] = _IndexAwareCueTerm(
                        term=term,
                        unresolved_slot_support=bool(
                            previous.unresolved_slot_support
                            or offered.unresolved_slot_support
                        ),
                        action_support=bool(
                            previous.action_support or offered.action_support
                        ),
                        temporal_support=bool(
                            previous.temporal_support or offered.temporal_support
                        ),
                        personalization_support=bool(
                            previous.personalization_support
                            or offered.personalization_support
                        ),
                        specificity_support=bool(
                            previous.specificity_support
                            or offered.specificity_support
                        ),
                        posting_fanout=fanout,
                        expansion_posting_fanout=(
                            offered.expansion_posting_fanout
                        ),
                        field_order=min(previous.field_order, field_order),
                        surface_order=min(
                            previous.surface_order, surface_order
                        ),
                    )
                surface_order += 1
    return tuple(sorted(merged.values(), key=lambda row: row.priority_key))


def _validate_index_aware_candidate_pair(
    index: FullStoreWindowIndex,
    candidate: FullStoreSlotCandidate,
    binding: LocalCitationBinding,
) -> int:
    """Fail closed unless a cue parent is an exact excerpt of this index."""

    _require(
        type(candidate) is FullStoreSlotCandidate
        and type(binding) is LocalCitationBinding
        and candidate.candidate_id == binding.candidate_id
        and candidate.source_group_handle == binding.source_group_handle
        and candidate.citation_binding_receipt_sha256 == binding.receipt_sha256
        and candidate.quote_sha256 == binding.quote_sha256
        and binding.namespace_id == index.cache.namespace_id
        and binding.cache_receipt_sha256 == index.cache.cache_receipt_sha256
        and binding.source_database_sha256
        == index.cache.source_database_sha256
        and binding.source_store_receipt_sha256
        == index.cache.source_store_receipt_sha256,
        "index-aware cue parent escaped its exact provenance",
    )
    span = binding.span
    lookup = active_index_lookup(index)
    matching = tuple(
        (window_index, window)
        for window_index in lookup.window_indices_by_chunk_id.get(
            span.chunk_id, ()
        )
        for window in (index.windows[window_index],)
        if window.row.source_id == binding.source_id
        and window.row.partition_id == binding.partition_id
        and window.row.chunk_id == span.chunk_id
        and window.start_char <= span.start_char < span.end_char <= window.end_char
        and _expected_excerpt_span(
            window, span.start_char, span.end_char
        ).identity_payload()
        == span.identity_payload()
    )
    _require(matching, "index-aware cue parent span is absent from its index")
    window_index, window = min(
        matching,
        key=lambda pair: pair[1].end_char - pair[1].start_char,
    )
    quote = window.row.text[span.start_char : span.end_char]
    _require(
        candidate.quote == quote
        and candidate.quote_sha256 == quote_sha256(quote)
        and candidate.token_count == count_tokens(quote)
        and candidate.role == window.row.role
        and candidate.created_at == window.row.created_at
        and candidate.event_date == window.event_date
        and candidate.event_date_basis == window.event_date_basis,
        "index-aware cue parent fields changed from their exact excerpt",
    )
    return window_index


def derive_index_aware_active_reconstruction_cues(
    index: FullStoreWindowIndex,
    /,
    *,
    hop: int,
    items: tuple[TypedEvidenceItem, ...],
    candidate_pairs: tuple[
        tuple[FullStoreSlotCandidate, LocalCitationBinding], ...
    ],
    operator_spec: TypedOperatorSpec,
    temporal_target: QuestionTemporalTarget,
    budget: ActiveReconstructionBudget,
) -> tuple[tuple[ActiveReconstructionCue, ...], bool]:
    """Derive bounded low-fanout cues from exact, provenance-bound facts.

    The index is read only for immutable posting fanout.  No question text,
    provider, embedding, or retained transformer state enters the derivation.
    Terms that have no posting, exceed the sealed global-expansion fanout, or
    originate only in role/timestamp/selection/protocol metadata are excluded.
    """

    _require(type(index) is FullStoreWindowIndex, "cue ranking index changed")
    _require(type(hop) is int and 1 <= hop <= 2, "cue ranking hop changed")
    _require(
        type(items) is tuple
        and all(type(item) is TypedEvidenceItem for item in items),
        "cue ranking typed items changed",
    )
    _require(
        type(candidate_pairs) is tuple
        and all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is FullStoreSlotCandidate
            and type(pair[1]) is LocalCitationBinding
            for pair in candidate_pairs
        ),
        "cue ranking candidate pairs changed",
    )
    _require(
        type(operator_spec) is TypedOperatorSpec
        and type(temporal_target) is QuestionTemporalTarget
        and type(budget) is ActiveReconstructionBudget,
        "cue ranking compiled objects changed",
    )
    parent_window_indices = frozenset(
        _validate_index_aware_candidate_pair(index, candidate, binding)
        for candidate, binding in candidate_pairs
    )

    resolved_slots = {
        slot_id
        for item in items
        for slot_id in item.supported_slot_ids
    } | {
        slot_id
        for candidate, _binding in candidate_pairs
        for slot_id in candidate.supported_slot_ids
    }
    unresolved_slot_terms = frozenset(
        term
        for slot in operator_spec.required_slots
        if slot.slot_id not in resolved_slots
        for term in slot.match_terms
    )
    temporal_required = bool(
        operator_spec.temporal_mode.value != "none"
        or temporal_target.mode is not TemporalTargetMode.NONE
    )

    offered: list[_IndexAwareCueOffer] = []
    for item in items:
        semantic = _index_aware_typed_item_semantic_projection(item)
        offered.append(
            _IndexAwareCueOffer(
                parent_kind="typed_item",
                parent_receipt_sha256=item.receipt_sha256,
                semantic=semantic,
                ranked_terms=_rank_index_aware_semantic_terms(
                    index,
                    _typed_item_index_aware_fields(item),
                    unresolved_slot_terms=unresolved_slot_terms,
                    temporal_required=temporal_required,
                    personalization_required=(
                        operator_spec.personalization_required
                    ),
                    parent_window_indices=parent_window_indices,
                ),
                action_concepts=canonical_action_concepts(item.summary),
                affinity=None,
                stable_key=item.receipt_sha256,
            )
        )
    for candidate, binding in candidate_pairs:
        semantic = _index_aware_candidate_semantic_projection(candidate)
        affinity = (
            selected_evidence_affinity(candidate, binding)
            if budget.use_selected_provenance_affinity
            else None
        )
        offered.append(
            _IndexAwareCueOffer(
                parent_kind="candidate",
                parent_receipt_sha256=candidate_projection_receipt_sha256(
                    candidate
                ),
                semantic=semantic,
                ranked_terms=_rank_index_aware_semantic_terms(
                    index,
                    _candidate_index_aware_fields(candidate),
                    unresolved_slot_terms=unresolved_slot_terms,
                    temporal_required=temporal_required,
                    personalization_required=(
                        operator_spec.personalization_required
                    ),
                    parent_window_indices=parent_window_indices,
                ),
                action_concepts=canonical_action_concepts(candidate.quote),
                affinity=affinity,
                stable_key=citation_span_receipt_sha256(binding),
            )
        )

    ranked = sorted(
        (
            row
            for row in offered
            if row.ranked_terms
            or (
                budget.use_selected_provenance_affinity
                and row.affinity is not None
            )
        ),
        key=lambda row: (
            0 if row.ranked_terms else 1,
            (
                row.ranked_terms[0].priority_key
                if row.ranked_terms
                else ()
            ),
            0 if row.action_concepts else 1,
            0 if row.parent_kind == "candidate" else 1,
            0 if row.affinity is not None else 1,
            row.stable_key,
        ),
    )
    deduped: list[_IndexAwareCueOffer] = []
    seen_offer_keys: set[tuple[object, ...]] = set()
    for row in ranked:
        key = (
            row.parent_kind,
            row.stable_key,
            identity_sha256(row.semantic),
            row.action_concepts,
            None if row.affinity is None else row.affinity.receipt_sha256,
        )
        if key in seen_offer_keys:
            continue
        seen_offer_keys.add(key)
        deduped.append(row)

    selected: list[_IndexAwareCueOffer] = []
    initially_allocated_terms = 0
    for row in deduped:
        if len(selected) >= budget.max_cues_per_hop:
            break
        if row.ranked_terms:
            if initially_allocated_terms >= budget.max_cue_terms_per_hop:
                continue
            initially_allocated_terms += 1
        selected.append(row)
    allocations: list[list[_IndexAwareCueTerm]] = [
        ([row.ranked_terms[0]] if row.ranked_terms else [])
        for row in selected
    ]
    remaining_terms = (
        budget.max_cue_terms_per_hop - initially_allocated_terms
    )
    term_index = 1
    while remaining_terms > 0:
        progressed = False
        for index_value, row in enumerate(selected):
            if (
                len(allocations[index_value]) >= budget.max_terms_per_cue
                or term_index >= len(row.ranked_terms)
            ):
                continue
            allocations[index_value].append(row.ranked_terms[term_index])
            remaining_terms -= 1
            progressed = True
            if remaining_terms == 0:
                break
        if not progressed:
            break
        term_index += 1

    cues = tuple(
        ActiveReconstructionCue(
            hop=hop,
            parent_kind=row.parent_kind,
            parent_receipt_sha256=row.parent_receipt_sha256,
            semantic_projection_sha256=identity_sha256(row.semantic),
            terms=tuple(term.term for term in allocated),
            action_concepts=row.action_concepts,
            selected_evidence_affinity=row.affinity,
        )
        for row, allocated in zip(selected, allocations, strict=True)
    )
    truncated = bool(
        len(selected) < len(deduped)
        or any(
            len(allocated) < len(row.ranked_terms)
            for row, allocated in zip(selected, allocations, strict=True)
        )
    )
    return cues, truncated


def _derive_cues(
    *,
    hop: int,
    items: tuple[TypedEvidenceItem, ...],
    candidate_pairs: tuple[
        tuple[FullStoreSlotCandidate, LocalCitationBinding], ...
    ],
    operator_spec: TypedOperatorSpec,
    budget: ActiveReconstructionBudget,
) -> tuple[tuple[ActiveReconstructionCue, ...], bool]:
    offered: list[_CueOffer] = []
    for item in items:
        semantic = _typed_item_semantic_projection(item)
        offered.append(
            _CueOffer(
                parent_kind="typed_item",
                parent_receipt_sha256=item.receipt_sha256,
                semantic=semantic,
                terms=_rank_cue_terms(
                    _terms_from_typed_item(item), operator_spec
                ),
                action_concepts=canonical_action_concepts(item.summary),
                affinity=None,
                stable_key=item.receipt_sha256,
            )
        )
    for candidate, binding in candidate_pairs:
        semantic = _candidate_semantic_projection(candidate)
        affinity = (
            selected_evidence_affinity(candidate, binding)
            if budget.use_selected_provenance_affinity
            else None
        )
        offered.append(
            _CueOffer(
                parent_kind="candidate",
                parent_receipt_sha256=candidate_projection_receipt_sha256(
                    candidate
                ),
                semantic=semantic,
                terms=_rank_cue_terms(
                    _terms_from_candidate(candidate), operator_spec
                ),
                action_concepts=canonical_action_concepts(candidate.quote),
                affinity=affinity,
                stable_key=citation_span_receipt_sha256(binding),
            )
        )

    # Candidate/span seeds are reserved before typed restatements, and affinity
    # parents before lexical-only parents.  Stable receipts remove input-order
    # dependence.  Exact duplicate semantic/span offers are discarded.
    ranked = sorted(
        (row for row in offered if row.terms),
        key=lambda row: (
            0 if row.parent_kind == "candidate" and row.affinity is not None else 1
            if row.parent_kind == "candidate"
            else 2,
            0 if row.action_concepts else 1,
            row.stable_key,
        ),
    )
    deduped: list[_CueOffer] = []
    seen_offer_keys: set[tuple[object, ...]] = set()
    for row in ranked:
        key = (
            row.parent_kind,
            row.stable_key,
            identity_sha256(row.semantic),
            row.action_concepts,
            None if row.affinity is None else row.affinity.receipt_sha256,
        )
        if key in seen_offer_keys:
            continue
        seen_offer_keys.add(key)
        deduped.append(row)

    selected = deduped[
        : min(budget.max_cues_per_hop, budget.max_cue_terms_per_hop)
    ]
    allocations: list[list[str]] = [[row.terms[0]] for row in selected]
    remaining_terms = budget.max_cue_terms_per_hop - len(selected)
    term_index = 1
    while remaining_terms > 0:
        progressed = False
        for index, row in enumerate(selected):
            if (
                len(allocations[index]) >= budget.max_terms_per_cue
                or term_index >= len(row.terms)
            ):
                continue
            allocations[index].append(row.terms[term_index])
            remaining_terms -= 1
            progressed = True
            if remaining_terms == 0:
                break
        if not progressed:
            break
        term_index += 1

    cues: list[ActiveReconstructionCue] = []
    for row, allocated in zip(selected, allocations, strict=True):
        cues.append(
            ActiveReconstructionCue(
                hop=hop,
                parent_kind=row.parent_kind,
                parent_receipt_sha256=row.parent_receipt_sha256,
                semantic_projection_sha256=identity_sha256(row.semantic),
                terms=tuple(allocated),
                action_concepts=row.action_concepts,
                selected_evidence_affinity=row.affinity,
            )
        )
    truncated = bool(
        len(selected) < len(deduped)
        or any(len(allocated) < len(row.terms) for row, allocated in zip(
            selected, allocations, strict=True
        ))
    )
    return tuple(cues), truncated


_ACTIVE_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w])|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|dozen|hundred|thousand)\b",
    re.IGNORECASE,
)
def active_supported_slot_ids(
    operator_spec: TypedOperatorSpec, quote: str, /
) -> tuple[str, ...]:
    """Conservatively recompute slot support from exact child text."""

    eligible_slots = tuple(
        slot
        for slot in operator_spec.required_slots
        if slot.relation_constraint is None
    )
    if not eligible_slots:
        return ()
    quote_terms = frozenset(normalized_terms(quote))
    quote_actions = set(canonical_action_concepts(quote))
    contains_numeric = bool(_ACTIVE_NUMBER_RE.search(quote))
    supported: list[str] = []
    for slot in eligible_slots:
        overlap = len(set(slot.match_terms) & quote_terms)
        slot_actions = set(
            canonical_action_concepts(
                " ".join((slot.label, *slot.match_terms))
            )
        )
        if slot_actions & quote_actions:
            overlap = max(overlap, 1)
        if (
            overlap >= slot.minimum_match_term_count
            and (not slot.requires_numeric or contains_numeric)
        ):
            supported.append(slot.slot_id)
    return tuple(supported)


def _parse_active_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_active_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def active_temporal_support(
    event_date: str | None, target: QuestionTemporalTarget, /
) -> tuple[int | None, bool]:
    """Recompute distance/support using the unchanged first-pass target."""

    event = _parse_active_date(event_date)
    if event is None or target.mode is TemporalTargetMode.NONE:
        return None, False
    if target.mode is TemporalTargetMode.EXACT_DAY:
        wanted = _parse_active_date(target.target_date)
        if wanted is None:
            return None, False
        distance = abs((event - wanted).days)
        return distance, distance == 0
    asked = _parse_active_datetime(target.asked_at)
    if asked is None:
        return None, False
    signed_distance = (asked.date() - event).days
    return abs(signed_distance), bool(
        0 <= signed_distance <= (target.lookback_days or 0)
    )


def active_history_obligation_supported(
    request: ActiveReconstructionScanRequest,
    window_index: int,
    /,
) -> bool:
    """Require a question-bound reason before broad history affinity admits a row.

    Exact-source affinity remains a local-neighborhood mechanism.  An enclosing
    history component is much wider (thousands of windows in the 1M stores), so
    it is only a route into rows that also satisfy a cue, action, original slot,
    numeric-answer, or temporal obligation.
    """

    _require(
        type(request) is ActiveReconstructionScanRequest
        and type(window_index) is int
        and 0 <= window_index < len(request.index.windows),
        "active history obligation coordinates changed",
    )
    window = request.index.windows[window_index]
    lookup = active_index_lookup(request.index)
    selective_terms = active_selective_cue_terms(request)
    direct = bool(selective_terms & window.terms)
    cue_actions = {
        concept for cue in request.cues for concept in cue.action_concepts
    }
    action = bool(
        cue_actions & set(lookup.action_concepts_by_window[window_index])
    )
    quote = window.row.text[window.start_char : window.end_char]
    slots = bool(active_supported_slot_ids(request.operator_spec, quote))
    _distance, temporal = active_temporal_support(
        window.event_date, request.temporal_target
    )
    numeric = bool(
        request.operator_spec.answer_shape is AnswerShape.NUMBER
        and not request.operator_spec.required_slots
        and window.contains_numeric_value
    )
    return bool(direct or action or slots or temporal or numeric)


def active_cue_posting_fanout_cap(index: FullStoreWindowIndex, /) -> int:
    """Bound one expansion term to at most one percent of a large namespace."""

    _require(type(index) is FullStoreWindowIndex, "active cue index changed")
    return min(
        ACTIVE_CUE_POSTING_FANOUT_CEILING,
        max(
            ACTIVE_CUE_POSTING_FANOUT_FLOOR,
            len(index.windows) // ACTIVE_CUE_POSTING_FANOUT_DIVISOR,
        ),
    )


def active_selective_cue_terms(
    request: ActiveReconstructionScanRequest, /
) -> frozenset[str]:
    """Return cue terms selective enough to act as global expansion edges."""

    _require(
        type(request) is ActiveReconstructionScanRequest,
        "active selective cue request changed",
    )
    cap = active_cue_posting_fanout_cap(request.index)
    return frozenset(
        term
        for cue in request.cues
        for term in cue.terms
        if 0 < len(request.index.term_postings.get(term, ())) <= cap
    )


def _expected_excerpt_span(
    window: IndexedContentWindow, start_char: int, end_char: int
) -> EvidenceSpan:
    row = window.row
    quote = row.text[start_char:end_char]
    return EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=start_char,
        end_char=end_char,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )


def _exact_index_window_for_match(
    request: ActiveReconstructionScanRequest,
    match: ActiveReconstructionCandidateMatch,
) -> tuple[int, IndexedContentWindow]:
    local = match.local_binding
    candidate = match.candidate
    index = request.index
    _require(
        local.namespace_id == index.cache.namespace_id
        and local.cache_receipt_sha256 == index.cache.cache_receipt_sha256
        and local.source_database_sha256
        == index.cache.source_database_sha256
        and local.source_store_receipt_sha256
        == index.cache.source_store_receipt_sha256,
        "candidate scanner escaped the supplied full-store index",
    )
    lookup = active_index_lookup(index)
    matching: list[tuple[int, IndexedContentWindow]] = []
    span = local.span
    for window_index in lookup.window_indices_by_chunk_id.get(span.chunk_id, ()):
        window = index.windows[window_index]
        row = window.row
        if not (
            row.namespace_id == local.namespace_id
            and row.partition_id == local.partition_id
            and row.source_id == local.source_id
            and row.chunk_id == span.chunk_id
            and window.start_char <= span.start_char < span.end_char <= window.end_char
        ):
            continue
        expected_span = _expected_excerpt_span(
            window, span.start_char, span.end_char
        )
        if expected_span.identity_payload() == span.identity_payload():
            matching.append((window_index, window))
    _require(matching, "candidate span is not an excerpt of the supplied index")
    window_index, window = min(
        matching,
        key=lambda pair: (
            pair[1].end_char - pair[1].start_char,
            pair[0],
        ),
    )
    span = local.span
    quote = window.row.text[span.start_char : span.end_char]
    _require(
        candidate.candidate_id
        == active_candidate_id_for_index_span(
            index,
            window_index,
            start_char=span.start_char,
            end_char=span.end_char,
            text_sha256=local.quote_sha256,
        ),
        "candidate ID is not the recomputed index-span identity",
    )
    distance, temporal = active_temporal_support(
        window.event_date, request.temporal_target
    )
    slots = active_supported_slot_ids(request.operator_spec, quote)
    expected_numeric = (
        window.contains_numeric_value
        if span.start_char == window.start_char and span.end_char == window.end_char
        else bool(_ACTIVE_NUMBER_RE.search(quote))
    )
    expected_axes = [f"active_support:{match.support_kind.value}"]
    if slots:
        expected_axes.append("original_operator_slot_support")
    if temporal:
        expected_axes.append("original_temporal_target_support")
    _require(
        candidate.quote == quote
        and candidate.quote_sha256 == quote_sha256(quote) == local.quote_sha256
        and candidate.token_count == count_tokens(quote)
        and candidate.role == window.row.role == span.role
        and candidate.created_at == window.row.created_at == span.created_at
        and candidate.event_date == window.event_date
        and candidate.event_date_basis == window.event_date_basis
        and candidate.contains_numeric_value == expected_numeric
        and candidate.temporal_distance_days == distance
        and candidate.supported_slot_ids == slots
        and candidate.selection_axes == tuple(expected_axes),
        "candidate provider fields do not match the supplied index excerpt",
    )
    return window_index, window


def _validate_match_support(
    request: ActiveReconstructionScanRequest,
    match: ActiveReconstructionCandidateMatch,
    window_index: int,
    window: IndexedContentWindow,
) -> None:
    cue_by_receipt = {row.receipt_sha256: row for row in request.cues}
    _require(
        set(match.supporting_cue_receipt_sha256s) <= set(cue_by_receipt),
        "candidate scanner cited a cue outside the request",
    )
    cues = tuple(
        cue_by_receipt[receipt]
        for receipt in match.supporting_cue_receipt_sha256s
    )
    child_terms = frozenset(indexed_surface_terms(match.candidate.quote))
    if match.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL:
        selective_terms = active_selective_cue_terms(request)
        _require(
            all(term in child_terms for term in match.matched_child_terms)
            and all(
                any(term in cue.terms for cue in cues)
                for term in match.matched_cue_terms
            )
            and all(
                any(term in cue.terms for term in match.matched_cue_terms)
                for cue in cues
            )
            and match.candidate.matched_query_terms == match.matched_cue_terms,
            "direct lexical support is false",
        )
        _require(
            set(match.matched_cue_terms) <= selective_terms,
            "direct lexical support exceeds the sealed posting-fanout bound",
        )
    elif (
        match.support_kind
        is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
    ):
        concept = match.action_concept or ""
        expected_proof_terms = canonical_action_proof_terms(
            match.candidate.quote, concept
        )
        _require(
            concept in canonical_action_concepts(match.candidate.quote)
            and all(concept in cue.action_concepts for cue in cues)
            and match.matched_child_terms == expected_proof_terms
            and match.candidate.matched_query_terms == (),
            "sealed action-equivalence support is false",
        )
    else:
        lookup = active_index_lookup(request.index)
        source_key = lookup.source_key_by_window[window_index]
        history_key = lookup.history_key_by_window[window_index]
        _require(
            match.candidate.matched_query_terms == ()
            and all(cue.selected_evidence_affinity is not None for cue in cues),
            "affinity support carried lexical attribution or no selected proof",
        )
        if (
            match.support_kind
            is ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY
        ):
            _require(
                all(
                    cue.selected_evidence_affinity.source_key_sha256 == source_key
                    for cue in cues
                    if cue.selected_evidence_affinity is not None
                ),
                "selected source affinity is false",
            )
        else:
            _require(
                all(
                    cue.selected_evidence_affinity.component_key_sha256
                    == history_key
                    for cue in cues
                    if cue.selected_evidence_affinity is not None
                ),
                "selected history affinity is false",
            )
            _require(
                active_history_obligation_supported(request, window_index),
                "selected history affinity lacks question-bound obligation support",
            )


def _canonical_match_order(
    request: ActiveReconstructionScanRequest,
    matches: tuple[ActiveReconstructionCandidateMatch, ...],
) -> tuple[ActiveReconstructionCandidateMatch, ...]:
    remaining = list(matches)
    ordered: list[ActiveReconstructionCandidateMatch] = []
    source_counts: dict[str, int] = {}
    while remaining:
        match = min(
            remaining,
            key=lambda row: (
                -int(
                    row.support_kind
                    is ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY
                ),
                -int(
                    row.support_kind
                    is ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
                ),
                -len(row.candidate.supported_slot_ids),
                -int(
                    row.support_kind
                    is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
                ),
                -int("original_temporal_target_support" in row.candidate.selection_axes),
                -len(row.supporting_cue_receipt_sha256s),
                source_counts.get(row.local_binding.source_id, 0),
                row.candidate.token_count,
                citation_span_receipt_sha256(row.local_binding),
                row.receipt_sha256,
            ),
        )
        remaining.remove(match)
        ordered.append(match)
        source = match.local_binding.source_id
        source_counts[source] = source_counts.get(source, 0) + 1
    return tuple(ordered)


_COVERAGE_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i've|me|my|mine|we|we've|our|ours)\b",
    re.IGNORECASE,
)


def _coverage_hydration_match_order(
    request: ActiveReconstructionScanRequest,
    matches: tuple[ActiveReconstructionCandidateMatch, ...],
    window_index_by_receipt: Mapping[str, int],
) -> tuple[ActiveReconstructionCandidateMatch, ...]:
    """Canonically order selected membership for cumulative hydration.

    Fixed subchannels choose membership under separate 1:1:2 reservations, so
    their per-channel selection order cannot itself be replayed as one global
    greedy sequence.  Once the callback's membership and support are verified,
    this second stateful pass orders hydration by cumulative coverage.  It
    intentionally promotes source/history affinity ahead of broad global
    action noise, while recomputing every feature from the sealed request,
    exact matches, and immutable index rather than trusting callback order.
    """

    cue_by_receipt = {row.receipt_sha256: row for row in request.cues}
    required_slots = {
        row.slot_id for row in request.operator_spec.required_slots
    }
    requested_actions = {
        value for cue in request.cues for value in cue.action_concepts
    }
    lookup = active_index_lookup(request.index)
    features: dict[str, dict[str, object]] = {}
    for match in matches:
        window_index = window_index_by_receipt[match.receipt_sha256]
        window = request.index.windows[window_index]
        row = window.row
        quote = row.text[window.start_char : window.end_char]
        supporting = tuple(
            cue_by_receipt[value]
            for value in match.supporting_cue_receipt_sha256s
        )
        slots = frozenset(
            set(match.candidate.supported_slot_ids) & required_slots
        )
        direct = frozenset(
            (
                "direct:lexical",
                *(f"direct_term:{value}" for value in match.matched_cue_terms),
            )
            if match.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL
            else ()
        )
        temporal = frozenset(
            (
                "temporal:evidence",
                *(
                    (f"event_date:{window.event_date}",)
                    if window.event_date is not None
                    else ()
                ),
            )
            if "original_temporal_target_support"
            in match.candidate.selection_axes
            else ()
        )
        actions = frozenset(
            f"action:{value}"
            for value in set(canonical_action_concepts(quote))
            & requested_actions
        )
        personalization = frozenset(
            ("personalization:first_person",)
            if request.operator_spec.personalization_required
            and _COVERAGE_FIRST_PERSON_RE.search(quote)
            else ()
        )
        roles = set()
        if row.role == "user":
            roles.add("exact_role:user")
        if row.role == request.operator_spec.required_evidence_role:
            roles.add(f"required_role:{row.role}")
        locality = (
            2
            if match.support_kind
            is ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY
            else 1
            if match.support_kind
            is ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
            else 0
        )
        direct_quality = (
            3
            if match.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL
            else 2
            if slots
            else 1
            if locality
            else 0
        )
        protocol_only = bool(
            match.support_kind
            is ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
            and match.candidate.token_count <= 4
            and not slots
            and not match.matched_cue_terms
            and not temporal
        )
        density = (
            4 * len(slots)
            + 3 * len(direct)
            + 2 * len({row.parent_receipt_sha256 for row in supporting})
            + len(supporting)
            + 2 * len(temporal)
            + 2 * len(actions)
            + 2 * len(personalization)
            + 2 * len(roles)
            + int(window.contains_numeric_value)
        )
        if protocol_only:
            density = max(0, density - 8)
        features[match.receipt_sha256] = {
            "actions": actions,
            "cue_parents": frozenset(
                row.parent_receipt_sha256 for row in supporting
            ),
            "cue_receipts": frozenset(match.supporting_cue_receipt_sha256s),
            "density": density,
            "direct": direct,
            "direct_quality": direct_quality,
            "history": lookup.history_key_by_window[window_index],
            "locality": locality,
            "personalization": personalization,
            "protocol_only": protocol_only,
            "roles": frozenset(roles),
            "slots": slots,
            "source": lookup.source_key_by_window[window_index],
            "temporal": temporal,
            "turn": identity_sha256(
                {
                    "chunk_id": row.chunk_id,
                    "format": f"{SCAN_MATCH_FORMAT}-turn-key-v1",
                    "namespace_id": row.namespace_id,
                    "ordinal": row.ordinal,
                    "source_id": row.source_id,
                    "turn_id": row.turn_id,
                }
            ),
        }

    covered: dict[str, set[str]] = {
        key: set()
        for key in (
            "actions",
            "cue_parents",
            "cue_receipts",
            "direct",
            "histories",
            "personalization",
            "roles",
            "slots",
            "sources",
            "temporal",
            "turns",
        )
    }

    def new_count(values: object, key: str) -> int:
        _require(type(values) is frozenset, "coverage match feature changed")
        return len(set(values) - covered[key])

    remaining = list(matches)
    ordered: list[ActiveReconstructionCandidateMatch] = []
    while remaining:
        match = min(
            remaining,
            key=lambda candidate: (
                -new_count(features[candidate.receipt_sha256]["slots"], "slots"),
                -new_count(features[candidate.receipt_sha256]["direct"], "direct"),
                -int(features[candidate.receipt_sha256]["locality"]),
                -new_count(
                    features[candidate.receipt_sha256]["cue_parents"],
                    "cue_parents",
                ),
                -new_count(
                    features[candidate.receipt_sha256]["cue_receipts"],
                    "cue_receipts",
                ),
                -new_count(
                    features[candidate.receipt_sha256]["temporal"], "temporal"
                ),
                -new_count(
                    features[candidate.receipt_sha256]["personalization"],
                    "personalization",
                ),
                -new_count(features[candidate.receipt_sha256]["roles"], "roles"),
                -int(
                    features[candidate.receipt_sha256]["source"]
                    not in covered["sources"]
                ),
                -int(
                    features[candidate.receipt_sha256]["history"]
                    not in covered["histories"]
                ),
                -int(
                    features[candidate.receipt_sha256]["turn"]
                    not in covered["turns"]
                ),
                -new_count(
                    features[candidate.receipt_sha256]["actions"], "actions"
                ),
                -int(features[candidate.receipt_sha256]["direct_quality"]),
                bool(features[candidate.receipt_sha256]["protocol_only"]),
                -Fraction(
                    int(features[candidate.receipt_sha256]["density"]),
                    max(1, candidate.candidate.token_count),
                ),
                candidate.candidate.token_count,
                citation_span_receipt_sha256(candidate.local_binding),
                candidate.receipt_sha256,
            ),
        )
        remaining.remove(match)
        ordered.append(match)
        row = features[match.receipt_sha256]
        for feature_key, covered_key in (
            ("actions", "actions"),
            ("cue_parents", "cue_parents"),
            ("cue_receipts", "cue_receipts"),
            ("direct", "direct"),
            ("personalization", "personalization"),
            ("roles", "roles"),
            ("slots", "slots"),
            ("temporal", "temporal"),
        ):
            value = row[feature_key]
            _require(type(value) is frozenset, "coverage admission feature changed")
            covered[covered_key].update(value)
        covered["sources"].add(str(row["source"]))
        covered["histories"].add(str(row["history"]))
        covered["turns"].add(str(row["turn"]))
    return tuple(ordered)


def validate_active_reconstruction_scan_batch(
    request: ActiveReconstructionScanRequest,
    batch: ActiveReconstructionScanBatch,
) -> ActiveReconstructionScanBatch:
    """Validate and canonically order one untrusted callback batch."""

    _require(
        type(batch) is ActiveReconstructionScanBatch,
        "candidate scanner must return an exact sealed scan batch",
    )
    _require(
        batch.request_receipt_sha256 == request.receipt_sha256,
        "candidate scanner returned a batch for another request",
    )
    _require(
        len(batch.matches) <= request.max_selected_candidates
        and sum(row.candidate.token_count for row in batch.matches)
        <= request.max_selected_tokens,
        "candidate scanner exceeded its sealed per-hop budget",
    )
    window_index_by_receipt: dict[str, int] = {}
    for match in batch.matches:
        window_index, window = _exact_index_window_for_match(request, match)
        _validate_match_support(request, match, window_index, window)
        window_index_by_receipt[match.receipt_sha256] = window_index
    ordered = (
        _coverage_hydration_match_order(
            request, batch.matches, window_index_by_receipt
        )
        if request.use_coverage_aware_callback_selection
        else _canonical_match_order(request, batch.matches)
    )
    return ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=ordered,
        candidate_population_count=batch.candidate_population_count,
        selection_truncated=batch.selection_truncated,
    )


def _validate_parent_candidate_in_index(
    index: FullStoreWindowIndex,
    candidate: FullStoreSlotCandidate,
    binding: LocalCitationBinding,
) -> None:
    _require(
        candidate.candidate_id == binding.candidate_id
        and candidate.citation_binding_receipt_sha256 == binding.receipt_sha256
        and binding.namespace_id == index.cache.namespace_id
        and binding.cache_receipt_sha256 == index.cache.cache_receipt_sha256
        and binding.source_database_sha256
        == index.cache.source_database_sha256
        and binding.source_store_receipt_sha256
        == index.cache.source_store_receipt_sha256,
        "first-pass candidate escaped its supplied index",
    )
    lookup = active_index_lookup(index)
    matches: list[IndexedContentWindow] = []
    span = binding.span
    for window_index in lookup.window_indices_by_chunk_id.get(span.chunk_id, ()):
        window = index.windows[window_index]
        row = window.row
        if not (
            row.namespace_id == binding.namespace_id
            and row.partition_id == binding.partition_id
            and row.source_id == binding.source_id
            and row.chunk_id == span.chunk_id
            and window.start_char <= span.start_char < span.end_char <= window.end_char
        ):
            continue
        if (
            _expected_excerpt_span(window, span.start_char, span.end_char)
            .identity_payload()
            == span.identity_payload()
        ):
            matches.append(window)
    _require(matches, "first-pass candidate span is absent from the supplied index")
    window = min(matches, key=lambda row: row.end_char - row.start_char)
    span = binding.span
    quote = window.row.text[span.start_char : span.end_char]
    expected_candidate_id = identity_sha256(
        {
            "atom_id": make_atom_id(span),
            "mechanism_id": FULL_STORE_MECHANISM_ID,
        }
    )
    _require(
        candidate.candidate_id == expected_candidate_id
        and candidate.quote == quote
        and candidate.quote_sha256 == quote_sha256(quote) == binding.quote_sha256
        and candidate.token_count == count_tokens(quote)
        and candidate.role == window.row.role
        and candidate.created_at == window.row.created_at
        and candidate.event_date == window.event_date
        and candidate.event_date_basis == window.event_date_basis,
        "first-pass candidate fields do not match its supplied index excerpt",
    )


def _validated_parent_seed_items(
    parent_result: FullStoreSlotClosureResult,
    parent_contribution: TypedEvidenceContribution | None,
) -> tuple[TypedEvidenceItem, ...]:
    if parent_contribution is None:
        return ()
    _require(
        type(parent_contribution) is TypedEvidenceContribution,
        "parent seed must be an exact typed contribution",
    )
    expected_artifact = identity_sha256(parent_result.local_audit_projection())
    _require(
        parent_contribution.mechanism_id == FULL_STORE_MECHANISM_ID
        and parent_contribution.sealed_artifact_sha256 == expected_artifact
        and len(parent_contribution.bindings) == len(parent_result.candidates),
        "parent seed contribution belongs to another first-pass result",
    )
    handle_to_candidate: dict[str, FullStoreSlotCandidate] = {}
    local_to_global_group: dict[str, str] = {}
    global_to_local_group: dict[str, str] = {}
    for candidate, local, binding in zip(
        parent_result.candidates,
        parent_result.local_bindings,
        parent_contribution.bindings,
        strict=True,
    ):
        _require(
            binding.origin is EvidenceOrigin.DIRECT_POINTER
            and binding.provenance_grade is ProvenanceGrade.DIRECT_POINTER
            and binding.sealed_artifact_sha256 == expected_artifact
            and binding.parent_receipt_sha256
            == parent_result.receipt.receipt_sha256
            and binding.evidence_receipt_sha256 == local.receipt_sha256
            and binding.payload_sha256
            == candidate_projection_receipt_sha256(candidate)
            and binding.citation_sha256 == candidate.quote_sha256
            and binding.citation_char_count == len(candidate.quote)
            and binding.local_source_locator_sha256 == local.receipt_sha256,
            "parent seed binding lost its exact candidate/local relation",
        )
        previous_global = local_to_global_group.setdefault(
            candidate.source_group_handle, binding.source_group_handle
        )
        previous_local = global_to_local_group.setdefault(
            binding.source_group_handle, candidate.source_group_handle
        )
        _require(
            previous_global == binding.source_group_handle
            and previous_local == candidate.source_group_handle,
            "parent seed source co-membership changed",
        )
        handle_to_candidate[binding.handle_id] = candidate
    slot_ids = {
        slot.slot_id for slot in parent_result.operator_spec.required_slots
    }
    items = parent_contribution.parsed.accepted_items
    _require(
        all(
            item.included
            and len(item.handle_ids) == 1
            and item.handle_ids[0] in handle_to_candidate
            and item.summary == handle_to_candidate[item.handle_ids[0]].quote
            and set(item.supported_slot_ids) <= slot_ids
            for item in items
        ),
        "parent seed item is not an exact same-tick candidate restatement",
    )
    return items


def run_typed_active_reconstruction(
    index: FullStoreWindowIndex,
    parent_result: FullStoreSlotClosureResult,
    /,
    *,
    candidate_scanner: ActiveReconstructionCandidateScanner,
    parent_contribution: TypedEvidenceContribution | None = None,
    budget: ActiveReconstructionBudget = ActiveReconstructionBudget(),
) -> TypedActiveReconstructionResult:
    """Run one or two bounded active-reconstruction hops without provider calls."""

    _require(type(index) is FullStoreWindowIndex, "reconstruction needs a prebuilt index")
    _require(
        type(parent_result) is FullStoreSlotClosureResult,
        "reconstruction needs an exact first-pass result",
    )
    _require(
        callable(candidate_scanner), "reconstruction needs an injected candidate scanner"
    )
    from .typed_active_full_store_scanner import scan_typed_active_full_store

    _require(
        candidate_scanner is scan_typed_active_full_store,
        "production active reconstruction requires the trusted local scanner",
    )
    _require(type(budget) is ActiveReconstructionBudget, "reconstruction budget changed")
    _require(
        parent_result.receipt.window_index_receipt_sha256 == index.receipt_sha256
        and parent_result.receipt.cache_receipt_sha256
        == index.cache.cache_receipt_sha256,
        "reconstruction parent does not belong to the supplied prebuilt index",
    )
    for candidate, binding in zip(
        parent_result.candidates, parent_result.local_bindings, strict=True
    ):
        _validate_parent_candidate_in_index(index, candidate, binding)
    seed_items = _validated_parent_seed_items(
        parent_result, parent_contribution
    )

    seen_candidate_spans: dict[str, str] = {}
    seen_span_receipts: set[str] = set()
    for candidate, local in zip(
        parent_result.candidates, parent_result.local_bindings, strict=True
    ):
        span_receipt = citation_span_receipt_sha256(local)
        seen_candidate_spans[candidate.candidate_id] = span_receipt
        seen_span_receipts.add(span_receipt)

    total_admitted = 0
    total_tokens = 0
    hops: list[ActiveReconstructionHopResult] = []
    current_items = seed_items
    current_candidates = parent_result.candidates
    current_bindings = parent_result.local_bindings
    lineage_parent = parent_result.receipt.receipt_sha256

    for hop_number in range(1, budget.max_hops + 1):
        candidate_pairs = tuple(
            zip(current_candidates, current_bindings, strict=True)
        )
        if budget.use_index_aware_cue_ranking:
            cues, cue_truncated = (
                derive_index_aware_active_reconstruction_cues(
                    index,
                    hop=hop_number,
                    items=current_items,
                    candidate_pairs=candidate_pairs,
                    operator_spec=parent_result.operator_spec,
                    temporal_target=parent_result.temporal_target,
                    budget=budget,
                )
            )
        else:
            cues, cue_truncated = _derive_cues(
                hop=hop_number,
                items=current_items,
                candidate_pairs=candidate_pairs,
                operator_spec=parent_result.operator_spec,
                budget=budget,
            )
        if not cues:
            break
        request = ActiveReconstructionScanRequest(
            index=index,
            operator_spec=parent_result.operator_spec,
            temporal_target=parent_result.temporal_target,
            hop=hop_number,
            lineage_parent_receipt_sha256=lineage_parent,
            cues=cues,
            max_selected_candidates=budget.max_selected_candidates_per_hop,
            max_selected_tokens=budget.max_selected_tokens_per_hop,
            use_fixed_scan_subchannels=budget.use_fixed_scan_subchannels,
            use_coverage_aware_callback_selection=(
                budget.use_coverage_aware_callback_selection
            ),
        )
        batch = validate_active_reconstruction_scan_batch(
            request, candidate_scanner(request)
        )
        cue_by_receipt = {row.receipt_sha256: row for row in cues}

        decisions: list[ActiveReconstructionDecision] = []
        admitted: list[ActiveReconstructionCandidateMatch] = []
        lineages: list[ActiveReconstructionLineage] = []
        admission_truncated = batch.selection_truncated
        for match in batch.matches:
            candidate_receipt = candidate_projection_receipt_sha256(match.candidate)
            span_receipt = citation_span_receipt_sha256(match.local_binding)
            previous_span = seen_candidate_spans.get(match.candidate.candidate_id)
            if previous_span is not None and previous_span != span_receipt:
                raise TypedActiveReconstructionError(
                    "one candidate ID resolved to different exact spans"
                )
            if previous_span is not None or span_receipt in seen_span_receipts:
                status: DecisionStatus = "duplicate_exact_candidate_or_span"
                admission_truncated = True
            elif (
                total_admitted + 1 > budget.max_admitted_candidates
                or total_tokens + match.candidate.token_count
                > budget.max_admitted_tokens
            ):
                status = "aggregate_budget_excluded"
                admission_truncated = True
            else:
                status = "admitted"
                total_admitted += 1
                total_tokens += match.candidate.token_count
                seen_candidate_spans[match.candidate.candidate_id] = span_receipt
                seen_span_receipts.add(span_receipt)
                admitted.append(match)
                cue_receipts = match.supporting_cue_receipt_sha256s
                parent_receipts = tuple(
                    dict.fromkeys(
                        cue_by_receipt[receipt].parent_receipt_sha256
                        for receipt in cue_receipts
                    )
                )
                affinity_receipts = tuple(
                    dict.fromkeys(
                        affinity.receipt_sha256
                        for receipt in cue_receipts
                        if (
                            affinity := cue_by_receipt[
                                receipt
                            ].selected_evidence_affinity
                        )
                        is not None
                    )
                )
                lineages.append(
                    ActiveReconstructionLineage(
                        hop=hop_number,
                        parent_receipt_sha256s=parent_receipts,
                        cue_receipt_sha256s=cue_receipts,
                        selected_affinity_receipt_sha256s=affinity_receipts,
                        scan_match_receipt_sha256=match.receipt_sha256,
                        child_candidate_projection_receipt_sha256=(
                            candidate_receipt
                        ),
                        child_local_binding_receipt_sha256=(
                            match.local_binding.receipt_sha256
                        ),
                        child_span_receipt_sha256=span_receipt,
                    )
                )
            decisions.append(
                ActiveReconstructionDecision(
                    match_receipt_sha256=match.receipt_sha256,
                    candidate_projection_receipt_sha256=candidate_receipt,
                    span_receipt_sha256=span_receipt,
                    status=status,
                )
            )

        hop_result = ActiveReconstructionHopResult(
            request=request,
            batch=batch,
            decisions=tuple(decisions),
            admitted_matches=tuple(admitted),
            lineages=tuple(lineages),
            cue_derivation_truncated=cue_truncated,
            admission_truncated=admission_truncated,
        )
        hops.append(hop_result)
        if not admitted:
            break
        current_items = ()
        current_candidates = tuple(row.candidate for row in admitted)
        current_bindings = tuple(row.local_binding for row in admitted)
        lineage_parent = hop_result.receipt_sha256

    return TypedActiveReconstructionResult(
        index=index,
        parent_result=parent_result,
        parent_contribution=parent_contribution,
        operator_spec=parent_result.operator_spec,
        temporal_target=parent_result.temporal_target,
        hops=tuple(hops),
        budget=budget,
    )


def adapt_typed_active_reconstruction_to_contribution(
    result: TypedActiveReconstructionResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> TypedEvidenceContribution:
    """Expose admitted exact child spans as one bounded direct-pointer lane."""

    _require(
        type(result) is TypedActiveReconstructionResult,
        "typed contribution requires an exact reconstruction result",
    )
    for value, label in (
        (handle_start, "global handle start"),
        (group_start, "global group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        not result.candidates
        or handle_start + len(result.candidates) - 1 <= 999_999,
        "global handle range exceeds the opaque contract",
    )

    source_keys = tuple(
        dict.fromkeys(
            (local.namespace_id, local.source_id) for local in result.local_bindings
        )
    )
    _require(
        not source_keys or group_start + len(source_keys) - 1 <= 999_999,
        "global group range exceeds the opaque contract",
    )
    groups = {
        source_key: f"G{group_start + index:03d}"
        for index, source_key in enumerate(source_keys)
    }
    sealed_artifact_sha256 = identity_sha256(result.local_audit_projection())
    bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    numeric_slot_ids = {
        slot.slot_id for slot in result.operator_spec.required_slots if slot.requires_numeric
    }
    for index, (candidate, local, lineage) in enumerate(
        zip(result.candidates, result.local_bindings, result.lineages, strict=True)
    ):
        handle_id = f"H{handle_start + index:03d}"
        group_handle = groups[(local.namespace_id, local.source_id)]
        numeric = None
        if numeric_slot_ids & set(candidate.supported_slot_ids):
            numeric = conservative_numeric_value(candidate.quote)
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.DIRECT_POINTER,
            provenance_grade=ProvenanceGrade.DIRECT_POINTER,
            source_group_handle=group_handle,
            sealed_artifact_sha256=sealed_artifact_sha256,
            parent_receipt_sha256=lineage.receipt_sha256,
            evidence_receipt_sha256=local.receipt_sha256,
            payload_sha256=candidate_projection_receipt_sha256(candidate),
            citation_sha256=candidate.quote_sha256,
            citation_char_count=len(candidate.quote),
            local_source_locator_sha256=local.receipt_sha256,
        )
        if numeric is not None:
            kind = "operand"
        elif result.operator_spec.temporal_mode.value != "none":
            kind = "event"
        elif result.operator_spec.answer_shape.value == "set_list":
            kind = "member"
        elif result.operator_spec.style.value == "state_chain":
            kind = "state"
        else:
            kind = "direct"
        semantic_slots = tuple(
            slot
            for slot in result.operator_spec.required_slots
            if slot.slot_id in candidate.supported_slot_ids
            and slot.kind in {SlotKind.OPERAND, SlotKind.COMPARISON_SIDE}
        )
        raw: dict[str, Any] = {
            "handle_ids": [handle_id],
            "included": True,
            "kind": kind,
            "numeric_role": "operand" if numeric is not None else "none",
            "specificity_terms": [],
            "summary": candidate.quote,
            "value_authority": "explicit",
        }
        if len(semantic_slots) == 1:
            raw["entity_key"] = semantic_slots[0].label
        if candidate.event_date is not None:
            raw["date"] = candidate.event_date
        if candidate.role in {"user", "assistant"}:
            raw["relation"] = f"authored_by_{candidate.role}"
        if numeric is not None:
            raw["numeric_value"] = numeric
        bindings.append(binding)
        raw_items.append(raw)

    frozen_bindings = tuple(bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=result.operator_spec,
        bindings=frozen_bindings,
    )
    return TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=frozen_bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed_artifact_sha256,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.truncated,
    )


__all__ = [
    "ActiveIndexLookup",
    "ActiveReconstructionBudget",
    "ActiveReconstructionCandidateMatch",
    "ActiveReconstructionCandidateScanner",
    "ActiveReconstructionCue",
    "ActiveReconstructionDecision",
    "ActiveReconstructionHopResult",
    "ActiveReconstructionLineage",
    "ActiveReconstructionScanBatch",
    "ActiveReconstructionScanRequest",
    "ActiveReconstructionSupportKind",
    "MECHANISM_ID",
    "SelectedEvidenceAffinity",
    "TypedActiveReconstructionError",
    "TypedActiveReconstructionResult",
    "adapt_typed_active_reconstruction_to_contribution",
    "active_candidate_id_for_index_span",
    "active_candidate_id_for_window",
    "active_cue_posting_fanout_cap",
    "active_history_obligation_supported",
    "active_index_lookup",
    "active_index_lookup_cache_audit",
    "active_selective_cue_terms",
    "active_supported_slot_ids",
    "active_temporal_support",
    "candidate_projection_receipt_sha256",
    "citation_span_receipt_sha256",
    "derive_index_aware_active_reconstruction_cues",
    "local_component_key_sha256",
    "local_history_key_sha256",
    "local_source_key_sha256",
    "run_typed_active_reconstruction",
    "selected_evidence_affinity",
    "validate_active_reconstruction_scan_batch",
]
