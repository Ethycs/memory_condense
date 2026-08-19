"""Deterministic, budgeted context assembly.

The point of this module is that context cost is *predictable*: every section
has a hard token ceiling, so a long conversation can never produce a surprise
token spike. Anything that does not fit is dropped and counted, never silently
truncated away without a record.

Section order follows the design:

    1. system / policies
    2. memory header   (typed bullets — active + pinned + top-ranked only)
    3. recent turns    (chronological)
    4. expansions      (verbatim chunk quotes, only when precision matters)
    5. the current user message
"""

from __future__ import annotations

import inspect
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

import pysbd

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.derived_scalar import (
    filter_conflicting_approximate_duration_recaps,
)
from memory_condense.lexical import tokenize
from memory_condense.schemas import (
    MemoryItem,
    MemoryResult,
    PackedContext,
    RetrievalResult,
)
from memory_condense.transcript_store import parse_source_metadata


@dataclass(frozen=True)
class ContextBudget:
    """Hard per-section token ceilings (design defaults)."""

    recent_window_tokens: int = 4500
    memory_header_tokens: int = 900
    expansion_tokens: int = 800
    # Retrieval asks for ten candidates by default.  The token ceiling, not an
    # unrelated count of three, should decide how many of those candidates
    # reach the prompt.  This raised assembled recall in the B0 investigation
    # without increasing the 800-token expansion budget.
    max_expansions: int = 10
    # Learned candidates are additive and may use otherwise-idle token budget;
    # they never consume one of the direct-retrieval slots.
    max_consolidation_expansions: int = 3
    max_expansion_tokens: int = 250
    # A hard coverage reservation is useful only when every admitted event
    # receives enough raw body content, after provenance-label overhead, to
    # convey a value.  When every requested event cannot meet this floor, the
    # packer deterministically reserves the largest feasible prefix and lets
    # the rest degrade to ordinary evidence.
    min_coverage_expansion_tokens: int = 24
    budget_aware_expansions: bool = False
    # Opt-in: apply diminishing returns to repeated excerpts from the same
    # durable source while performing budget-aware selection.
    source_diverse_expansions: bool = False
    # Opt-in lexical sentence extraction after chunk retrieval. This spends
    # prompt tokens on the sentences most directly tied to the live query
    # while retaining the durable chunk ID for provenance.
    query_aware_sentence_expansions: bool = False
    max_sentences_per_expansion: int = 2
    # Opt-in rate-distortion filter. Candidate-set IDF supplies information
    # weights; relevance and marginal concept/source novelty are divided by
    # rendered token cost without disturbing the retriever's evidence order.
    information_gain_expansions: bool = False
    min_information_gain_per_token: float = 0.0
    # Opt-in: treat standalone source/session timestamps as provenance rather
    # than independent evidence. The timestamp is bound to each selected
    # excerpt from that source, making temporal order recoverable without
    # spending a candidate slot on an anonymous date-only chunk.
    source_metadata_expansions: bool = False
    # Opt-in: use diffused source heat as weighted-fair prompt exposure. The
    # default preserves the established retrieval ordering exactly.
    heat_weighted_expansions: bool = False
    max_source_expansion_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.max_consolidation_expansions < 0:
            raise ValueError("max_consolidation_expansions must be non-negative")
        if self.min_coverage_expansion_tokens < 1:
            raise ValueError("min_coverage_expansion_tokens must be positive")
        if self.max_sentences_per_expansion < 1:
            raise ValueError("max_sentences_per_expansion must be positive")
        if self.min_information_gain_per_token < 0.0:
            raise ValueError("min_information_gain_per_token must be non-negative")
        if not 0.0 < self.max_source_expansion_fraction <= 1.0:
            raise ValueError("max_source_expansion_fraction must lie in (0, 1]")

    def total(self) -> int:
        return (
            self.recent_window_tokens
            + self.memory_header_tokens
            + self.expansion_tokens
        )


@dataclass(frozen=True, slots=True)
class _PostCoverageClosure:
    """Exact closed evidence IDs plus the scope of the closure proof."""

    chunk_ids: tuple[str, ...]
    scope: str
    global_recall_guaranteed: bool


MEMORY_HEADER_PREFIX = "Relevant memory:"
EXPANSION_PREFIX = "Supporting excerpts:"

# Post-coverage closure is intentionally narrower than ordinary coverage
# reservation.  These bases are emitted only for deterministic typed
# frontiers; a neural credible-set reservation is useful for ordering, but is
# not strong enough to justify destroying the fail-open tail.
_POST_COVERAGE_SCAN_CONTRACT_BASES = {
    "canonical_venue_episode_aligned_v1": "canonical_fixed_frontier",
    "direct_performance_source_occurrence_v1": "direct_performance_frontier",
}
_PROVENANCE_TIMESTAMP_RE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})[/-](?P<month>\d{1,2})"
    r"[/-](?P<day>\d{1,2})(?:\D+(?P<hour>\d{1,2})"
    r":(?P<minute>\d{2}))?"
)


def _report_value(report: Any, field: str) -> Any:
    """Read one diagnostic field without coupling to a report class."""

    if isinstance(report, Mapping):
        return report.get(field)
    return getattr(report, field, None)


def _exact_report_int(report: Any, field: str) -> int | None:
    """Return a report integer, rejecting bools and coercion-friendly values."""

    value = _report_value(report, field)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _provenance_timestamp_key(value: str | None) -> float | None:
    """Parse a full source date conservatively for closure-order validation."""

    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        match = _PROVENANCE_TIMESTAMP_RE.search(cleaned)
        if match is None:
            return None
        try:
            return datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour") or 0),
                int(match.group("minute") or 0),
            ).timestamp()
        except (OverflowError, ValueError):
            return None


def is_source_metadata_text(text: str) -> bool:
    """Whether ``text`` is a synthetic source timestamp, not evidence."""

    return parse_source_metadata(text) is not None


class ExpansionSelector(Protocol):
    """Transient query-conditioned ordering over a bounded evidence subset."""

    last_report: Any
    allow_selected_scope_fixed_k_closure: bool

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        max_results: int | None = None,
        source_timestamps: Mapping[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...


class ContextPacker:
    """Packs memory, recent turns, and expansions into a budgeted message list."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        expansion_selector: ExpansionSelector | None = None,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.expansion_selector = expansion_selector
        # Text-free, per-candidate diagnostics for the most recent expansion
        # packet.  This is intentionally ephemeral: it explains where a
        # bounded candidate was reordered or cut without entering PackedContext
        # or the durable memory store.
        self.last_expansion_trace: list[dict[str, Any]] = []
        self.last_closure_report: dict[str, Any] = {
            "applied": False,
            "closure_scope": "",
            "closure_global_recall_guaranteed": False,
        }
        self._sentence_segmenter = (
            pysbd.Segmenter(language="en", clean=False)
            if self.budget.query_aware_sentence_expansions
            else None
        )

    # -- public API ---------------------------------------------------------

    def pack(
        self,
        system_prompt: str = "",
        memories: list[MemoryResult] | list[MemoryItem] | None = None,
        recent_turns: list[tuple[str, str]] | None = None,
        expansions: list[RetrievalResult] | None = None,
        user_text: str | None = None,
        source_metadata: dict[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
    ) -> PackedContext:
        """Assemble a `PackedContext`. Every section is independently capped."""
        memories = memories or []
        recent_turns = recent_turns or []
        expansions = expansions or []

        header, header_tokens, header_dropped, memory_ids = (
            self._build_memory_header(memories)
        )
        kept_turns, turn_tokens, turns_dropped = self._fit_recent_turns(recent_turns)
        (
            exp_texts,
            expansion_chunk_ids,
            exp_tokens,
            exp_dropped,
            source_tokens,
        ) = self._build_expansions(
            expansions,
            query=user_text or "",
            source_metadata=source_metadata or {},
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=active_partition_scan,
        )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if header:
            messages.append({"role": "system", "content": header})
        for role, text in kept_turns:
            messages.append({"role": role, "content": text})
        if exp_texts:
            block = EXPANSION_PREFIX + "\n" + "\n".join(exp_texts)
            messages.append({"role": "system", "content": block})
        if user_text is not None:
            messages.append({"role": "user", "content": user_text})

        token_counts = {
            "system": count_tokens(system_prompt) if system_prompt else 0,
            "memory_header": header_tokens,
            "recent_turns": turn_tokens,
            "expansions": exp_tokens,
            "user": count_tokens(user_text) if user_text else 0,
        }
        dropped = {
            "memories": header_dropped,
            "recent_turns": turns_dropped,
            "expansions": exp_dropped,
        }

        return PackedContext(
            messages=messages,
            memory_header=header,
            memory_ids=memory_ids,
            expansions=exp_texts,
            expansion_chunk_ids=expansion_chunk_ids,
            recent_turns=kept_turns,
            token_counts=token_counts,
            expansion_source_token_counts=source_tokens,
            dropped=dropped,
        )

    # -- section builders ---------------------------------------------------

    def _build_memory_header(
        self, memories: list[MemoryResult] | list[MemoryItem]
    ) -> tuple[str, int, int, list[str]]:
        """Typed bullets, highest-ranked first, capped at the header budget."""
        items = [m.item if isinstance(m, MemoryResult) else m for m in memories]
        active = [i for i in items if i.status.value == "active"]

        if not active:
            return "", 0, 0, []

        lines: list[str] = []
        memory_ids: list[str] = []
        used = count_tokens(MEMORY_HEADER_PREFIX)
        dropped = 0

        for item in active:
            bullet = self._format_memory(item)
            cost = count_tokens(bullet) + 1  # +1 for the newline
            if used + cost > self.budget.memory_header_tokens:
                dropped += 1
                continue
            lines.append(bullet)
            memory_ids.append(item.mem_id)
            used += cost

        if not lines:
            return "", 0, dropped, []

        header = MEMORY_HEADER_PREFIX + "\n" + "\n".join(lines)
        return header, count_tokens(header), dropped, memory_ids

    @staticmethod
    def _format_memory(item: MemoryItem) -> str:
        pin_marker = "*" if item.is_pinned else ""
        line = f"- [{item.type.value}]{pin_marker} {item.content.strip()}"
        if item.details:
            line += f" ({item.details.strip()})"
        return line

    def _fit_recent_turns(
        self, recent_turns: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int, int]:
        """Keep the most recent turns that fit, returned oldest-first."""
        kept: list[tuple[str, str]] = []
        used = 0

        for role, text in reversed(recent_turns):
            cost = count_tokens(text)
            if used + cost > self.budget.recent_window_tokens:
                break
            kept.append((role, text))
            used += cost

        kept.reverse()
        return kept, used, len(recent_turns) - len(kept)

    def _build_expansions(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str = "",
        source_metadata: dict[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Verbatim excerpts, each capped, and capped again in aggregate.

        The final excerpt is shortened to the remaining aggregate budget.  The
        old implementation dropped it wholesale, often leaving a material
        fraction of the fixed budget unused even though more ranked evidence
        was available.
        """
        self.last_expansion_trace = []
        self.last_closure_report = {
            "applied": False,
            "closure_scope": "",
            "closure_global_recall_guaranteed": False,
        }
        original = list(expansions)
        original_rank: dict[str, int] = {}
        for rank, result in enumerate(original, start=1):
            original_rank.setdefault(result.chunk.chunk_id, rank)
        ranked = (
            self._heat_weighted_order(expansions)
            if self.budget.heat_weighted_expansions
            else expansions
        )
        source_timestamps: dict[str, str] = {}
        if self.budget.source_metadata_expansions:
            # Provenance is not evidence. Resolve synthetic timestamp rows
            # before estimating information gain so their unique source IDs
            # and date numbers cannot crowd real conversational content out
            # of the packet. The timestamp remains attached to every emitted
            # excerpt from that source below.
            source_timestamps, ranked = self._bind_source_metadata(
                ranked,
                candidate_pool=ranked,
                source_metadata=source_metadata,
            )
        metadata_rank = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(ranked, start=1)
        }
        selector_report: Any = None
        selector_report_is_current = False
        selector_output_rejected = False
        returned_ids: set[str] = set()
        if self.expansion_selector is not None:
            selector_report_before = getattr(
                self.expansion_selector,
                "last_report",
                None,
            )
            complete_frontier_for = getattr(
                self.expansion_selector,
                "requires_complete_frontier_for",
                None,
            )
            requires_complete_frontier = (
                bool(complete_frontier_for(query))
                if callable(complete_frontier_for)
                else bool(
                    getattr(
                        self.expansion_selector,
                        "requires_complete_frontier",
                        False,
                    )
                )
            )
            # The frozen-prefix arm is deliberately monotonic: preserve the
            # measured baseline ranker and only demote likely duplicate
            # support. A semantic selector may instead replace that ranking.
            if getattr(
                self.expansion_selector,
                "requires_baseline_ranking",
                False,
            ) and not requires_complete_frontier:
                if self.budget.information_gain_expansions:
                    ranked = self._information_gain_order(ranked, query=query)
                elif self.budget.budget_aware_expansions:
                    ranked = self._budget_aware_order(ranked, query=query)
            selector_input = list(ranked)
            select_kwargs: dict[str, Any] = {
                "source_timestamps": source_timestamps,
            }
            scan_fields = dict(active_partition_scan or {})
            scan_total = scan_fields.get("active_partition_total")
            scan_inspected = scan_fields.get("active_partition_inspected")
            if active_partition_total is None and scan_total is not None:
                active_partition_total = scan_total
            if active_partition_inspected is None and scan_inspected is not None:
                active_partition_inspected = scan_inspected
            if (
                active_partition_total is not None
                or active_partition_inspected is not None
                or scan_fields
            ):
                try:
                    selector_parameters = inspect.signature(
                        self.expansion_selector.select
                    ).parameters.values()
                except (TypeError, ValueError):
                    selector_parameters = ()
                accepts_partition = {
                    parameter.name for parameter in selector_parameters
                }
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in selector_parameters
                )
                if accepts_kwargs or "active_partition_total" in accepts_partition:
                    select_kwargs["active_partition_total"] = (
                        active_partition_total
                    )
                if (
                    accepts_kwargs
                    or "active_partition_inspected" in accepts_partition
                ):
                    select_kwargs["active_partition_inspected"] = (
                        active_partition_inspected
                    )
                if accepts_kwargs or "active_partition_scan" in accepts_partition:
                    select_kwargs["active_partition_scan"] = scan_fields
            returned = self.expansion_selector.select(
                query,
                ranked,
                **select_kwargs,
            )
            allowed_by_id: dict[str, RetrievalResult] = {}
            for result in selector_input:
                allowed_by_id.setdefault(result.chunk.chunk_id, result)
            ranked = []
            rejected_selector_rows: list[tuple[RetrievalResult, str, int]] = []
            for returned_rank, result in enumerate(returned, start=1):
                if not isinstance(result, RetrievalResult):
                    selector_output_rejected = True
                    if bool(getattr(self.expansion_selector, "strict", False)):
                        raise TypeError("selector returned a non-RetrievalResult row")
                    continue
                chunk_id = result.chunk.chunk_id
                expected = allowed_by_id.get(chunk_id)
                if expected is None:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_injected_rejected", returned_rank)
                    )
                    continue
                if expected is not result:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_replacement_rejected", returned_rank)
                    )
                    continue
                if chunk_id in returned_ids:
                    selector_output_rejected = True
                    rejected_selector_rows.append(
                        (result, "selector_duplicate_rejected", returned_rank)
                    )
                    continue
                returned_ids.add(chunk_id)
                ranked.append(result)
            if rejected_selector_rows and bool(
                getattr(self.expansion_selector, "strict", False)
            ):
                reasons = ", ".join(
                    reason for _result, reason, _rank in rejected_selector_rows
                )
                raise ValueError(f"unsafe selector output: {reasons}")
            # A selector may prioritize exact inputs but cannot silently erase
            # an omitted row. Preserve every unreturned original at the tail;
            # downstream budget accounting, not a malformed/partial selector,
            # remains the only destructive cutoff.
            ranked.extend(
                result
                for result in selector_input
                if result.chunk.chunk_id not in returned_ids
            )
            selector_report = getattr(
                self.expansion_selector,
                "last_report",
                None,
            )
            # Closure may delete evidence, so a stale report from an earlier
            # query is never sufficient. The production selector replaces its
            # frozen report on every call; mutable/reused reports fail open.
            selector_report_is_current = bool(
                selector_report is not None
                and selector_report is not selector_report_before
            )
        elif self.budget.information_gain_expansions:
            ranked = self._information_gain_order(ranked, query=query)
            selector_input = list(ranked)
            rejected_selector_rows = []
        elif self.budget.budget_aware_expansions:
            ranked = self._budget_aware_order(ranked, query=query)
            selector_input = list(ranked)
            rejected_selector_rows = []
        else:
            selector_input = list(ranked)
            rejected_selector_rows = []

        # A derived duration query needs two explicit temporal boundaries, not
        # exhaustive set coverage.  The normal IG path above restores those
        # operands; remove only a separately proven, approximate recap that
        # conflicts with their provenance dates.  The helper is fail-open and
        # returns the exact surviving RetrievalResult objects unchanged.
        ranked, temporal_conflicts = (
            filter_conflicting_approximate_duration_recaps(
                ranked,
                query=query,
                source_timestamps=source_timestamps,
            )
        )

        selector_input_rank: dict[str, int] = {}
        for result in selector_input:
            selector_input_rank.setdefault(
                result.chunk.chunk_id,
                len(selector_input_rank) + 1,
            )
        post_selector_rank: dict[str, int] = {}
        for result in ranked:
            post_selector_rank.setdefault(
                result.chunk.chunk_id,
                len(post_selector_rank) + 1,
            )
        selector_details: dict[str, Mapping[str, Any]] = {}
        selector_trace_rows: list[Mapping[str, Any]] = []
        if self.expansion_selector is not None:
            for row in getattr(
                self.expansion_selector,
                "last_candidate_trace",
                (),
            ):
                if not isinstance(row, Mapping):
                    continue
                selector_trace_rows.append(row)
                chunk_id = row.get("chunk_id")
                if isinstance(chunk_id, str):
                    selector_details[chunk_id] = row
        rejected_input_reason = {
            result.chunk.chunk_id: reason
            for result, reason, _rank in rejected_selector_rows
            if result.chunk.chunk_id in original_rank
        }

        detail_fields = (
            "cross_encoder_input_rank",
            "cross_encoder_score",
            "cross_encoder_rank",
            "group_id",
            "group_role",
            "representative_chunk_id",
            "merge_similarity",
            "merge_threshold",
            "qk_score",
            "ov_transport",
            "prefix_utility",
            "semantic_score",
            "answer_object_key_present",
            "semantic_score_kind",
            "answerability_score",
            "answerability_score_kind",
            "membership_score",
            "preferred_evidence_role",
            "role_match",
            "value_evidence",
            "assignment_hypothesis",
            "p_existing",
            "p_new",
            "p_null",
            "existing_energy",
            "new_energy",
            "null_energy",
            "temporal_in_scope",
            "posterior_entropy",
            "posterior_kind",
            "semantic_surprisal",
            "posterior_uncertain",
            "credible_cluster",
            "coverage_reserved",
            "reservation_basis",
        )
        trace_by_id: dict[str, dict[str, Any]] = {}
        for result in original:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            if chunk_id not in metadata_rank:
                reason = "source_metadata_filtered"
            elif chunk_id not in selector_input_rank:
                if self.budget.information_gain_expansions:
                    reason = "preselector_information_gain_filtered"
                elif self.budget.budget_aware_expansions:
                    reason = "preselector_budget_filtered"
                else:
                    reason = "preselector_filtered"
            elif chunk_id not in post_selector_rank:
                reason = (
                    "temporal_conflict_suppressed"
                    if chunk_id in temporal_conflicts
                    else rejected_input_reason.get(
                        chunk_id,
                        "selector_filtered",
                    )
                )
            else:
                reason = "pending"
            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "source_id": self._result_source_id(result),
                "route": result.route or "",
                "anchor_chunk_id": result.anchor_chunk_id,
                "original_rank": original_rank[chunk_id],
                "selector_input_rank": selector_input_rank.get(chunk_id),
                "post_selector_rank": post_selector_rank.get(chunk_id),
                "packed_rank": None,
                "cutoff_reason": reason,
                "chunk_tokens": int(result.chunk.token_count),
                "content_tokens": None,
                "rendered_tokens": None,
                "cumulative_tokens": None,
                "selector_output_rejection": rejected_input_reason.get(chunk_id),
            }
            details = selector_details.get(chunk_id, {})
            for field in detail_fields:
                value = details.get(field)
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[field] = value
            conflict = temporal_conflicts.get(chunk_id)
            if conflict is not None:
                row.update(
                    {
                        "temporal_conflict_action": "suppressed",
                        "temporal_conflict_basis": conflict.reason,
                        "temporal_onset_chunk_id": conflict.onset_chunk_id,
                        "temporal_endpoint_chunk_id": conflict.endpoint_chunk_id,
                    }
                )
            trace_by_id[chunk_id] = row

        # A selector is an ordering policy, not an evidence source. Retain a
        # text-free diagnostic for rejected fabrications without ever letting
        # their payload reach the prompt.
        for result, rejection_reason, returned_rank in rejected_selector_rows:
            chunk_id = result.chunk.chunk_id
            if chunk_id in trace_by_id:
                continue
            trace_by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "source_id": self._result_source_id(result),
                "route": result.route or "",
                "anchor_chunk_id": result.anchor_chunk_id,
                "original_rank": None,
                "selector_input_rank": None,
                "post_selector_rank": returned_rank,
                "packed_rank": None,
                "cutoff_reason": rejection_reason,
                "chunk_tokens": int(result.chunk.token_count),
                "content_tokens": None,
                "rendered_tokens": None,
                "cumulative_tokens": None,
            }
        reserved_ids = {
            result.chunk.chunk_id
            for result in ranked
            if bool(
                selector_details.get(result.chunk.chunk_id, {}).get(
                    "coverage_reserved",
                    False,
                )
            )
            and selector_details.get(result.chunk.chunk_id, {}).get("group_role")
            == "representative"
        }
        # Enforce the selector's coverage contract even if a future composite
        # interleaves support rows. The objects themselves remain untouched,
        # preserving exact chunk/source provenance.
        packing_ranked = [
            result for result in ranked if result.chunk.chunk_id in reserved_ids
        ]
        packing_ranked.extend(
            result for result in ranked if result.chunk.chunk_id not in reserved_ids
        )
        for rank, result in enumerate(packing_ranked, start=1):
            trace_by_id[result.chunk.chunk_id]["coverage_pack_rank"] = rank

        def expansion_label(result: RetrievalResult, ordinal: int) -> str:
            source_id = self._result_source_id(result)
            timestamp = source_timestamps.get(source_id)
            role = (
                result.turn.role.strip().lower()
                if result.turn is not None and result.turn.role.strip()
                else ""
            )
            provenance = ""
            if timestamp:
                provenance += f" @ {timestamp}"
            if role:
                provenance += f" | {role}"
            return f"[{ordinal}{provenance}] "

        texts: list[str] = []
        chunk_ids: list[str] = []
        used = count_tokens(EXPANSION_PREFIX)
        source_tokens: dict[str, int] = defaultdict(int)
        direct_kept = 0
        consolidation_kept = 0
        token_cutoff = False
        requested_reserved_rows = [
            result
            for result in packing_ranked
            if result.chunk.chunk_id in reserved_ids
        ]
        # Query-aware sentence packing is a lossy optimization.  A coverage
        # representative has already been selected as the one row that must
        # carry its event, so applying that optimization here can turn a
        # 60-token evidence body into a 16-token sentence and then incorrectly
        # declare the 24-token reservation fulfilled.  Reservations therefore
        # allocate from the raw chunk body; ordinary evidence remains free to
        # use query-aware sentence packing below.
        reservation_bodies = {
            result.chunk.chunk_id: result.chunk.text.strip()
            for result in requested_reserved_rows
        }
        minimum_content = min(
            self.budget.min_coverage_expansion_tokens,
            self.budget.max_expansion_tokens,
        )
        active_reserved_rows: list[RetrievalResult] = []
        reservation_minimums: dict[str, int] = {}
        reservation_snippets: dict[str, str] = {}
        projected = used
        # Admit only a deterministic prefix that can carry labels plus a
        # useful excerpt from every event.  This prevents a large ALL set from
        # dividing the fair share to zero and aborting before any evidence is
        # emitted.
        for result in requested_reserved_rows:
            body = reservation_bodies[result.chunk.chunk_id]
            body_tokens = count_tokens(body)
            required_content = min(minimum_content, body_tokens)
            ordinal = len(active_reserved_rows) + 1
            minimum_snippet = truncate_to_tokens(body, required_content)
            minimum_snippet_tokens = count_tokens(minimum_snippet)
            required_cost = count_tokens(
                expansion_label(result, ordinal) + minimum_snippet
            ) + 1
            if minimum_snippet_tokens < required_content:
                # ``truncate_to_tokens`` is expected to round-trip token
                # prefixes, but keep the reservation invariant explicit if a
                # tokenizer implementation ever changes.
                break
            if required_content < 1 or (
                projected + required_cost > self.budget.expansion_tokens
            ):
                break
            active_reserved_rows.append(result)
            reservation_minimums[result.chunk.chunk_id] = required_content
            projected += required_cost

        active_reserved_ids = {
            result.chunk.chunk_id for result in active_reserved_rows
        }
        reservation_content_cap: int | None = None
        if active_reserved_rows:
            lower = max(reservation_minimums.values())
            upper = self.budget.max_expansion_tokens
            # Equal-cap water filling gives every active event its raw-body
            # minimum, while short rows return unused capacity to longer rows.
            # Cost the exact rendered label+body pair: estimating labels and
            # bodies independently lets BPE boundary differences accumulate
            # and can starve the final representative.
            while lower <= upper:
                midpoint = (lower + upper) // 2
                candidate_snippets = {
                    result.chunk.chunk_id: truncate_to_tokens(
                        reservation_bodies[result.chunk.chunk_id],
                        midpoint,
                    )
                    for result in active_reserved_rows
                }
                rendered_cost = used + sum(
                    count_tokens(
                        expansion_label(result, index)
                        + candidate_snippets[result.chunk.chunk_id]
                    )
                    + 1
                    for index, result in enumerate(
                        active_reserved_rows,
                        start=1,
                    )
                )
                if rendered_cost <= self.budget.expansion_tokens:
                    reservation_content_cap = midpoint
                    reservation_snippets = candidate_snippets
                    lower = midpoint + 1
                else:
                    upper = midpoint - 1
            if reservation_content_cap is None:
                # The prefix admission calculation proves the minimum fits;
                # this guard keeps that invariant explicit if token accounting
                # changes later.
                reservation_content_cap = max(reservation_minimums.values())
                reservation_snippets = {
                    result.chunk.chunk_id: truncate_to_tokens(
                        reservation_bodies[result.chunk.chunk_id],
                        reservation_content_cap,
                    )
                    for result in active_reserved_rows
                }

        for result in requested_reserved_rows:
            chunk_id = result.chunk.chunk_id
            active = chunk_id in active_reserved_ids
            trace_by_id[chunk_id].update(
                {
                    "coverage_reservation_requested": True,
                    "coverage_reservation_active": active,
                    "coverage_reservation_degraded": not active,
                    "coverage_reservation_feasible": active,
                    "coverage_content_cap": (
                        count_tokens(
                            reservation_snippets.get(chunk_id, "")
                        )
                        if active
                        else None
                    ),
                }
            )

        closure = self._post_coverage_closure_ids(
            selector_report=selector_report,
            selector_report_is_current=selector_report_is_current,
            selector_output_rejected=selector_output_rejected,
            selector_input=selector_input,
            returned_ids=returned_ids,
            selector_trace_rows=selector_trace_rows,
            requested_reserved_rows=requested_reserved_rows,
            active_reserved_ids=active_reserved_ids,
            reservation_bodies=reservation_bodies,
            reservation_snippets=reservation_snippets,
            source_timestamps=source_timestamps,
        )
        closure_id_set = set(closure.chunk_ids if closure is not None else ())
        closure_applied = closure is not None
        closure_scope = closure.scope if closure is not None else ""
        closure_global_recall_guaranteed = bool(
            closure is not None and closure.global_recall_guaranteed
        )
        scope_provenance = {
            "partition_scope_kind": (
                _report_value(selector_report, "partition_scope_kind")
                if selector_report_is_current
                else None
            ),
            "partition_inventory_total": (
                _report_value(selector_report, "partition_inventory_total")
                if selector_report_is_current
                else None
            ),
            "selected_partition_count": (
                _report_value(selector_report, "selected_partition_count")
                if selector_report_is_current
                else None
            ),
            "partition_scope_exhaustive": (
                _report_value(selector_report, "partition_scope_exhaustive")
                if selector_report_is_current
                else None
            ),
            "selected_scope_structurally_complete": (
                _report_value(
                    selector_report,
                    "selected_scope_structurally_complete",
                )
                if selector_report_is_current
                else None
            ),
            "global_semantic_complete": (
                _report_value(selector_report, "global_semantic_complete")
                if selector_report_is_current
                else None
            ),
        }
        self.last_closure_report = {
            "applied": closure_applied,
            "closure_scope": closure_scope,
            "closure_global_recall_guaranteed": (
                closure_global_recall_guaranteed
            ),
            **scope_provenance,
        }
        for diagnostic in trace_by_id.values():
            diagnostic["post_coverage_closure_applied"] = closure_applied
            diagnostic["post_coverage_closed"] = False
            diagnostic["closure_scope"] = closure_scope
            diagnostic["closure_global_recall_guaranteed"] = (
                closure_global_recall_guaranteed
            )
            diagnostic.update(scope_provenance)
        if closure_applied:
            # The proof above includes full raw-body preflight for every
            # member, so no ordinary alternative is needed to complete this
            # exact FIXED answer set. Keep the exact trusted objects and mark
            # every suppressed row explicitly in the text-free trace.
            for chunk_id, diagnostic in trace_by_id.items():
                if chunk_id not in closure_id_set:
                    diagnostic["cutoff_reason"] = "post_coverage_closed"
                    diagnostic["post_coverage_closed"] = True
            packing_ranked = [
                result
                for result in active_reserved_rows
                if result.chunk.chunk_id in closure_id_set
            ]

        for result in packing_ranked:
            diagnostic = trace_by_id[result.chunk.chunk_id]
            is_reserved = result.chunk.chunk_id in active_reserved_ids
            is_consolidation = result.route == "live_consolidation"
            if is_consolidation:
                if (
                    not is_reserved
                    and consolidation_kept
                    >= self.budget.max_consolidation_expansions
                ):
                    diagnostic["cutoff_reason"] = "consolidation_count_cap"
                    continue
            elif not is_reserved and direct_kept >= self.budget.max_expansions:
                diagnostic["cutoff_reason"] = "direct_count_cap"
                continue
            remaining = self.budget.expansion_tokens - used
            source_id = self._result_source_id(result)
            label = expansion_label(result, len(texts) + 1)
            if is_reserved:
                # This exact raw-body snippet was preflighted together with
                # every other active representative.  Never shrink it based
                # on what earlier rows happened to consume.
                snippet = reservation_snippets[result.chunk.chunk_id]
                content_budget = count_tokens(snippet)
            else:
                # Reserve the label and newline accounted for by this packer.
                content_budget = min(
                    self.budget.max_expansion_tokens,
                    remaining - count_tokens(label) - 1,
                )
                if content_budget <= 0:
                    diagnostic["cutoff_reason"] = "token_budget_exhausted"
                    token_cutoff = True
                    break
                prepared = self._prepare_expansion_text(result.chunk.text, query)
                snippet = truncate_to_tokens(prepared, content_budget)
            if not snippet:
                diagnostic["cutoff_reason"] = "empty_after_prepare"
                continue
            entry = label + snippet
            cost = count_tokens(entry) + 1
            # Token boundaries can shift where the label meets the excerpt.
            # Tighten by the exact overage so the hard ceiling remains exact.
            if not is_reserved and used + cost > self.budget.expansion_tokens:
                snippet = truncate_to_tokens(
                    snippet, max(0, content_budget - (used + cost - self.budget.expansion_tokens))
                )
                entry = label + snippet
                cost = count_tokens(entry) + 1
            if not snippet or used + cost > self.budget.expansion_tokens:
                diagnostic["cutoff_reason"] = "token_budget_no_fit"
                token_cutoff = True
                break
            texts.append(entry)
            chunk_ids.append(result.chunk.chunk_id)
            if is_consolidation:
                consolidation_kept += 1
            else:
                direct_kept += 1
            used += cost
            source_tokens[source_id] += count_tokens(snippet)
            diagnostic.update(
                {
                    "packed_rank": len(texts),
                    "cutoff_reason": "packed",
                    "content_tokens": count_tokens(snippet),
                    "rendered_tokens": cost,
                    "cumulative_tokens": used,
                }
            )

        for diagnostic in trace_by_id.values():
            if diagnostic["cutoff_reason"] == "pending":
                diagnostic["cutoff_reason"] = (
                    "after_token_cutoff" if token_cutoff else "not_packed"
                )
        self.last_expansion_trace = list(trace_by_id.values())

        if not texts:
            return [], [], 0, len(expansions), {}

        return (
            texts,
            chunk_ids,
            used,
            len(expansions) - len(texts),
            dict(source_tokens),
        )

    def _post_coverage_closure_ids(
        self,
        *,
        selector_report: Any,
        selector_report_is_current: bool,
        selector_output_rejected: bool,
        selector_input: Sequence[RetrievalResult],
        returned_ids: set[str],
        selector_trace_rows: Sequence[Mapping[str, Any]],
        requested_reserved_rows: Sequence[RetrievalResult],
        active_reserved_ids: set[str],
        reservation_bodies: Mapping[str, str],
        reservation_snippets: Mapping[str, str],
        source_timestamps: Mapping[str, str],
    ) -> _PostCoverageClosure | None:
        """Prove when a typed FIXED frontier can safely close the prompt tail.

        Coverage ranking normally fails open: unselected and uncertain rows
        remain available after the reserved representatives.  Closure is the
        narrow exception for a fully inspected, structurally typed, ordered
        FIXED-K result whose exact raw bodies have all been preflighted.  Any
        absent, stale, malformed, contradictory, truncated, or rejected
        diagnostic returns ``None`` and preserves the ordinary fail-open path.
        """

        if (
            self.expansion_selector is None
            or not selector_report_is_current
            or selector_report is None
            or selector_output_rejected
        ):
            return None
        if (
            _report_value(selector_report, "selection_status") != "applied"
            or _report_value(selector_report, "fallback_reason") not in (None, "")
            or _report_value(selector_report, "bypass_reason") not in (None, "")
            or _report_value(selector_report, "score_provider_fallback")
            not in (None, "")
            or _report_value(selector_report, "operator") != "fixed_cardinality"
            or _report_value(selector_report, "quantifier") != "fixed_cardinality"
            or _report_value(selector_report, "requires_completeness") is not True
        ):
            return None

        cardinality = _exact_report_int(selector_report, "cardinality")
        if cardinality is None or cardinality < 1:
            return None
        active_partition_exhaustive = _report_value(
            selector_report,
            "active_partition_exhaustive",
        )
        selected_scope_complete = _report_value(
            selector_report,
            "selected_scope_structurally_complete",
        )
        legacy_selected_scope_complete = _report_value(
            selector_report,
            "active_partition_semantically_complete",
        )
        if (
            selected_scope_complete is not True
            or legacy_selected_scope_complete is not True
        ):
            return None
        scope_kind = _report_value(selector_report, "partition_scope_kind")
        if scope_kind not in {
            "approximate_top_k",
            "global",
            "authoritative",
        }:
            return None
        partition_scope_exhaustive = _report_value(
            selector_report,
            "partition_scope_exhaustive",
        )
        if (
            partition_scope_exhaustive is not None
            and not isinstance(partition_scope_exhaustive, bool)
        ):
            return None
        inventory_total_value = _report_value(
            selector_report,
            "partition_inventory_total",
        )
        selected_partition_value = _report_value(
            selector_report,
            "selected_partition_count",
        )
        inventory_total = (
            None
            if inventory_total_value is None
            else _exact_report_int(selector_report, "partition_inventory_total")
        )
        selected_partition_count = (
            None
            if selected_partition_value is None
            else _exact_report_int(selector_report, "selected_partition_count")
        )
        if (
            (inventory_total_value is not None and inventory_total is None)
            or (
                selected_partition_value is not None
                and selected_partition_count is None
            )
            or (inventory_total is not None and inventory_total < 0)
            or (
                selected_partition_count is not None
                and selected_partition_count < 0
            )
            or (
                inventory_total is not None
                and selected_partition_count is not None
                and selected_partition_count > inventory_total
            )
            or inventory_total is None
            or selected_partition_count is None
            or inventory_total < 1
            or selected_partition_count < 1
            or partition_scope_exhaustive is None
        ):
            return None
        if inventory_total is not None and selected_partition_count is not None:
            if partition_scope_exhaustive is not (
                inventory_total == selected_partition_count
            ):
                return None
        global_semantic_complete = _report_value(
            selector_report,
            "global_semantic_complete",
        )
        if (
            global_semantic_complete is not None
            and not isinstance(global_semantic_complete, bool)
        ):
            return None
        global_proof = bool(
            global_semantic_complete is True
            and scope_kind in {"global", "authoritative"}
            and (
                scope_kind == "authoritative"
                or partition_scope_exhaustive is True
            )
        )
        if (
            (scope_kind == "global" and partition_scope_exhaustive is not True)
            or (global_semantic_complete is True and not global_proof)
        ):
            return None
        selected_scope_policy = bool(
            not global_proof
            and scope_kind == "approximate_top_k"
            and partition_scope_exhaustive is False
            and global_semantic_complete is False
            and _report_value(
                selector_report,
                "allow_selected_scope_fixed_k_closure",
            )
            is True
            and getattr(
                self.expansion_selector,
                "allow_selected_scope_fixed_k_closure",
                False,
            )
            is True
        )
        if not global_proof and not selected_scope_policy:
            return None
        scan_contract = _report_value(
            selector_report,
            "active_partition_scan_contract",
        )
        required_reservation_basis = _POST_COVERAGE_SCAN_CONTRACT_BASES.get(
            scan_contract
        )
        structural_hypotheses = _exact_report_int(
            selector_report,
            "active_partition_structural_hypotheses",
        )
        structural_rows = _exact_report_int(
            selector_report,
            "active_partition_structural_rows",
        )
        active_sources = _exact_report_int(
            selector_report,
            "active_partition_sources_total",
        )
        if (
            required_reservation_basis is None
            or structural_hypotheses != cardinality
            or structural_rows is None
            or structural_rows < cardinality
            or active_sources is None
            or active_sources < 1
        ):
            return None
        if (
            _report_value(selector_report, "routed_frontier_exhaustive") is not True
            or _exact_report_int(selector_report, "frontier_uninspected") != 0
            # Closing the tail is destructive.  Scoring every row that happened
            # to reach the bounded route union is not proof that the active
            # durable partition was searched.  Require the typed structural
            # scan to state physical and semantic completeness explicitly.
            or active_partition_exhaustive is not True
            or _exact_report_int(selector_report, "cardinality_deficit") != 0
            or _exact_report_int(
                selector_report,
                "structural_eligible_clusters",
            )
            != cardinality
            or _exact_report_int(
                selector_report,
                "structural_reserved_representatives",
            )
            != cardinality
            or _exact_report_int(selector_report, "reserved_representatives")
            != cardinality
        ):
            return None

        for field in (
            "active_partition_candidates_truncated",
            "active_partition_structural_overflow",
        ):
            if _exact_report_int(selector_report, field) != 0:
                return None

        input_ids = [result.chunk.chunk_id for result in selector_input]
        if len(input_ids) != len(set(input_ids)):
            return None
        input_count = len(input_ids)
        for field in (
            "input_candidates",
            "inspected_candidates",
            "classified_candidates",
            "frontier_candidates",
            "frontier_attempted",
        ):
            if _exact_report_int(selector_report, field) != input_count:
                return None
        if _exact_report_int(selector_report, "output_candidates") != len(
            returned_ids
        ):
            return None

        active_total = _report_value(selector_report, "active_partition_total")
        active_inspected = _report_value(
            selector_report,
            "active_partition_inspected",
        )
        if active_total is not None or active_inspected is not None:
            if (
                isinstance(active_total, bool)
                or not isinstance(active_total, int)
                or isinstance(active_inspected, bool)
                or not isinstance(active_inspected, int)
                or active_total < 0
                or active_inspected != active_total
            ):
                return None

        trace_ids = [
            row.get("chunk_id")
            for row in selector_trace_rows
            if isinstance(row.get("chunk_id"), str)
        ]
        if (
            len(trace_ids) != len(selector_trace_rows)
            or len(trace_ids) != len(set(trace_ids))
            or set(trace_ids) != set(input_ids)
        ):
            return None
        structural_rows = [
            row
            for row in selector_trace_rows
            if row.get("coverage_reserved") is True
            and row.get("group_role") == "representative"
        ]
        if len(structural_rows) != cardinality:
            return None
        structural_ids = tuple(str(row["chunk_id"]) for row in structural_rows)
        if (
            len(set(structural_ids)) != cardinality
            or not set(structural_ids).issubset(returned_ids)
            or any(
                not isinstance(row.get("reservation_basis"), str)
                or row.get("reservation_basis")
                != required_reservation_basis
                or row.get("group_id") is None
                or row.get("role_match") is False
                or (
                    row.get("temporal_in_scope") is not None
                    and row.get("temporal_in_scope") is not True
                )
                for row in structural_rows
            )
            or len({str(row["group_id"]) for row in structural_rows})
            != cardinality
        ):
            return None

        requested_ids = tuple(
            result.chunk.chunk_id for result in requested_reserved_rows
        )
        if set(requested_ids) != set(structural_ids):
            return None
        if set(requested_ids) != active_reserved_ids:
            return None
        if any(
            not reservation_bodies.get(chunk_id)
            or reservation_snippets.get(chunk_id)
            != reservation_bodies.get(chunk_id)
            for chunk_id in requested_ids
        ):
            return None

        ordering = _report_value(selector_report, "ordering")
        if ordering not in ("ascending", "descending"):
            return None
        timestamps = [
            _provenance_timestamp_key(
                source_timestamps.get(self._result_source_id(result))
            )
            for result in requested_reserved_rows
        ]
        if any(timestamp is None for timestamp in timestamps):
            return None
        ordered_values = [float(timestamp) for timestamp in timestamps]
        if ordering == "ascending":
            ordered = all(
                left < right
                for left, right in zip(
                    ordered_values,
                    ordered_values[1:],
                    strict=False,
                )
            )
        else:
            ordered = all(
                left > right
                for left, right in zip(
                    ordered_values,
                    ordered_values[1:],
                    strict=False,
                )
            )
        if not ordered:
            return None
        return _PostCoverageClosure(
            chunk_ids=requested_ids,
            scope=("global_semantic" if global_proof else "selected_scope_policy"),
            global_recall_guaranteed=global_proof,
        )

    def _bind_source_metadata(
        self,
        selected: list[RetrievalResult],
        *,
        candidate_pool: list[RetrievalResult] | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], list[RetrievalResult]]:
        """Bind source timestamps and force routed sources to carry content.

        LongMemEval represents a session date as a synthetic system turn. If
        that chunk and a content chunk from the same source are both in the
        candidate set, emitting them independently destroys the relation the
        responder needs. A selected timestamp is therefore replaced by the
        highest-ranked real content candidate from the same routed source,
        even when the information-rate filter did not independently retain
        that content. A timestamp with no available companion is kept so
        direct date questions do not regress.
        """

        pool = candidate_pool or selected
        timestamps: dict[str, str] = {}
        persisted_metadata_sources: set[str] = set()
        for source_id, text in (source_metadata or {}).items():
            parsed = parse_source_metadata(text)
            if parsed is not None:
                timestamps[source_id] = parsed[1]
                persisted_metadata_sources.add(source_id)
        companions: dict[str, RetrievalResult] = {}
        for result in pool:
            source_id = self._result_source_id(result)
            parsed = parse_source_metadata(result.chunk.text)
            if parsed is None:
                companions.setdefault(source_id, result)
                continue
            timestamps.setdefault(source_id, parsed[1])

        evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()
        resolved_metadata_sources: set[str] = set()
        for result in selected:
            source_id = self._result_source_id(result)
            is_metadata = is_source_metadata_text(result.chunk.text)
            candidate = result
            if is_metadata and source_id not in resolved_metadata_sources:
                companion = companions.get(source_id)
                resolved_metadata_sources.add(source_id)
                if companion is not None:
                    candidate = companion
                elif source_id in persisted_metadata_sources:
                    # The durable store already supplied this timestamp. With
                    # no content from the same source in the bounded candidate
                    # pool, rendering the anonymous metadata row adds no fact
                    # the responder can associate with the user's question.
                    continue
            elif is_metadata:
                continue
            if candidate.chunk.chunk_id in seen_chunks:
                continue
            seen_chunks.add(candidate.chunk.chunk_id)
            evidence.append(candidate)
        return timestamps, evidence


    def _budget_aware_order(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str = "",
    ) -> list[RetrievalResult]:
        """Choose high-utility evidence under the hard token ceiling.

        Retrieval score divided by square-root token cost is a conservative
        length correction: it stops a few long, marginal candidates from
        hiding short precise evidence without collapsing into a pure
        score-per-token policy that over-favors tiny fragments. Selected rows
        return in original rank order for deterministic prompt rendering.
        """

        prefix_cost = count_tokens(EXPANSION_PREFIX)
        available = max(0, self.budget.expansion_tokens - prefix_cost)
        ranked: list[tuple[float, int, int, bool, RetrievalResult]] = []
        for index, result in enumerate(expansions):
            prepared = self._prepare_expansion_text(result.chunk.text, query)
            snippet = truncate_to_tokens(
                prepared, self.budget.max_expansion_tokens
            )
            if not snippet:
                continue
            # Two tokens safely approximate the rendered label and newline;
            # the exact pack below remains the authoritative hard cap.
            cost = count_tokens(snippet) + 2
            utility = max(0.0, float(result.score)) / math.sqrt(max(1, cost))
            ranked.append(
                (
                    utility,
                    index,
                    cost,
                    result.route == "live_consolidation",
                    result,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected: list[tuple[int, RetrievalResult]] = []
        used = 0
        direct = 0
        consolidation = 0
        source_counts: dict[str, int] = defaultdict(int)
        remaining = list(ranked)
        while remaining:
            if self.budget.source_diverse_expansions:
                remaining.sort(
                    key=lambda item: (
                        -item[0]
                        / (
                            1
                            + source_counts[
                                self._result_source_id(item[4])
                            ]
                        ),
                        item[1],
                    )
                )
            _utility, index, cost, is_consolidation, result = remaining.pop(0)
            if is_consolidation:
                if consolidation >= self.budget.max_consolidation_expansions:
                    continue
            elif direct >= self.budget.max_expansions:
                continue
            if used + cost > available:
                continue
            selected.append((index, result))
            used += cost
            if is_consolidation:
                consolidation += 1
            else:
                direct += 1
            source_counts[self._result_source_id(result)] += 1
        selected.sort(key=lambda item: item[0])
        return [result for _index, result in selected]

    def _prepare_expansion_text(self, text: str, query: str) -> str:
        """Return a deterministic query-focused excerpt when enabled.

        The retriever still chooses and scores durable chunks. This method is
        deliberately only a packing transform: it keeps the best lexical
        sentence matches in their original order and stores no model state.
        If neither the query nor any sentence has a usable lexical match, the
        original text is retained so dense-only semantic hits are not erased.
        """

        stripped = text.strip()
        if not self.budget.query_aware_sentence_expansions or not query.strip():
            return stripped
        if self._sentence_segmenter is None:
            return stripped

        sentences = [
            segment.strip()
            for segment in self._sentence_segmenter.segment(stripped)
            if segment.strip()
        ]
        if len(sentences) <= self.budget.max_sentences_per_expansion:
            return stripped

        query_terms = set(tokenize(query))
        if not query_terms:
            return stripped

        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = set(tokenize(sentence))
            overlap = query_terms.intersection(sentence_terms)
            if not overlap:
                continue
            # Exact numbers and long identifiers are unusually discriminative
            # in long-chat recall. Length normalization prevents a long
            # sentence from winning merely by containing more words.
            overlap_weight = sum(
                3.0 if term.isdigit() or len(term) >= 8 else 1.0
                for term in overlap
            )
            score = overlap_weight / math.sqrt(max(1, len(sentence_terms)))
            scored.append((score, index))

        if not scored:
            return stripped

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(
            index
            for _score, index in scored[
                : self.budget.max_sentences_per_expansion
            ]
        )
        return " ".join(sentences[index] for index in selected)

    def _information_gain_order(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str,
    ) -> list[RetrievalResult]:
        """Filter low-yield evidence using estimated information per token.

        This is a deterministic rate-distortion proxy rather than a claim to
        know true answer mutual information. Candidate-set IDF estimates term
        surprise, normalized retrieval score estimates semantic relevance,
        and accepted evidence discounts repeated concepts, sources, and numeric
        facts. Crucially, this is a monotone filter over retrieval order: it can
        remove a low-yield item but cannot promote a weaker candidate over a
        required higher-ranked item.
        """

        if not expansions:
            return []

        prepared: list[dict[str, object]] = []
        document_frequency: dict[str, int] = defaultdict(int)
        raw_scores = [max(0.0, float(result.score)) for result in expansions]
        low_score = min(raw_scores, default=0.0)
        high_score = max(raw_scores, default=0.0)
        for index, result in enumerate(expansions):
            text = self._prepare_expansion_text(result.chunk.text, query)
            snippet = truncate_to_tokens(text, self.budget.max_expansion_tokens)
            terms = set(tokenize(snippet))
            for term in terms:
                document_frequency[term] += 1
            cost = count_tokens(snippet) + 2 if snippet else 0
            relative_score = (
                (raw_scores[index] - low_score) / (high_score - low_score)
                if high_score > low_score
                else 0.0
            )
            normalized_score = max(
                min(1.0, raw_scores[index]),
                0.5 * relative_score,
            )
            prepared.append(
                {
                    "index": index,
                    "result": result,
                    "terms": terms,
                    "cost": cost,
                    "score": normalized_score,
                    "source": self._result_source_id(result),
                    "consolidation": result.route == "live_consolidation",
                }
            )

        count = len(expansions)
        idf = {
            term: math.log2((count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in document_frequency.items()
        }
        query_terms = set(tokenize(query))
        # A set/sequence answer has a higher distortion cost than a singleton:
        # superficially repetitive excerpts may each carry a different required
        # member. Retain more evidence for these queries instead of teaching the
        # redundancy filter that "another concert" or "another change" is noise.
        multi_fact_markers = {
            "all",
            "each",
            "order",
            "ordered",
            "earliest",
            "latest",
            "sequence",
            "chronological",
            "compare",
            "differences",
            "between",
        }
        effective_threshold = self.budget.min_information_gain_per_token
        if query_terms.intersection(multi_fact_markers):
            effective_threshold *= 0.70
        query_weight = sum(idf.get(term, math.log2(count + 1.0)) for term in query_terms)
        selected: list[RetrievalResult] = []
        selected_terms: set[str] = set()
        selected_numbers: set[str] = set()
        selected_sources: set[str] = set()
        for item in prepared:
            cost = int(item["cost"])
            if cost <= 0:
                continue
            terms = set(item["terms"])
            total_information = sum(idf.get(term, 1.0) for term in terms)
            new_information = sum(
                idf.get(term, 1.0) for term in terms - selected_terms
            )
            lexical_relevance = (
                sum(
                    idf.get(term, 1.0)
                    for term in terms.intersection(query_terms)
                )
                / query_weight
                if query_weight > 0.0
                else 0.0
            )
            semantic_relevance = float(item["score"])
            relevance = max(lexical_relevance, semantic_relevance)
            concept_novelty = (
                new_information / total_information
                if total_information > 0.0
                else 0.0
            )
            source_novelty = float(str(item["source"]) not in selected_sources)
            numbers = {term for term in terms if term.isdigit()}
            temporal_novelty = (
                len(numbers - selected_numbers) / len(numbers)
                if numbers
                else 0.0
            )
            novelty = (
                0.65 * concept_novelty
                + 0.25 * source_novelty
                + 0.10 * temporal_novelty
            )
            marginal_information = relevance * (0.60 + 0.40 * novelty)
            gain_rate = marginal_information / max(1, cost)
            if gain_rate < effective_threshold:
                continue
            result = item["result"]
            if not isinstance(result, RetrievalResult):
                continue
            selected.append(result)
            selected_terms.update(terms)
            selected_numbers.update(term for term in terms if term.isdigit())
            selected_sources.add(str(item["source"]))

        return selected

    @staticmethod
    def _result_source_id(result: RetrievalResult) -> str:
        if result.memory_source_id:
            return result.memory_source_id
        if result.turn is not None:
            return str(result.turn.source_id or result.turn.turn_id)
        return result.chunk.turn_id

    def _heat_weighted_order(
        self, expansions: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Order a prefix by weighted-fair source exposure.

        Heat is source-level purchasing power, while chunk length is its cost.
        Sources with insufficient material naturally yield their unused share
        to the remaining queues. Nothing transformer-shaped is retained here.
        """

        source_heat: dict[str, float] = {}
        queues: dict[str, deque[RetrievalResult]] = defaultdict(deque)
        for result in expansions:
            source_id = result.memory_source_id or result.chunk.turn_id
            queues[source_id].append(result)
            if result.source_heat is not None:
                source_heat[source_id] = max(
                    source_heat.get(source_id, 0.0), float(result.source_heat)
                )
        if not source_heat or sum(source_heat.values()) <= 0.0:
            return expansions

        served: dict[str, int] = defaultdict(int)
        ordered: list[RetrievalResult] = []
        source_cap = max(
            1,
            math.ceil(
                self.budget.expansion_tokens
                * self.budget.max_source_expansion_fraction
            ),
        )
        while any(queues.values()):
            choices: list[tuple[float, float, str, RetrievalResult]] = []
            capped: list[tuple[float, float, str, RetrievalResult]] = []
            for source_id, queue in queues.items():
                if not queue:
                    continue
                result = queue[0]
                cost = max(
                    1,
                    min(result.chunk.token_count, self.budget.max_expansion_tokens),
                )
                weight = max(source_heat.get(source_id, 0.0), 1e-12)
                choice = (
                    (served[source_id] + cost) / weight,
                    -float(result.diffusion_heat or 0.0),
                    source_id,
                    result,
                )
                choices.append(choice)
                if served[source_id] == 0 or served[source_id] + cost <= source_cap:
                    capped.append(choice)
            pool = capped or choices
            _, _, source_id, result = min(pool)
            queues[source_id].popleft()
            served[source_id] += max(
                1,
                min(result.chunk.token_count, self.budget.max_expansion_tokens),
            )
            ordered.append(result)
        return ordered
