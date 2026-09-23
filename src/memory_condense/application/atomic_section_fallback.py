"""Recover budget-rejected sections through their original atomic addresses."""
from dataclasses import dataclass

from memory_condense.application.additive_threaded_context import attempted_spans
from memory_condense.application.section_retrieval import (
    SectionHydrationDiagnostic, SectionRetrievalResult, _render, hydrate_section_plan,
)
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_routing import SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import bound_int


@dataclass(frozen=True, slots=True)
class AtomicFallbackResult(SealedIdentity):
    baseline: SectionRetrievalResult
    hydration: SectionRetrievalResult
    atomic_plan: SectionRoutePlan
    added_atomic_ids: tuple[str, ...]
    atomic_owners: tuple[tuple[str, tuple[str, ...]], ...]
    attempted_raw_spans: int
    receipt_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "added_atomic_ids", tuple(self.added_atomic_ids))
        object.__setattr__(self, "atomic_owners", tuple((a, tuple(owners)) for a, owners in self.atomic_owners))
        b, h = self.baseline, self.hydration
        if (h.sections[:len(b.sections)] != b.sections
                or (h.max_context_tokens, h.max_raw_spans) != (b.max_context_tokens, b.max_raw_spans)
                or h.plan.query_sha256 != b.plan.query_sha256
                or self.atomic_plan.query_sha256 != b.plan.query_sha256
                or h.raw_turn_read_count < b.raw_turn_read_count
                or tuple(s.section.section_id for s in h.sections[len(b.sections):]) != self.added_atomic_ids):
            raise ValueError("atomic fallback changed baseline evidence, query or budgets")
        bound_int(self.attempted_raw_spans, "attempted_raw_spans")
        if self.attempted_raw_spans > b.max_raw_spans:
            raise ValueError("atomic fallback exceeded the combined raw inspection budget")
        self._seal()


class AtomicSectionFallback:
    """Pre-index original atoms; query time accepts already-ranked summary plans."""

    def __init__(self, atomic_index, *, max_candidates=32, max_additions=8):
        if type(atomic_index) is not SectionSummaryIndex:
            raise TypeError("atomic fallback requires an authenticated summary index")
        bound_int(max_candidates, "max_candidates", 1)
        bound_int(max_additions, "max_additions", 1)
        self.index = atomic_index
        self.by_id = {s.section_id: s for s in atomic_index.sections}
        if (any(len(s.spans) != 1 or s.child_section_ids for s in atomic_index.sections)
                or len({s.spans[0].receipt_sha256 for s in atomic_index.sections}) != len(atomic_index.sections)):
            raise ValueError("fallback index must contain unique original single-span atoms")
        self.max_candidates, self.max_additions = max_candidates, max_additions

    def hydrate(self, primary, atoms, *, load_turn, max_raw_spans=128, max_context_tokens=3072):
        if type(primary) is not SectionRoutePlan or type(atoms) is not SectionRoutePlan:
            raise TypeError("atomic fallback requires two summary route plans")
        if (primary.query_sha256 != atoms.query_sha256 or atoms.index_sha256 != self.index.receipt_sha256
                or len(atoms.routes) > self.max_candidates
                or any(self.by_id.get(r.section.section_id) != r.section for r in atoms.routes)):
            raise ValueError("atomic fallback candidate query, identity or bound changed")
        primary_ids = {r.section.section_id: r.section for r in primary.routes}
        primary_spans = [s.receipt_sha256 for r in primary.routes for s in r.section.spans]
        if len(set(primary_spans)) != len(primary_spans):
            raise ValueError("primary selections must not duplicate or overlap raw evidence")
        for route in atoms.routes:
            if route.section.section_id in primary_ids and primary_ids[route.section.section_id] != route.section:
                raise ValueError("primary and atomic views disagree about a section identity")

        cache, failures, physical_reads = {}, {}, 0

        def cached_load(turn_id):
            nonlocal physical_reads
            if turn_id in failures:
                raise failures[turn_id]
            if turn_id not in cache:
                physical_reads += 1
                try:
                    cache[turn_id] = load_turn(turn_id)
                except Exception as error:
                    failures[turn_id] = error
                    raise
            return cache[turn_id]

        baseline = hydrate_section_plan(primary, load_turn=cached_load, max_raw_spans=max_raw_spans,
                                        max_context_tokens=max_context_tokens)
        owners = {}
        for diagnostic in baseline.diagnostics:
            if diagnostic.reason in {"context_budget", "raw_span_budget"}:
                for span in primary_ids[diagnostic.section_id].spans:
                    owners.setdefault(span.receipt_sha256, []).append(diagnostic.section_id)
        sections, diagnostics = list(baseline.sections), list(baseline.diagnostics)
        routes, added, attribution = list(primary.routes), [], []
        present = {e.span.receipt_sha256 for s in sections for e in s.evidence}
        attempts = attempted_spans(baseline)
        for route in atoms.routes:
            if len(added) >= self.max_additions:
                break
            section = route.section
            span = section.spans[0]
            if (span.receipt_sha256 not in owners or span.receipt_sha256 in present
                    or section.section_id in primary_ids):
                continue
            if primary.eligible_source_ids is not None and section.source_id not in primary.eligible_source_ids:
                raise ValueError("atomic fallback escaped the primary source scope")
            routes.append(route)
            attribution.append((section.section_id, tuple(owners[span.receipt_sha256])))
            reason = None
            if attempts + 1 > max_raw_spans:
                reason = "raw_span_budget"
            elif span.token_count > max_context_tokens:
                reason = "context_budget"
            else:
                attempts += 1
                single_plan = SectionRoutePlan(self.index.receipt_sha256, atoms.query_sha256, (route,), 1,
                                               atoms.eligible_source_ids, 1, routing_backend="summary_hybrid")
                single = hydrate_section_plan(single_plan, load_turn=cached_load, max_raw_spans=1,
                                              max_context_tokens=max_context_tokens)
                if single.diagnostics:
                    reason = single.diagnostics[0].reason
                elif count_tokens(_render((*sections, *single.sections))) > max_context_tokens:
                    reason = "context_budget"
                else:
                    sections.extend(single.sections)
                    present.add(span.receipt_sha256)
                    added.append(section.section_id)
            if reason is not None:
                diagnostics.append(SectionHydrationDiagnostic(section.section_id, reason))
        if len(routes) == len(primary.routes):
            combined = baseline
        else:
            # The plan explicitly binds both index views; rejected coarse routes
            # remain diagnosed even when some of their atoms are recovered.
            index_binding = identity_sha256({"primary_index_sha256": primary.index_sha256,
                                            "atomic_index_sha256": self.index.receipt_sha256})
            plan = SectionRoutePlan(index_binding, primary.query_sha256, tuple(routes), len(routes),
                                    primary.eligible_source_ids, len(routes), routing_backend="summary_hybrid")
            combined = SectionRetrievalResult(plan, tuple(sections), tuple(diagnostics), physical_reads,
                max_raw_spans, max_context_tokens, count_tokens(_render(tuple(sections))))
        return AtomicFallbackResult(baseline, combined, atoms, tuple(added), tuple(attribution), attempts)
