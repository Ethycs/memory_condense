"""Keep a summary exploration beam separate from the final raw-section budget.

A narrow answer packet must not force the same narrow choice between a broad
parent and an already-specific leaf. Expand all surviving branches while they
fit the exploration budget, then ask Qwen to select the final leaf sections.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_routing import (
    SectionRoute, SectionRoutePlan, SectionSummaryIndex, SummaryReasoningPass, SummaryReasoningReceipt,
)
from memory_condense.search.section_summary import bound_int, exact_text
from memory_condense.search.summary_reasoning import SummaryChoiceRequest, parse_summary_choice


@dataclass(frozen=True, slots=True)
class SpineBeamRoute(SealedIdentity):
    plan: SectionRoutePlan
    beam_sections: int
    explored_frontiers: tuple[tuple[str, ...], ...]
    pruned_frontier_count: int
    receipt_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "explored_frontiers", tuple(tuple(f) for f in self.explored_frontiers))
        bound_int(self.beam_sections, "beam_sections", self.plan.max_sections)
        bound_int(self.pruned_frontier_count, "pruned_frontier_count")
        if any(route.section.child_section_ids for route in self.plan.routes):
            raise ValueError("spine beam must terminate at whole leaf sections")
        self._seal()


def route_user_spine_hierarchy(
    query: str, index: SectionSummaryIndex, *, reasoner, max_sections: int = 3,
    beam_sections: int = 8, group_size: int = 16, max_depth: int = 32,
    max_calls: int = 64, max_prompt_tokens: int = 4096,
    eligible_source_ids: Sequence[str] | None = None,
) -> SpineBeamRoute:
    """Qwen sees summaries only; no raw reader or hydration capability is accepted.

    Groups and calls are bounded. Search can still lose relevant branches when
    the beam is exceeded; the receipt records those pruning rounds. An exhausted
    depth/call budget fails before hydration rather than substituting a parent.
    """
    exact_text(query, "query")
    for name, value in (("max_sections", max_sections), ("max_depth", max_depth),
                        ("max_calls", max_calls), ("max_prompt_tokens", max_prompt_tokens)):
        bound_int(value, name, 1)
    bound_int(beam_sections, "beam_sections", max_sections)
    bound_int(group_size, "group_size", beam_sections + 1)
    scope = None if eligible_source_ids is None else tuple(eligible_source_ids)
    if scope is not None and (len(set(scope)) != len(scope) or any(type(s) is not str or not s for s in scope)):
        raise ValueError("source scope must contain unique exact IDs")
    identity = (reasoner.model, reasoner.gateway_url)
    if "qwen" not in identity[0].casefold():
        raise ValueError("user-spine routing requires Qwen")
    by_id = {s.section_id: s for s in index.sections}
    children = {child for s in index.sections for child in s.child_section_ids}
    frontier = [s for s in index.sections if s.section_id not in children
                and (scope is None or s.source_id in scope)]
    passes, frontiers = [], []
    pruned = rounds = matched = 0
    selected = []

    def choose(group, limit):
        if len(passes) >= max_calls:
            raise ValueError("user-spine routing exhausted its call budget")
        request = SummaryChoiceRequest(query, tuple(s.summary for s in group), limit, max_prompt_tokens)
        response = reasoner.complete(request)
        labels = parse_summary_choice(response, candidate_count=len(group), max_choices=limit)
        winners = [group[label] for label in labels]
        passes.append(SummaryReasoningPass(tuple(s.section_id for s in group),
            tuple(s.section_id for s in winners), request.prompt_sha256, quote_sha256(response), request.prompt_token_proxy))
        return winners

    def rank(candidates, limit):
        while len(candidates) > group_size:
            candidates = [s for start in range(0, len(candidates), group_size)
                          for s in choose(candidates[start:start + group_size], limit)]
        return choose(candidates, limit) if candidates else []

    while frontier:
        if rounds == max_depth:
            raise ValueError("user-spine routing exhausted its depth budget")
        rounds += 1
        frontiers.append(tuple(s.section_id for s in frontier))
        if all(not s.child_section_ids for s in frontier):
            matched = len(frontier)
            selected = rank(frontier, max_sections)
            break
        if len(frontier) > beam_sections:
            pruned += 1
            frontier = rank(frontier, beam_sections)
        frontier = [child for s in frontier for child in (
            [by_id[c] for c in s.child_section_ids] if s.child_section_ids else [s])]
    if (reasoner.model, reasoner.gateway_url) != identity:
        raise ValueError("user-spine reasoner identity changed during routing")
    receipt = SummaryReasoningReceipt(*identity, tuple(passes), rounds,
                                     max_depth, max_calls, group_size, max_prompt_tokens)
    plan = SectionRoutePlan(index.receipt_sha256, quote_sha256(query),
        tuple(SectionRoute(s, 1.0 / rank, ()) for rank, s in enumerate(selected, 1)),
        matched, scope, max_sections, routing_backend="qwen_summary_reasoning", reasoning_receipt=receipt)
    return SpineBeamRoute(plan, beam_sections, tuple(frontiers), pruned)
