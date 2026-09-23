"""Prioritize unseen direct evidence within an existing summary-selected set."""
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_routing import SectionRoutePlan


def prioritize_unseen_direct_routes(baseline, expanded, *, visible_section_ids=(),
                                    excluded_turn_ids=(), deferred_turn_ids=()):
    """Reorder summary addresses without loading or altering raw evidence.

    Already supplied sections and caller-designated metadata turns are excluded.
    Current live observations are deferred, not dropped: their previews may be
    incomplete. Remaining direct matches precede hierarchy-neighborhood additions.
    """
    if (baseline.query_sha256 != expanded.query_sha256
            or baseline.eligible_source_ids != expanded.eligible_source_ids):
        raise ValueError('direct and expanded plans must share query and source scope')
    selected = {r.section.section_id: r.section for r in expanded.routes}
    if any(selected.get(r.section.section_id) != r.section for r in baseline.routes):
        raise ValueError('expanded plan must preserve every direct evidence address')
    visible, excluded, deferred = map(frozenset, (visible_section_ids, excluded_turn_ids, deferred_turn_ids))
    seen, first, last = set(), [], []
    for route in (*baseline.routes, *expanded.routes):
        section = route.section
        if section.section_id in seen:
            continue
        seen.add(section.section_id)
        if section.section_id in visible or any(s.turn_id in excluded for s in section.spans):
            continue
        (last if any(s.turn_id in deferred for s in section.spans) else first).append(route)
    routes = tuple(first + last)
    identity = identity_sha256({'format': 'unseen-direct-section-priority-v1',
        'baseline_sha256': baseline.receipt_sha256, 'expanded_sha256': expanded.receipt_sha256,
        'visible_section_ids': sorted(visible), 'excluded_turn_ids': sorted(excluded),
        'deferred_turn_ids': sorted(deferred)})
    return SectionRoutePlan(identity, baseline.query_sha256, routes,
        max(expanded.matched_section_count, len(routes)), expanded.eligible_source_ids,
        max(1, len(routes)), routing_backend='summary_hybrid')
