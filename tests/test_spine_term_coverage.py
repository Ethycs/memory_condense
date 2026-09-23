from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.spine_term_coverage import SpineTermCoverage


def fixture(name="Orion", other_source="gamma"):
    turns, atoms, leaves = {}, [], []
    rows = [("alpha", "Create a service roster.", "Generic roster advice. " * 90),
            ("alpha", "The roster needs four shifts.", "A generic four-shift roster."),
            ("beta", f"Use {name} and Lyra.", f"Sunday | {name} | 08:00-16:00"),
            (other_source, "Create another roster.", "An unrelated roster.")]
    for i, (source, user, context) in enumerate(rows):
        spans = []
        for role, raw in (("user", user), ("assistant", context)):
            key = f"{source}-{i}-{role}"
            turn = Turn(turn_id=key, source_id=source, role=role, text=raw,
                        created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
            span = RawSectionSpan.from_turn(turn)
            turns[key] = turn
            spans.append(span)
            summary = raw if role == "user" else "A proposed roster."
            atoms.append(SectionSummary(key, source, summary, (span,), "fixture"))
        summary = json.dumps({"user_spine": user, "attached_context_not_user_assertions": "A proposed roster.",
                              "transcript_date_range": [turn.created_at.isoformat()] * 2})
        leaves.append(SectionSummary(f"leaf-{i}", source, summary, tuple(spans), "fixture"))
    hierarchy = SectionSummaryIndex(leaves)
    source_spine = SourceSpineHydrationIndex(hierarchy, atoms)
    return turns, source_spine, hierarchy


def prior_plan(hierarchy, projection, query):
    selected = SectionRoutePlan(hierarchy.receipt_sha256, quote_sha256(query),
        (SectionRoute(hierarchy.sections[0], 1.0, ()),), 1, None, 1)
    return projection.expand(selected)[0]


@pytest.mark.parametrize("name", ("Orion", "Kestrel", "Cedar"))
def test_rare_summary_term_recovers_paired_exact_context_without_losing_users(name):
    turns, projection, hierarchy = fixture(name)
    query = f"When is {name} on duty?"
    prior = prior_plan(hierarchy, projection, query)
    result, audit = SpineTermCoverage(hierarchy, projection).expand(query, prior)
    old = hydrate_section_plan(prior, load_turn=turns.get, max_context_tokens=512, max_raw_spans=128)
    new = hydrate_section_plan(result, load_turn=turns.get, max_context_tokens=512, max_raw_spans=128)
    def users(hydrated):
        return [s for s in hydrated.sections if all(e.span.role == "user" for e in s.evidence)]
    assert users(new)[:len(users(old))] == users(old)
    assert f"Sunday | {name} | 08:00-16:00" in new.render_context()
    assert new.context_token_count <= 512
    assert audit["raw_reads"] == 0 and audit["rare_terms"] == [{"term": name.lower(), "document_frequency": 1}]
    assert all(e.text == turns[e.span.turn_id].text for s in new.sections for e in s.evidence)


def test_common_or_absent_terms_leave_the_original_plan_unchanged():
    _, projection, hierarchy = fixture()
    for query in ("roster", "nonexistent"):
        prior = prior_plan(hierarchy, projection, query)
        result, audit = SpineTermCoverage(hierarchy, projection).expand(query, prior)
        assert result is prior and not audit["selected_exchange_ids"]


def test_foreign_query_or_replaced_source_descriptor_is_rejected():
    _, projection, hierarchy = fixture()
    query = "Orion"
    prior = prior_plan(hierarchy, projection, query)
    coverage = SpineTermCoverage(hierarchy, projection)
    with pytest.raises(ValueError, match="another query"):
        coverage.expand("Lyra", prior)
    changed = replace(prior.routes[0].section, summary="replacement", receipt_sha256="")
    foreign = replace(prior, routes=(SectionRoute(changed, 1.0, ()), *prior.routes[1:]), receipt_sha256="")
    with pytest.raises(ValueError, match="changed source"):
        coverage.expand(query, foreign)
