"""Refine rare-term coverage inside already summary-selected conversations."""
from copy import copy

from memory_condense.search.spine_term_coverage import SpineTermCoverage


class ScopedSpineTermCoverage(SpineTermCoverage):
    def expand(self, query, prior):
        sources = frozenset(r.section.source_id for r in prior.routes)
        scoped = copy(self)
        scoped.leaves = tuple(s for s in self.leaves if s.source_id in sources)
        available_terms = set().union(*(self.terms[s.section_id] for s in scoped.leaves))
        # Keep global document frequencies; merely require independent source
        # selection before spending context on a rare term's matching exchange.
        scoped.frequencies = {term: count for term, count in self.frequencies.items() if term in available_terms}
        plan, audit = SpineTermCoverage.expand(scoped, query, prior)
        return plan, {**audit, "source_scope": sorted(sources),
                      "new_sources_admitted": False, "rarity_population": "complete memory"}
