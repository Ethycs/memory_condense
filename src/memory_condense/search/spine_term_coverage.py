"""Reserve rare query terms in stored summaries and retain their paired context.

This optional supplement reads no raw text. It keeps the existing user-section
prefix and then offers summary-matched exchanges before the old context tail.
Exact hydration remains the authority for bytes and the unchanged final budget.
"""
from collections import Counter
import json

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.indexes.lexical import tokenize
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.source_spine_overflow import SourceSpineOverflow


class SpineTermCoverage:
    def __init__(self, hierarchy, source_spine):
        if hierarchy.receipt_sha256 != source_spine.parent_sha256:
            raise ValueError("term addresses and source projection belong to different memories")
        self.hierarchy = hierarchy
        self.source_spine = source_spine
        self.users = SourceSpineOverflow(source_spine).by_turn
        self.leaves = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        self.terms = {}
        frequencies = Counter()
        for section in self.leaves:
            channels = json.loads(section.summary)
            terms = frozenset(tokenize((channels["user_spine"] or "") + "\n" +
                                      (channels["attached_context_not_user_assertions"] or "")))
            self.terms[section.section_id] = terms
            frequencies.update(terms)
        self.frequencies = dict(frequencies)
        # Corpus-relative rarity; no benchmark entities or outcome selectors.
        self.maximum_document_frequency = max(1, len(self.leaves) // 100)

    def expand(self, query, prior):
        if (prior.index_sha256 != self.source_spine.index.receipt_sha256 or
                prior.query_sha256 != quote_sha256(query)):
            raise ValueError("term supplement belongs to another query or source projection")
        originals = {s.section_id: s for s in self.source_spine.index.sections}
        if any(originals.get(r.section.section_id) != r.section for r in prior.routes):
            raise ValueError("term supplement cannot accept changed source descriptors")
        query_terms = set(tokenize(query))
        rare = sorted((term for term in query_terms
            if 0 < self.frequencies.get(term, 0) <= self.maximum_document_frequency),
            key=lambda term: (self.frequencies[term], term))[:2]
        selected = {}
        for term in rare:
            candidates = sorted((s for s in self.leaves if term in self.terms[s.section_id]),
                key=lambda s: (-len(query_terms & self.terms[s.section_id]),
                               len(self.terms[s.section_id]), s.section_id))
            for section in candidates[:2]:
                selected.setdefault(section.section_id, section)
        if not selected:
            return prior, {"rare_terms": [], "selected_exchange_ids": [], "raw_reads": 0,
                           "prior_user_order_preserved": True}
        users = [r.section for r in prior.routes if all(p.role == "user" for p in r.section.spans)]
        attached = [r.section for r in prior.routes if not all(p.role == "user" for p in r.section.spans)]
        seen_users = {s.section_id for s in users}
        additions, contexts = [], []
        for section in selected.values():
            for span in section.spans:
                if span.role == "user":
                    user = self.users[span.turn_id]
                    if user.section_id not in seen_users:
                        additions.append(user)
                        seen_users.add(user.section_id)
            context = self.source_spine.attachments.get(section.section_id)
            if context is not None:
                contexts.append(context)
        sections = {s.section_id: s for s in (*users, *additions, *contexts, *attached)}
        plan = SectionRoutePlan(prior.index_sha256, prior.query_sha256,
            tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(sections.values())),
            len(self.source_spine.index.sections), None, max(1, len(sections)), routing_backend="summary_hybrid")
        return plan, {"rare_terms": [{"term": term, "document_frequency": self.frequencies[term]} for term in rare],
            "maximum_document_frequency": self.maximum_document_frequency,
            "selected_exchange_ids": list(selected), "additional_user_ids": [s.section_id for s in additions],
            "prioritized_context_ids": [s.section_id for s in contexts], "raw_reads": 0,
            "prior_user_order_preserved": True, "prior_context_may_be_displaced": True}
