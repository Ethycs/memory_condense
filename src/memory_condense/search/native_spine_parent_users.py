"""Semantic addresses for the user spine of stored top-level summary groups."""
import json

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary


FORMAT = 'native-spine-root-user-summaries-v1'


def project_parent_users(hierarchy):
    """Use every stored root, including singleton groups, without raw reads.

    Roots retain exact user-span addresses. Only the stored ``user_spine`` string
    contributes to semantic ranking; attached assistant context is excluded.
    """
    if type(hierarchy) is not SectionSummaryIndex:
        raise TypeError('parent-user projection requires an authenticated hierarchy')
    children = {c for s in hierarchy.sections for c in s.child_section_ids}
    projected = []
    for root in hierarchy.sections:
        if root.section_id in children:
            continue
        spans = tuple(s for s in root.spans if s.role == 'user')
        if not spans:
            continue
        try:
            summary = json.loads(root.summary)['user_spine']
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError('a user-containing root lacks a stored user-spine summary') from error
        if type(summary) is not str or not summary.strip():
            raise ValueError('a user-containing root has an empty user-spine summary')
        provenance = {'format': FORMAT, 'hierarchy_sha256': hierarchy.receipt_sha256,
                      'original_root_sha256': root.receipt_sha256}
        projected.append(SectionSummary('native-parent-user-' + identity_sha256(provenance),
            root.source_id, summary, spans, canonical_json(provenance)))
    return SectionSummaryIndex(projected)
