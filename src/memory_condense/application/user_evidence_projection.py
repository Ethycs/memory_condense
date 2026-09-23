"""A whole-section assistant-context ablation for user-fact recall."""
from dataclasses import dataclass

from memory_condense.application import user_spine_section_context_v2 as layout
from memory_condense.application.section_retrieval import (
    SectionHydrationDiagnostic, SectionRetrievalResult, _render,
)
from memory_condense.domain._tokenizer import count_tokens

TranscriptOrder = layout.TranscriptOrder
FORMAT = 'user-bearing-evidence-whole-section-projection-v1'


@dataclass(frozen=True)
class UserEvidenceContext:
    rendered: layout.UserSpineSectionContext
    source_hydration_sha256: str
    omitted_section_ids: tuple[str, ...]

    @property
    def text(self):
        return self.rendered.text

    def identity_payload(self):
        return {**self.rendered.identity_payload(), 'format': FORMAT,
                'source_hydration_sha256': self.source_hydration_sha256,
                'omitted_section_ids': list(self.omitted_section_ids)}


def render_user_spine_sections(hydrated, order):
    """Omit only entire assistant-only sections, without reading their content.

    All user sections, mixed-role sections and other-role sections stay intact.
    If there is no user-bearing section, retain the original packet. This is an
    explicit ablation, not a claim that assistant context is generally redundant.
    Authenticate the full input first, including sections that may be omitted.
    """
    original = layout.render_user_spine_sections(hydrated, order)
    has_user = any(row.span.role == 'user' for s in hydrated.sections for row in s.evidence)
    omitted = tuple(s for s in hydrated.sections if has_user and s.evidence
                    and all(row.span.role == 'assistant' for row in s.evidence))
    omitted_ids = tuple(s.section.section_id for s in omitted)
    if not omitted:
        return UserEvidenceContext(original, hydrated.receipt_sha256, ())
    omitted_set = set(omitted_ids)
    sections = tuple(s for s in hydrated.sections if s.section.section_id not in omitted_set)
    projected = SectionRetrievalResult(hydrated.plan, sections,
        (*hydrated.diagnostics, *(SectionHydrationDiagnostic(s, 'assistant_context_ablation')
                                  for s in omitted_ids)),
        hydrated.raw_turn_read_count, hydrated.max_raw_spans, hydrated.max_context_tokens,
        count_tokens(_render(sections)))
    rendered = layout.render_user_spine_sections(projected, order)
    return UserEvidenceContext(rendered, hydrated.receipt_sha256, omitted_ids)
