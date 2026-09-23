"""Use spare grouped-context space without displacing any prior raw evidence."""
from dataclasses import dataclass, replace
from datetime import datetime

from memory_condense.application.section_retrieval import (
    SectionRetrievalResult, _render, hydrate_section_plan,
)
from memory_condense.application.threaded_section_context import ThreadedSectionContext, render_threaded_sections
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.as_of_spine_routing import _plan


CONTEXT_LIMIT = 3072
SPAN_LIMIT = 128
ADDITION_LIMIT = 8
CANDIDATE_LIMIT = 32
# Internal legacy framing may be larger than the grouped reader context.
# This staging value is never the reader prompt budget.
STAGING_LIMIT = 8192


def attempted_spans(result):
    rejected = {d.section_id: d.reason for d in result.diagnostics}
    return sum(len(r.section.spans) for r in result.plan.routes
               if rejected.get(r.section.section_id) != 'raw_span_budget'
               and sum(p.token_count for p in r.section.spans) <= result.max_context_tokens)


@dataclass(frozen=True)
class SupplementedContext:
    hydration: SectionRetrievalResult
    rendered: ThreadedSectionContext
    base_hydration_sha256: str
    candidate_plan_sha256: str
    added_section_ids: tuple
    extra_raw_read_count: int
    attempted_raw_spans: int
    diagnostics: tuple

    def identity_payload(self):
        return {'hydration': self.hydration.identity_payload(), 'rendered': self.rendered.identity_payload(),
                'base_hydration_sha256': self.base_hydration_sha256,
                'candidate_plan_sha256': self.candidate_plan_sha256,
                'added_section_ids': list(self.added_section_ids),
                'extra_raw_read_count': self.extra_raw_read_count,
                'attempted_raw_spans': self.attempted_raw_spans,
                'diagnostics': [{'section_id': sid, 'reason': reason} for sid, reason in self.diagnostics],
                'reader_context_limit': CONTEXT_LIMIT, 'staging_context_limit': STAGING_LIMIT}


def supplement_threaded_context(base, order, candidates, *, load_turn, asked_day):
    if base.max_context_tokens != CONTEXT_LIMIT or base.max_raw_spans != SPAN_LIMIT:
        raise ValueError('supplement requires the original reader and span budgets')
    if candidates.query_sha256 != base.plan.query_sha256 or len(candidates.routes) > CANDIDATE_LIMIT:
        raise ValueError('supplement changed the query or candidate budget')
    rendered = render_threaded_sections(base, order)
    combined, accepted = base, []
    existing_turns = {r.span.turn_id for s in base.sections for r in s.evidence}
    original_spans = {r.span.receipt_sha256 for s in base.sections for r in s.evidence}
    attempts, extra_reads, diagnostics = attempted_spans(base), 0, []
    for route in candidates.routes:
        section = route.section
        if len(accepted) >= ADDITION_LIMIT:
            break
        if (any(p.role != 'user' for p in section.spans) or len({p.turn_id for p in section.spans}) != 1
                or section.spans[0].start_char != 0):
            raise ValueError('supplements must be whole user-turn sections')
        if any(p.turn_id in existing_turns for p in section.spans):
            diagnostics.append((section.section_id, 'existing_turn'))
            continue
        if any(datetime.fromisoformat(p.created_at).date() > asked_day for p in section.spans):
            diagnostics.append((section.section_id, 'future_turn'))
            continue
        if attempts + len(section.spans) > SPAN_LIMIT:
            diagnostics.append((section.section_id, 'raw_span_budget'))
            continue
        if sum(p.token_count for p in section.spans) > CONTEXT_LIMIT:
            diagnostics.append((section.section_id, 'raw_content_budget'))
            continue
        attempts += len(section.spans)
        single_plan = _plan(candidates.index_sha256, candidates.query_sha256, (section,))
        single = hydrate_section_plan(single_plan, load_turn=load_turn,
                                      max_raw_spans=len(section.spans), max_context_tokens=STAGING_LIMIT)
        extra_reads += single.raw_turn_read_count
        if single.diagnostics:
            diagnostics.extend((d.section_id, d.reason) for d in single.diagnostics)
            continue
        if quote_sha256(''.join(r.text for r in single.sections[0].evidence)) != section.spans[0].turn_text_sha256:
            raise ValueError('supplements must preserve the whole user turn')
        sections = (*combined.sections, *single.sections)
        context_tokens = count_tokens(_render(sections))
        if context_tokens > STAGING_LIMIT:
            diagnostics.append((section.section_id, 'staging_budget'))
            continue
        plan = _plan(identity_sha256([s.section.receipt_sha256 for s in sections]),
                     base.plan.query_sha256, [s.section for s in sections])
        proposal = SectionRetrievalResult(plan, sections, (), base.raw_turn_read_count + extra_reads,
                                         SPAN_LIMIT, STAGING_LIMIT, context_tokens)
        proposal_rendered = render_threaded_sections(proposal, order)
        if proposal_rendered.token_count > CONTEXT_LIMIT:
            diagnostics.append((section.section_id, 'reader_context_budget'))
            continue
        combined, rendered = proposal, proposal_rendered
        accepted.append(section.section_id)
        existing_turns.update(p.turn_id for p in section.spans)
    if accepted and combined.raw_turn_read_count != base.raw_turn_read_count + extra_reads:
        combined = replace(combined, raw_turn_read_count=base.raw_turn_read_count + extra_reads, receipt_sha256='')
        rendered = render_threaded_sections(combined, order)
    if not original_spans <= {sha for sha, _, _ in rendered.placements}:
        raise ValueError('supplement displaced original evidence')
    return SupplementedContext(combined, rendered, base.receipt_sha256, candidates.receipt_sha256,
                               tuple(accepted), extra_reads, attempts, tuple(diagnostics))
