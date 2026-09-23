"""Use the existing ordered renderer on unchanged native-spine raw evidence."""
from datetime import datetime
from types import SimpleNamespace

from memory_condense.application.section_retrieval import (
    HydratedSection, HydratedSectionSpan, SectionHydrationDiagnostic, SectionRetrievalResult,
)
from memory_condense.application import threaded_section_context as renderer
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.native_spine_summary import body_identity
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tools import native_spine_context_policy as context_policy


def build(memory, question, policy, order):
    result = memory.retrieve(question['retrieval_query'], question['prompt_question'],
                             **context_policy.validate_policy(policy))
    rendered = renderer.render_threaded_sections(result.hydration, order)
    messages = context_policy.messages(question, SimpleNamespace(render_context=lambda: rendered.text))
    return (messages, result.hydration.identity_payload(), result.routing.identity_payload(),
            rendered.identity_payload())


def hydration_from_payload(payload):
    """Revalidate the complete saved hydration receipt for independent rendering."""
    p = dict(payload['plan'])
    if p['attention_receipt'] is not None or p['reasoning_receipt'] is not None:
        raise ValueError('native context rendering expects offline attention topology')
    p['routes'] = tuple(SectionRoute(SectionSummary.from_dict(row['section']), row['score'],
        tuple(row['matched_terms']), row['receipt_sha256']) for row in p['routes'])
    plan = SectionRoutePlan(**p)
    sections = tuple(HydratedSection(SectionSummary.from_dict(row['section']),
        tuple(HydratedSectionSpan(RawSectionSpan(**e['span']), e['text'], e['receipt_sha256'])
              for e in row['evidence']), row['receipt_sha256']) for row in payload['sections'])
    result = SectionRetrievalResult(plan, sections,
        tuple(SectionHydrationDiagnostic(**d) for d in payload['diagnostics']),
        payload['raw_turn_read_count'], payload['max_raw_spans'], payload['max_context_tokens'],
        payload['context_token_count'], payload['receipt_sha256'])
    if result.identity_payload() != payload:
        raise ValueError('saved native hydration changed during reconstruction')
    return result


def source_order(sessions, load_body):
    """Recreate ingestion order directly from original bodies for audit only."""
    turns = []
    for source in sessions:
        body = load_body(source['body_sha256'])
        if body_identity(body) != source['body_sha256']:
            raise ValueError('transcript-order source body identity changed')
        for ordinal, turn in enumerate(body['turns']):
            turn_id = 'native-turn-' + identity_sha256({'occurrence_id': source['occurrence_id'],
                'body_sha256': source['body_sha256'], 'turn_ordinal': ordinal})
            turns.append(Turn(turn_id=turn_id, source_id='native-source-' + source['occurrence_id'],
                role=turn['role'], text=turn['text'], created_at=datetime.fromisoformat(source['created_at'])))
    return renderer.TranscriptOrder(turns)


def verify_packet(question, hydration, routing, rendered, sessions, load_body, policy, order):
    _, count = context_policy.verify_packet(question, hydration, routing, sessions, load_body, policy)
    rebuilt = renderer.render_threaded_sections(hydration_from_payload(hydration), order)
    if rebuilt.identity_payload() != rendered:
        raise ValueError('served conversation layout differs from exact source reconstruction')
    return context_policy.messages(question, SimpleNamespace(render_context=lambda: rebuilt.text)), count
