"""Bounded, question-independent policies for existing summary/context retrieval."""
from datetime import datetime
from types import SimpleNamespace

from memory_condense.application.section_retrieval import HydratedSection, HydratedSectionSpan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.spine_reader_policy_v6 import complete_reader_messages
from memory_condense.search.native_spine_summary import body_identity
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_time_prior_v2 import question_day
from tools import evaluate_frozen_native_spine_full100 as frozen


DENSE_PARENT_2048 = {'max_context_tokens': 2048, 'max_raw_spans': 128, 'max_direct': 32,
    'lexical_reserve': 0, 'context_seed_limit': 4, 'max_additions': 16,
    'protected_direct': 0, 'ancestor_hops': 2}


def validate_policy(policy):
    bounds = {'max_context_tokens': (256, 4096), 'max_raw_spans': (1, 128),
        'max_direct': (1, 64), 'lexical_reserve': (0, 8), 'context_seed_limit': (1, 8),
        'max_additions': (0, 64), 'protected_direct': (0, 64), 'ancestor_hops': (0, 2)}
    if (set(policy) != set(bounds) or any(type(policy[k]) is not int or not lo <= policy[k] <= hi
            for k, (lo, hi) in bounds.items())
            or max(policy['lexical_reserve'], policy['protected_direct'], policy['context_seed_limit']) > policy['max_direct']):
        raise ValueError('context policy must contain bounded, question-independent retrieval limits')
    return dict(policy)


def messages(question, hydration):
    return complete_reader_messages(frozen.design.reader_messages(
        frozen.serving.protocol.messages(question, hydration), 'v5'))


def build(memory, question, policy):
    limits = validate_policy(policy)
    result = memory.retrieve(question['retrieval_query'], question['prompt_question'], **limits)
    return messages(question, result.hydration), result.hydration.identity_payload(), result.routing.identity_payload()


def verify_packet(question, hydration, routing, sessions, load_body, policy):
    """Reconstruct exact source excerpts, with limits from the sealed policy."""
    p = validate_policy(policy)
    if (hydration['max_context_tokens'] != p['max_context_tokens']
            or hydration['max_raw_spans'] != p['max_raw_spans']
            or hydration['raw_turn_read_count'] > p['max_raw_spans']
            or hydration['context_token_count'] > p['max_context_tokens']
            or routing['query_qwen_passes'] != 0 or routing['raw_reads_during_routing'] != 0
            or routing['protected_direct'] != p['protected_direct']
            or routing['ancestor_hops'] != p['ancestor_hops']
            or len(routing['context_atomic_ids']) > p['max_additions']
            or len(routing['consulted_chunk_ids']) > p['context_seed_limit']
            or routing['baseline']['max_sections'] != p['max_direct']
            or hydration['plan'] != routing['expanded']):
        raise ValueError('audited packet violates its sealed context policy')
    sources = {'native-source-' + s['occurrence_id']: s for s in sessions}
    raw_turns, rendered = {}, []
    asked = question_day(question['retrieval_query'], question['prompt_question'])
    span_count = 0
    selected = {r['section']['section_id']: r['section'] for r in routing['expanded']['routes']}
    for number, row in enumerate(hydration['sections'], 1):
        section = SectionSummary.from_dict(row['section'])
        if selected.get(section.section_id) != row['section']:
            raise ValueError('served evidence escaped summary-selected sections')
        evidence = []
        for item in row['evidence']:
            span = RawSectionSpan(**item['span'])
            source = sources.get(span.source_id)
            if source is None or source['created_at'] != span.created_at or datetime.fromisoformat(span.created_at).date() > asked:
                raise ValueError('served evidence belongs to a foreign or future occurrence')
            if span.source_id not in raw_turns:
                body = load_body(source['body_sha256'])
                if body_identity(body) != source['body_sha256']:
                    raise ValueError('raw audit loader returned a different body')
                raw_turns[span.source_id] = {'native-turn-' + identity_sha256({
                    'occurrence_id': source['occurrence_id'], 'body_sha256': source['body_sha256'],
                    'turn_ordinal': ordinal}): turn for ordinal, turn in enumerate(body['turns'])}
            turn = raw_turns[span.source_id].get(span.turn_id)
            if (turn is None or span.role != turn['role'] or quote_sha256(turn['text']) != span.turn_text_sha256
                    or span.end_char > len(turn['text'])
                    or item['text'] != turn['text'][span.start_char:span.end_char]):
                raise ValueError('served text differs from its exact original raw section')
            evidence.append(HydratedSectionSpan(span, item['text']))
            span_count += 1
        hydrated = HydratedSection(section, tuple(evidence))
        if hydrated.identity_payload() != row:
            raise ValueError('hydrated section receipt differs from its raw evidence')
        rendered.append(hydrated.render_raw(f'S{number}'))
    context = '\n\n'.join(rendered)
    if count_tokens(context) != hydration['context_token_count'] or span_count > p['max_raw_spans']:
        raise ValueError('served context count differs from actual raw packet')
    return messages(question, SimpleNamespace(render_context=lambda: context)), span_count
