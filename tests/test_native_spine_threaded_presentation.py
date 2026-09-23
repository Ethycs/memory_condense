from copy import deepcopy

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.threaded_section_context import TranscriptOrder
from memory_condense.search.native_spine_memory import materialize_history
from tests.test_native_spine_context_routing import parent_fixture
from tests.test_native_spine_routing import QUERY, DATED
from tests.test_native_spine_memory import body_and_atoms, source
from tools import native_spine_context_policy as policy
from tools import native_spine_threaded_presentation as presentation


def test_grouped_native_packet_preserves_every_selected_span_and_reader():
    history, semantic, hierarchy, encoder = parent_fixture(future=True)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    question = {'retrieval_query': QUERY, 'prompt_question': DATED}
    flat, old_hydration, old_routing = policy.build(memory, question, policy.DENSE_PARENT_2048)
    messages, hydration, routing, rendered = presentation.build(memory, question,
        policy.DENSE_PARENT_2048, TranscriptOrder(history.turns.values()))
    assert messages[0] == flat[0]
    assert hydration == old_hydration and routing == old_routing
    assert encoder.calls == [QUERY, QUERY]
    assert rendered['token_count'] < hydration['context_token_count']
    evidence = {e['span']['receipt_sha256']: e['text'] for s in hydration['sections'] for e in s['evidence']}
    assert set(evidence) == {p['span_sha256'] for p in rendered['placements']}
    assert all(rendered['text'][p['start_char']:p['end_char']] == evidence[p['span_sha256']]
               for p in rendered['placements'])
    assert presentation.hydration_from_payload(hydration).identity_payload() == hydration
    altered = deepcopy(hydration)
    altered['sections'][0]['evidence'][0]['text'] += ' invented'
    with pytest.raises(ValueError):
        presentation.hydration_from_payload(altered)


def test_audit_source_order_matches_ingestion_order_including_repeated_body():
    body, summaries = body_and_atoms()
    sessions = [source(body, 0, '2026-02-02'), source(body, 1, '2026-02-03')]
    history = materialize_history(sessions, load_body=lambda _: body,
        load_summaries=lambda _: summaries, compiler_identity='fixture')
    original = TranscriptOrder(history.turns.values())
    rebuilt = presentation.source_order(sessions, lambda _: body)
    assert rebuilt.receipt_sha256 == original.receipt_sha256
    assert dict(rebuilt.positions) == dict(original.positions)
    altered = deepcopy(body)
    altered['turns'][0]['text'] += ' changed'
    with pytest.raises(ValueError, match='source body identity'):
        presentation.source_order(sessions, lambda _: altered)
