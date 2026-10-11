"""Shipping dialogue policy, snapshot lifecycle, and exact-source learning."""
from dataclasses import replace
import json

import numpy as np
import pytest

from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.runtime.config import dialogue_policy
from tests.test_native_spine_context_routing import parent_fixture, QUERY, DATED


def prepared(policy=None, *, future=False):
    history, semantic, hierarchy, encoder = parent_fixture(future=future)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    leaves = [s for s in hierarchy.sections if not s.child_section_ids]
    matrix = np.array([[1., 0.]] * len(leaves), dtype=np.float32)
    memory.configure_dialogue(policy or DialogueRetrievalPolicy(), matrix,
                             embedding_identity=memory._query_embedding_identity)
    return history, memory, encoder


@pytest.mark.parametrize('value', [[], {'unknown': 1}, {'version': 2}, {'version': True},
    {'episode_limit': 0}, {'episode_limit': 257}, {'chronology_bands': 13},
    {'neighbor_turns': 5}, {'max_context_tokens': True}, {'max_context_tokens': 65537},
    {'max_raw_spans': -1}, {'closing_responses': 'false'}, {'receipt_sha256': 'changed'}])
def test_invalid_policy_fails_before_model_start(value):
    with pytest.raises(ValueError):
        DialogueRetrievalPolicy.from_mapping(value)


def test_profiles_roundtrip_and_explicit_rollback(tmp_path):
    path = tmp_path/'policy.json'
    policy = DialogueRetrievalPolicy(episode_limit=8, max_context_tokens=4096)
    path.write_text(json.dumps(policy.identity_payload()), encoding='utf-8-sig')
    assert dialogue_policy('dialogue', path) == policy
    assert dialogue_policy('dialogue') == DialogueRetrievalPolicy()
    assert dialogue_policy() is None
    with pytest.raises(ValueError, match='requires'):
        dialogue_policy('cap8', path)
    assert replace(policy, neighbor_turns=0, receipt_sha256='').receipt_sha256 != policy.receipt_sha256


def test_prepared_route_matches_assay_with_one_query_embedding_and_no_raw_routing():
    history, memory, encoder = prepared(future=True)
    options = dict(max_direct=1, lexical_reserve=0, protected_direct=0, max_additions=0)
    # Independent prototype entry point is the evaluated control.
    base = memory.router.route_vector(QUERY, DATED, np.array([1., 0.]),
        embedding_identity=memory.router.semantic.embedding_identity, **options)
    plan, _ = memory.dialogue.router.route(base, np.array([1., 0.]),
        embedding_identity=memory._query_embedding_identity, closing_responses=True)
    expected = hydrate_section_plan(plan, load_turn=history.get_turn, max_context_tokens=6144, max_raw_spans=128)
    result = memory.retrieve(QUERY, DATED, **options)
    assert encoder.calls == [QUERY]
    assert result.hydration == expected
    assert result.routing.expanded == result.hydration.plan
    assert result.routing.policy == DialogueRetrievalPolicy()
    assert result.routing.raw_reads_during_routing == result.routing.query_qwen_passes == 0
    assert not result.routing.exhaustive
    assert result.routing.eligible_episodes == 2  # Future occurrence excluded.
    assert result.hydration.sections[0].section == history.atoms[1]  # Closing response.
    with pytest.raises(ValueError, match='scope or evidence'):
        replace(result.routing, exhaustive=True, receipt_sha256='')


def test_explicit_budgets_and_corrupt_raw_fail_closed():
    history, memory, _ = prepared(DialogueRetrievalPolicy(max_context_tokens=100, max_raw_spans=1))
    limits = retrieval_limits(memory, {})
    assert limits == dict(max_context_tokens=100, max_raw_spans=1)
    result = memory.retrieve(QUERY, DATED, **limits)
    assert result.hydration.context_token_count <= 100
    assert sum(len(s.evidence) for s in result.hydration.sections) <= 1
    assert retrieval_limits(memory, {'max_context_tokens': 80})['max_context_tokens'] == 80
    memory.load_turn = lambda tid: history.get_turn(tid).model_copy(update={'text': 'Changed'})
    assert not memory.retrieve(QUERY, DATED).hydration.sections


def test_prepared_index_cannot_follow_a_changed_encoder_or_router():
    _, memory, encoder = prepared()
    encoder.execution_identity = {'changed': True}
    with pytest.raises(ValueError, match='encoder changed'):
        memory.retrieve(QUERY, DATED)
    assert not encoder.calls
    _, memory, encoder = prepared()
    memory.router = object()
    with pytest.raises(ValueError, match='another native snapshot'):
        memory.retrieve(QUERY, DATED)
    assert not encoder.calls


def test_resident_recall_reopen_reuses_summary_vectors_and_preserves_provenance(tmp_path, monkeypatch):
    from tests.test_engineering_research_resident import setup, STAMP
    from memory_condense.application.chat_session import ChatEvent
    from memory_condense.runtime.resident import ResidentNativeBackend
    backend, events, _, actor = setup(tmp_path, monkeypatch)
    policy = DialogueRetrievalPolicy(max_context_tokens=160, max_raw_spans=3)
    backend.dialogue_policy = policy
    try:
        backend.sync(events)
        vectors_before = set(backend.vectors)
        encoded = []
        def embed(texts):
            encoded.extend(texts)
            return np.asarray([[1., 0.]] * len(texts), dtype=np.float32)
        monkeypatch.setattr(backend.encoder, 'embed_queries', embed)
        first = backend.recall_published('Deployment?')
        direct = backend.recall('Deployment?')
        assert first['routing'] == direct['routing']
        assert first['hydration'] == direct['hydration']
        assert first['references'] == direct['references']
        assert first['routing']['policy'] == policy.identity_payload()
        assert first['hydration']['max_context_tokens'] == 160
        assert first['hydration']['max_raw_spans'] == 3
        assert first['references']
        assert encoded == []  # Recall embeds the query only.
        events.extend([ChatEvent('new-user', 'user', 'Keep migration pending.', STAMP),
                       ChatEvent('new-answer', 'assistant', 'Migration is still pending.', STAMP)])
        backend.sync(events)
        assert not set(encoded) & vectors_before
        after = backend.recall_published('Deployment?')
        assert after['published_events'] == len(events)
        assert after['routing']['hierarchy_sha256'] != first['routing']['hierarchy_sha256']
    finally:
        backend.close()
    # Cold admission must reuse durable vectors, with no re-summarization.
    reopened = ResidentNativeBackend(tmp_path, tmp_path/'live', actor, dialogue_policy=policy)
    try:
        monkeypatch.setattr(backend.encoder, 'embed_queries', lambda _: pytest.fail('Unexpected summary embedding'))
        reopened.sync(events)
        assert reopened.recall_published('Deployment?') == after
    finally:
        reopened.close()


def test_failed_dialogue_preparation_keeps_previous_published_prefix(tmp_path, monkeypatch):
    from tests.test_engineering_research_resident import setup, STAMP
    from memory_condense.application.chat_session import ChatEvent
    backend, events, _, _ = setup(tmp_path, monkeypatch)
    backend.dialogue_policy = DialogueRetrievalPolicy()
    try:
        backend.sync(events)
        before = backend.recall_published('Deployment?')
        acknowledged = backend.last_reopen
        prepare = backend._prepare_dialogue
        def fail(_):
            raise ValueError('simulated invalid summary vectors')
        monkeypatch.setattr(backend, '_prepare_dialogue', fail)
        events.append(ChatEvent('later', 'user', 'The deployment is still a plan.', STAMP))
        with pytest.raises(ValueError, match='invalid summary vectors'):
            backend.sync(events)
        assert backend.last_reopen == acknowledged
        assert backend.recall_published('Deployment?') == before
        monkeypatch.setattr(backend, '_prepare_dialogue', prepare)
        backend.sync(events)
        assert backend.recall_published('Deployment?')['published_events'] == len(events)
    finally:
        backend.close()


def test_dialogue_io_credits_original_memories_and_keeps_recall_ancestry(tmp_path):
    from tests.test_chat_native import NativeFixtureBackend, graph
    from memory_condense.application.chat_session import ChatEvent, ChatSession
    class Backend(NativeFixtureBackend):
        def sync(self, events):
            super().sync(events)
            memory = self.app._load_native_spine()[2]
            matrix = np.asarray([[1., 0.]] * len(memory.router.leaf_atoms), dtype=np.float32)
            memory.configure_dialogue(DialogueRetrievalPolicy(), matrix,
                                      embedding_identity=memory._query_embedding_identity)
    backend = Backend(tmp_path/'memory')
    with ChatSession(tmp_path/'chat', 'session', backend) as chat:
        chat.ingest_many([ChatEvent('past-1', 'user', 'Use one shared event stream.'),
                          ChatEvent('past-2', 'assistant', 'Keep original source pointers.'),
                          ChatEvent('now', 'user', 'Continue the design.')])
        first = chat.recall('Original design?', packet_id='first', input_event_id='now')
        assert first.references
        trace = chat.event('_chat:recall:first').metadata['_chat']['retrieval']
        assert trace['policy'] == DialogueRetrievalPolicy().identity_payload()
        assert trace['mode'] == 'dialogue' and not trace['exhaustive']
        chat.feedback('first', successful=True)
        chat.flush()
        assert dict(graph(tmp_path/'memory')['nodes']) == {'past-1': 1, 'past-2': 1, 'now': 1}
        second = chat.recall('Original design?', packet_id='second', input_event_id='now')
        copies = [r for r in second.references if r['span']['turn_id'] == '_chat:recall:first']
        assert copies and not copies[0]['independent']
        assert copies[0]['original_references'] == list(first.references)
        chat.flush()
        old_graph = graph(tmp_path/'memory')
        chat._call(lambda: backend.learn(replace(second, references=tuple(copies)), access_event_id='copy-only'))
        assert graph(tmp_path/'memory') == old_graph
