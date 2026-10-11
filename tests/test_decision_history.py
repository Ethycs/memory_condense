from dataclasses import replace
from types import SimpleNamespace

import pytest

from memory_condense.application.chat_native import native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.decision_history import annotate, snapshot_revisions, StoredRevision
from memory_condense.application.history_intent import HistoryIntent, FORMAT
from memory_condense.application.tactic_router import BGETacticRouter
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn, CreateOp, SupersedeOp, Provenance, MemoryType
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tests.test_tactic_router import Encoder, model
from tests.test_dialogue_retrieval import prepared, QUERY, DATED


def history_model(encoder, **changes):
    value = dict(format=FORMAT, dimension=2, embedding_identity=summary_embedding_identity(encoder),
        input_format='bounded-recent-v1', weights=[0.,0.], intercept=10., threshold=.8)
    value.update(changes)
    return value | dict(receipt_sha256=identity_sha256(value))


def history_router(encoder):
    return BGETacticRouter(encoder, model(encoder), history_artifact=history_model(encoder))


def test_history_head_reuses_embedding_and_custom_old_head_remains_unchanged():
    encoder = Encoder()
    decision = history_router(encoder).prioritize('Which attempted fixes worked?')
    assert decision.preferred_tactic == 'decision_history' and not decision.abstained
    assert decision.history_intent['selected'] and decision.embedding_calls == 1
    assert len(encoder.calls) == 1
    old = BGETacticRouter(encoder,model(encoder)).prioritize('Which attempted fixes worked?')
    assert old.preferred_tactic == 'cap8' and old.history_intent is None


@pytest.mark.parametrize('changes', [dict(weights=[0.]),dict(threshold=0),dict(threshold=True),
    dict(intercept=float('inf')),dict(input_format='other'),dict(embedding_identity='other')])
def test_invalid_history_models_fail_before_inference(changes):
    encoder = Encoder()
    with pytest.raises((ValueError,TypeError)):
        BGETacticRouter(encoder,model(encoder),history_artifact=history_model(encoder,**changes))
    assert not encoder.calls


def test_automatic_history_packet_preserves_baseline_spans_and_source_scope():
    history,memory,encoder = prepared(future=True)
    # Same retrieval, different evidence annotations; compare independently.
    from tests.test_automatic_tactics import head
    memory.configure_tactics(head(encoder,'overview'))
    baseline = memory.retrieve(QUERY,DATED)
    memory.configure_tactics(history_router(encoder))
    result = memory.retrieve(QUERY,DATED)
    assert result.hydration == baseline.hydration
    events = [ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat(),dict(source_id=t.source_id))
              for t in {a.spans[0].turn_id:history.get_turn(a.spans[0].turn_id) for a in history.atoms}.values()]
    app = SimpleNamespace(transcript=history,retrieve_native_spine=memory.retrieve,decision_revisions=())
    packet = native_packet(app,QUERY,DATED,events)
    assert packet['retrieval']['mode'] == 'decision_history'
    receipt = packet['retrieval']['decision_history']
    assert receipt['raw_reads'] == receipt['provider_calls'] == 0
    assert receipt['token_count'] <= receipt['max_tokens']
    assert packet['retrieval']['packet_token_count'] == count_tokens(packet['text'])
    assert 'not effective dates' in packet['text']
    assert 'evidence_id' in packet['text'] and receipt['revision_links'] == []
    assert packet['references'] == [dict(section_id=s.section.section_id,
        span=e.span.identity_payload(),independent=True) for s in baseline.hydration.sections for e in s.evidence]
    assert not any('future' in r['span']['source_id'] for r in packet['references'])
    assert encoder.calls == [QUERY]*3


def evidence():
    events = [ChatEvent('first','user','Use batch ingestion.', '2026-01-01T10:00:00+00:00'),
              ChatEvent('second','assistant','I propose running it in the background.', '2026-01-01T10:01:00+00:00'),
              ChatEvent('third','user','Replace batch ingestion with streaming.', '2026-01-03T10:00:00+00:00'),
              ChatEvent('fourth','tool','Test failed: index did not advance.', '2026-01-03T10:01:00+00:00')]
    refs = []
    for e in events:
        turn = Turn(turn_id=e.event_id,role=e.role,text=e.text,source_id='s',created_at=e.created_at)
        refs.append(dict(section_id=e.event_id,span=RawSectionSpan.from_turn(turn).identity_payload(),independent=True))
    return events,refs


def test_chronology_role_and_episode_links_do_not_invent_revision_or_success():
    events,refs = evidence()
    text,labels,receipt = annotate(list(reversed(refs)),events)
    assert labels == {3:'D001',2:'D002',1:'D003',0:'D004'}
    rows = receipt['annotations']
    assert [r['turn'] for r in rows] == [e.event_id for e in events]
    assert [r['follows'] for r in rows] == [None,'D001',None,'D003']
    assert rows[1]['attribution']=='assistant_report' and rows[3]['attribution']=='tool_observation'
    assert receipt['revision_links']==[] and not receipt['exhaustive']
    assert 'unverified, not failed' in text
    assert all('status' not in r for r in rows)


def test_revision_requires_both_exact_quotes_and_independent_delivered_support():
    events,refs = evidence()
    revision = StoredRevision('old','new',(('first','Use batch ingestion.'),),
                              (('third','Replace batch ingestion with streaming.'),))
    _,_,receipt = annotate(refs,events,revisions=(revision,))
    assert receipt['revision_links'][0]['before']==['D001']
    assert receipt['revision_links'][0]['after']==['D003']
    for smaller in (refs[:2],refs[2:],[dict(r,independent=False) for r in refs]):
        assert annotate(smaller,events,revisions=(revision,))[2]['revision_links']==[]
    wrong = replace(revision,after=(('third','Absent source quote.'),))
    assert annotate(refs,events,revisions=(wrong,))[2]['revision_links']==[]
    # A source pointer to the right turn cannot certify a quote outside the delivered slice.
    t = Turn(turn_id='third',role='user',text=events[2].text,source_id='s',created_at=events[2].created_at)
    partial = dict(refs[2],span=RawSectionSpan.from_turn(t,start_char=0,end_char=7).identity_payload())
    assert annotate([refs[0],partial],events,revisions=(revision,))[2]['revision_links']==[]


def test_annotation_budget_is_exact_and_omissions_are_visible():
    events,refs = evidence()
    for budget in (1,100,300,500,2048):
        text,labels,receipt = annotate(refs,events,max_tokens=budget)
        assert count_tokens(text)==receipt['token_count']<=budget
        assert len(labels)==len(refs)
        assert len(receipt['annotations'])+receipt['omitted_annotations']==len(refs)


def test_revision_snapshot_uses_existing_store_without_mutating_it(tmp_path):
    from memory_condense.application.condenser import MemoryCondenser
    encoder = Encoder()
    encoder.dim = 2
    encoder.embed_chunks = lambda chunks: [c.model_copy(update={'embedding':[1.,0.]}) for c in chunks]
    with MemoryCondenser(tmp_path/'store',auto_extract=False,embedder=encoder) as app:
        old = app.memory.create(CreateOp(type=MemoryType.DECISION,content='Use batches',
            provenance=[Provenance(turn_id='first',quote='Use batch ingestion.')]))
        new = app.memory.supersede(SupersedeOp(mem_id=old.mem_id,
            replacement=CreateOp(type=MemoryType.DECISION,content='Use streaming',
                provenance=[Provenance(turn_id='third',quote='Replace batch ingestion with streaming.')])) )
        before = app.memory.list_items(status=None)
        captured = snapshot_revisions(app.memory)
        assert len(captured)==1 and captured[0].predecessor==old.mem_id and captured[0].successor==new.mem_id
        assert app.memory.list_items(status=None)==before
        events,refs=evidence()
        assert annotate(refs,events,revisions=captured)[2]['revision_links']


def test_forced_history_uses_published_snapshot_and_rollback_skips_annotations(tmp_path,monkeypatch):
    from tests.test_engineering_research_resident import setup
    from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
    backend,events,_,_=setup(tmp_path,monkeypatch)
    backend.retrieval_policy='decision_history'
    backend.dialogue_policy=DialogueRetrievalPolicy()
    try:
        backend.sync(events)
        published=backend.recall_published('Which fixes did we try?')
        direct=backend.recall('Which fixes did we try?')
        assert published['text']==direct['text']
        assert published['retrieval']['mode']=='decision_history'
        assert published['retrieval']['tactic']['reason']=='forced'
        assert backend._tactic_classifier is None
        memory=backend.app._load_native_spine()[2]
        result=memory.retrieve(QUERY,DATED,augment=False)
        assert result.tactic is None and result.hierarchy_context is None
    finally:
        backend.close()
