"""The handoff presentation must not change source scope or learning evidence."""
from types import SimpleNamespace

import pytest

from memory_condense.application.chat_native import native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.continuity import contract, evidence_labels, FORMAT
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from tests.test_automatic_tactics import head
from tests.test_decision_history import evidence, history_router
from tests.test_dialogue_retrieval import prepared, QUERY, DATED


def test_citation_manifest_preserves_exact_addresses_and_copy_attribution():
    events,refs = evidence()
    refs[1] = dict(refs[1],independent=False,original_references=[refs[0]])
    labels = evidence_labels(refs,events)
    text,receipt = contract(refs,labels,hierarchy={},source_scope=['s'],
                            index_sha256='index',as_of_day='2026-01-03')
    assert receipt['format'] == FORMAT
    assert receipt['raw_reads'] == receipt['provider_calls'] == 0
    assert receipt['snapshot']['source_ids'] == ['s']
    assert not receipt['coverage']['exhaustive']
    assert receipt['token_count'] == count_tokens(text) < 700
    assert receipt['evidence'][1]['kind'] == 'recalled_copy'
    for entry in receipt['evidence']:
        original = refs[entry['reference_index']]
        assert entry['span_sha256'] == original['span']['receipt_sha256']
    assert receipt['receipt_sha256'] == identity_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    assert contract(refs,labels,hierarchy={},source_scope=['s'],
                    index_sha256='index',as_of_day='2026-01-03') == (text,receipt)
    assert refs[1]['original_references'] == [refs[0]]


@pytest.mark.parametrize('labels', [{},{0:'D001'},{0:'D001',1:'D001',2:'D003',3:'D004'}])
def test_missing_or_ambiguous_citation_labels_are_rejected(labels):
    _,refs = evidence()
    with pytest.raises(ValueError,match='every delivered'):
        contract(refs,labels,hierarchy={},source_scope=['s'],index_sha256='i',as_of_day='2026-01-03')


@pytest.mark.parametrize('foreign_summary',[False,True])
def test_manifest_cannot_expand_authorized_source_scope(foreign_summary):
    events,refs = evidence()
    hierarchy = dict(references=[dict(source_id='other')]) if foreign_summary else {}
    with pytest.raises(ValueError,match='source scope'):
        contract(refs,evidence_labels(refs,events),hierarchy=hierarchy,
                 source_scope=['s'] if foreign_summary else [],index_sha256='i',as_of_day='2026-01-03')


@pytest.mark.parametrize('mode',['overview','history','focused'])
def test_continuity_is_integrated_without_new_retrieval_or_raw_reads(mode):
    history,memory,encoder = prepared(future=True)
    memory.configure_tactics(history_router(encoder) if mode=='history' else head(encoder,mode))
    baseline = memory.retrieve(QUERY,DATED)
    turns = {a.spans[0].turn_id:history.get_turn(a.spans[0].turn_id) for a in history.atoms}.values()
    events = [ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat(),dict(source_id=t.source_id)) for t in turns]
    app = SimpleNamespace(transcript=history,retrieve_native_spine=memory.retrieve,decision_revisions=())
    packet = native_packet(app,QUERY,DATED,events)
    expected = [dict(section_id=s.section.section_id,span=e.span.identity_payload(),independent=True)
                for s in baseline.hydration.sections for e in s.evidence]
    assert packet['references'] == expected
    assert packet['hydration'] == baseline.hydration.identity_payload()
    assert encoder.calls == [QUERY,QUERY]
    assert packet['retrieval']['packet_token_count'] == count_tokens(packet['text'])
    if mode=='focused':
        assert 'continuity' not in packet['retrieval']
        assert 'CONTINUITY OUTPUT' not in packet['text']
        return
    manifest = packet['retrieval']['continuity']
    assert manifest['snapshot']['index_sha256'] == baseline.hydration.plan.index_sha256
    assert manifest['snapshot']['source_ids'] == list(baseline.routing.baseline.eligible_source_ids)
    exact = [r for r in manifest['evidence'] if r['kind']=='exact']
    summaries = [r for r in manifest['evidence'] if r['kind']=='summary']
    assert len(exact) == len(expected)
    assert len(summaries) == len(packet['retrieval']['hierarchy_context']['references'])
    for r in exact:
        assert '"evidence_id": "'+r['label']+'"' in packet['text']
    for r in summaries:
        ref = packet['retrieval']['hierarchy_context']['references'][r['reference_index']]
        assert r['section_sha256'] == ref['section_sha256']
        assert '['+r['label']+']' in packet['text']
    assert 'future' not in str(manifest)
    assert 'Never default a suggestion to Accepted' in packet['text']

