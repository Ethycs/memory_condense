import json
import sqlite3

import pytest

from tools.engineering_research_gateway import read,save
from tools.evaluate_chat_io_local100 import prepare_resume


def stopped_run(tmp_path):
    old=tmp_path/'old'
    runtime=tmp_path/'old-runtime'
    (runtime/'chat').mkdir(parents=True)
    (runtime/'store').mkdir()
    (runtime/'store'/'preserved').write_text('original indexed history')
    question={'question_id':'q0','retrieval_query':'event'}
    response={'content':'original answer'}
    save(old/'run-plan.json',dict(runtime_root=str(runtime),questions=[question]*100))
    save(old/'actor.json',{'case_id':'test'})
    save(old/'bootstrap.json',{'initial_events':0})
    save(old/'shutdown.json',{'completed_answers':1})
    save(old/'answers/000.json',dict(ordinal=0,question=question,prediction='original answer',
                                   result={'response':response}))
    with sqlite3.connect(runtime/'chat/chat-events.sqlite') as db:
        db.executescript('CREATE TABLE events(sequence INTEGER PRIMARY KEY,event_id,role,text,metadata);'
                         'CREATE TABLE packets(packet_id,input_event_id); CREATE TABLE feedback(successful,applied);')
        ids=['local-q000','_chat:recall:local-a000','local-a000:assistant',
             '_chat:feedback:local-a000','local-q001','_chat:recall:local-a001','local-a001:error:failed']
        for i,event_id in enumerate(ids):
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(i,event_id,'assistant' if i==2 else 'user',
                'original answer' if i==2 else 'event',json.dumps({'response':response} if i==2 else {})))
        db.executemany('INSERT INTO packets VALUES(?,?)',
                       [('local-a000','local-q000'),('local-a001','local-q001')])
        db.execute('INSERT INTO feedback VALUES(1,1)')
    return old,runtime


def test_resume_clones_state_and_preserves_completed_outputs(tmp_path):
    old,source=stopped_run(tmp_path)
    root=tmp_path/'new'
    runtime=tmp_path/'new-runtime'
    original=(old/'answers/000.json').read_bytes()
    prepare_resume(root,runtime,old)
    plan=read(root/'run-plan.json')
    assert plan['resume']['retained_answers']==1
    assert plan['resume']['restored_events']==7
    assert plan['resume']['recovery_packet_id']=='local-a001-recovery-1'
    assert plan['resume']['extra_events']==2 and plan['resume']['extra_packets']==1
    assert (root/'answers/000.json').read_bytes()==original
    assert (old/'answers/000.json').read_bytes()==original
    assert (runtime/'store/preserved').read_text()=='original indexed history'
    assert source!=runtime


def test_repeated_recovery_preserves_all_failed_packets_and_answers(tmp_path):
    old,_=stopped_run(tmp_path)
    root,runtime=tmp_path/'first',tmp_path/'first-runtime'
    prepare_resume(root,runtime,old)
    save(root/'bootstrap.json',{'initial_events':0})
    save(root/'shutdown.json',{'completed_answers':1})
    with sqlite3.connect(runtime/'chat/chat-events.sqlite') as db:
        db.execute('INSERT INTO events VALUES(7,?,?,?,?)',
                   ('_chat:recall:local-a001-recovery-1','system','recall','{}'))
        db.execute('INSERT INTO events VALUES(8,?,?,?,?)',
                   ('local-a001:error:failed-again','system','error','{}'))
        db.execute('INSERT INTO packets VALUES(?,?)',('local-a001-recovery-1','local-q001'))
    final=tmp_path/'second'
    prepare_resume(final,tmp_path/'second-runtime',root,inline_envelope_retries=2)
    plan=read(final/'run-plan.json')
    assert plan['resume']['extra_events']==4 and plan['resume']['extra_packets']==2
    assert plan['resume']['recovery_packet_id']=='local-a001-recovery-2'
    assert plan['resume']['failed_packet_ids']==['local-a001','local-a001-recovery-1']
    assert plan['inline_envelope_retries']==2
    assert (final/'answers/000.json').read_bytes()==(old/'answers/000.json').read_bytes()


def test_resume_rejects_disagreement_with_captured_answer(tmp_path):
    old,runtime=stopped_run(tmp_path)
    with sqlite3.connect(runtime/'chat/chat-events.sqlite') as db:
        db.execute("UPDATE events SET text='different answer' WHERE event_id='local-a000:assistant'")
    with pytest.raises(AssertionError):
        prepare_resume(tmp_path/'new',tmp_path/'new-runtime',old)
    assert not (tmp_path/'new').exists()
