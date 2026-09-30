import json
from pathlib import Path

import pytest

from tools.engineering_research_gateway import check_budget
from tools.engineering_research_execution import execute_tests
from tools.run_engineering_research_battery import safe_path, execute_action, messages_for, parse_json, validate_review


def test_budget_counts_reserved_calls_and_tokens():
    plan={'budgets':{'actor':{'calls':1,'prompt_cap':100,'output_cap':32,'input_token_budget':100}}}
    job={'kind':'actor','messages':[{'role':'user','content':'test'}],'max_tokens':32}
    assert check_budget(plan,[],job)>0
    with pytest.raises(ValueError,match='campaign'):
        check_budget(plan,[{'kind':'actor','prompt_tokens':1}],job)
    with pytest.raises(ValueError,match='request'):
        check_budget(plan,[],dict(job,max_tokens=33))


@pytest.mark.parametrize('path',['../rubric.json','/outside','C:/secret','a\\b'])
def test_tools_cannot_escape_workspace(tmp_path,path):
    with pytest.raises(ValueError):
        safe_path(tmp_path,path)


def test_write_many_validates_every_path_before_mutating(tmp_path):
    with pytest.raises(ValueError):
        execute_action(tmp_path,{'action':'write_many','artifacts':{'good.py':'pass','../bad.py':'pass'}})
    assert not (tmp_path/'good.py').exists()


def test_actor_frame_matches_between_arms():
    actor=dict(case_id='E01',domain='engineering',task_kind='python_component',original_request='Do it.',
               current_turn_id='x:T0002',task='Build it',deliverables=['a.py'],public_checks=[],external_dependencies='none')
    a=messages_for(actor,'full source',[])
    b=messages_for(actor,'memory excerpt',[])
    assert a[0]==b[0]
    assert a[1]['content'].replace('full source','memory excerpt')==b[1]['content']


def test_candidate_tests_execute_with_limits_and_cannot_read_private_file(tmp_path):
    workspace=tmp_path/'workspace'
    workspace.mkdir()
    secret=tmp_path/'private.txt'
    secret.write_text('private rubric')
    (workspace/'test_candidate.py').write_text('''import unittest, os
class Check(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(3+4,7)
    def test_no_secret(self):
        with self.assertRaises(PermissionError):
            open('''+repr(str(secret))+''').read()
    def test_no_network(self):
        import socket
        with self.assertRaises(PermissionError):
            socket.socket()
    def test_mock(self):
        from unittest.mock import Mock
        self.assertEqual(Mock(return_value=3)(),3)
    def test_no_credentials(self):
        self.assertNotIn('LITELLM_KEY',os.environ)
''',encoding='utf-8')
    result=execute_tests(workspace)
    assert result['exit_code']==0,result
    assert 'Ran 5 tests' in result['output']
    assert result['limits']['memory_mib']==512


def test_bad_candidate_is_recorded_as_failure(tmp_path):
    (tmp_path/'test_bad.py').write_text('import unittest\nclass Bad(unittest.TestCase):\n def test_bad(self): self.assertEqual(1,2)\n')
    result=execute_tests(tmp_path)
    assert result['exit_code']!=0 and 'FAILED' in result['output']


def test_acceptance_checks_execute_in_restricted_process(tmp_path):
    (tmp_path/'feedback_loop.py').write_text('def run_cycle(state, trainer, metrics, analyzers, evolvers, scheduler, orchestrator):\n for fn in (trainer, metrics, analyzers, evolvers, scheduler, orchestrator, trainer):\n  state=fn(state)\n return state\n')
    result=execute_tests(tmp_path,acceptance_case='E08')
    assert result['exit_code']==0,result
    assert json.loads(result['output'])['all_passed']


def test_repeated_identical_source_fragments_compile_once_with_distinct_raw_addresses(tmp_path,monkeypatch):
    from tools import engineering_research_memory as memory
    calls=[]
    class FakeGateway:
        def __init__(self,root):pass
        def call(self,kind,messages,**kwargs):
            fragments=json.loads(messages[1]['content'])['fragments']
            calls.append(fragments)
            return {'request_sha256':'0'*64,'content':json.dumps({'atoms':[
                    {'label':f['label'],'summary':'User requires deterministic output.','support':['Make output deterministic.']}
                for f in fragments]})}
    monkeypatch.setattr(memory,'Gateway',FakeGateway)
    rows=[{'turn_id':f'source:T{i:04d}','role':'user','text':'Make output deterministic. '*64} for i in range(2)]
    atoms=memory.Compiler(tmp_path,'E01/test').atoms(rows,'source','2026-01-01T00:00:00+00:00')
    assert len(calls)==1 and len(calls[0])==1
    assert len(atoms)==2 and atoms[0].spans[0].turn_id!=atoms[1].spans[0].turn_id
    assert atoms[0].summary==atoms[1].summary


def test_compiler_fast_forward_only_summarizes_new_input_and_output(tmp_path, monkeypatch):
    from tools import engineering_research_memory as memory
    calls = []
    class Gateway:
        def __init__(self, root):
            pass
        def call(self, kind, messages, **kwargs):
            assert kind == 'raw'
            fragments = json.loads(messages[1]['content'])['fragments']
            calls.extend(f['fragment'] for f in fragments)
            return {'request_sha256': '0'*64, 'content': json.dumps({'atoms': [
                dict(label=f['label'], summary='A stored event summary.', support=[f['fragment'][:40]]) for f in fragments]})}
    monkeypatch.setattr(memory, 'Gateway', Gateway)
    compiler = memory.Compiler(tmp_path, 'live')
    rows = [dict(turn_id='past', role='user', text='Original project requirements. '*64)]
    stamp = '2026-09-29T00:00:00+00:00'
    initial = compiler.atoms(rows, 'source', stamp)
    rows.append(dict(turn_id='input', role='user', text='Deploy to the cobalt cluster. '*64))
    after_input = compiler.atoms(rows, 'source', stamp)
    rows.append(dict(turn_id='output', role='assistant', text='The migration identifier is migration-482. '*64))
    after_output = compiler.atoms(rows, 'source', stamp)
    assert calls == [r['text'] for r in rows]
    assert after_input[:1] == initial and after_output[:2] == after_input
    assert [a.spans[0].turn_id for a in after_output] == ['past', 'input', 'output']
    assert compiler.atoms(rows, 'source', stamp) == after_output
    assert len(calls) == 3


def test_exhausted_arm_is_saved_as_failure_without_generation_retry(tmp_path, monkeypatch):
    from tools import run_engineering_research_battery as runner
    calls=[]
    def fail(*args):
        calls.append(args)
        raise RuntimeError('support validation exhausted')
    monkeypatch.setattr(runner,'run_arm',fail)
    monkeypatch.setattr(runner.battery,'read_binding',lambda _: {'deliverables':['analysis.md']})
    result=runner.run_arm_recording_failure(tmp_path,{'id':'R01','actor':{}},'memory')
    assert len(calls)==1
    assert not result['finished'] and not result['structural']['structurally_complete']
    assert result['actor_calls']==0 and 'support validation exhausted' in result['lifecycle_error']
    assert (tmp_path/'cases/R01/memory/result.json').is_file()


def test_resume_never_regenerates_completed_grading(tmp_path, monkeypatch):
    from tools import run_engineering_research_battery as runner
    folder=tmp_path/'cases/E02/grading'
    folder.mkdir(parents=True)
    (folder/'reviews.json').write_text('{}')
    monkeypatch.setattr(runner.battery,'read_binding',lambda _: pytest.fail('Completed grading must be skipped'))
    runner.grade_pair(tmp_path,{'id':'E02'}, {})
