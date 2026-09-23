from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools import run_hierarchy_after_local_compilation as handoff
from tools.matched_eval.artifacts import publish_sealed_json,read_sealed_json


def test_live_process_identity_includes_creation_time_and_access_errors_are_not_terminal():
    row = {'pid':123,'create_time':10.}
    assert handoff.dependency_alive(row,lambda _:SimpleNamespace(is_running=lambda:True,create_time=lambda:10.))
    assert not handoff.dependency_alive(row,lambda _:SimpleNamespace(is_running=lambda:True,create_time=lambda:11.))
    def missing(_): raise psutil.NoSuchProcess(123)
    assert not handoff.dependency_alive(row,missing)
    def denied(_): raise psutil.AccessDenied(123)
    with pytest.raises(psutil.AccessDenied):
        handoff.dependency_alive(row,denied)


def test_observation_deadline_does_not_turn_a_live_compiler_into_completion():
    elapsed = [0.]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds
    with pytest.raises(TimeoutError,match='still live'):
        handoff.wait_for_terminal({'pid':123},25,sleep=sleep,now=lambda:elapsed[0],alive=lambda _:True)
    assert sleeps==[10,10,5]


def test_wait_proceeds_only_after_observed_terminal_state():
    elapsed = [0.]
    def sleep(seconds): elapsed[0]+=seconds
    handoff.wait_for_terminal({'pid':123},30,sleep=sleep,now=lambda:elapsed[0],alive=lambda _:elapsed[0]<20)
    assert elapsed[0]==20


def test_incomplete_parent_population_cannot_release_evaluation(tmp_path,monkeypatch):
    preflight = SimpleNamespace(payload={'dependency':{},'compiler_root':str(tmp_path)})
    monkeypatch.setattr(handoff,'dependency_alive',lambda _:False)
    publish_sealed_json(tmp_path/'parents'/'populations'/'partial.json',{'complete_namespace_population':False})
    with pytest.raises(ValueError,match='complete bound ten-memory'):
        handoff.completed_population(preflight)
    monkeypatch.setattr(handoff,'dependency_alive',lambda _:True)
    with pytest.raises(ValueError,match='still running'):
        handoff.completed_population(preflight)


class Client:
    def __init__(self,finish_reason='stop'):
        self.calls = []
        self.closed = False
        self.finish_reason = finish_reason
        self.chat = SimpleNamespace(completions=self)
    def create(self,**kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish_reason,
            message=SimpleNamespace(content='OK'))])
    def close(self): self.closed=True


def test_readiness_uses_only_two_synthetic_calls_and_never_retries_existing_reservations(tmp_path):
    preflight,_ = publish_sealed_json(tmp_path/'preflight.json',{'fixture':True})
    client = Client()
    result = handoff.readiness(tmp_path,preflight,lambda:client)
    assert [c['model'] for c in client.calls]==list(handoff.MODELS)
    assert all(c['messages']==handoff.READINESS_MESSAGES and c['max_tokens']==64 for c in client.calls)
    assert result.payload['new_provider_calls']==2 and client.closed
    repeated = Client()
    with pytest.raises(FileExistsError):
        handoff.readiness(tmp_path,preflight,lambda:repeated)
    assert not repeated.calls and repeated.closed


def test_unfinished_readiness_stops_after_first_call(tmp_path):
    preflight,_ = publish_sealed_json(tmp_path/'preflight.json',{'fixture':True})
    client = Client('length')
    with pytest.raises(ValueError,match='readiness failed'):
        handoff.readiness(tmp_path,preflight,lambda:client)
    assert len(client.calls)==1 and client.closed
    assert not (tmp_path/'readiness'/'complete.json').exists()


def setup_run(tmp_path,monkeypatch):
    root = tmp_path/'handoff'
    evaluation = tmp_path/'evaluation'
    preflight,_ = publish_sealed_json(root/'preflight.json',{
        'dependency':{'pid':123},'maximum_wait_seconds':10,
        'evaluation_root':str(evaluation),'source_root':str(tmp_path/'source')})
    population,_ = publish_sealed_json(tmp_path/'population.json',{'complete_fixture':True})
    events = []
    monkeypatch.setattr(handoff,'validate_preflight',lambda _:preflight)
    monkeypatch.setattr(handoff,'wait_for_terminal',lambda *_:events.append('terminal'))
    monkeypatch.setattr(handoff,'completed_population',lambda _:events.append('population') or population)
    monkeypatch.setattr(handoff,'require_idle',lambda:events.append('idle'))
    monkeypatch.setattr(handoff,'readiness',lambda *_:events.append('readiness'))
    def child(command,check):
        assert check is True
        phase = command[5]
        events.append(phase)
        if phase=='run':
            report,_ = publish_sealed_json(evaluation/'joint-report.json',{'target_gate_passed':False})
            publish_sealed_json(evaluation/'complete.json',{'joint_report_sha256':report.sha256,'target_gate_passed':False})
    monkeypatch.setattr(handoff.subprocess,'run',child)
    return root,events


def test_handoff_orders_complete_admission_prepare_readiness_streams_and_replay(tmp_path,monkeypatch):
    root,events = setup_run(tmp_path,monkeypatch)
    handoff.run(root,True)
    assert events==['terminal','population','idle','prepare','readiness','run','replay']
    result = read_sealed_json(root/'complete.json')
    assert result.payload['target_gate_passed'] is False
    assert result.payload['judge_replay_completed'] is True
    with pytest.raises(FileExistsError):
        handoff.run(root,True)
    assert len(events)==7


def test_dependency_failure_never_prepares_or_sends_answers(tmp_path,monkeypatch):
    root,events = setup_run(tmp_path,monkeypatch)
    def unavailable(_): raise ValueError('incomplete parent trees')
    monkeypatch.setattr(handoff,'completed_population',unavailable)
    with pytest.raises(ValueError,match='incomplete parent'):
        handoff.run(root,True)
    assert events==['terminal']
    failure = read_sealed_json(root/'failure.json')
    assert failure.payload['automatic_retry_performed'] is False
    assert not (root/'complete.json').exists()


def test_execution_requires_existing_provider_authorization_flag(tmp_path,monkeypatch):
    root,events = setup_run(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='provider flag'):
        handoff.run(root,False)
    assert not events and not (root/'execution.reserved').exists()
