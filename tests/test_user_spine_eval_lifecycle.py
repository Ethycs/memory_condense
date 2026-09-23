"""Exercise the pending answer/judge lifecycle without a network or real gold."""

import json
from types import SimpleNamespace

import pytest

from tests.test_fast_completion_runtime import _FakeClient
from tools import evaluate_user_spine_real_pilot as pilot
from tools import run_hot_reduced30_answer_judge as lifecycle
from tools.matched_eval.artifacts import SealedArtifactError, publish_sealed_json, read_sealed_json


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'synthetic-lifecycle-only'
    source = tmp_path/'synthetic-source'
    publish_sealed_json(source/'preflight.json', {'binding': {'synthetic': True}})
    preflight, _ = publish_sealed_json(root/'preflight.json', {
        'source_root': str(source), 'gateway_url': 'https://invalid.test',
        'questions': [{'id': 'locked_q86', 'question_id': 'SYNTHETIC_QUESTION'}],
        'arms': ['first', 'second']})
    rows = [{'case_id': case_id, 'group': group, 'arm': arm, 'question': question,
             'messages': [{'role': 'user', 'content': question}]} for case_id, group, question in (
        ('locked_q86', 'locked_benchmark_development', 'SYNTHETIC_BENCHMARK_QUERY'),
        ('synthetic_probe', 'source_derived_development', 'SYNTHETIC_DEVELOPMENT_QUERY'))
        for arm in ('first', 'second')]
    publish_sealed_json(root/'selection.json', {'preflight_sha256': preflight.sha256, 'rows': rows})
    refs = tmp_path/'synthetic-references.json'
    refs.write_text(json.dumps({'source_turn_population_sha256': 'synthetic-population',
        'cases': [{'id': 'synthetic_probe', 'turn_ordinal': 0,
                   'quote': 'EXACT_SUPPORT', 'answer': 'DEVELOPMENT_GOLD_CANARY'}]}))
    monkeypatch.setattr(pilot, 'REFERENCES', refs)
    monkeypatch.setattr(pilot, '_turns', lambda binding: (
        [SimpleNamespace(role='user', text='EXACT_SUPPORT')], 'synthetic-population'))
    gold_calls = []

    def load_gold(*args):
        # Gold cannot be loaded until immutable predictions exist.
        assert len(read_sealed_json(root/'answers.json').payload['rows']) == 4
        gold_calls.append(True)
        return 'synthetic-benchmark-population', [None]*86 + [SimpleNamespace(
            question_id='SYNTHETIC_QUESTION', answer='BENCHMARK_GOLD_CANARY')]

    monkeypatch.setattr(lifecycle, '_load_locked_validation_question_population', load_gold)
    return root, gold_calls


def test_pending_answer_judge_workflow_deduplicates_and_replays_without_clients(tmp_path, monkeypatch):
    root, gold_calls = fixture(tmp_path, monkeypatch)
    answer_client = _FakeClient(root/'answers-checkpoints')
    monkeypatch.setattr(pilot, '_completion_client', lambda *args: answer_client)
    pilot.answers(root, True)
    assert not gold_calls
    assert len(answer_client.chat.completions.requests) == 2
    assert all('GOLD_CANARY' not in json.dumps(r['messages']) for r in answer_client.chat.completions.requests)
    answer_artifact = read_sealed_json(root/'answers.json')
    rows = answer_artifact.payload['rows']
    assert [(r['case_id'], r['arm']) for r in rows] == [
        ('locked_q86', 'first'), ('locked_q86', 'second'),
        ('synthetic_probe', 'first'), ('synthetic_probe', 'second')]
    assert rows[0]['prediction'] == rows[1]['prediction']
    assert rows[2]['prediction'] == rows[3]['prediction']
    assert rows[0]['prediction'] != rows[2]['prediction']
    pilot.judge_preflight(root)
    assert len(gold_calls) == 1
    judge_client = _FakeClient(root/'judge-checkpoints')
    original_create = judge_client.chat.completions.create

    def create(**request):
        response = original_create(**request)
        response.choices[0].message.content = (
            'CORRECT' if 'BENCHMARK_GOLD_CANARY' in json.dumps(request['messages']) else 'INCORRECT')
        return response

    judge_client.chat.completions.create = create
    monkeypatch.setattr(pilot, '_completion_client', lambda *args: judge_client)
    pilot.judge(root, True)
    judged = read_sealed_json(root/'judgments.json')
    for arm in ('first', 'second'):
        assert judged.payload['aggregates']['locked_benchmark_development'][arm] == {'correct': 1, 'count': 1}
        assert judged.payload['aggregates']['source_derived_development'][arm] == {'correct': 0, 'count': 1}
    assert not judged.payload['promotion']
    assert len(judge_client.chat.completions.requests) == 2

    def forbid_client(*args):
        pytest.fail('replay must not create a provider client')

    monkeypatch.setattr(pilot, '_completion_client', forbid_client)
    pilot.answers(root, False)
    pilot.judge(root, False)
    assert read_sealed_json(root/'answers.json').sha256 == answer_artifact.sha256
    assert read_sealed_json(root/'judgments.json').sha256 == judged.sha256


def test_judge_cannot_join_gold_before_answers_exist(tmp_path, monkeypatch):
    root, gold_calls = fixture(tmp_path, monkeypatch)
    with pytest.raises(SealedArtifactError, match='regular file'):
        pilot.judge_preflight(root)
    assert not gold_calls


def test_invalid_judge_output_is_checkpointed_but_never_published_as_accuracy(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot, '_completion_client', lambda *args: _FakeClient(root/'answers-checkpoints'))
    pilot.answers(root, True)
    pilot.judge_preflight(root)
    # Default fake content is an answer hash, deliberately not a binary verdict.
    client = _FakeClient(root/'judge-checkpoints')
    monkeypatch.setattr(pilot, '_completion_client', lambda *args: client)
    with pytest.raises(RuntimeError, match='malformed verdict'):
        pilot.judge(root, True)
    assert len(list((root/'judge-checkpoints').glob('*.response.json'))) == 2
    assert not (root/'judgments.json').exists()
    monkeypatch.setattr(pilot, '_completion_client', lambda *args: pytest.fail('no retry on malformed completion'))
    with pytest.raises(RuntimeError, match='malformed verdict'):
        pilot.judge(root, False)
    assert len(client.chat.completions.requests) == 2
