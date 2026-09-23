from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import quote_sha256
from tools import evaluate_threaded_spine_full100_v2 as evaluation


def test_complete_judging_path_binds_real_prompt_builder_and_streamed_population(monkeypatch, tmp_path):
    events, observations = [], []
    for ordinal in range(100):
        for arm in evaluation.call_order(ordinal):
            prediction = f'Answer {ordinal}'
            call = {'question': {'ordinal': ordinal, 'question_id': str(ordinal),
                                'retrieval_query': f'Question {ordinal}'}, 'arm': arm}
            response = SimpleNamespace(sha256=quote_sha256(f'{ordinal}-{arm}'), payload={'measurement': {
                'prediction': prediction, 'prediction_sha256': quote_sha256(prediction),
                'prepare_s': .2, 'e2e_ttft_s': 5., 'e2e_total_s': 5., 'finish_reason': 'stop'}})
            observations.append((call, response))
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: SimpleNamespace(sha256='a' * 64))

    def seal(*args):
        events.append('sealed')
        return SimpleNamespace(sha256='b' * 64), observations

    def references():
        assert events == ['sealed']
        events.append('references')
        return None, [SimpleNamespace(question_id=str(i), answer=f'Answer {i}') for i in range(100)]

    def batch(root, phase, prompts, *args):
        assert events == ['sealed', 'references']
        assert phase == 'judge' and len(prompts) == 200
        assert all(f'Question {i}' in str(prompts[2 * i]) for i in range(100))
        events.append('judged')
        return SimpleNamespace(logical_completions=['CORRECT'] * 200, unique_records=[]), 0, 200, 0.

    monkeypatch.setattr(evaluation, 'seal_answers', seal)
    monkeypatch.setattr(evaluation, 'load_references', references)
    monkeypatch.setattr(evaluation, '_batch', batch)
    report = evaluation.judge(tmp_path)
    assert events == ['sealed', 'references', 'judged']
    assert report.payload['accuracy'] == {'flat': 100, 'threaded': 100}
    assert report.payload['target_gate_passed']
    assert len(report.payload['rows']) == 200
    assert (tmp_path / 'joint-report.json').exists()


def test_repair_rejects_predecessor_that_started_execution(monkeypatch, tmp_path):
    from tools import evaluate_threaded_spine_full100 as predecessor
    monkeypatch.setattr(predecessor, 'load_preflight', lambda _: SimpleNamespace(payload={}))
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'execution.reserved').write_text('reserved', encoding='utf-8')
    with pytest.raises(ValueError, match='unexecuted'):
        evaluation.prepare(tmp_path / 'successor', source)
