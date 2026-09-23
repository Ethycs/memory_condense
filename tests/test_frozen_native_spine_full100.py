from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.spine_reader_policy_v5 import SPINE_READER_SYSTEM_PROMPT_V5
from tests.test_native_joint_population import case, manifests
from tests.test_native_spine_routing import fixture, QUERY, DATED
from tools import evaluate_frozen_native_spine_full100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def synthetic_population(root):
    calls = []
    for ordinal in range(100):
        c = dict(case(ordinal), question=f'Question {ordinal}')
        q = evaluation.question(c)
        prompts = {a: evaluation.serving.protocol.messages(q) for a in evaluation.ARMS}
        evidence, _ = publish_sealed_json(root/'evidence'/f'{ordinal:03d}.json', {
            'question': q, 'case': c, 'messages': prompts,
            'hydration': {a: {'fixture': a, 'ordinal': ordinal} for a in evaluation.MEMORY_ARMS},
            'routing': {a: {'fixture_route': a, 'ordinal': ordinal} for a in evaluation.MEMORY_ARMS}})
        for arm in evaluation.call_order(ordinal):
            calls.append({'call_index': len(calls), 'question': q, 'arm': arm,
                'messages': prompts[arm], 'messages_sha256': identity_sha256(prompts[arm]),
                'evidence_sha256': evidence.sha256})
    return calls


def test_frozen_packet_keeps_exact_date_eligible_raw_and_embeds_each_query():
    history, semantic, hierarchy, encoder = fixture(future=True)
    memory = ResidentNativeSpineContextMemory(semantic, hierarchy, encoder=encoder, load_turn=history.get_turn)
    q = {'retrieval_query': QUERY, 'prompt_question': DATED}
    for arm in evaluation.MEMORY_ARMS:
        messages, hydration, routing = evaluation.build(memory, q, arm)
        assert messages[0]['content'] == SPINE_READER_SYSTEM_PROMPT_V5
        assert 'botanical garden' in messages[1]['content']
        assert hydration['context_token_count'] <= 1024
        assert routing['protected_direct'] == 0
        assert routing['query_qwen_passes'] == routing['raw_reads_during_routing'] == 0
        assert '2026-09-13' not in str(hydration)
    assert encoder.calls == [QUERY, QUERY]


def test_partial_corpus_rejected_before_runtime_or_provider(tmp_path, monkeypatch):
    source, store, hierarchy = manifests()
    store.payload['complete_source_compilation'] = False
    settings = {key: tmp_path/key for key in ('sources', 'store', 'vectors', 'm_dataset', 'candidate')}
    settings['hierarchies'] = tmp_path/'hierarchies.json'
    actual, _ = publish_sealed_json(settings['sources']/'sources.json', source.payload)
    store.payload['sources_sha256'] = actual.sha256
    publish_sealed_json(settings['store']/'summary-bodies.json', store.payload)
    publish_sealed_json(settings['hierarchies'], hierarchy.payload)
    monkeypatch.setattr(evaluation, 'validate_candidate', lambda _: None)
    for name in ('EmbeddingService', 'FrozenParentNativeSpineCorpus', '_completion_client'):
        monkeypatch.setattr(evaluation, name, lambda *a, **k: pytest.fail('incomplete corpus constructed runtime'))
    with pytest.raises(ValueError, match='every source body'):
        evaluation.prepare(tmp_path/'evaluation', settings)


def test_partial_answers_cannot_open_gold(tmp_path, monkeypatch):
    plan = SimpleNamespace(payload={'calls': synthetic_population(tmp_path)}, sha256='fixture')
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: plan)
    monkeypatch.setattr(evaluation, 'load_references', lambda _: pytest.fail('gold opened early'))
    with pytest.raises(ValueError, match='all 300'):
        evaluation.judge(tmp_path)


@pytest.mark.parametrize('damage', ['missing', 'control', 'duplicate_history'])
def test_schedule_rejects_incomplete_or_unmatched_population(tmp_path, damage):
    calls = synthetic_population(tmp_path)
    evaluation.validate_calls(calls)
    if damage == 'missing':
        calls.pop()
    elif damage == 'control':
        row = next(c for c in calls if c['arm'] == 'parent_context_api')
        row['messages'] = [{'role': 'user', 'content': 'different control'}]
        row['messages_sha256'] = identity_sha256(row['messages'])
    else:
        for row in calls[3:6]:
            row['question']['namespace_id'] = calls[0]['question']['namespace_id']
    with pytest.raises(ValueError):
        evaluation.validate_calls(calls)


def fake_observations(calls, correct=95, total=4.8):
    observations, judged = [], []
    for call in calls:
        i, arm = call['question']['ordinal'], call['arm']
        response = SimpleNamespace(sha256=f'response-{call["call_index"]}', payload={'measurement': {
            'prepare_s': .3 if arm in evaluation.MEMORY_ARMS else .001,
            'e2e_ttft_s': 3.0, 'e2e_total_s': (7.0 if i >= 94 else total),
            'finish_reason': 'stop', 'prediction_sha256': quote_sha256(str(i))}})
        observations.append((call, response))
        if arm in evaluation.MEMORY_ARMS:
            judged.append({'ordinal': i, 'arm': arm, 'prediction_sha256': quote_sha256(str(i)),
                'response_sha256': response.sha256, 'correct': i < correct,
                'verdict': 'CORRECT' if i < correct else 'INCORRECT'})
    return observations, judged


def test_joint_gate_uses_actual_accuracy_and_median_reports_tail(tmp_path):
    calls = synthetic_population(tmp_path)
    report = evaluation.joint_statistics(*fake_observations(calls))
    assert report['target_gate_passed'] is True
    assert report['accuracy']['parent_context'] == {'correct': 95, 'questions': 100}
    assert report['latency']['parent_context']['e2e_total_s']['p95_s'] == 7.0
    assert report['candidate_answers_under_five_seconds'] == 94
    assert report['tail_latency_threshold_applied'] is False
    assert not evaluation.joint_statistics(*fake_observations(calls, correct=94))['target_gate_passed']
    assert not evaluation.joint_statistics(*fake_observations(calls, total=5.0))['target_gate_passed']
    observations, judged = fake_observations(calls)
    judged[0]['response_sha256'] = 'different-response'
    with pytest.raises(ValueError, match='same judged responses'):
        evaluation.joint_statistics(observations, judged)


def test_300_streams_fresh_timed_retrieval_sealed_before_200_judgments_and_replay(tmp_path, monkeypatch):
    calls = synthetic_population(tmp_path)
    plan, _ = publish_sealed_json(tmp_path/'preflight.json', {'calls': calls,
        'settings': {'vectors': 'fixture-vectors'}, 'embedding_identity': 'fixture',
        'population_admission_sha256': 'fixture'})
    monkeypatch.setattr(evaluation, 'load_preflight', lambda _: plan)
    monkeypatch.setattr(evaluation, 'require_idle', lambda: None)
    monkeypatch.setattr(evaluation, 'open_corpus', lambda _: SimpleNamespace(
        close=lambda: None, load_namespace=lambda *a, **k: object()))
    monkeypatch.setattr(evaluation.vector_compiler, 'NativeSummaryVectors', lambda _: object())
    monkeypatch.setattr(evaluation.population, 'namespace_receipt', lambda *a: None)
    monkeypatch.setattr(evaluation, 'resident', lambda *a: object())
    monkeypatch.setattr(evaluation, 'EmbeddingService', lambda **k: SimpleNamespace(
        close=lambda: None, embed_query=lambda q: None))
    monkeypatch.setattr(evaluation, 'summary_embedding_identity', lambda _: 'fixture')
    builds, sent, timing = [], [], []
    real_measure = evaluation.measure_streaming_answer
    def measure(**kwargs):
        timing.append(True)
        try:
            return real_measure(**kwargs)
        finally:
            timing.pop()
    monkeypatch.setattr(evaluation, 'measure_streaming_answer', measure)
    def build(memory, q, arm):
        assert timing, 'fresh retrieval must be inside the measurement callback'
        builds.append((q['ordinal'], arm))
        return (evaluation.serving.protocol.messages(q), {'fixture': arm, 'ordinal': q['ordinal']},
                {'fixture_route': arm, 'ordinal': q['ordinal']})
    monkeypatch.setattr(evaluation, 'build', build)
    class Stream:
        def __iter__(self):
            yield {'model': evaluation.MODEL, 'choices': [{'index': 0,
                'delta': {'role': 'assistant'}, 'finish_reason': None}]}
            yield {'model': evaluation.MODEL, 'choices': [{'index': 0,
                'delta': {'content': 'fact'}, 'finish_reason': 'stop'}]}
        def close(self):
            pass
    class Client:
        max_retries = 0
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        def with_options(self, **kwargs):
            return self
        def close(self):
            pass
        def create(self, **kwargs):
            sent.append(kwargs)
            if kwargs.get('stream'):
                return Stream()
            assert (tmp_path/'answers.json').exists()
            return SimpleNamespace(id=f'judge-{len(sent)}', model='codex_sdk/gpt-5.6-sol', usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT'), finish_reason='stop')])
    monkeypatch.setattr(evaluation, '_completion_client', lambda *a: Client())
    def references(settings):
        assert len(list((tmp_path/'journal').glob('*.response.json'))) == 300
        assert (tmp_path/'answers.json').exists()
        return {f'q{i}': 'fact' for i in range(100)}
    monkeypatch.setattr(evaluation, 'load_references', references)
    report = evaluation.run(tmp_path, True)
    assert len(builds) == 200
    assert sum(bool(c.get('stream')) for c in sent) == 300
    assert sum(not c.get('stream') for c in sent) == 100  # Identical judge prompts deduplicate.
    assert len(report.payload['rows']) == 200
    assert report.payload['accuracy'] == {a: {'correct': 100, 'questions': 100} for a in evaluation.MEMORY_ARMS}
    monkeypatch.setattr(evaluation, '_completion_client', lambda *a: pytest.fail('replay called provider'))
    assert evaluation.judge(tmp_path, False).sha256 == report.sha256
    with pytest.raises(ValueError, match='another answer release'):
        evaluation.run(tmp_path, True)
