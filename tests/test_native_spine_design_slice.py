from types import SimpleNamespace

import pytest

from tools.prepare_native_spine_design_slice import select_case, selected_artifacts


def test_design_selection_admits_one_locked_namespace_without_gold():
    sources = SimpleNamespace(sha256='source', payload={'namespaces': [
        {'namespace_id': 'first', 'sha256': 'a'}, {'namespace_id': 'second', 'sha256': 'b'}]})
    cases = SimpleNamespace(payload={'sources_sha256': 'source', 'gold_answer_text_included': False,
        'ingest_use_permitted': False, 'cases': [
            {'ordinal': 0, 'namespace_id': 'first', 'namespace_sha256': 'a'},
            {'ordinal': 1, 'namespace_id': 'second', 'namespace_sha256': 'b'}]})
    case, namespace = select_case(sources, cases, 0)
    assert case['namespace_id'] == namespace['namespace_id'] == 'first'
    cases.payload['gold_answer_text_included'] = True
    with pytest.raises(ValueError, match='gold policy'):
        select_case(sources, cases, 0)


@pytest.mark.parametrize('complete', [False, True])
def test_body_selection_does_not_require_or_return_unrelated_histories(complete):
    rows = [{'path': f'exchanges/{sha}.json', 'sha256': sha} for sha in ('a', 'b', 'unrelated')]
    selected = selected_artifacts(rows, {'a', 'b'}, require_complete=complete)
    assert set(selected) == {'a', 'b'}
    assert selected_artifacts(rows, {'missing'}, require_complete=False) == {}
    with pytest.raises(ValueError, match='missing compiled exchanges'):
        selected_artifacts(rows, {'a', 'missing'}, require_complete=True)
    with pytest.raises(ValueError, match='duplicate selected'):
        selected_artifacts([rows[0], rows[0]], {'a'}, require_complete=True)


def test_single_history_run_seals_five_fresh_answers_before_two_judgments(tmp_path, monkeypatch):
    from tools import evaluate_native_spine_design_slice as evaluation
    from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
    from tools.prepare_native_spine_design_slice import binding
    source, _ = publish_sealed_json(tmp_path/'source.json', {})
    case = {'ordinal': 0, 'question_id': 'q0', 'namespace_id': 'one-history', 'question': 'What is the fact?',
        'question_date': '2023/05/30 (Tue) 23:18', 'reference_sha256': evaluation.quote_sha256('fact')}
    selected = SimpleNamespace(sha256='one-scope', payload={'case': case, 'source': binding(source)})
    namespace = SimpleNamespace(audit={'body_tokens': 1_098_417})
    monkeypatch.setattr(evaluation, 'load_namespace', lambda _: (selected, namespace))
    monkeypatch.setattr(evaluation.serving, 'require_idle', lambda: None)
    monkeypatch.setattr(evaluation.serving.population, 'namespace_receipt', lambda *a: None)
    monkeypatch.setattr(evaluation, 'NativeSummaryVectors', lambda _: SimpleNamespace(
        preflight=SimpleNamespace(payload={'design_scope_sha256': 'one-scope'}), embedding_identity='encoder'))
    monkeypatch.setattr(evaluation, 'EmbeddingService', lambda **k: SimpleNamespace(
        embed_query=lambda _: None, close=lambda: None))
    monkeypatch.setattr(evaluation, 'summary_embedding_identity', lambda _: 'encoder')
    monkeypatch.setattr(evaluation.serving, 'resident', lambda *a: object())
    builds, sent = [], []
    def build(memory, question, arm):
        builds.append(arm)
        return ([{'role': 'user', 'content': 'Question and exact evidence'}],
            {'sections': [{'evidence': 'fact'}]}, {'arm': arm})
    monkeypatch.setattr(evaluation.serving, 'build', build)
    def reference(*args):
        assert len(list((tmp_path/'evaluation/journal').glob('*.response.json'))) == 5
        assert (tmp_path/'evaluation/answers.json').exists()
        return 'fact'
    monkeypatch.setattr(evaluation, 'reference', reference)
    class Stream:
        def __iter__(self):
            yield {'model': evaluation.serving.MODEL, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
            yield {'model': evaluation.serving.MODEL, 'choices': [{'index': 0, 'delta': {'content': 'fact'}, 'finish_reason': 'stop'}]}
        def close(self):
            pass
    class Client:
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        def close(self):
            pass
        def create(self, **kwargs):
            sent.append(kwargs)
            if kwargs.get('stream'):
                return Stream()
            assert (tmp_path/'evaluation/answers.json').exists()
            return SimpleNamespace(model='codex_sdk/gpt-5.6-sol', choices=[SimpleNamespace(
                message=SimpleNamespace(content='CORRECT' if len(sent) == 6 else 'INCORRECT'), finish_reason='stop')])
    monkeypatch.setattr(evaluation.serving, '_completion_client', lambda *a: Client())
    result = evaluation.run(tmp_path, tmp_path/'unused-dataset', True)
    assert builds == ['flat', 'hierarchy', 'flat', 'hierarchy']
    assert len(sent) == 7 and sum(bool(c.get('stream')) for c in sent) == 5
    assert result.payload['accuracy'] == {'flat': {'correct': 1, 'questions': 1}, 'hierarchy': {'correct': 0, 'questions': 1}}
    assert result.payload['history_count'] == result.payload['question_count'] == 1
    assert result.payload['accuracy_generalization_established'] is False
    plan = read_sealed_json(tmp_path/'evaluation/preflight.json')
    assert evaluation.report(tmp_path/'evaluation', plan).sha256 == result.sha256 and len(sent) == 7
    with pytest.raises(ValueError, match='already started'):
        evaluation.run(tmp_path, tmp_path/'unused-dataset', True)
