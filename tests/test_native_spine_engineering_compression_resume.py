import json

import pytest

from tools import native_spine_engineering_compression_resume as recovery


def request():
    s = recovery.s
    return s.SpineSummaryRequest('attached_context', (
        s.SpineSummaryFragment('assistant', '2026-08-16', 'Assistant implemented tests.'),
    ), user_spine='Proceed with the work.', max_output_tokens=128)


def test_recovery_keeps_inputs_and_enforces_the_stricter_budget(tmp_path, monkeypatch):
    def exhausted(self, request):
        raise ValueError('bounded Qwen compression retries exhausted')

    monkeypatch.setattr(recovery.OriginalCompiler, 'merge', exhausted)
    calls = []

    class Gateway:
        def call(self, kind, messages, **kwargs):
            calls.append(kwargs['typed_request'])
            # The first reply fits the original 128-token cap but exceeds 96.
            content = 'word ' * 110 if len(calls) == 1 else 'Assistant implemented tests.'
            return {'content': json.dumps({'summary': content})}

    original = request()
    result = recovery.RecoveringCompiler(tmp_path, Gateway()).merge(original)
    assert result == 'Assistant implemented tests.'
    assert [r['max_output_tokens'] for r in calls] == [96, 64]
    assert all(r['user_spine'] == original.user_spine for r in calls)
    assert all(r['fragments'] == recovery.asdict(original)['fragments'] for r in calls)
    saved = recovery.s.payload(next((tmp_path / 'merged-summaries').glob('*.json')))
    assert saved['request']['max_output_tokens'] == 128
    assert saved['summary'] == result


def test_unrelated_validation_errors_are_not_retried(tmp_path, monkeypatch):
    def invalid(self, request):
        raise ValueError('cache binding changed')

    monkeypatch.setattr(recovery.OriginalCompiler, 'merge', invalid)
    with pytest.raises(ValueError, match='cache binding changed'):
        recovery.RecoveringCompiler(tmp_path, object()).merge(request())
