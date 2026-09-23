from dataclasses import asdict
import json

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest, parse_spine_summary
from tools.local_qwen_spine_backend import job_messages
from tools.local_spine_json_recovery import (
    JsonBatchFourQwen, repaired_row, repair_response, repair_snapshot, verify_repaired_row,
)
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.reuse_local_spine_parent_summaries import INPUT_KEYS, freeze_namespace
from tools.run_local_spine_parent_batch4 import BatchFourQwen


BAD = r'''{"summary":"User likes Sweet Child O\' Mine."}'''


def request(cap=128):
    return SpineSummaryRequest('user_spine', (
        SpineSummaryFragment('user_summary', '2026-09-11', "User likes Sweet Child O' Mine."),),
        max_output_tokens=cap)


def test_apostrophe_encoding_repair_changes_only_the_invalid_backslash():
    fixed, receipt = repair_response(BAD, request())
    assert parse_spine_summary(fixed, request()) == "User likes Sweet Child O' Mine."
    positions = receipt['removed_backslash_positions']
    assert len(positions) == 1
    assert fixed == ''.join(c for i, c in enumerate(BAD) if i not in positions)
    assert repair_response(fixed, request()) == (fixed, None)


@pytest.mark.parametrize('value', ["User likes O'Brien.", r"User likes O\'Brien.",
    'User said "hello".', 'User uses a backslash: \\', 'User likes café.'])
def test_valid_json_is_byte_for_byte_unchanged(value):
    response = json.dumps({'summary': value})
    assert repair_response(response, request()) == (response, None)


@pytest.mark.parametrize('response', [
    r'''{"summary":"O\'Brien","extra":"not allowed"}''',
    r'''{"summary":"O\'Brien","summary":"ambiguous"}''',
    r'''{"summary":"O\'Brien and \q"}''',
    r'''{"summary":"O\'Brien"''',
    r'''```json {"summary":"O\'Brien"} ```''',
    r'''\' {"summary":"O\'Brien"}''',
])
def test_repair_does_not_admit_other_malformed_or_ambiguous_responses(response):
    assert repair_response(response, request()) == (response, None)


def test_encoding_repair_never_relaxes_the_summary_budget():
    assert repair_response(BAD, request(1)) == (BAD, None)


def test_valid_escaped_backslash_is_preserved_while_invalid_apostrophe_escape_is_removed():
    response = r'''{"summary":"User likes O\\\'Brien."}'''
    fixed, receipt = repair_response(response, request())
    assert receipt is not None
    assert parse_spine_summary(fixed, request()) == r"User likes O\'Brien."


def test_encoding_cannot_rescue_truncated_generation_or_wrong_attribution():
    job = request()
    row = {'job_sha256': job.prompt_sha256, 'stopped': False, 'response': BAD}
    assert repaired_row(row, job) == row
    with pytest.raises(ValueError, match='attribution'):
        repaired_row({**row, 'job_sha256': 'wrong'}, job)


def test_saved_repair_receipt_must_reproduce_from_the_original_text():
    job = request()
    row = repaired_row({'job_sha256': job.prompt_sha256, 'stopped': True, 'response': BAD}, job)
    verify_repaired_row(row, job)
    with pytest.raises(ValueError, match='reproduce'):
        verify_repaired_row({**row, 'original_response': '{"summary":"Different text."}'}, job)
    with pytest.raises(ValueError, match='reproduce'):
        verify_repaired_row({**row, 'response': '{"summary":"Different text."}'}, job)


def test_backend_preserves_original_output_and_makes_only_the_original_generation_call(monkeypatch):
    job = request()
    calls = []
    def generate(self, jobs, attempt):
        calls.append((jobs, attempt))
        return {'rows': [{'job_sha256': job.prompt_sha256, 'stopped': True,
            'response': BAD, 'output_tokens': 55}], 'raw_inputs_to_qwen': False, 'remote_provider_calls': 0}
    monkeypatch.setattr(BatchFourQwen, 'generate', generate)
    backend = JsonBatchFourQwen.__new__(JsonBatchFourQwen)
    result = backend.generate((job,), 2)
    row = result['rows'][0]
    assert calls == [((job,), 2)]
    assert row['original_response'] == BAD and row['output_tokens'] == 55
    assert parse_spine_summary(row['response'], job) == "User likes Sweet Child O' Mine."
    verify_repaired_row(row, job)


def snapshot_fixture(root, *, stopped=True, incomplete=False):
    fields = {key: 'fixture-' + key for key in INPUT_KEYS}
    cache, _ = publish_sealed_json(root / 'cached' / 'input-cache.json', {
        'atoms_sha256': fields['atoms_sha256'], 'summaries': {'a' * 64: 'Prior completed summary.'}})
    cached, _ = publish_sealed_json(root / 'cached' / 'preflight.json', {
        **fields, 'input_cache_sha256': cache.sha256, 'implementation': {}, 'raw_inputs_to_qwen': False})
    local, _ = publish_sealed_json(root / 'local' / 'preflight.json', {
        **fields, 'input_cache_sha256': cache.sha256, 'cached_preflight_sha256': cached.sha256,
        'backend_sha256': 'b' * 64, 'implementation': {}, 'raw_inputs_to_qwen': False})
    job = request()
    response_paths = []
    for attempt, text in ((0, r'''{"summary":"Earlier O\'Brien text."}'''), (2, BAD)):
        body = {'preflight_sha256': local.sha256, 'backend_sha256': 'b' * 64,
            'attempt': attempt, 'jobs': [asdict(job)], 'messages': [job_messages(job, attempt)],
            'raw_inputs_to_qwen': False}
        request_artifact, _ = publish_sealed_json(root / 'local' / 'requests' / (identity_sha256(body) + '.json'), body)
        path = root / 'local' / 'responses' / (request_artifact.sha256 + '.json')
        if not incomplete:
            publish_sealed_json(path, {'request_sha256': request_artifact.sha256,
                'backend_sha256': 'b' * 64, 'raw_inputs_to_qwen': False, 'remote_provider_calls': 0,
                'rows': [{'job_sha256': job.prompt_sha256, 'stopped': stopped, 'response': text}]})
            response_paths.append(path)
    freeze_namespace(root / 'cached', root / 'local', root / 'snapshot')
    return job, response_paths


def test_snapshot_recovers_latest_completed_response_without_mutating_the_source_and_replays(tmp_path):
    job, paths = snapshot_fixture(tmp_path)
    originals = {path: path.read_bytes() for path in paths}
    result = repair_snapshot(tmp_path / 'snapshot', tmp_path / 'recovered')
    cache = read_sealed_json(tmp_path / 'recovered' / 'input-cache.json')
    assert result['encoding_repaired_jobs'] == 1 and result['cached_jobs'] == 2
    assert cache.payload['summaries'][job.prompt_sha256] == "User likes Sweet Child O' Mine."
    assert cache.payload['encoding_repairs'][0]['attempt'] == 2
    assert cache.payload['new_calls'] == 0
    assert repair_snapshot(tmp_path / 'snapshot', tmp_path / 'recovered') == result
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_snapshot_refuses_unacknowledged_executions(tmp_path):
    snapshot_fixture(tmp_path, incomplete=True)
    with pytest.raises(ValueError, match='unacknowledged'):
        repair_snapshot(tmp_path / 'snapshot', tmp_path / 'recovered')
    assert not (tmp_path / 'recovered' / 'input-cache.json').exists()


def test_snapshot_refuses_recovery_of_generation_without_eos(tmp_path):
    snapshot_fixture(tmp_path, stopped=False)
    with pytest.raises(ValueError, match='separate recovery decision'):
        repair_snapshot(tmp_path / 'snapshot', tmp_path / 'recovered')
