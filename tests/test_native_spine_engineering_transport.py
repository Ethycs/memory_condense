import pytest

from tools import native_spine_engineering_transport_resume as transport


def test_failed_call_retries_same_payload_with_new_identity_and_preserves_failure(tmp_path, monkeypatch):
    s = transport.session
    jobs = []

    def fake_call(self, kind, messages, **options):
        job = dict(kind=kind, messages=messages, **options)
        jobs.append(job)
        key = s.identity_sha256(job)
        request = s.save(tmp_path / 'gateway' / f'{key}.request.json', job)
        if job['nonce'] == 7:
            s.save(tmp_path / 'gateway' / f'{key}.response.json', {
                'request_sha256': request.sha256, 'error_type': 'InternalServerError'})
            raise RuntimeError('server error')
        return {'content': 'accepted'}

    monkeypatch.setattr(transport.OriginalGateway, 'call', fake_call)
    result = transport.RecoveringGateway(tmp_path).call('raw', [{'role': 'user', 'content': 'fixture'}], nonce=7)
    assert result == {'content': 'accepted'}
    assert len(jobs) == 2
    assert jobs[0]['messages'] == jobs[1]['messages']
    assert jobs[1]['nonce']['transport_retry'] == 1
    receipt, = (tmp_path / 'transport-retries').glob('*.json')
    assert s.payload(receipt)['retry_job_sha256'] == s.identity_sha256(jobs[1])
    failure, = (tmp_path / 'gateway').glob('*.response.json')
    assert s.payload(failure)['error_type'] == 'InternalServerError'


def test_success_does_not_retry(tmp_path, monkeypatch):
    calls = []
    def success(*args, **kwargs):
        calls.append(kwargs)
        return {'content': 'accepted'}
    monkeypatch.setattr(transport.OriginalGateway, 'call', success)
    assert transport.RecoveringGateway(tmp_path).call('raw', [])['content'] == 'accepted'
    assert len(calls) == 1
    assert not (tmp_path / 'transport-retries').exists()


def test_unacknowledged_call_does_not_retry(tmp_path, monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise TimeoutError('request retained without a response')
    monkeypatch.setattr(transport.OriginalGateway, 'call', timeout)
    with pytest.raises(TimeoutError):
        transport.RecoveringGateway(tmp_path).call('answer', [])
    assert len(calls) == 1


def test_persistent_server_failure_stops_after_two_retries(tmp_path, monkeypatch):
    s = transport.session
    calls = []
    def fail(self, kind, messages, **options):
        job = dict(kind=kind, messages=messages, **options)
        calls.append(job)
        key = s.identity_sha256(job)
        request = s.save(tmp_path / 'gateway' / f'{key}.request.json', job)
        s.save(tmp_path / 'gateway' / f'{key}.response.json', {
            'request_sha256': request.sha256, 'error_type': 'InternalServerError'})
        raise RuntimeError('server error')
    monkeypatch.setattr(transport.OriginalGateway, 'call', fail)
    with pytest.raises(RuntimeError):
        transport.RecoveringGateway(tmp_path).call('raw', [])
    assert len(calls) == 3
    assert len(list((tmp_path / 'transport-retries').glob('*.json'))) == 2


def test_report_counts_failed_attempt_and_verifies_exact_retry(tmp_path, monkeypatch):
    from tools.report_native_spine_engineering_session import audit_gateway
    s = transport.session
    def fake_call(self, kind, messages, **options):
        job = dict(kind=kind, messages=messages, **options)
        key = s.identity_sha256(job)
        request = s.save(tmp_path / 'gateway' / f'{key}.request.json', job)
        response = {'request_sha256': request.sha256, 'elapsed_s': 2.5}
        if job['nonce'] is None:
            response['error_type'] = 'InternalServerError'
        else:
            response.update(content='accepted', finish_reason='stop')
        s.save(tmp_path / 'gateway' / f'{key}.response.json', response)
        if 'error_type' in response:
            raise RuntimeError('server error')
        return response
    monkeypatch.setattr(transport.OriginalGateway, 'call', fake_call)
    transport.RecoveringGateway(tmp_path).call('raw', [{'role': 'user', 'content': 'fixture'}])
    totals, links = audit_gateway(tmp_path)
    assert totals['raw']['count'] == 2
    assert totals['raw']['successful_count'] == totals['raw']['failed_count'] == 1
    assert totals['raw']['total_s'] == 5
    assert len(links) == 1
    assert links[0]['failed_job_sha256'] != links[0]['successful_job_sha256']
