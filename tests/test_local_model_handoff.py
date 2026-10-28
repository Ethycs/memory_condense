"""Service levels, memory modes and the ingest-rate handoff.

Design note ``docs/01 - Design/05 - Local Model Handoff and Concurrent
Linking.md``: the proxy serves each conversation at the best level available,
never refuses a request because a model is busy, and the receipt records the
level. The handoff starts when the adjusted steady-state ingest z-score -- each
preparation standardised against its own start, steady or stop regime -- is
abnormally high.
"""
import json
import math
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.interfaces.proxy_memory import MemoryProxy, MemoryProxyError
from memory_condense.interfaces.proxy_receipts import receipt_for_augment, receipt_for_passthrough
from memory_condense.interfaces.proxy_server import ProxyConfig, build_app, memory_health
from memory_condense.runtime.ingest_rate import IngestRateMonitor
from memory_condense.runtime.model_lock import RecallPriorityLock
from memory_condense.runtime.summary_fallback import LocalFirstSummaries
from tests.test_chat_streaming import StreamingBackend
from tests.test_proxy_memory import HEADERS, Factory, completion
from tests.test_summary_api_fallback import API, Local, raw_messages, response

PATH = '/v1/chat/completions'
PAYLOAD = dict(model='test', messages=[dict(role='user', content='Question')])


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        time.sleep(.005)


# -- the ingest-rate monitor --------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def sample(monitor, clock, seconds, *, tokens=1000, ordinal=10, forced=False):
    ticket = monitor.begin(tokens=tokens, ordinal=ordinal, forced=forced)
    clock.now += seconds
    monitor.end(ticket)


def warmed(clock=None, **options):
    """A monitor whose regimes have learned their regular rates.

    Steady ingest runs near 1.0 s per 1k tokens; starts near 4 s (model
    loads); drains near 6 s (the backlog queues behind itself).
    """
    clock = clock or Clock()
    monitor = IngestRateMonitor(clock=clock, min_samples=20, **options)
    def jitter(i):
        return 1 + .05 * ((i % 5) - 2)
    for i in range(30):
        # Offset phases so adjacent samples are not all high at once.
        sample(monitor, clock, 1.0 * jitter(i))
        sample(monitor, clock, 4.0 * jitter(i + 2), ordinal=0)
        sample(monitor, clock, 6.0 * jitter(i + 4), forced=True)
    assert monitor.status()['episodes'] == 0, 'regular ingest must not hand off'
    return monitor, clock


def test_no_handoff_before_the_steady_baseline_is_warm():
    clock = Clock()
    monitor = IngestRateMonitor(clock=clock, min_samples=20)
    for _ in range(19):
        sample(monitor, clock, 1.0)
    sample(monitor, clock, 50.0)
    assert not monitor.handoff() and monitor.zscore() is None


def test_regular_start_and_stop_behaviour_never_triggers_a_handoff():
    monitor, clock = warmed()
    # Each is four to six times the steady rate, but normal for its regime.
    for _ in range(5):
        sample(monitor, clock, 4.1, ordinal=0)
        sample(monitor, clock, 5.9, forced=True)
    assert not monitor.handoff()
    assert abs(monitor.zscore()) < 2


def test_rate_is_per_token_so_a_large_exchange_is_not_slow():
    monitor, clock = warmed()
    for _ in range(3):
        sample(monitor, clock, 3.0, tokens=3000)
    assert not monitor.handoff()


def test_slow_steady_ingest_hands_off_and_a_hold_then_probes_again():
    monitor, clock = warmed(hold_s=30.0)
    sample(monitor, clock, 1.15)          # slower, but within regular variation
    assert not monitor.handoff()
    sample(monitor, clock, 2.0)           # twice the steady rate: about 10 sd
    assert monitor.handoff()
    status = monitor.status()
    assert status['z'] > 3 and status['reason'].startswith('ingest z=')
    # Work during the episode partly ran on the provider: it measures nothing.
    learned = status['baselines']['steady']['samples']
    sample(monitor, clock, 0.5)
    assert monitor.status()['baselines']['steady']['samples'] == learned
    assert monitor.status()['excluded_samples'] == 1
    clock.now += 30
    assert not monitor.handoff()          # hold over: probe local again
    sample(monitor, clock, 1.0)
    assert not monitor.handoff()
    for _ in range(3):
        sample(monitor, clock, 2.0)
    assert monitor.handoff()              # still slow: a new episode


def test_an_overdue_in_flight_preparation_hands_off_before_it_finishes():
    monitor, clock = warmed()
    ticket = monitor.begin(tokens=1000, ordinal=10, forced=False)
    clock.now += 0.5
    assert not monitor.handoff()
    clock.now += 2.5
    assert monitor.handoff()
    monitor.end(ticket)


def test_saturation_does_not_teach_the_baseline_that_saturation_is_normal():
    monitor, clock = warmed(hold_s=0.0)
    before = monitor.status()['baselines']['steady']['seconds_per_1k_tokens']
    for _ in range(40):
        sample(monitor, clock, 3.0)
        monitor.handoff()
    after = monitor.status()['baselines']['steady']['seconds_per_1k_tokens']
    assert math.isclose(before, after, rel_tol=.01)


def test_failed_preparations_are_not_rate_samples():
    monitor, clock = warmed()
    learned = monitor.status()['baselines']['steady']['samples']
    ticket = monitor.begin(tokens=1000, ordinal=10, forced=False)
    clock.now += 100
    monitor.end(ticket, ok=False)
    assert monitor.status()['baselines']['steady']['samples'] == learned
    assert not monitor.handoff()


def test_the_baseline_survives_a_restart(tmp_path):
    path = tmp_path / 'ingest-baseline.json'
    monitor, clock = warmed(path=path)
    monitor.save()
    reopened = IngestRateMonitor(clock=clock, path=path)
    assert reopened.status()['baselines'] == monitor.status()['baselines']
    for _ in range(3):
        sample(reopened, clock, 2.0)
    assert reopened.handoff()


def test_a_corrupt_baseline_costs_only_a_warm_up(tmp_path):
    path = tmp_path / 'ingest-baseline.json'
    path.write_text('{not json', encoding='utf-8')
    assert IngestRateMonitor(path=path).status()['baselines']['steady']['samples'] == 0


class Observer:
    def __init__(self):
        self.begun, self.ended = [], []

    def begin(self, **kwargs):
        self.begun.append(kwargs)
        return len(self.begun)

    def end(self, ticket, *, ok=True):
        self.ended.append((ticket, ok))


def test_streaming_ingest_reports_each_preparation_with_its_regime(tmp_path):
    observer = Observer()
    session = ChatSession(tmp_path, 'observed', StreamingBackend(), streaming=True,
                          recent_exchanges=2, ingest_observer=observer)
    with session:
        for turn in range(3):
            session.ingest_many([ChatEvent(f'u{turn}', 'user', f'Question {turn}.'),
                                 ChatEvent(f'a{turn}', 'assistant', f'Answer {turn}.')])
        session.ingest(ChatEvent('u3', 'user', 'Last question.'))
        session.flush()
    assert observer.begun and len(observer.ended) == len(observer.begun)
    assert all(ok for _, ok in observer.ended)
    # Up to three preparations run at once, so they may report out of order.
    assert sorted(b['ordinal'] for b in observer.begun) == list(range(len(observer.begun)))
    assert all(b['tokens'] > 0 for b in observer.begun)
    # The flush drains the open tail: that preparation is a stop-regime sample.
    assert observer.begun[-1]['forced']


# -- the summary handoff ------------------------------------------------------

VALID = response(json.dumps(dict(atoms=[dict(label='T0', summary='Release r18 is planned',
                                             support=['Release r18 is planned'])])))


def test_failure_policy_ignores_saturation(tmp_path):
    local, api = Local(VALID), API()
    runtime = LocalFirstSummaries(local, tmp_path, api, saturated=lambda: True)
    assert runtime.call('raw', raw_messages(), scope='live') is VALID
    assert local.calls == 1 and not api.calls


@pytest.mark.parametrize('policy', ['saturation', 'both'])
def test_saturation_hands_off_without_touching_the_local_model(tmp_path, policy):
    local, api = Local(VALID), API()
    runtime = LocalFirstSummaries(local, tmp_path, api, handoff=policy, saturated=lambda: True)
    result = runtime.call('raw', raw_messages(), scope='live')
    assert result['summary_route'] == 'api_handoff_saturated'
    assert result['local_failure'] == 'gpu_saturated'
    assert local.calls == 0 and len(api.calls) == 1
    assert runtime.metrics['handoff_saturated'] == 1 and not runtime.metrics['local_failed']


def test_strict_policy_never_escalates_an_unsaturated_failure(tmp_path):
    failed = response('invalid JSON')
    local, api = Local(failed), API()
    runtime = LocalFirstSummaries(local, tmp_path, api, handoff='saturation', saturated=lambda: False)
    assert runtime.call('raw', raw_messages(), scope='live') is failed
    assert local.calls == 1 and not api.calls
    assert runtime.metrics['local_without_handoff'] == 1


def test_both_policy_still_escalates_an_unsaturated_failure(tmp_path):
    local, api = Local(response('invalid JSON')), API()
    runtime = LocalFirstSummaries(local, tmp_path, api, handoff='both', saturated=lambda: False)
    assert runtime.call('raw', raw_messages(), scope='live')['summary_route'] == 'api_escalation'
    assert local.calls == 1 and len(api.calls) == 1


def test_without_a_signal_nothing_is_saturated(tmp_path):
    local, api = Local(VALID), API()
    runtime = LocalFirstSummaries(local, tmp_path, api, handoff='saturation')
    assert runtime.call('raw', raw_messages(), scope='live') is VALID
    assert not api.calls


def test_handoff_configuration_is_validated(tmp_path):
    with pytest.raises(ValueError):
        LocalFirstSummaries(Local(), tmp_path, API(), handoff='sometimes')
    with pytest.raises(TypeError):
        LocalFirstSummaries(Local(), tmp_path, API(), saturated=True)
    with pytest.raises(ValueError):
        IngestRateMonitor(enter_z=0)


def test_gate_reports_queued_work_by_priority():
    gate = RecallPriorityLock()
    assert gate.queued() == dict(foreground=0, background=0, held=False)
    release = threading.Event()

    def hold():
        with gate:
            release.wait(5)

    def wait(foreground):
        with (gate.foreground() if foreground else gate):
            pass

    holder = threading.Thread(target=hold)
    holder.start()
    _wait_until(lambda: gate.queued()['held'])
    waiters = [threading.Thread(target=wait, args=(fg,)) for fg in (False, True)]
    for thread in waiters:
        thread.start()
    _wait_until(lambda: gate.queued()['foreground'] + gate.queued()['background'] == 2)
    assert gate.queued() == dict(foreground=1, background=1, held=True)
    release.set()
    for thread in (holder, *waiters):
        thread.join(5)
    assert gate.queued() == dict(foreground=0, background=0, held=False)


# -- memory modes and service levels -----------------------------------------

class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        return httpx.Response(200, json=completion('openai', 'Answer'))


def app_with(tmp_path, upstream, **options):
    memory = MemoryProxy(tmp_path, Factory(), inline_memory=False, **options)
    app = build_app(config=ProxyConfig(mode='augment', data_dir=str(tmp_path),
                                       upstreams={'openai': 'https://example.test/v1',
                                                  'anthropic': 'https://example.test'}),
                    memory=memory, client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    return app, memory


def serving(client):
    return client.get('/_control/status').json()['serving']


def test_off_passes_through_before_any_session_and_strips_memory_headers(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream, memory_mode='off')
    with TestClient(app) as client:
        reply = client.post(PATH, headers=HEADERS, json=PAYLOAD)
        assert reply.status_code == 200
        assert not memory.entries and not memory.factory.backends
        sent = upstream.calls[0]
        assert json.loads(sent.content) == PAYLOAD
        assert not [k for k in sent.headers if k.lower().startswith('x-memory-')]
        assert sent.headers['authorization'] == 'Bearer test-secret'
        status = memory.status()
        assert status['passthrough']['forced'] == 1 and status['passthrough']['degraded'] == 0
        # A forced passthrough is the operator's choice, never a health reason.
        assert memory_health(status) == ('ok', [])
        assert status['profile'] == 'B'
        counters = serving(client)
        assert counters['levels'] == dict(local=0, handoff=0, passthrough=1)
        assert counters['passthrough_forced'] == 1 and counters['passthrough_degraded'] == 0


def test_a_header_selects_the_mode_per_request(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream, memory_mode='require')
    with TestClient(app) as client:
        off = dict(HEADERS, **{'x-memory-mode': 'off'})
        assert client.post(PATH, headers=off, json=PAYLOAD).status_code == 200
        assert not memory.entries
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        assert len(memory.entries) == 1
        bad = dict(HEADERS, **{'x-memory-mode': 'sometimes'})
        assert client.post(PATH, headers=bad, json=PAYLOAD).status_code == 400
        assert serving(client)['levels'] == dict(local=1, handoff=0, passthrough=1)


def test_auto_passes_through_past_capacity_and_degrades_health(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream, max_sessions=1)
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        other = dict(HEADERS, **{'x-memory-conversation-id': 'second'})
        reply = client.post(PATH, headers=other, json=PAYLOAD)
        assert reply.status_code == 200 and len(upstream.calls) == 2
        assert json.loads(upstream.calls[-1].content) == PAYLOAD
        assert len(memory.entries) == 1
        status = memory.status()
        assert status['passthrough']['degraded'] == 1
        assert 'memory session limit' in status['passthrough']['last_degraded']['reason']
        assert 'memory_passthrough_degraded' in memory_health(status)[1]
        counters = serving(client)
        assert counters['passthrough_degraded'] == 1 and counters['levels']['local'] == 1


def test_require_still_refuses_past_capacity(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream, max_sessions=1, memory_mode='require')
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        other = dict(HEADERS, **{'x-memory-conversation-id': 'second'})
        assert client.post(PATH, headers=other, json=PAYLOAD).status_code == 429
        assert len(upstream.calls) == 1


def test_a_skipped_turn_is_ingested_by_the_next_memory_backed_turn(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)
    with TestClient(app) as client:
        first = dict(model='test', messages=[dict(role='user', content='First turn.')])
        assert client.post(PATH, headers=dict(HEADERS, **{'x-memory-mode': 'off'}), json=first).status_code == 200
        later = dict(model='test', messages=[
            dict(role='user', content='First turn.'), dict(role='assistant', content='Answer'),
            dict(role='user', content='Second turn.')])
        assert client.post(PATH, headers=dict(HEADERS, **{'x-memory-request-id': 'turn2'}),
                           json=later).status_code == 200
        session = next(iter(memory.entries.values())).session
        texts = [event.text for event in session.events()]
        assert 'First turn.' in texts and 'Second turn.' in texts


def test_a_model_failure_before_capture_passes_through_under_auto(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)

    def broken(namespace, directory):
        raise RuntimeError('local models failed to open')
    memory.factory = broken
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        assert len(upstream.calls) == 1
        assert memory.status()['passthrough']['last_degraded']['reason'] == 'memory unavailable: RuntimeError'
    app, memory = app_with(tmp_path, upstream, memory_mode='require')
    memory.factory = broken
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 503
        assert len(upstream.calls) == 1


def test_an_unknown_provider_outcome_is_never_resent(tmp_path):
    error = MemoryProxyError('Previous provider outcome is unknown; request was retained and will not be resent',
                             409, degradable=False)
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)

    def raising(*_args, **_kwargs):
        raise error
    memory._exchange = raising
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 409
    assert not upstream.calls and error.memory_degradable is False


def test_a_turn_that_spent_provider_tokens_is_recorded_as_a_handoff(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)
    spent = {'count': 0}

    def api_calls(namespace):
        spent['count'] += 1  # one provider call during each exchange
        return spent['count'] // 2
    memory.factory.api_calls = api_calls
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        assert serving(client)['levels'] == dict(local=0, handoff=1, passthrough=0)


def test_a_turn_that_begins_during_an_ingest_handoff_is_recorded_as_one(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)
    memory.factory.handoff_active = lambda: True
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        memory.factory.handoff_active = lambda: False
        second = dict(PAYLOAD, messages=[*PAYLOAD['messages'], dict(role='assistant', content='Answer'),
                                         dict(role='user', content='Next')])
        assert client.post(PATH, headers=dict(HEADERS, **{'x-memory-request-id': 'turn2'}),
                           json=second).status_code == 200
        assert serving(client)['levels'] == dict(local=1, handoff=1, passthrough=0)


def test_receipts_carry_level_mode_and_the_guard_passthrough():
    body = json.dumps(PAYLOAD).encode()
    guarded = receipt_for_augment('chat', body, body, level='passthrough', memory_mode='auto')
    assert guarded.level == 'passthrough' and guarded.degraded and guarded.memory_mode == 'auto'
    # An unchanged body alone is an empty packet, not a failure.
    assert not receipt_for_augment('chat', body, body).degraded
    served = json.dumps(dict(PAYLOAD, messages=[dict(role='user', content='Packet. Question')])).encode()
    assert receipt_for_augment('chat', body, served, level='handoff').level == 'handoff'
    forced = receipt_for_passthrough('chat', body, degraded=False, reason='memory mode off', memory_mode='off')
    assert (forced.level, forced.degraded, forced.rewritten) == ('passthrough', False, False)


@pytest.mark.parametrize('options,profile', [
    (dict(memory_mode='require', max_sessions=1, summary_api_fallback=False), 'A'),
    (dict(memory_mode='off'), 'B'),
    (dict(), 'custom'),
])
def test_status_names_the_profile(tmp_path, options, profile):
    assert MemoryProxy(tmp_path, Factory(), **options).status()['profile'] == profile


def test_the_designated_summary_model_is_validated(tmp_path):
    with pytest.raises(ValueError):
        MemoryProxy(tmp_path, Factory(), summary_model=' ')
    with pytest.raises(ValueError):
        MemoryProxy(tmp_path, Factory(), memory_mode='sometimes')


def test_release_closes_every_session_and_keeps_the_store(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        entry = next(iter(memory.entries.values()))
        memory.release()
        assert not memory.entries and entry.stale and memory.factory.forgotten == [None]
        assert any((tmp_path / 'sessions').iterdir())


# -- the control-plane override -----------------------------------------------

def test_an_override_forces_and_releases_the_handoff_before_warm_up():
    clock = Clock()
    monitor = IngestRateMonitor(clock=clock)
    status = monitor.override('handoff')
    assert status['handoff'] and status['override'] == dict(mode='handoff', expires_in_s=None)
    assert status['reason'].startswith('override')
    # Work that overlapped a forced episode partly ran on the provider.
    sample(monitor, clock, 1.0)
    assert monitor.status()['excluded_samples'] == 1
    clock.now += 3600
    assert monitor.handoff()              # no hold expiry: held until released
    assert not monitor.override('auto')['handoff']


def test_a_timed_override_returns_to_the_zscore():
    monitor, clock = warmed()
    assert monitor.override('handoff', for_s=60)['override']['expires_in_s'] == 60
    clock.now += 59
    assert monitor.handoff()
    clock.now += 1
    assert not monitor.handoff()
    assert monitor.status()['override'] == dict(mode='auto', expires_in_s=None)


def test_local_suppresses_the_handoff_but_still_reports_the_zscore():
    monitor, clock = warmed()
    for _ in range(3):
        sample(monitor, clock, 2.0)
    assert monitor.handoff()
    status = monitor.override('local')
    assert not status['handoff'] and status['reason'] is None
    for _ in range(3):
        sample(monitor, clock, 2.0)
    assert not monitor.handoff() and monitor.zscore() > monitor.enter_z
    # Back to auto, the same evidence starts an episode at once.
    assert monitor.override('auto')['handoff']


@pytest.mark.parametrize('mode, for_s', [('sometimes', None), ('auto', 10), ('handoff', 0),
                                         ('local', -1), ('handoff', True), ('handoff', 'soon'),
                                         ('handoff', math.inf)])
def test_invalid_overrides_are_refused(mode, for_s):
    monitor = IngestRateMonitor()
    with pytest.raises(ValueError):
        monitor.override(mode, for_s=for_s)
    assert monitor.status()['override']['mode'] == 'auto'


class HandoffFactory(Factory):
    """The parts of ``NativeProxySessions`` the override route reaches."""
    from memory_condense.runtime.sessions import NativeProxySessions as _Sessions
    override_handoff = _Sessions.override_handoff
    handoff_active = _Sessions.handoff_active

    def __init__(self, policy='both'):
        super().__init__()
        self.summary_handoff, self.ingest = policy, IngestRateMonitor()

    def gate_status(self):
        return dict(policy=self.summary_handoff, ingest=self.ingest.status())


def test_the_control_route_overrides_the_handoff(tmp_path):
    upstream = Recorder()
    app, memory = app_with(tmp_path, upstream)
    memory.factory = HandoffFactory()
    with TestClient(app) as client:
        reply = client.post('/_control/handoff', json=dict(mode='handoff', for_s=120))
        assert reply.status_code == 200
        assert reply.json()['handoff'] and reply.json()['policy'] == 'both'
        ingest = client.get('/_control/status').json()['memory']['gpu_gate']['ingest']
        assert ingest['override']['mode'] == 'handoff' and ingest['handoff']
        assert client.post(PATH, headers=HEADERS, json=PAYLOAD).status_code == 200
        assert serving(client)['levels']['handoff'] == 1
        assert client.post('/_control/handoff', json=dict(mode='auto')).json()['handoff'] is False
        assert client.post('/_control/handoff', json=dict(mode='never')).status_code == 400
        assert client.post('/_control/handoff', json={}).status_code == 400


def test_the_control_route_respects_the_failure_policy(tmp_path):
    app, memory = app_with(tmp_path, Recorder())
    memory.factory = HandoffFactory(policy='failure')
    with TestClient(app) as client:
        refused = client.post('/_control/handoff', json=dict(mode='handoff'))
        assert refused.status_code == 409 and refused.json()['error']['type'] == 'policy'
        assert client.post('/_control/handoff', json=dict(mode='local')).status_code == 200


def test_the_control_route_needs_a_handoff(tmp_path):
    app, _memory = app_with(tmp_path, Recorder())   # the plain factory has none
    with TestClient(app) as client:
        assert client.post('/_control/handoff', json=dict(mode='auto')).status_code == 501


def test_ctl_handoff_posts_the_override_and_status_shows_it(monkeypatch):
    from memory_condense.control import cli
    sent = []
    monkeypatch.setattr(cli, 'post_control', lambda base, route, payload: sent.append((route, payload)) or {})
    assert cli.main(['handoff', 'handoff', '--for', '90', '--proxy', 'http://x']) == cli.EXIT_OK
    assert cli.main(['handoff', 'auto', '--for', '5']) == cli.EXIT_REFUSED
    assert sent == [('handoff', dict(mode='handoff', for_s=90.0))]
    monitor = IngestRateMonitor()
    monitor.override('local', for_s=90)
    line = cli.describe_handoff(dict(policy='both', ingest=monitor.status()))
    assert line.startswith('off (override local,') and 'z warming up' in line and 'policy both' in line
    assert cli.describe_handoff(None) is None
