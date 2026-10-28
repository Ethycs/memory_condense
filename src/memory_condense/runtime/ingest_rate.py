"""Adjusted steady-state ingest rate, and the handoff decision it drives.

Every exchange preparation (``ChatSession`` streaming ingest) is one sample:
``log(seconds per 1k tokens)``. Preparation time includes every wait on the
GPU gate, so a slower rate is the measurement of local saturation.

Ingest has two regular transients that are not saturation:

* ``start`` -- a session's first preparations, which pay lazy model loads,
  cold caches and journal catch-up;
* ``stop`` -- forced drains (flush, close), which submit the whole backlog at
  once, so jobs queue behind each other by design.

Each regime keeps its own baseline, and each sample is standardised against
the baseline of its own regime. The resulting z-scores are therefore
comparable across regimes: a slow start or a long drain is ``z ~ 0`` when it is
as slow as starts and drains usually are, and only an abnormal sample moves
the score. The decision uses the last ``window`` adjusted z-scores combined as
the z of their mean, together with any in-flight preparation whose elapsed
time alone already exceeds the threshold (a lower bound, so a job stuck behind
the gate triggers the handoff before it finishes).

Hysteresis is a hold, not a second threshold: once the score passes
``enter_z`` the handoff lasts ``hold_s``, then the window is cleared and local
ingest is probed again, so only fresh local evidence can start another
episode. Samples from jobs that overlapped an episode are partly provider
time, so they are never used -- which is also why there is no clean local
signal to end an episode early. Baselines only learn from samples at or below
``enter_z``, so saturation cannot teach the baseline that saturation is
normal.

An operator can override the decision from the control plane (``POST
/_control/handoff``): ``handoff`` forces an episode, ``local`` suppresses
one, ``auto`` returns to the z-score. An override may carry a duration, after
which the decision is the z-score's again. Overrides are runtime state, never
persisted: a restart is always ``auto``. While ``local`` is forced the z-score
is still computed and reported, so the operator can see what auto would do.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

PHASES = ('start', 'steady', 'stop')
OVERRIDES = ('auto', 'handoff', 'local')
_FORMAT = 'memory-condense-ingest-baseline-v1'


class _Baseline:
    """Exponentially weighted mean and variance with a 1/n warm-up.

    While ``n <= horizon`` the weight is ``1/n``, which makes the estimates the
    exact sample mean and population variance; after that the weight stays at
    ``1/horizon`` so the baseline can follow slow drift (new drivers, a
    different model build) without forgetting it within one burst.
    """
    __slots__ = ('n', 'mean', 'var')

    def __init__(self, n=0, mean=0.0, var=0.0):
        self.n, self.mean, self.var = int(n), float(mean), float(var)

    def update(self, x, horizon):
        self.n += 1
        alpha = 1.0 / min(self.n, horizon)
        delta = x - self.mean
        self.mean += alpha * delta
        self.var = (1.0 - alpha) * (self.var + alpha * delta * delta)

    def sd(self, floor):
        return max(math.sqrt(self.var), floor)

    def as_dict(self):
        return dict(n=self.n, mean=self.mean, var=self.var)


@dataclass
class IngestTicket:
    phase: str
    tokens: int
    started: float
    episode: int


class IngestRateMonitor:
    """Service-wide, because every session's local work shares one GPU gate."""

    def __init__(self, *, enter_z=3.0, window=3, min_samples=20, start_samples=2,
                 horizon=200, min_tokens=128, min_sd=0.05, hold_s=30.0, path=None,
                 clock=time.monotonic):
        if not enter_z > 0 or hold_s < 0:
            raise ValueError('The handoff threshold must be a positive z-score')
        if window < 1 or min_samples < 2 or start_samples < 0 or horizon < min_samples:
            raise ValueError('Invalid ingest-rate sampling configuration')
        self.enter_z, self.window = float(enter_z), int(window)
        self.min_samples, self.start_samples, self.horizon = min_samples, start_samples, horizon
        self.min_tokens, self.min_sd, self.hold_s = min_tokens, min_sd, hold_s
        self.path, self.clock = (Path(path) if path is not None else None), clock
        self._lock = threading.Lock()
        self._baselines = {phase: _Baseline() for phase in PHASES}
        self._recent = deque(maxlen=self.window)
        self._inflight = {}
        self._active, self._since, self._episodes = False, None, 0
        self._last_z, self._reason = None, None
        self._excluded, self._unsaved = 0, 0
        self._override, self._override_until = 'auto', None
        self._load()

    # -- samples ------------------------------------------------------------
    def begin(self, *, tokens, ordinal, forced):
        phase = 'stop' if forced else 'start' if ordinal < self.start_samples else 'steady'
        with self._lock:
            ticket = IngestTicket(phase, max(int(tokens), 0), self.clock(), self._episodes)
            self._inflight[id(ticket)] = ticket
        return ticket

    def end(self, ticket, *, ok=True):
        with self._lock:
            self._inflight.pop(id(ticket), None)
            if not ok:
                return  # a failed preparation measures nothing about rate
            if self._active or self._episodes != ticket.episode:
                # Some of this job may have run on the provider.
                self._excluded += 1
                return
            x = self._x(self.clock() - ticket.started, ticket.tokens)
            z = self._z(ticket.phase, x)
            if z is not None:
                self._recent.append(z)
            if z is None or z <= self.enter_z:
                self._baselines[ticket.phase].update(x, self.horizon)
                self._unsaved += 1
            self._decide()
            save = self._unsaved >= 10
        if save:
            self.save()

    # -- decision -----------------------------------------------------------
    def handoff(self):
        """True while local ingest is abnormally slow for its regime."""
        with self._lock:
            self._decide()
            return self._active

    def zscore(self):
        with self._lock:
            self._decide()
            return self._last_z

    def override(self, mode, *, for_s=None):
        """Force (``handoff``), suppress (``local``) or release (``auto``) the handoff.

        ``for_s`` bounds a ``handoff`` or ``local`` override; ``None`` holds it
        until the next override. Returns the status after the change.
        """
        if mode not in OVERRIDES:
            raise ValueError(f'override must be one of {", ".join(OVERRIDES)}')
        if for_s is not None:
            if mode == 'auto':
                raise ValueError('auto takes no duration')
            if isinstance(for_s, bool) or not isinstance(for_s, (int, float)) or not for_s > 0                     or not math.isfinite(for_s):
                raise ValueError('the override duration must be a positive number of seconds')
        with self._lock:
            self._override = mode
            self._override_until = None if for_s is None else self.clock() + float(for_s)
        return self.status()

    def _expire_override(self, now):
        if self._override_until is not None and now >= self._override_until:
            self._override, self._override_until = 'auto', None

    def _start_episode(self, now, reason):
        self._active, self._since, self._reason = True, now, reason
        self._episodes += 1

    def _end_episode(self):
        # Probe: forget the evidence that started this episode; only local
        # work begun from here on can start another one.
        self._recent.clear()
        self._active, self._since, self._reason = False, None, None
        self._episodes += 1  # jobs that overlapped the episode stay excluded

    def _x(self, seconds, tokens):
        return math.log(max(seconds, 1e-6) / (max(tokens, self.min_tokens) / 1000.0))

    def _z(self, phase, x):
        baseline = self._baselines[phase]
        if baseline.n < self.min_samples:
            return None
        return (x - baseline.mean) / baseline.sd(self.min_sd)

    def _current(self):
        """The adjusted z: recent window, raised by any overdue in-flight job."""
        candidates = []
        if self._recent:
            candidates.append(sum(self._recent) / math.sqrt(len(self._recent)))
        now = self.clock()
        for ticket in self._inflight.values():
            if ticket.episode != self._episodes:
                continue
            z = self._z(ticket.phase, self._x(now - ticket.started, ticket.tokens))
            if z is not None and z > 0:
                candidates.append(z)
        return max(candidates) if candidates else None

    def _decide(self):
        now = self.clock()
        self._expire_override(now)
        warm = self._baselines['steady'].n >= self.min_samples
        if self._override == 'handoff':
            if not self._active:
                self._start_episode(now, 'override: handoff forced from the control plane')
            elif not self._reason.startswith('override'):
                self._reason = 'override: handoff forced from the control plane'
            self._last_z = self._current() if warm else None
            return
        if self._active and self._reason.startswith('override'):
            self._end_episode()  # released: probe local at once
        if self._override == 'local':
            if self._active:
                self._end_episode()
            self._last_z = self._current() if warm else None
            return
        if not warm:
            self._active, self._last_z = False, None
            return
        if self._active:
            if now - self._since < self.hold_s:
                return
            self._end_episode()
            self._last_z = None
            return
        current = self._current()
        self._last_z = current
        if current is not None and current > self.enter_z:
            self._start_episode(now, f'ingest z={current:.2f} above {self.enter_z:.2f}')

    # -- reporting and persistence -----------------------------------------
    def status(self):
        with self._lock:
            self._decide()
            baselines = {}
            for phase, baseline in self._baselines.items():
                baselines[phase] = dict(
                    samples=baseline.n,
                    seconds_per_1k_tokens=round(math.exp(baseline.mean), 4) if baseline.n else None,
                    log_sd=round(baseline.sd(self.min_sd), 4) if baseline.n else None,
                    warmed=baseline.n >= self.min_samples)
            override = dict(mode=self._override, expires_in_s=None if self._override_until is None
                            else round(max(self._override_until - self.clock(), 0.0), 1))
            return dict(handoff=self._active, z=None if self._last_z is None else round(self._last_z, 3),
                        reason=self._reason, override=override, enter_z=self.enter_z, hold_s=self.hold_s,
                        window=list(round(z, 3) for z in self._recent), in_flight=len(self._inflight),
                        excluded_samples=self._excluded, episodes=self._episodes, baselines=baselines)

    def save(self):
        if self.path is None:
            return
        with self._lock:
            value = dict(format=_FORMAT, baselines={p: b.as_dict() for p, b in self._baselines.items()})
            self._unsaved = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value), encoding='utf-8')
            os.replace(temporary, self.path)
        except OSError:
            pass  # a lost baseline costs one warm-up, never a request

    def _load(self):
        if self.path is None or not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
            if value.get('format') != _FORMAT:
                return
            for phase in PHASES:
                row = value['baselines'][phase]
                self._baselines[phase] = _Baseline(row['n'], row['mean'], row['var'])
        except (OSError, ValueError, KeyError, TypeError):
            self._baselines = {phase: _Baseline() for phase in PHASES}
