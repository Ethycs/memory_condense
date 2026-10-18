"""Proxy binding to the evaluated resident local model stack."""
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4

from memory_condense.application.chat_session import ChatSession
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.local import LocalRuntime
from memory_condense.runtime.resident import ResidentNativeBackend
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.semantic_summary import (
    SEMANTIC_VALIDATION_POLICY,
    SEMANTIC_VALIDATION_SHA256,
    SemanticLocalFirstSummaries,
    policy_root_name,
)
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy


class NativeProxySessions:
    """Own models once; each conversation owns its journal and native index."""
    def __init__(self, directory, *, recent_exchanges=12, recent_token_budget=8192, assets=None,
                 summary_api_fallback=True, dialogue_policy=None, retrieval_policy='auto', experiment_policy=None):
        if dialogue_policy is not None and type(dialogue_policy) is not DialogueRetrievalPolicy:
            raise TypeError('Expected a validated DialogueRetrievalPolicy')
        self.directory = Path(directory)
        if retrieval_policy not in ('auto', 'cap8', 'dialogue', 'hierarchy_neighborhoods', 'decision_history'):
            raise ValueError('Unknown memory retrieval policy')
        if retrieval_policy == 'cap8' and dialogue_policy is not None:
            raise ValueError('cap8 cannot configure dialogue retrieval')
        self.retrieval_policy = retrieval_policy
        if experiment_policy is not None:
            from memory_condense.application.native_experiments import NativeExperimentPolicy
            if type(experiment_policy) is not NativeExperimentPolicy:
                raise TypeError('Expected NativeExperimentPolicy')
        self.experiment_policy = experiment_policy
        self.dialogue_policy = (dialogue_policy or DialogueRetrievalPolicy()
                               if retrieval_policy != 'cap8' else None)
        self.summary_api_fallback = summary_api_fallback
        self.recent_exchanges, self.recent_token_budget = recent_exchanges, recent_token_budget
        self.assets = assets or RuntimeAssets.resolve()
        self.runtime = LocalRuntime(self.directory/'runtime'/uuid4().hex, assets=self.assets)
        # model_start_lock serialises actual model construction and is therefore
        # held for minutes on a cold start. Registry reads must never queue
        # behind it, or status reporting waits on the very work it describes.
        self.model_start_lock = RLock()
        self.registry_lock = RLock()
        self.summary_runtimes = {}
        self._lifecycle = {}

    # -- lifecycle ---------------------------------------------------------
    # Observations are recorded under registry_lock only, so a status request
    # can read the current stage of a session whose models are still loading.
    def _observe(self, namespace, stage, *, error=None):
        now = time.monotonic()
        with self.registry_lock:
            record = self._lifecycle.get(namespace)
            if record is None:
                record = dict(started_at=now, stage=None, stage_started_at=now, error=None)
                self._lifecycle[namespace] = record
            if record['stage'] != stage:
                record['stage'], record['stage_started_at'] = stage, now
            record['error'] = error

    def startup_status(self):
        """Per-namespace stage and elapsed time; never blocks on model work."""
        now = time.monotonic()
        with self.registry_lock:
            records = {namespace: dict(record) for namespace, record in self._lifecycle.items()}
        return {namespace: dict(stage=record['stage'], error=record['error'],
                                elapsed_s=round(now-record['started_at'], 3),
                                stage_elapsed_s=round(now-record['stage_started_at'], 3))
                for namespace, record in records.items()}

    def _summary_runtime(self, namespace, api, *, configure):
        with self.registry_lock:
            runtime = self.summary_runtimes.get(namespace)
            if runtime is None:
                runtime = SemanticLocalFirstSummaries(self.runtime,
                    self.directory/'sessions'/namespace/'summary-fallback'/policy_root_name(), api,
                    require_provider=self.summary_api_fallback)
                self.summary_runtimes[namespace] = runtime
                return runtime
        if configure:
            runtime.configure(api)
        return runtime

    def configure_summary_provider(self, namespace, api):
        self._summary_runtime(namespace, api, configure=True)

    def summary_status(self):
        from collections import Counter
        counts = Counter()
        with self.registry_lock:
            runtimes = list(self.summary_runtimes.values())
        for runtime in runtimes:
            with runtime._lock:
                counts.update(runtime.metrics)
        return dict(counts, validation_policy=SEMANTIC_VALIDATION_POLICY,
                    validation_policy_sha256=SEMANTIC_VALIDATION_SHA256)

    def retrieval_status(self):
        result = dict(mode=self.retrieval_policy,
                      policy=self.dialogue_policy.identity_payload() if self.dialogue_policy else None)
        if self.experiment_policy is not None:
            result['experiments'] = self.experiment_policy.identity_payload()
        return result

    def __call__(self, namespace, directory):
        directory = Path(directory)
        self._observe(namespace, 'constructing')
        summary_runtime = self._summary_runtime(namespace, None, configure=False)
        actor_path = directory/'actor.json'
        actor = read(actor_path) if actor_path.exists() else dict(case_id=namespace,
            source=dict(family='proxy:'+namespace,export_timestamp=datetime.now(timezone.utc).isoformat()))
        save(actor_path,actor)
        owner = self
        class Backend(ResidentNativeBackend):
            def _open(self, rows):
                # Each stage is published before the wait it describes, so a
                # cold start is visible as "waiting" or "loading", never idle.
                owner._observe(namespace, 'awaiting_model_start')
                with owner.model_start_lock:
                    owner._observe(namespace, 'opening_models')
                    try:
                        super()._open(rows)
                    except BaseException as exc:
                        owner._observe(namespace, 'failed', error=f'{type(exc).__name__}: {exc}')
                        raise
                owner._observe(namespace, 'ready')

            def close(self):
                # Models belong to the service, not an individual conversation.
                if self.app is not None:
                    self.app.close()
        backend = Backend(directory/'compiler',directory,actor,
                          runtime=summary_runtime, dialogue_policy=self.dialogue_policy,
                          retrieval_policy=self.retrieval_policy,experiment_policy=self.experiment_policy)
        # Constructing the session schedules its first sync immediately, so the
        # stage must be set before that worker can report a later one.
        self._observe(namespace, 'admitting')
        return ChatSession(directory/'chat','proxy:'+namespace,backend,streaming=True,
                           recent_exchanges=self.recent_exchanges,recent_token_budget=self.recent_token_budget)

    def close(self):
        self.runtime.close()
