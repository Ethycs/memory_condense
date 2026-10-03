"""Proxy binding to the evaluated resident local model stack."""
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from memory_condense.application.chat_session import ChatSession
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.local import LocalRuntime
from memory_condense.runtime.resident import ResidentNativeBackend
from memory_condense.runtime.config import RuntimeAssets


class NativeProxySessions:
    """Own models once; each conversation owns its journal and native index."""
    def __init__(self, directory, *, recent_exchanges=12, recent_token_budget=8192, assets=None):
        self.directory = Path(directory)
        self.recent_exchanges, self.recent_token_budget = recent_exchanges, recent_token_budget
        self.assets = assets or RuntimeAssets.resolve()
        self.runtime = LocalRuntime(self.directory/'runtime'/uuid4().hex, assets=self.assets)
        self.model_start_lock = RLock()

    def __call__(self, namespace, directory):
        directory = Path(directory)
        actor_path = directory/'actor.json'
        actor = read(actor_path) if actor_path.exists() else dict(case_id=namespace,
            source=dict(family='proxy:'+namespace,export_timestamp=datetime.now(timezone.utc).isoformat()))
        save(actor_path,actor)
        owner = self
        class Backend(ResidentNativeBackend):
            def _open(self, rows):
                with owner.model_start_lock:
                    super()._open(rows)

            def close(self):
                # Models belong to the service, not an individual conversation.
                if self.app is not None:
                    self.app.close()
        backend = Backend(directory/'compiler',directory,actor,runtime=self.runtime)
        return ChatSession(directory/'chat','proxy:'+namespace,backend,streaming=True,
                           recent_exchanges=self.recent_exchanges,recent_token_budget=self.recent_token_budget)

    def close(self):
        self.runtime.close()
