"""Opt-in native summary retrieval on the normal durable application transcript."""
import sqlite3

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.persistence import native_spine_store


class NativeSpineWorkflowMixin:
    def install_native_spine(self, atomic_index, hierarchy, matrix, *, embedding_identity):
        """Publish compiled summary indexes after normal ``ingest_many`` completes.

        Existing ingestion may reuse compiled summary/model outputs. Every exact
        address is validated against this application's fully ingested raw text.
        The default chunk retrieval API remains available alongside this path.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError('attempt to write a readonly database')
        if self.pending_ingest_count():
            raise ValueError('complete pending raw ingestion before installing native memory')
        receipt = native_spine_store.publish(self.database_path.parent / 'native-spine.sqlite',
            atomic_index=atomic_index, hierarchy=hierarchy, matrix=matrix,
            embedding_identity=embedding_identity, turns=self.transcript.get_all())
        self._native_spine_loaded = None
        return receipt

    def _load_native_spine(self):
        # Reading the application revision also makes use after close fail and
        # detects new turns added by this or another application connection.
        revision = self._db.current_turn()
        loaded = getattr(self, '_native_spine_loaded', None)
        if loaded is None:
            if self.pending_ingest_count():
                raise ValueError('native memory requires completed raw ingestion')
            snapshot = native_spine_store.load(self.database_path.parent / 'native-spine.sqlite',
                                               turns=self.transcript.get_all())
            memory = ResidentNativeSpineContextMemory(snapshot.semantic, snapshot.hierarchy,
                encoder=self._embedder, load_turn=self.transcript.get_turn)
            loaded = (revision, snapshot, memory)
            self._native_spine_loaded = loaded
        if revision != loaded[0]:
            raise ValueError('raw transcript advanced; install an updated native snapshot')
        return loaded

    def native_spine_receipt(self):
        """Cold-admit saved summary indexes and return their source binding."""
        return dict(self._load_native_spine()[1].receipt)

    def retrieve_native_spine(self, query, dated_question, **limits):
        """Embed a fresh query and hydrate selected spans from the application DB."""
        return self._load_native_spine()[2].retrieve(query, dated_question, **limits)
