"""Opt-in native summary retrieval on the normal durable application transcript."""
import sqlite3
import os
from uuid import uuid4

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.application.native_spine_policy import CAP8_LIMITS
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.persistence import native_spine_incremental_store
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter


class NativeSpineWorkflowMixin:
    # Complete snapshots use the measured cap-8 policy. Historical parent-only
    # evaluators explicitly retain their original routing through their subclass.
    _native_spine_completion_default = True

    def capture_native_many(self, turns):
        """Store exact IO and learning chunks; native publication indexes summaries."""
        from memory_condense.application.native_capture import capture_native_many
        return capture_native_many(self, turns)

    def _native_retrievable_chunks(self, chunk_ids):
        """Admit Hebbian nodes only after their exact sources are published."""
        directory = self.database_path.parent
        if not any((directory / name).exists() for name in
                   (native_spine_incremental_store.FILENAME, 'native-spine.sqlite')):
            return set()
        _, snapshot, _ = self._load_native_spine()
        cached = getattr(self, '_native_learning_sources', None)
        if cached is None or cached[0] is not snapshot:
            cached = (snapshot, {s.spans[0].turn_id: s.spans[0].turn_text_sha256
                                 for s in snapshot.semantic.sections})
            self._native_learning_sources = cached
        admitted = set()
        for cid in chunk_ids:
            row = self._db.execute('SELECT turn_id,start_char,end_char,text FROM chunks WHERE chunk_id=?',
                                   (cid,)).fetchone()
            if row is None or row[0] not in cached[1]:
                continue
            turn = self.transcript.get_turn(row[0])
            if (turn is not None and 0 <= row[1] < row[2] <= len(turn.text)
                    and turn.text[row[1]:row[2]] == row[3]
                    and quote_sha256(turn.text) == cached[1][row[0]]):
                admitted.add(cid)
        return admitted

    def install_native_spine(self, atomic_index, hierarchy, matrix, *, embedding_identity):
        """Publish compiled summary indexes after normal ``ingest_many`` completes.

        Existing ingestion may reuse compiled summary/model outputs. Every exact
        address is validated against this application's fully ingested raw text.
        The default chunk retrieval API remains available alongside this path.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError('attempt to write a readonly database')
        if (self.database_path.parent / native_spine_incremental_store.FILENAME).exists():
            raise ValueError('use incremental publication for an existing live snapshot')
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
            incremental_path = self.database_path.parent / native_spine_incremental_store.FILENAME
            incremental = None
            if incremental_path.exists():
                incremental = native_spine_incremental_store.load(incremental_path, turns=self.transcript.get_all())
                self._native_spine_incremental = incremental
                snapshot = incremental.native
            else:
                snapshot = native_spine_store.load(self.database_path.parent / 'native-spine.sqlite',
                                                   turns=self.transcript.get_all())
            memory = ResidentNativeSpineContextMemory(snapshot.semantic, snapshot.hierarchy,
                encoder=self._embedder, load_turn=self.transcript.get_turn)
            parent_path = self.database_path.parent / native_spine_parent_store.FILENAME
            if self._native_spine_completion_default and (incremental is not None or parent_path.exists()):
                if incremental is not None:
                    parents, parent_receipt = incremental.parents, incremental.parent_receipt
                else:
                    parents, parent_receipt = native_spine_parent_store.load(parent_path,
                        hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
                memory.router = NativeSpineUserCompletionRouter(
                    snapshot.semantic, snapshot.hierarchy, parents)
                self._native_parent_user_receipt = parent_receipt
            loaded = (revision, snapshot, memory)
            self._native_spine_loaded = loaded
        if revision != loaded[0]:
            raise ValueError('raw transcript advanced; install an updated native snapshot')
        return loaded

    def install_native_spine_incremental(self, atomic_index, hierarchy, matrix, *,
                                         embedding_identity, projection, parent_matrix):
        """Commit changed native/parent rows together, retaining the warm indexes."""
        if self._db.read_only:
            raise sqlite3.OperationalError('attempt to write a readonly database')
        if self.pending_ingest_count():
            raise ValueError('complete pending raw ingestion before installing native memory')
        directory = self.database_path.parent
        destination = directory / native_spine_incremental_store.FILENAME
        previous = getattr(self, '_native_spine_incremental', None)
        # A fresh connection may append before recalling. Admit the persisted
        # prefix before using its row identities for incremental publication.
        turns = self.transcript.native_snapshot()
        if previous is None and destination.exists():
            count = native_spine_incremental_store.saved_turn_count(destination)
            previous = native_spine_incremental_store.load(destination, turns=turns[:count])
        revision = self._db.current_turn()
        source_revision = self.transcript.source_revision()
        def still_current():
            if revision != self._db.current_turn() or source_revision != self.transcript.source_revision():
                raise ValueError('raw transcript advanced during native publication')
        path = destination if previous is not None else directory / (uuid4().hex + '.live.sqlite')
        try:
            state = native_spine_incremental_store.publish(path, atomic_index=atomic_index,
                hierarchy=hierarchy, matrix=matrix, projection=projection, parent_matrix=parent_matrix,
                embedding_identity=embedding_identity, turns=turns, previous=previous,
                before_commit=still_current)
            snapshot = state.native
            memory = ResidentNativeSpineContextMemory(snapshot.semantic, hierarchy,
                encoder=self._embedder, load_turn=self.transcript.get_turn)
            if self._native_spine_completion_default:
                memory.router = NativeSpineUserCompletionRouter(snapshot.semantic, hierarchy, state.parents)
            if path != destination:
                os.replace(path, destination)
            self._native_spine_incremental = state
            self._native_parent_user_receipt = state.parent_receipt
            self._native_spine_loaded = (revision, snapshot, memory)
            return dict(snapshot.receipt)
        finally:
            if path != destination:
                path.unlink(missing_ok=True)

    def install_native_spine_complete(self, atomic_index, hierarchy, matrix, *,
                                      embedding_identity, parent_matrix):
        """Publish both indexes and keep their verified objects in this session.

        Cold opens still validate persisted bytes and every raw address. Staged
        files prevent a validation failure from replacing a valid pair. A crash
        between the two replacements is detected by their cross-bound receipts.
        """
        if self._db.read_only:
            raise sqlite3.OperationalError('attempt to write a readonly database')
        if (self.database_path.parent / native_spine_incremental_store.FILENAME).exists():
            raise ValueError('use incremental publication for an existing live snapshot')
        if self.pending_ingest_count():
            raise ValueError('complete pending raw ingestion before installing native memory')
        directory = self.database_path.parent
        token = uuid4().hex
        native_path = directory / (token + '.native.sqlite')
        parent_path = directory / (token + '.parents.sqlite')
        revision = self._db.current_turn()
        try:
            snapshot = native_spine_store.publish_snapshot(native_path,
                atomic_index=atomic_index, hierarchy=hierarchy, matrix=matrix,
                embedding_identity=embedding_identity, turns=self.transcript.get_all())
            parents, parent_receipt = native_spine_parent_store.publish_snapshot(parent_path,
                hierarchy=hierarchy, matrix=parent_matrix, native_receipt=snapshot.receipt)
            memory = ResidentNativeSpineContextMemory(snapshot.semantic, snapshot.hierarchy,
                encoder=self._embedder, load_turn=self.transcript.get_turn)
            if self._native_spine_completion_default:
                memory.router = NativeSpineUserCompletionRouter(snapshot.semantic, hierarchy, parents)
            if revision != self._db.current_turn():
                raise ValueError('raw transcript advanced during native publication')
            self._native_spine_loaded = None
            os.replace(native_path, directory / 'native-spine.sqlite')
            os.replace(parent_path, directory / native_spine_parent_store.FILENAME)
            self._native_parent_user_receipt = parent_receipt
            self._native_spine_loaded = (revision, snapshot, memory)
            return dict(snapshot.receipt)
        finally:
            native_path.unlink(missing_ok=True)
            parent_path.unlink(missing_ok=True)

    def native_spine_receipt(self):
        """Cold-admit saved summary indexes and return their source binding."""
        return dict(self._load_native_spine()[1].receipt)

    def retrieve_native_spine(self, query, dated_question, **limits):
        """Retrieve exact evidence, using cap-8 when parent summaries are installed.

        Complete native snapshots automatically use the tested 2,048-token,
        eight-direct-route policy with at most eight additional user atoms.
        Explicit limits override these defaults. Older snapshots without a
        parent index keep their original context router; an invalid parent
        index raises instead of silently falling back. Ordinary chunk retrieval
        remains a separate API.
        """
        memory = self._load_native_spine()[2]
        if isinstance(memory.router, NativeSpineUserCompletionRouter):
            limits = {**CAP8_LIMITS, **limits}
        return memory.retrieve(query, dated_question, **limits)

    def native_parent_user_receipt(self):
        """Return the validated parent index binding for a complete snapshot."""
        self._load_native_spine()
        if not isinstance(self._native_spine_loaded[2].router, NativeSpineUserCompletionRouter):
            raise ValueError('native parent-user summary index is not installed')
        return dict(self._native_parent_user_receipt)
