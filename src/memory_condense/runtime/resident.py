"""Resident implementation of the live chat compiler and cap-eight reader.

Independent jobs prepare exchanges; one writer publishes. Readers use the last committed
immutable index and its authenticated raw turn snapshot. Model operations share
a short lock; generation and index publication do not hold up recall.
"""
from dataclasses import asdict, dataclass
from collections import Counter, OrderedDict, deque
from contextlib import closing, contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime
from functools import wraps
import json
from pathlib import Path
import sqlite3
import time
import threading
from types import SimpleNamespace, MappingProxyType

import numpy as np

from memory_condense.application.chat_native import native_packet, learn_native_packet
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_policy import CAP8_LIMITS
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.persistence import native_spine_incremental_store
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.episodes.parent_budgeted_spine_hierarchy import build_parent_budgeted_spine_hierarchy
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from memory_condense.search.summary_semantic_index import summary_embedding_identity, compatible_summary_embedding
from memory_condense.runtime.attention import cache_method, user_windows
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.compiler import Compiler, LocalAttention, PreparationSuperseded, storage_rows
from memory_condense.runtime.seed import load_seed
from memory_condense.runtime.helpers import StagedEmbedding


def model_operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.model_lock:
            return method(self, *args, **kwargs)
    return guarded


class RecallPriorityLock:
    """Reentrant model admission with FIFO queues and foreground priority.

    A running operation finishes normally. On release, queued recalls precede
    every queued background operation, including jobs that arrived earlier.
    Admission and priority share one condition so there is no check/lock race.
    """
    def __init__(self):
        self._condition = threading.Condition()
        self._foreground, self._background = deque(), deque()
        self._owner, self._depth = None, 0

    def _acquire(self, foreground=False):
        owner = threading.get_ident()
        with self._condition:
            if self._owner == owner:
                self._depth += 1
                return
            queue = self._foreground if foreground else self._background
            ticket = object()
            queue.append(ticket)
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: self._owner is None
                    and queue[0] is ticket and (foreground or not self._foreground))
                self._owner, self._depth = owner, 1
            finally:
                queue.remove(ticket)
                self._condition.notify_all()

    def _release(self):
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError('Model admission released by a non-owner')
            self._depth -= 1
            if not self._depth:
                self._owner = None
                self._condition.notify_all()

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, *_exc):
        self._release()

    @contextmanager
    def foreground(self):
        self._acquire(foreground=True)
        try:
            yield
        finally:
            self._release()


class SharedEmbedding:
    """Keep one encoder resident, serializing only actual model operations."""
    def __init__(self, encoder, *, background_batch_size=None):
        self.encoder, self.model_lock = encoder, RecallPriorityLock()
        if background_batch_size is None:
            device = str(encoder.execution_identity.get('device', 'auto'))
            background_batch_size = 8 if device.startswith('cuda') else 1
        if type(background_batch_size) is not int or background_batch_size < 1:
            raise ValueError('Background embedding batch size must be a positive integer')
        self.background_batch_size = background_batch_size
        self._chunk_vectors = OrderedDict()
        self._cache_lock = threading.Lock()
        self.metrics = Counter()

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    @model_operation
    def _load_model(self):
        return self.encoder._load_model()

    def embed_query(self, query):
        waiting=time.perf_counter()
        with self.model_lock.foreground():
            elapsed=time.perf_counter()-waiting
            self.metrics['query_lock_wait_s'] += elapsed
            self.metrics['query_lock_wait_max_s'] = max(self.metrics['query_lock_wait_max_s'],elapsed)
            started=time.perf_counter()
            result=self.encoder.embed_query(query)
            self.metrics['query_compute_s'] += time.perf_counter()-started
            self.metrics['query_calls'] += 1
            return result

    def _background(self, call, values, metric):
        waiting=time.perf_counter()
        with self.model_lock:
            self.metrics[metric+'_lock_wait_s'] += time.perf_counter()-waiting
            started=time.perf_counter()
            result=call(values)
            elapsed=time.perf_counter()-started
            self.metrics[metric+'_compute_s'] += elapsed
            self.metrics[metric+'_batch_max_s'] = max(self.metrics[metric+'_batch_max_s'],elapsed)
            self.metrics[metric+'_batches'] += 1
            return result

    def embed_queries(self, queries):
        if not queries:
            return self.encoder.embed_queries(queries)
        size=self.background_batch_size
        return np.concatenate([self._background(self.encoder.embed_queries,queries[i:i+size],'summary')
                               for i in range(0,len(queries),size)],axis=0)

    def embed_chunks(self, chunks):
        # Chunk vectors depend on exact text and encoder identity, not turn
        # coordinates. Keep source ownership on each returned chunk unchanged.
        identity=summary_embedding_identity(self.encoder)
        keys=[(identity,c.text) for c in chunks]
        values,missing={},{}
        with self._cache_lock:
            for key,chunk in zip(keys,chunks):
                if key in self._chunk_vectors:
                    values[key]=self._chunk_vectors[key]
                    self._chunk_vectors.move_to_end(key)
                    self.metrics['chunk_cache_hits'] += 1
                elif key in missing:
                    self.metrics['chunk_batch_duplicates'] += 1
                else:
                    missing[key]=chunk
        pending=list(missing.items())
        size=self.background_batch_size
        for i in range(0,len(pending),size):
            batch=pending[i:i+size]
            generated=self._background(self.encoder.embed_chunks,[c for _,c in batch],'chunk')
            if len(generated)!=len(batch):
                raise ValueError('Chunk embedding cardinality changed')
            for (key,source),value in zip(batch,generated,strict=True):
                if (source.model_copy(update={'embedding':value.embedding,'lexical_weights':value.lexical_weights})!=value
                        or value.embedding is None or not np.isfinite(value.embedding).all()):
                    raise ValueError('Chunk embedding changed source ownership or vector validity')
                stored=(tuple(value.embedding),dict(value.lexical_weights) if value.lexical_weights is not None else None)
                values[key]=stored
                with self._cache_lock:
                    self._chunk_vectors[key]=stored
                    self.metrics['chunks_embedded'] += 1
                    while len(self._chunk_vectors)>1024:
                        self._chunk_vectors.popitem(last=False)
        return [chunk.model_copy(update={'embedding':list(values[key][0]),
                    'lexical_weights':dict(values[key][1]) if values[key][1] is not None else None})
                for key,chunk in zip(keys,chunks)]

    @model_operation
    def park(self):
        return self.encoder.park()

    @model_operation
    def close(self):
        return self.encoder.close()


class ResidentAttention(LocalAttention):
    def __init__(self, root, preflight, embedding, *, retain_gpu=True):
        super().__init__(root, preflight)
        self.embedding = embedding
        self.model_lock = embedding.model_lock
        self.metrics = Counter()
        self.retain_gpu = retain_gpu
        self.host_embeddings = retain_gpu

    @model_operation
    def prepare(self):
        """Admit the resident footprint at startup, before interactive work."""
        import torch
        self.embedding._load_model()
        free, _ = torch.cuda.mem_get_info()
        # The six FP16 layers occupy about 2.2 GiB without the vocabulary
        # table. Leave workspace headroom beyond their persistent allocation.
        if free < 3.0 * 1024**3:
            self.retain_gpu = self.host_embeddings = False
        if not self.retain_gpu:
            self.embedding.park()
        self.load_scorer()
        self.park()

    def load_scorer(self):
        started = time.perf_counter()
        super().load_scorer()
        self.metrics['model_load_s'] += time.perf_counter()-started
        self.metrics['model_loads'] += 1

    @model_operation
    def score_sequence(self, texts):
        started = time.perf_counter()
        key = identity_sha256({'preflight_sha256': self.preflight.sha256, 'texts': list(texts)})
        if key not in self.values and not (self.root / 'attention' / (key + '.json')).exists():
            self.metrics['cache_misses'] += 1
            now = time.perf_counter()
            if not self.retain_gpu:
                self.embedding.park()
            self.metrics['embedding_park_s'] += time.perf_counter()-now
            if self.scorer is not None and not self.retain_gpu:
                now = time.perf_counter()
                encoder = self.scorer.linker.encoder
                encoder.model.to(encoder.device)
                self.metrics['model_resume_s'] += time.perf_counter()-now
        else:
            self.metrics['cache_hits'] += 1
        try:
            result = super().score_sequence(texts)
        finally:
            # In low-memory staging mode return Qwen to CPU before releasing
            # the lock, so a reader can safely resume BGE immediately.
            self.park()
        self.metrics['score_sequence_s'] += time.perf_counter()-started
        return result

    @model_operation
    def park(self):
        if self.scorer is not None and not self.retain_gpu:
            self.scorer.linker.encoder.model.to('cpu')
            import torch
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class PreparedExchange:
    events: tuple
    atoms: tuple
    exchanges: tuple


class ResidentNativeBackend:
    """Append-only live memory; cold validation once, durable publication per sync."""
    def __init__(self, run, arm_root, actor, *, runtime=None):
        self.run, self.arm_root, self.actor = Path(run), Path(arm_root), actor
        self.runtime = runtime
        self.rows, self.events, self.live_rows, self.atoms = [], (), [], ()
        self.app = self.encoder = self.attention = self.seed = None
        self.vectors = {}
        self.last_reopen = None
        self.timings = []
        self._timing_lock = threading.Lock()
        self._atomic_index = self._hierarchy_index = None
        self._parent_projection = None
        self._published = None
        self._publication_lock = threading.Lock()
        self._stream_base = None
        self._stream_prepared = ()
        self._stream_groups = {}
        self._stream_finalized = 0
        self._stream_needs_finalization = False

    def start_stream(self):
        """Writer-owned admission; workers never open or mutate application DBs."""
        if self.app is None:
            self._open(self.rows)
        if self._stream_base is None:
            if self.last_reopen is None:
                self._stream_base = ((), ())
            else:
                snapshot = self.app._load_native_spine()[1]
                self._stream_base = (snapshot.semantic.hierarchy.sections, snapshot.hierarchy.sections)
                self._atomic_index = snapshot.semantic.hierarchy
                self._hierarchy_index = snapshot.hierarchy
                incremental = getattr(self.app, '_native_spine_incremental', None)
                if incremental is not None:
                    self._parent_projection = incremental.parents.hierarchy
            self._stream_originals = self._stream_base[0]

    def prepare_exchange(self, events):
        """Independent, cache-only work over one completed durable exchange."""
        started = time.perf_counter()
        rows = storage_rows([e.row(self.actor['source']['family']) for e in events])
        if self.seed and any(r['source_id'] in {t.source_id for t in self.seed.turns} for r in rows):
            raise ValueError('A continuation must have its own live source identity')
        atoms = self.compiler.atoms(rows,self.actor['source']['family'],self.actor['source']['export_timestamp'],
                                    original_atoms=self._stream_originals)
        exchanges = compile_user_spine_exchanges(atoms,summarize=self.compiler.summarizer(),
            summarizer_identity='engineering-battery-summary-only-v1',max_channel_tokens=128,
            max_prompt_tokens=2048,max_workers=1)
        self.timings.append(dict(operation='prepare_exchange',elapsed_s=time.perf_counter()-started,
                                 events=len(events),first_event_id=events[0].event_id))
        return PreparedExchange(tuple(events),atoms,exchanges)

    def _compile_stream(self, prepared, *, finalize=False):
        """Seal eight-exchange attention groups once; keep the open tail exact."""
        started=time.perf_counter()
        atoms=tuple(a for p in prepared for a in p.atoms)
        exchanges=tuple(e for p in prepared for e in p.exchanges)
        sources={}
        for exchange in exchanges:
            sources.setdefault(exchange.section.source_id,[]).append(exchange)
        sections=[]
        sealed=open_exchanges=0
        for group in sources.values():
            for start in range(0,len(group),8):
                block=tuple(group[start:start+8])
                key=tuple(e.receipt_sha256 for e in block)
                if len(block)==8 or finalize:
                    if key not in self._stream_groups:
                        self._stream_groups[key]=build_parent_budgeted_spine_hierarchy(block,
                            scorer=self.attention,summarize=self.compiler.summarizer(),
                            summarizer_identity='engineering-battery-summary-only-v1',leaf_token_cap=512,
                            max_leaf_exchanges=2,max_exchange_channel_tokens=128,max_parent_channel_tokens=512,
                            window_exchange_cap=8,max_prompt_tokens=2048,max_workers=3).sections
                    sections.extend(self._stream_groups[key])
                    sealed+=1
                else:
                    sections.extend(e.section for e in block)
                    open_exchanges+=len(block)
        hierarchy_s=time.perf_counter()-started
        old_atoms,old_hierarchy=self._stream_base
        atomic=SectionSummaryIndex((*old_atoms,*atoms),previous=self._atomic_index)
        hierarchy=SectionSummaryIndex((*old_hierarchy,*sections),previous=self._hierarchy_index)
        self._atomic_index,self._hierarchy_index=atomic,hierarchy
        return atomic,hierarchy,dict(total_s=time.perf_counter()-started,hierarchy_s=hierarchy_s,
            sealed_groups=sealed,open_exchanges=open_exchanges,streaming=True,finalize=finalize,exchange_count=len(exchanges))

    def sync_prepared(self, events, prepared):
        # Failed publication can retry these same immutable results. Advance
        # the accepted preparation list only after durable publication succeeds.
        if any(type(p) is not PreparedExchange for p in prepared):
            raise TypeError('Expected authenticated exchange preparations')
        if tuple(e for p in prepared for e in p.events)!=tuple(events[len(self.rows):]):
            raise ValueError('Prepared exchanges do not match the next chronological prefix')
        combined=(*self._stream_prepared,*prepared)
        self.sync(events,_compiler=lambda _:self._compile_stream(combined))
        self._stream_prepared=combined
        self._stream_needs_finalization=any(n%8 for n in Counter(
            e.section.source_id for p in combined for e in p.exchanges).values())
        self._stream_originals=(*self._stream_base[0],*(a for p in combined for a in p.atoms))

    def finalize_stream(self, events):
        if not self._stream_prepared or not self._stream_needs_finalization or self._stream_finalized==len(events):
            return
        self.sync(events,_compiler=lambda _:self._compile_stream(self._stream_prepared,finalize=True))
        self._stream_finalized=len(events)
        self._stream_needs_finalization=False

    def _open(self, rows):
        import torch
        torch.set_num_threads(4)
        self.compiler = Compiler(self.run, self.actor['case_id'] + '/live', report=lambda **_: None)
        if self.runtime is not None:
            self.compiler.gateway = self.runtime
        if self.actor.get('native_seed'):
            self.seed = load_seed(self.actor['native_seed'], rows)
        self.encoder = (self.runtime.embedding() if self.runtime is not None else
                        SharedEmbedding(StagedEmbedding(device='cuda', batch_size=8)))
        self.embedding_identity = summary_embedding_identity(self.encoder)
        if self.seed and not compatible_summary_embedding(self.encoder, self.seed.embedding_identity):
            raise ValueError('Live encoder differs from the native seed')
        cache = self.run / 'cache' / 'attention'
        self.attention = (self.runtime.attention(cache) if self.runtime is not None else
                          ResidentAttention(cache, cache_method(cache, versioned=True), self.encoder))
        prepare = getattr(self.attention, 'prepare', None)
        if prepare is not None:
            prepare()
            self.attention.preflight = cache_method(cache, host_embeddings=self.attention.host_embeddings, versioned=True)
        if self.runtime is not None:
            self.runtime.start_reader()
        self.app = MemoryCondenser(self.arm_root / 'store' / 'memory',
                                   embedder=self.encoder, auto_extract=False)

    def _stamp(self, ordinal):
        if self.seed and ordinal < len(self.seed.turns):
            return self.seed.turns[ordinal].created_at
        return datetime.fromisoformat(self.actor['source']['export_timestamp'])

    def _check_raw_prefix(self, rows):
        existing = self.app.transcript.native_snapshot()
        if len(existing) > len(rows) or any(
            (t.turn_id, t.source_id, t.role, t.text, t.created_at) !=
            (r['turn_id'], r['source_id'], r['role'], r['text'], self._stamp(i))
            for i, (t, r) in enumerate(zip(existing, rows))):
            raise ValueError('Memory is not the exact chronological prefix')
        return len(existing)

    def matrix(self, index):
        result, missing = [], {}
        for section in index.sections:
            if self.seed and section.section_id in self.seed.vectors:
                summary, vector = self.seed.vectors[section.section_id]
                if summary != section.summary:
                    raise ValueError('Seed summary changed while reusing its vector')
                result.append(vector)
                continue
            text = section.summary
            if text not in self.vectors:
                key = identity_sha256(dict(text=text, embedding=self.embedding_identity))
                path = self.run / 'cache' / 'vectors' / (key + '.json')
                if path.exists():
                    self.vectors[text] = np.asarray(read(path)['vector'], dtype=np.float32)
                else:
                    missing[text] = path
            result.append(text)
        if missing:
            values = np.asarray(self.encoder.embed_queries(list(missing)), dtype=np.float32)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if not np.isfinite(values).all() or np.any(norms == 0):
                raise ValueError('Summary embeddings must be finite and nonzero')
            values /= norms
            for (text, path), vector in zip(missing.items(), values, strict=True):
                save(path, dict(vector=vector.tolist()))
                self.vectors[text] = vector
        return np.asarray([self.vectors[v] if isinstance(v, str) else v for v in result], dtype=np.float32)

    def _record_timing(self, value):
        with self._timing_lock:
            self.timings.append(value)
            try:
                self.arm_root.mkdir(parents=True, exist_ok=True)
                with (self.arm_root/'ingestion-timings.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(value, sort_keys=True)+'\n')
            except OSError as exc:
                # Diagnostics must not change the durable publication outcome.
                value['timing_log_error'] = str(exc)

    def sync(self, events, *, _compiler=None):
        started = time.perf_counter()
        self._sync_details = dict(phase='admission')
        try:
            return self._sync_native(events, _compiler=_compiler)
        except BaseException as exc:
            self._record_timing(dict(operation='sync_failed', history_turns=len(events),
                elapsed_s=time.perf_counter()-started, error=f'{type(exc).__name__}: {exc}',
                **self._sync_details))
            raise

    def _sync_native(self, events, *, _compiler=None):
        started = time.perf_counter()
        original = [e.row(self.actor['source']['family']) for e in events]
        if len(original) < len(self.rows) or original[:len(self.rows)] != self.rows:
            raise ValueError('Chat memory can only fast-forward the acknowledged turn prefix')
        if self.last_reopen is not None and original == self.rows and _compiler is None:
            return
        rows = storage_rows(original)
        if self.app is None:
            self._open(rows)
        embedding_before=dict(self.encoder.metrics)
        existing_count = self._check_raw_prefix(rows)
        prefix = len(self.seed.turns) if self.seed else 0
        seed_sources = {t.source_id for t in self.seed.turns} if self.seed else set()
        if any(r['source_id'] in seed_sources for r in rows[prefix:]):
            raise ValueError('A continuation must have its own live source identity')
        # Admit complete durable indexes once. Changed persisted bytes fail
        # closed; they must never become trusted merely because caches exist.
        directory = self.app.database_path.parent
        if (self.last_reopen is None and existing_count == len(rows)
                and not self.app.pending_ingest_count()
                and ((directory / native_spine_incremental_store.FILENAME).exists()
                     or ((directory / 'native-spine.sqlite').exists()
                         and (directory / 'native-spine-parent-users-v1.sqlite').exists()))):
            path = directory / 'native-spine.sqlite'
            live_path = directory / native_spine_incremental_store.FILENAME
            if live_path.exists():
                count = native_spine_incremental_store.saved_turn_count(live_path)
            else:
                with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                    saved = db.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()
                count = json.loads(saved[0])['turn_count'] if saved else None
            if type(count) is not int or not 0 < count <= len(rows):
                raise ValueError('Persisted native prefix is invalid')
            if count == len(rows):
                snapshot = self.app.native_spine_receipt()
                parents = self.app.native_parent_user_receipt()
                migration_s = 0.0
                if not live_path.exists():
                    # Convert older checkpoints during cold admission. The
                    # first interactive append must already have warm source
                    # validation and stable per-section publication receipts.
                    migration_start = time.perf_counter()
                    _, admitted, memory = self.app._load_native_spine()
                    old_parents = memory.router.parent_semantic
                    by_root = {json.loads(s.summarizer_identity)['original_root_sha256']: vector
                        for s,vector in zip(old_parents.sections, old_parents._dense._matrix, strict=True)}
                    projection = project_parent_users(admitted.hierarchy, stable_ids=True)
                    parent_matrix = np.asarray([by_root[json.loads(s.summarizer_identity)['original_root_sha256']]
                                                for s in projection.sections], dtype=np.float32)
                    snapshot = self.app.install_native_spine_incremental(admitted.semantic.hierarchy,
                        admitted.hierarchy, admitted.semantic._dense._matrix,
                        embedding_identity=admitted.semantic.embedding_identity,
                        projection=projection, parent_matrix=parent_matrix)
                    parents = self.app.native_parent_user_receipt()
                    migration_s = time.perf_counter()-migration_start
                self._ack(events, original, snapshot, parents, started,
                          {'cold_admission': True, 'legacy_migration_s': migration_s})
                return
            # A crash may leave fully indexed raw turns ahead of the last
            # published pair. Authenticate that *older* prefix before repair;
            # corrupted bytes or a mismatched pair still fail closed.
            if live_path.exists():
                self.app._native_spine_incremental = native_spine_incremental_store.load(
                    live_path, turns=self.app.transcript.get_all()[:count])
            else:
                old = native_spine_store.load(path, turns=self.app.transcript.get_all()[:count])
                native_spine_parent_store.load(directory / native_spine_parent_store.FILENAME,
                    hierarchy=old.hierarchy, native_receipt=old.receipt)
        new = [(r['role'], r['text'], r['source_id'], self._stamp(i), r['turn_id'])
               for i, r in enumerate(rows) if i >= existing_count]
        preparation_start = time.perf_counter()
        self._sync_details.update(phase='capture_and_compile', new_turns=len(new))
        # Compilation owns no application DB connection. Generation can run
        # while the single writer captures exact IO and its learning topology;
        # the existing lock. Publish only after BOTH prerequisites succeed.
        with ThreadPoolExecutor(max_workers=1) as pool:
            compilation = pool.submit(_compiler or self._compile, rows[prefix:])
            ingest_start = time.perf_counter()
            for i in range(0, len(new), 32):
                self.app.capture_native_many(new[i:i+32])
            if self.app.pending_ingest_count():
                self.app.recover_pending_ingests()
            raw_ingest_s = time.perf_counter()-ingest_start
            self._sync_details['raw_ingest_s'] = raw_ingest_s
            atomic, hierarchy, compile_timings = compilation.result()
        prepared = time.perf_counter()
        self._sync_details.update(phase='summary_embedding', compile_timings=compile_timings)
        matrix = self.matrix(atomic)
        self._parent_projection = project_parent_users(hierarchy, stable_ids=True, previous=self._parent_projection)
        parent_matrix = self.matrix(self._parent_projection)
        embedded = time.perf_counter()
        self._sync_details.update(phase='publication', summary_embedding_s=embedded-prepared)
        snapshot = self.app.install_native_spine_incremental(atomic, hierarchy, matrix,
            embedding_identity=self.embedding_identity, projection=self._parent_projection, parent_matrix=parent_matrix)
        self._ack(events, original, snapshot, self.app.native_parent_user_receipt(), started,
            dict(cold_admission=False, new_turns=len(new), compile_s=compile_timings['total_s'],
                 compile_timings=compile_timings,
                 raw_ingest_s=raw_ingest_s, preparation_wall_s=prepared-preparation_start,
                 compilation_overlaps_raw_ingest=True, summary_embedding_s=embedded-prepared,
                 embedding_detail={k:v-embedding_before.get(k,0) for k,v in self.encoder.metrics.items()},
                 publish_s=time.perf_counter()-embedded))

    def prepare(self, events, *, should_yield=lambda:False):
        """Warm summary caches for a durable journal prefix without publication.

        Called only by the session's existing single writer. The next sync still
        captures new raw events, embeds new summaries, publishes, then learns.
        """
        if self.last_reopen is None:
            return  # Initial admission stays on the ordinary synchronization path.
        started=time.perf_counter()
        original=[e.row(self.actor['source']['family']) for e in events]
        if len(original)<len(self.rows) or original[:len(self.rows)]!=self.rows:
            raise ValueError('Preparation must extend the published chat prefix')
        rows=storage_rows(original)
        prefix=len(self.seed.turns) if self.seed else 0
        seed_sources={t.source_id for t in self.seed.turns} if self.seed else set()
        if any(r['source_id'] in seed_sources for r in rows[prefix:]):
            raise ValueError('A continuation must have its own live source identity')
        superseded,detail=False,None
        self.compiler.preparation_should_yield=should_yield
        try:
            _,_,detail=self._compile(rows[prefix:])
        except PreparationSuperseded:
            superseded=True
        finally:
            self.compiler.preparation_should_yield=None
        self.timings.append(dict(operation='prepare',elapsed_s=time.perf_counter()-started,
                                 history_turns=len(rows),compile_timings=detail,superseded=superseded))

    def _compile(self, live):
        """Compile the live suffix and combine it with the immutable seed."""
        phases = {}
        attention_before = dict(getattr(self.attention, 'metrics', {}))
        started = time.perf_counter()
        if live[:len(self.live_rows)] != self.live_rows:
            raise ValueError('Live compilation prefix changed')
        additions = self.compiler.atoms(live[len(self.live_rows):],
            self.actor['source']['family'], self.actor['source']['export_timestamp'],
            original_atoms=(*(self.seed.atoms if self.seed else ()), *self.atoms)) if len(live) > len(self.live_rows) else ()
        self.atoms = (*self.atoms, *additions)
        self.live_rows = live
        now = time.perf_counter()
        phases['atoms_s'] = now-started
        summarizer = self.compiler.summarizer()
        exchanges = compile_user_spine_exchanges(self.atoms, summarize=summarizer,
            summarizer_identity='engineering-battery-summary-only-v1', max_channel_tokens=128, max_prompt_tokens=2048,
            max_workers=self.compiler.summary_workers)
        phases['exchanges_s'] = time.perf_counter()-now
        now = time.perf_counter()
        sources = {}
        for exchange in exchanges:
            sources.setdefault(exchange.section.source_id, []).append(exchange)
        try:
            for group in sources.values():
                for window in user_windows(group):
                    self.attention.score_sequence(tuple(window['texts']))
            phases['attention_windows_s'] = time.perf_counter()-now
            now = time.perf_counter()
            hierarchy = build_parent_budgeted_spine_hierarchy(exchanges, scorer=self.attention, summarize=summarizer,
                summarizer_identity='engineering-battery-summary-only-v1', leaf_token_cap=512, max_leaf_exchanges=2,
                max_exchange_channel_tokens=128, max_parent_channel_tokens=512, window_exchange_cap=8,
                max_prompt_tokens=2048, max_workers=self.compiler.summary_workers).summary_index()
            phases['hierarchy_s'] = time.perf_counter()-now
        finally:
            now = time.perf_counter()
            self.attention.park()
            phases['attention_park_s'] = time.perf_counter()-now
        now = time.perf_counter()
        if self._atomic_index is None:
            self._atomic_index = self.seed.atomic_index if self.seed else SectionSummaryIndex(())
            self._hierarchy_index = self.seed.hierarchy_index if self.seed else SectionSummaryIndex(())
        if self.seed:
            hierarchy = self._hierarchy_index.updated((*self.seed.hierarchy, *hierarchy.sections))
        else:
            hierarchy = self._hierarchy_index.updated(hierarchy.sections)
        atomic = self._atomic_index.updated((*(self.seed.atoms if self.seed else ()), *self.atoms))
        self._atomic_index, self._hierarchy_index = atomic, hierarchy
        phases['combined_indexes_s'] = time.perf_counter()-now
        phases['total_s'] = time.perf_counter()-started
        phases['attention_detail'] = {k: v-attention_before.get(k, 0)
                                     for k, v in getattr(self.attention, 'metrics', {}).items()}
        return atomic, hierarchy, phases

    def _ack(self, events, rows, snapshot, parents, started, phases):
        if snapshot['turn_count'] != len(rows) or not compatible_summary_embedding(self.encoder, snapshot['embedding_identity']):
            raise ValueError('Native chat index has not reached the captured turn prefix')
        self.events, self.rows = tuple(events), rows
        self.last_reopen = dict(snapshot=snapshot, parent_snapshot=parents,
            history_sha256=identity_sha256(rows), history_turns=len(rows),
            separate_process_reopen=False, resident=True)
        # A reader takes this complete tuple once. Future writer mutations to
        # the application's loaded index cannot change its routing namespace.
        self._publish_read_view()
        timing = dict(operation='sync', history_turns=len(rows), elapsed_s=time.perf_counter()-started, **phases)
        self._record_timing(timing)

    def _publish_read_view(self):
        # Capture the committed, authenticated prefix once. An active recall
        # never opens the mutable SQLite/WAL files during writer checkpoints.
        # Turn records are frozen; the mapping is read-only and namespace-bound
        # to this exact published index, including after a failed future sync.
        turns = self.app.transcript.native_snapshot()
        if len(turns) != len(self.events):
            raise ValueError('Published hydration prefix differs from the native index')
        raw = MappingProxyType({t.turn_id: t for t in turns})
        memory = self.app._load_native_spine()[2]
        with self._publication_lock:
            self._published = (memory, self.events, dict(self.last_reopen), raw)

    def recall(self, query):
        if self.last_reopen is None:
            raise ValueError('Chat recall requires an indexed turn prefix')
        started = time.perf_counter()
        anchor = max(datetime.fromisoformat(e.created_at) for e in self.events if e.created_at).strftime('%Y/%m/%d (%a)')
        result = native_packet(self.app, query, f'[Question asked at {anchor} 23:59] ' + query, self.events)
        self.timings.append(dict(operation='recall', elapsed_s=time.perf_counter()-started))
        return {**self.last_reopen, **result}

    def recall_published(self, query):
        """Read the last successful prefix while the next batch is building."""
        with self._publication_lock:
            published = self._published
        if published is None:
            raise ValueError('Chat recall requires an indexed turn prefix')
        original, events, receipt, raw = published
        started = time.perf_counter()
        anchor = max(datetime.fromisoformat(e.created_at) for e in events if e.created_at).strftime('%Y/%m/%d (%a)')
        transcript = SimpleNamespace(get_turn=raw.get)
        memory = copy(original)
        memory.load_turn = transcript.get_turn
        def retrieve(query, dated_question, **limits):
            if isinstance(memory.router, NativeSpineUserCompletionRouter):
                limits = {**CAP8_LIMITS, **limits}
            return memory.retrieve(query, dated_question, **limits)
        view = SimpleNamespace(transcript=transcript, retrieve_native_spine=retrieve)
        result = native_packet(view, query, f'[Question asked at {anchor} 23:59] ' + query, events)
        self.timings.append(dict(operation='recall', elapsed_s=time.perf_counter()-started,
                                 published_events=len(events)))
        return {**receipt, **result, 'published_events': len(events)}

    def learn(self, packet, *, access_event_id):
        started = time.perf_counter()
        learned = learn_native_packet(self.app, packet, self.events, access_event_id=access_event_id)
        self._record_timing(dict(operation='learn', elapsed_s=time.perf_counter()-started,
                                 access_event_id=access_event_id))
        return dict(learning=asdict(learned) if learned is not None else None)

    def close(self):
        try:
            if self.app is not None:
                self.app.close()
        finally:
            if self.attention is not None:
                close=getattr(self.attention,'close',None)
                if close is not None:
                    close()
                else:
                    self.attention.park()
                    self.attention.scorer = None
            if self.encoder is not None:
                self.encoder.close()
