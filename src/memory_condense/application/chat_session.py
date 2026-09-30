"""One durable chat stream for all I/O, recalled packets, and explicit outcomes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Protocol, Sequence


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ChatEvent:
    event_id: str
    role: str
    text: str
    created_at: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError('A stable event_id is required')
        if self.role not in ('user', 'assistant', 'tool', 'system', 'source'):
            raise ValueError('Unsupported chat event role')
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError('Completed events require content')
        if not isinstance(self.metadata, dict):
            raise ValueError('Event metadata must be an object')
        _json(self.metadata)
        if self.created_at is not None:
            stamp = datetime.fromisoformat(self.created_at)
            if stamp.tzinfo is None:
                raise ValueError('Event timestamps must include a timezone')
            object.__setattr__(self, 'created_at', stamp.astimezone(timezone.utc).isoformat())

    def row(self, session_id):
        return dict(turn_id=self.event_id, source_id=self.metadata.get('source_id', session_id),
                    role=self.role, text=self.text, created_at=self.created_at, metadata=self.metadata)


@dataclass(frozen=True)
class RecallPacket:
    packet_id: str
    query: str
    text: str
    # Exact span receipts, section IDs, and original pointers for recalled copies.
    references: tuple[dict, ...]
    input_event_id: str
    # Exact journal events accompany archive recall; the rolling window is not
    # copied back into every recall-receipt transcript event.
    recent_events: tuple[dict, ...] = ()

    @property
    def context_text(self):
        if not self.recent_events:
            return self.text
        recent = '\n'.join(_json(event) for event in self.recent_events)
        return (self.text + '\n\nRecent conversation (chronological source data):\n' + recent).strip()


class ChatBackend(Protocol):
    def sync(self, events: Sequence[ChatEvent]) -> None: ...
    def recall(self, query: str) -> dict: ...
    def learn(self, packet: RecallPacket, *, access_event_id: str) -> None: ...
    def close(self) -> None: ...


class ChatSession:
    """One writer and identical admission rules for live chat and replay.

    Capture commits before acknowledgement. Sync and learning share one writer.
    Backends with recall_published can serve an immutable committed snapshot on
    a separate reader while ingestion runs; other backends remain serialized.
    Eager recall is a freshness barrier. Background recall uses the committed archive
    plus exact recent journal events. Failed indexing leaves events durable.
    """

    def __init__(self, directory: str | Path, session_id: str, backend: ChatBackend, *, batch_exchanges: int = 0,
                 prepare_exchanges: tuple[int, ...] = (), streaming: bool = False,
                 recent_exchanges: int | None = None, recent_token_budget: int | None = None):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError('A session_id is required')
        if type(batch_exchanges) is not int or batch_exchanges < 0:
            raise ValueError('batch_exchanges must be a nonnegative integer')
        if type(streaming) is not bool:
            raise ValueError('streaming must be a boolean')
        if streaming and (prepare_exchanges or not all(callable(getattr(backend,n,None)) for n in
                ('start_stream','prepare_exchange','sync_prepared','finalize_stream','recall_published'))):
            raise ValueError('Streaming requires independent preparation and ordered publication')
        recent_exchanges = (12 if streaming else batch_exchanges) if recent_exchanges is None else recent_exchanges
        if type(recent_exchanges) is not int or recent_exchanges < 0 or (streaming and not recent_exchanges):
            raise ValueError('recent_exchanges must retain a positive window in streaming mode')
        if recent_token_budget is not None and (type(recent_token_budget) is not int or recent_token_budget < 1):
            raise ValueError('recent_token_budget must be positive')
        if (not isinstance(prepare_exchanges, tuple)
                or any(type(n) is not int or not 0 < n < batch_exchanges for n in prepare_exchanges)
                or tuple(sorted(set(prepare_exchanges))) != prepare_exchanges):
            raise ValueError('Preparation thresholds must be ordered distinct exchanges inside a batch')
        if prepare_exchanges and (not callable(getattr(backend, 'prepare', None))
                                  or not callable(getattr(backend, 'recall_published', None))):
            raise ValueError('Preparation requires a cache-only preparer and published recall')
        self.batch_exchanges = batch_exchanges
        self.streaming, self.recent_exchanges, self.recent_token_budget = streaming, recent_exchanges, recent_token_budget
        self._stream_jobs, self._stream_queued = [], False
        self.prepare_exchanges, self._prepared = prepare_exchanges, 0
        self._capture_depth = 0
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id, self.backend = session_id, backend
        self.path = self.directory / 'chat-events.sqlite'
        self._lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._closed, self._verified = False, False
        self._installed, self._error = 0, None
        self._owner = (self.directory / 'chat-owner.lock').open('a+b')
        try:
            self._lock_owner()
        except BaseException:
            self._owner.close()
            raise
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chat-memory')
        self._preparers = ThreadPoolExecutor(max_workers=3, thread_name_prefix='chat-summary') if streaming else None
        self._reader = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='chat-recall')
                        if (streaming or batch_exchanges) and callable(getattr(backend, 'recall_published', None)) else None)
        try:
            with self._connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS session (id INTEGER PRIMARY KEY CHECK(id=1), name TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS packets (packet_id TEXT PRIMARY KEY, query TEXT NOT NULL, text TEXT NOT NULL, refs TEXT NOT NULL, input_event_id TEXT NOT NULL REFERENCES events(event_id))')
                db.execute('CREATE INDEX IF NOT EXISTS packets_input ON packets(input_event_id)')
                if 'recent_events' not in {r[1] for r in db.execute('PRAGMA table_info(packets)')}:
                    db.execute("ALTER TABLE packets ADD COLUMN recent_events TEXT NOT NULL DEFAULT '[]'")
                db.execute('CREATE TABLE IF NOT EXISTS feedback (packet_id TEXT PRIMARY KEY REFERENCES packets(packet_id), successful INTEGER NOT NULL, event_id TEXT UNIQUE NOT NULL, applied INTEGER NOT NULL DEFAULT 0)')
                db.execute('INSERT OR IGNORE INTO session VALUES (1, ?)', (session_id,))
                if db.execute('SELECT name FROM session WHERE id=1').fetchone()[0] != session_id:
                    raise ValueError('Session directory belongs to a different session')
                # A legacy journal is admitted in full once. New journals start
                # at zero; subsequent opens restore only the committed prefix
                # or a durable in-flight target after a crash.
                db.execute('CREATE TABLE IF NOT EXISTS ingestion_state (id INTEGER PRIMARY KEY CHECK(id=1), committed INTEGER NOT NULL, target INTEGER)')
                db.execute('INSERT OR IGNORE INTO ingestion_state VALUES (1, (SELECT COUNT(*) FROM events), NULL)')
            self._startup = self._schedule_sync()
        except BaseException:
            if self._preparers is not None:
                self._preparers.shutdown(wait=True)
            if self._reader is not None:
                self._reader.shutdown(wait=True)
            self._executor.shutdown(wait=True)
            self._owner.close()
            raise

    def _lock_owner(self):
        import os
        if os.name == 'nt':
            import msvcrt
            self._owner.seek(0)
            if not self._owner.read(1):
                self._owner.write(b'0')
                self._owner.flush()
            self._owner.seek(0)
            msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db
        finally:
            db.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Chat session is closed')

    def events(self) -> tuple[ChatEvent, ...]:
        with self._connect() as db:
            rows = db.execute('SELECT event_id, role, text, created_at, metadata FROM events ORDER BY sequence').fetchall()
        return tuple(ChatEvent(*row[:4], metadata=json.loads(row[4])) for row in rows)

    def event(self, event_id):
        with self._connect() as db:
            row = db.execute('SELECT event_id, role, text, created_at, metadata FROM events WHERE event_id=?', (event_id,)).fetchone()
        if row is None:
            raise KeyError('Unknown chat event')
        return ChatEvent(*row[:4], metadata=json.loads(row[4]))

    @staticmethod
    def _insert(db, event):
        metadata = _json(event.metadata)
        existing = db.execute('SELECT role, text, created_at, metadata FROM events WHERE event_id=?', (event.event_id,)).fetchone()
        if existing is not None:
            if ((existing[0], existing[1], existing[3]) != (event.role, event.text, metadata)
                    or (event.created_at is not None and existing[2] != event.created_at)):
                raise ValueError('Event identity reused with different content')
            return 0
        db.execute('INSERT INTO events (event_id, role, text, created_at, metadata) VALUES (?, ?, ?, ?, ?)',
                   (event.event_id, event.role, event.text,
                    event.created_at or datetime.now(timezone.utc).isoformat(), metadata))
        return 1

    def ingest(self, event: ChatEvent) -> dict:
        return self.ingest_many((event,))

    def ingest_many(self, events: Sequence[ChatEvent]) -> dict:
        events = tuple(events)
        for event in events:
            if type(event) is not ChatEvent:
                raise TypeError('Ingest accepts ChatEvent objects')
            if event.event_id.startswith('_chat:') or '_chat' in event.metadata:
                raise ValueError('Reserved chat receipt identity/metadata')
        with self._lock:
            self._check_open()
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                added = sum(self._insert(db, event) for event in events)
                count = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            # Each job reads the latest prefix; queued jobs become no-ops after
            # the first one catches up. No extra compiler call per queued event.
            self._schedule_sync()
            return dict(session_id=self.session_id, accepted_events=added,
                        durable_events=count, indexed_events=self._installed)

    def _schedule_sync(self):
        if self.streaming:
            with self._lock:
                if self._closed or self._capture_depth or self._stream_queued:
                    return
                self._stream_queued = True
                def tick():
                    with self._lock:
                        self._stream_queued = False
                    self._sync(force=False)
                return self._executor.submit(tick)
        if not self.batch_exchanges:
            return self._executor.submit(self._sync)
        elif not self._capture_depth:
            return self._executor.submit(lambda: self._sync(force=False))

    @contextmanager
    def capture_exchange(self):
        """Group an answer and its feedback before testing the batch boundary."""
        with self._lock:
            self._check_open()
            self._capture_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._capture_depth -= 1
                if not self._closed and not self._capture_depth:
                    self._schedule_sync()

    def _batch_end(self, db, *, exchanges=None, completed_only=False, after=None):
        threshold = self.batch_exchanges if exchanges is None else exchanges
        after = self._installed if after is None else after
        rows = db.execute('SELECT sequence, role, event_id FROM events WHERE sequence>? ORDER BY sequence',
                          (after,)).fetchall()
        if completed_only and self._capture_depth:
            # An active exchange may have captured its answer but not feedback.
            # Only prepare fully closed earlier exchanges, never that open tail.
            users = [sequence for sequence, role, _ in rows if role=='user']
            if users:
                rows = [row for row in rows if row[0] < users[-1]]
        completed, has_user, answered = 0, False, False
        for sequence, role, event_id in rows:
            if event_id.startswith('_chat:'):
                continue
            if role == 'user':
                completed += int(has_user and answered)
                if completed >= threshold:
                    return sequence - 1
                has_user, answered = True, False
            elif role == 'assistant' and has_user:
                answered = True
        if completed + int(has_user and answered) >= threshold:
            return rows[-1][0]
        return after

    def _stream_sync(self, force_target):
        """Bound preparation to three exchanges; publish only a contiguous prefix."""
        self.backend.start_stream()
        while True:
            with self._lock, self._connect() as db:
                after = self._stream_jobs[-1][0] if self._stream_jobs else self._installed
                while len(self._stream_jobs) < 3:
                    end = self._batch_end(db, exchanges=1, completed_only=True, after=after)
                    if end<=after and not self._capture_depth:
                        # Tool results may arrive after their assistant exchange
                        # was published. Process that closed observation prefix
                        # without waiting for another user/assistant exchange.
                        pending=db.execute('SELECT sequence,role FROM events WHERE sequence>? ORDER BY sequence',(after,)).fetchall()
                        if pending and pending[0][1]!='user':
                            end=next((n-1 for n,role in pending if role=='user'),pending[-1][0])
                    if force_target is not None:
                        end = min(end, force_target)
                        if end <= after:
                            end = force_target
                    if end <= after:
                        break
                    events = self.events()[after:end]
                    future = self._preparers.submit(self.backend.prepare_exchange, events)
                    self._stream_jobs.append((end, events, future))
                    future.add_done_callback(lambda _: self._schedule_sync())
                    after = end
            if not self._stream_jobs:
                break
            first = self._stream_jobs[0][2]
            if force_target is not None:
                # A failed preparation may be retried by an explicit flush.
                # Publication failures retain successful preparation results.
                if first.done() and first.exception() is not None:
                    end, events, _ = self._stream_jobs[0]
                    first = self._preparers.submit(self.backend.prepare_exchange, events)
                    self._stream_jobs[0] = (end, events, first)
                first.result()
            elif not first.done():
                break
            ready = []
            for end, _, future in self._stream_jobs:
                if not future.done():
                    break
                ready.append((end, future.result()))
            if not ready:
                break
            target = ready[-1][0]
            with self._connect() as db:
                db.execute('UPDATE ingestion_state SET target=? WHERE id=1', (target,))
            self.backend.sync_prepared(self.events()[:target], tuple(value for _, value in ready))
            with self._connect() as db:
                db.execute('UPDATE ingestion_state SET committed=?,target=NULL WHERE id=1', (target,))
            with self._lock:
                self._installed = target
                del self._stream_jobs[:len(ready)]
            if force_target is not None and target >= force_target:
                break
        if force_target is not None:
            self.backend.finalize_stream(self.events()[:self._installed])

    def _prepare_pending(self):
        if not self.prepare_exchanges:
            return
        with self._lock, self._connect() as db:
            target = max(self._batch_end(db, exchanges=n, completed_only=True) for n in self.prepare_exchanges)
        if target > max(self._installed, self._prepared):
            # Same writer owns preparation and publication. Preparation changes
            # caches only: committed state, raw indexing and learning stay put.
            self.backend.prepare(self.events()[:target], should_yield=self._preparation_should_yield)
            self._prepared = target

    def _preparation_should_yield(self):
        with self._lock, self._connect() as db:
            return self._closed or self._batch_end(db, completed_only=True)>self._installed

    def _publish_prefix(self, target):
        events = self.events()[:target]
        if len(events) != target:
            raise ValueError('Ingestion checkpoint exceeds the durable journal')
        with self._connect() as db:
            db.execute('UPDATE ingestion_state SET target=? WHERE id=1', (target,))
        if events:
            self.backend.sync(events)
        with self._connect() as db:
            db.execute('UPDATE ingestion_state SET committed=?, target=NULL WHERE id=1', (target,))
        with self._lock:
            self._installed, self._verified = target, True

    def _sync(self, *, force=True, allow_batch=False):
        try:
            with self._connect() as db:
                force_target = db.execute('SELECT COUNT(*) FROM events').fetchone()[0] if force else None
            if not self._verified:
                with self._connect() as db:
                    committed, target = db.execute('SELECT committed,target FROM ingestion_state WHERE id=1').fetchone()
                self._publish_prefix(target if target is not None else committed)
            if self.streaming:
                self._stream_sync(force_target)
            while not self.streaming:
                with self._lock, self._connect() as db:
                    target = db.execute('SELECT target FROM ingestion_state WHERE id=1').fetchone()[0]
                    if target is None:
                        if force:
                            target = force_target
                        else:
                            target = self._batch_end(db, completed_only=not allow_batch)
                if target <= self._installed:
                    break
                self._publish_prefix(target)
                if force and target >= force_target:
                    break
            with self._connect() as db:
                pending = db.execute('SELECT f.packet_id, f.successful, f.event_id FROM feedback f JOIN events e ON e.event_id=f.event_id WHERE f.applied=0 AND e.sequence<=? ORDER BY f.rowid', (self._installed,)).fetchall()
            for packet_id, successful, event_id in pending:
                if successful:
                    self.backend.learn(self.packet(packet_id), access_event_id=event_id)
                # The backend uses the same stable ID on a crash/retry between
                # graph commit and acknowledgement here. Applied rows persist.
                with self._connect() as db:
                    db.execute('UPDATE feedback SET applied=1 WHERE packet_id=?', (packet_id,))
            if not force and not self.streaming:
                self._prepare_pending()
            with self._lock:
                self._error = None
        except BaseException as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
            raise

    def _call(self, operation, *, reader=False):
        with self._lock:
            self._check_open()
            future = (self._reader if reader and self._reader is not None else self._executor).submit(operation)
        return future.result()

    def flush(self):
        self._call(self._sync)
        return self.status()

    def packet(self, packet_id):
        with self._connect() as db:
            row = db.execute('SELECT query, text, refs, input_event_id, recent_events FROM packets WHERE packet_id=?', (packet_id,)).fetchone()
        if row is None:
            raise KeyError('Unknown recall packet')
        return RecallPacket(packet_id, row[0], row[1], tuple(json.loads(row[2])), row[3], tuple(json.loads(row[4])))

    def _recent_events(self, published_events=None):
        if not self.recent_exchanges:
            return ()
        with self._connect() as db:
            # Configured completed exchanges plus the current unanswered input. Tools
            # stay with their exchange; internal recall/feedback copies do not
            # recursively enlarge the recent conversation.
            users = db.execute("SELECT sequence FROM events WHERE role='user' ORDER BY sequence DESC LIMIT ?",
                               (self.recent_exchanges + 1,)).fetchall()
            start = users[-1][0] if users else 1
            latest_answered = bool(users and db.execute(
                "SELECT 1 FROM events WHERE sequence>? AND role='assistant' LIMIT 1", (users[0][0],)).fetchone())
            if len(users) > self.recent_exchanges and latest_answered:
                start = users[-2][0]
            # Unanswered inputs/errors may span more journal entries than six
            # completed exchanges. Never hide an uncommitted source event.
            committed = self._installed if published_events is None else min(self._installed, published_events)
            start = min(start, committed + 1)
            rows = db.execute('SELECT sequence,event_id,role,text,created_at,metadata FROM events WHERE sequence>=? ORDER BY sequence',
                              (start,)).fetchall()
        visible = [(sequence,dict(event_id=event_id, role=role, text=text, created_at=stamp,
                          session_id=self.session_id, source_id=json.loads(metadata).get('source_id')))
                     for sequence, event_id, role, text, stamp, metadata in rows
                     if sequence >= start and not event_id.startswith('_chat:')]
        if self.recent_token_budget is not None:
            from memory_condense.domain._tokenizer import count_tokens
            # Remove whole committed exchanges only. A lagging index never
            # makes uncommitted input disappear to satisfy a prompt budget.
            while len(visible)>1 and count_tokens('\n'.join(_json(e) for _,e in visible))>self.recent_token_budget:
                cut = next((i for i,(_,e) in enumerate(visible[1:],1) if e['role']=='user'),None)
                if cut is None or visible[cut-1][0]>committed:
                    break
                visible = visible[cut:]
        return tuple(e for _,e in visible)

    def recalls_for_input(self, event_id):
        """Follow an input's durable links to packets and their original sources."""
        self.event(event_id)
        with self._connect() as db:
            ids = db.execute('SELECT packet_id FROM packets WHERE input_event_id=? ORDER BY rowid', (event_id,)).fetchall()
        return tuple(self.packet(row[0]) for row in ids)

    def recall(self, query: str, *, packet_id: str, input_event_id: str | None = None) -> RecallPacket:
        """Ingest a delivered packet; retries return it verbatim without relearning."""
        if not isinstance(query, str) or not query.strip() or not isinstance(packet_id, str) or not packet_id.strip():
            raise ValueError('Recall requires a question and stable packet_id')
        def operation():
            try:
                prior = self.packet(packet_id)
            except KeyError:
                prior = None
            if prior is not None:
                if prior.query != query or (input_event_id is not None and prior.input_event_id != input_event_id):
                    raise ValueError('Packet identity reused with a different query')
                return prior
            linked_input = input_event_id
            if linked_input is None:
                with self._connect() as db:
                    row = db.execute("SELECT event_id FROM events WHERE role IN ('user', 'assistant', 'tool') AND substr(event_id,1,6)!='_chat:' ORDER BY sequence DESC LIMIT 1").fetchone()
                linked_input = row[0] if row else None
            event = self.event(linked_input)
            if event.role not in ('user', 'assistant', 'tool') or '_chat' in event.metadata:
                raise ValueError('Recall must link to an ingested input event')
            if self._reader is not None:
                # Initial admission is the only indexing barrier. Later failed
                # or slow batches leave the last published archive usable.
                if not self._verified:
                    self._startup.result()
                result = (self.backend.recall_published(query) if self._installed else
                          dict(text='', references=[], published_events=0))
                published = result['published_events']
            else:
                self._sync(force=not bool(self.batch_exchanges), allow_batch=True)
                result = self.backend.recall(query) if self._installed else dict(text='', references=[])
                published = self._installed
            text, references = result['text'], tuple(result['references'])
            recent = self._recent_events(published)
            if not isinstance(text, str):
                raise ValueError('Recall packet must contain text')
            receipt = dict(kind='recall', packet_id=packet_id, query=query, references=references, input_event_id=linked_input)
            if recent:
                receipt['recent_event_ids'] = [e['event_id'] for e in recent]
            event = ChatEvent('_chat:recall:' + packet_id, 'tool', text or 'No matching memory found.', metadata={'_chat': receipt})
            with self._connect() as db:
                db.execute('INSERT INTO packets (packet_id,query,text,refs,input_event_id,recent_events) VALUES (?, ?, ?, ?, ?, ?)',
                           (packet_id, query, text, _json(references), linked_input, _json(recent)))
                self._insert(db, event)
            # Retrieval is delivered without waiting for its own re-indexing.
            # Shutdown's final sync also covers a recall already queued at close.
            return RecallPacket(packet_id, query, text, references, linked_input, recent)
        packet = self._call(operation, reader=True)
        with self._lock:
            if not self._closed:
                self._schedule_sync()
        return packet

    def feedback(self, packet_id: str, *, successful: bool) -> dict:
        """Record use/success once. ChatIO calls this after a completed exchange.

        This is a co-access signal, not a quality grade. Standalone recall users
        can report their own outcome; retrieval alone does not reinforce memory.
        """
        if type(successful) is not bool:
            raise ValueError('Feedback requires a boolean outcome')
        with self._lock:
            self._check_open()
            packet = self.packet(packet_id)
            if successful and not packet.references:
                raise ValueError('An empty packet cannot reinforce memory')
            event_id = '_chat:feedback:' + packet_id
            receipt = dict(kind='recall_feedback', packet_id=packet_id, successful=successful)
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                prior = db.execute('SELECT successful FROM feedback WHERE packet_id=?', (packet_id,)).fetchone()
                if prior and bool(prior[0]) != successful:
                    raise ValueError('Packet already has a different final outcome')
                self._insert(db, ChatEvent(event_id, 'tool', _json(receipt), metadata={'_chat': receipt}))
                db.execute('INSERT OR IGNORE INTO feedback (packet_id, successful, event_id) VALUES (?, ?, ?)', (packet_id, successful, event_id))
            self._schedule_sync()
            return receipt

    def status(self):
        with self._lock, self._connect() as db:
            count = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            feedback = db.execute('SELECT COUNT(*) FROM feedback WHERE applied=0').fetchone()[0]
            return dict(session_id=self.session_id, durable_events=count, indexed_events=self._installed,
                        pending_events=count-self._installed, pending_feedback=feedback,
                        batch_exchanges=self.batch_exchanges,
                        streaming=self.streaming, recent_exchanges=self.recent_exchanges,
                        recent_token_budget=self.recent_token_budget,
                        preparation_jobs=len(self._stream_jobs),
                        prepare_exchanges=self.prepare_exchanges,
                        nonblocking_recall=self._reader is not None,
                        last_error=self._error, closed=self._closed)

    def close(self):
        with self._io_lock:
            self._close()

    def _close(self):
        def finish():
            try:
                self._sync()
            finally:
                if self._preparers is not None:
                    self._preparers.shutdown(wait=True)
                self.backend.close()
        with self._lock:
            if self._closed:
                return
            self._closed = True  # Close admissions before the final barrier.
        try:
            # Readers may still be hydrating the prior snapshot. Finish their
            # durable packet receipts before the writer's final drain/close.
            if self._reader is not None:
                self._reader.shutdown(wait=True)
            future = self._executor.submit(finish)
            future.result()
        finally:
            self._executor.shutdown(wait=True)
            self._owner.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except BaseException as exc:
            if value is None:
                raise
            value.add_note(f'Chat close also failed: {exc}')


def chat_request(session: ChatSession, request: dict) -> dict:
    """Single interface for live adapters and replay: message/recall/feedback."""
    if request.get('session_id', session.session_id) != session.session_id:
        raise ValueError('Request belongs to a different session')
    operation = request.get('operation', 'message')
    if operation == 'message':
        return session.ingest(ChatEvent(event_id=request['event_id'], role=request['role'],
            text=request['text'], created_at=request.get('created_at'), metadata=request.get('metadata', {})))
    if operation == 'recall':
        return asdict(session.recall(request['query'], packet_id=request['packet_id'], input_event_id=request.get('input_event_id')))
    if operation == 'feedback':
        return session.feedback(request['packet_id'], successful=request['successful'])
    if operation == 'recalls':
        return {'packets': [asdict(p) for p in session.recalls_for_input(request['event_id'])]}
    if operation == 'flush':
        return session.flush()
    if operation == 'status':
        return session.status()
    raise ValueError('Unknown chat operation')
