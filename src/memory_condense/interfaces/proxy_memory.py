"""Durable ChatSession integration for provider-compatible memory requests."""
from base64 import b64decode, b64encode
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
import time

from memory_condense.application.chat_io import ChatIO
from memory_condense.control.status import now_iso
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.inline_memory import parse_inline_response
from memory_condense.interfaces.proxy_units import guard_served_body
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.interfaces import proxy_memory_wire as wire


@dataclass(frozen=True)
class WireResponse:
    status: int
    headers: dict
    body: bytes

    def stored(self):
        # Persist only protocol/correlation headers; never cookies or credentials.
        allowed = {'content-type', 'request-id', 'x-request-id', 'x-litellm-call-id', 'retry-after'}
        return dict(status=self.status, headers={k:v for k,v in self.headers.items() if k.lower() in allowed},
                    body=b64encode(self.body).decode('ascii'))

    @classmethod
    def restore(cls, value):
        return cls(value['status'], value['headers'], b64decode(value['body']))


class MemoryProxyError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class UpstreamFailure(Exception):
    def __init__(self, response):
        super().__init__(f'Provider returned HTTP {response.status}')
        self.response = response


@dataclass
class SessionEntry:
    session: object
    messages: list = field(default_factory=list)
    policy: object = None
    #: Set by :meth:`MemoryProxy.swap` once this entry's session has been
    #: closed and dropped from the registry. A request that resolved the entry
    #: before the swap began and is only now winning its IO lock must not use
    #: it: the files underneath it may have been replaced.
    stale: bool = False

    def reload(self):
        self.messages = []
        for event in self.session.events():
            if 'proxy_policy' in event.metadata:
                self.policy = event.metadata['proxy_policy']
            message = event.metadata.get('proxy_message') or event.metadata.get('response', {}).get('proxy_message')
            if message is not None:
                self.messages.append((event.event_id, message))


class MemoryProxy:
    """One namespace per provider/conversation; one IO owner per session.

    session_factory(namespace, directory) returns the normal streaming ChatSession.
    The caller supplies HTTP transport. Credentials only bind the namespace;
    credential headers are never persisted with requests or responses.
    """
    def __init__(self, directory, session_factory, *, inline_memory=True, empty_response_retries=2,
                 max_sessions=8, recent_token_budget=8192, summary_api_fallback=True):
        if type(empty_response_retries) is not int or not 0 <= empty_response_retries <= 2:
            raise ValueError('Empty completion retries must be between zero and two')
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError('Session limit must be positive')
        self.directory, self.factory = Path(directory), session_factory
        self.inline_memory, self.empty_response_retries = inline_memory, empty_response_retries
        self.max_sessions, self.recent_token_budget = max_sessions, recent_token_budget
        self.summary_api_fallback = summary_api_fallback
        self.entries, self.lock, self.closed = {}, threading.RLock(), False
        # `lock` is held across session construction and journal replay, which
        # is exactly the work status must be able to describe. These mirrors are
        # written under a lock that is never held across slow work, so a status
        # request never queues behind a cold start.
        self._status_lock = threading.RLock()
        self._live, self._opening = {}, {}
        # Lock order everywhere in this class: `lock` -> a session's `_io_lock`
        # -> `_status_lock`. `quiesce`/`swap` take the first two in that order;
        # the request path takes `lock` (inside `entry`), releases it, then
        # takes `_io_lock`. Nothing ever takes `lock` while holding `_io_lock`.
        self.generation, self.last_swap_at = 0, None

    @staticmethod
    def namespace(provider, headers):
        """The store partition for one conversation. Credentials play no part.

        This assumes a privileged environment: a single user's own machine,
        with the proxy bound to localhost by default. Memory is partitioned by
        conversation, and the provider credential is forwarded upstream
        without ever being read here.

        It used to mix ``sha256(authorization, x-api-key, ...)`` into the key
        for tenant isolation. That was the wrong layer and it cost real data:
        provider OAuth tokens rotate, so every refresh minted a fresh empty
        namespace for a conversation the user was in the middle of -- silently,
        and burning one of ``max_sessions`` each time. Access control over a
        network belongs to a transport in front of this, not to the shape of
        the storage key; see ``docs/01 - Design/03 - System Controls.md``.
        """
        headers = {k.lower():v for k,v in headers.items()}
        conversation = headers.get('x-memory-conversation-id') or headers.get('x-conversation-id')
        if not conversation or not conversation.strip():
            raise MemoryProxyError('Memory mode requires x-memory-conversation-id')
        if len(conversation) > 512:
            raise MemoryProxyError('Conversation ID is too long')
        return hashlib.sha256(wire.encoded([provider, conversation.strip()])).hexdigest()

    def entry(self, namespace, *, summary_api=None):
        with self.lock:
            if self.closed:
                raise MemoryProxyError('Memory proxy is closing', 503)
            if namespace not in self.entries:
                if len(self.entries) >= self.max_sessions:
                    raise MemoryProxyError('Active memory session limit reached; restart to reopen another session', 429)
            configure = getattr(self.factory, 'configure_summary_provider', None)
            if summary_api is not None and configure is not None:
                configure(namespace, summary_api)
            if namespace not in self.entries:
                with self._status_lock:
                    self._opening[namespace] = time.monotonic()
                try:
                    session = self.factory(namespace, self.directory/'sessions'/namespace)
                    entry = SessionEntry(session)
                    entry.reload()
                except BaseException:
                    with self._status_lock:
                        self._opening.pop(namespace, None)
                    raise
                with session._connect() as db:
                    db.execute('CREATE TABLE IF NOT EXISTS proxy_requests (request_id TEXT PRIMARY KEY, body_sha TEXT NOT NULL, state TEXT NOT NULL, input_id TEXT, failure TEXT)')
                    db.execute('CREATE TABLE IF NOT EXISTS proxy_attempts (request_id TEXT NOT NULL, attempt INTEGER NOT NULL, status INTEGER, outcome TEXT NOT NULL, elapsed_s REAL NOT NULL, headers TEXT NOT NULL, PRIMARY KEY(request_id,attempt))')
                    columns = {r[1] for r in db.execute('PRAGMA table_info(proxy_requests)')}
                    if 'served_body' not in columns:
                        db.execute('ALTER TABLE proxy_requests ADD COLUMN served_body BLOB')
                    if 'served_guard' not in columns:
                        # The pre-send guard's verdict on the rewritten body,
                        # kept apart from served_body so the ledger can say
                        # both what was sent and why it was not the rewrite.
                        db.execute('ALTER TABLE proxy_requests ADD COLUMN served_guard TEXT')
                    if 'response_body' not in {r[1] for r in db.execute('PRAGMA table_info(proxy_attempts)')}:
                        db.execute('ALTER TABLE proxy_attempts ADD COLUMN response_body BLOB')
                self.entries[namespace] = entry
                with self._status_lock:
                    self._opening.pop(namespace, None)
                    self._live[namespace] = entry
            return self.entries[namespace]

    def status(self):
        # Deliberately does not take `self.lock`: a session being opened holds
        # it for the whole of construction and journal replay.
        now = time.monotonic()
        with self._status_lock:
            live = list(self._live.items())
            opening = {namespace: round(now-started, 3) for namespace, started in self._opening.items()}
        sessions, values = {}, []
        for namespace, entry in live:
            try:
                value = entry.session.status()
            except Exception as exc:
                sessions[namespace] = dict(unavailable=f'{type(exc).__name__}: {exc}')
                continue
            sessions[namespace] = value
            values.append(value)
        startup_status = getattr(self.factory, 'startup_status', lambda: {})
        summary_status = getattr(self.factory, 'summary_status', lambda: {})
        retrieval_status = getattr(self.factory, 'retrieval_status', lambda: {})
        try:
            startup = startup_status()
        except Exception as exc:
            startup = dict(unavailable=f'{type(exc).__name__}: {exc}')
        pending_events = sum(v['pending_events'] for v in values)
        # `entry` gates admission on `len(self.entries)`, so the capacity signal
        # must count the same thing or it reports room that does not exist. Read
        # without `self.lock` for the reason above: a length read is atomic, and
        # a monitoring value may be one admission stale. Publishing the limit is
        # what lets a surface warn before the next new conversation is refused,
        # instead of health reading `ok` up to the moment it fails.
        admitted = len(self.entries)
        return dict(active_sessions=len(values), pending_events=pending_events,
                    max_sessions=self.max_sessions, admitted_sessions=admitted,
                    sessions_at_capacity=admitted >= self.max_sessions,
                    pending_feedback=sum(v['pending_feedback'] for v in values),
                    failed_sessions=sum(v['last_error'] is not None for v in values),
                    opening_sessions=opening, startup=startup, sessions=sessions,
                    # Aggregation must not hide which session is blocked, and a
                    # session still opening is work in progress, not idle.
                    busy=bool(opening) or bool(pending_events)
                         or any(v['preparation_jobs'] for v in values)
                         or any(stage not in ('ready', None) for stage in
                                (r.get('stage') for r in startup.values() if isinstance(r, dict))),
                    inline_memory=self.inline_memory, empty_response_retries=self.empty_response_retries,
                    summary_api_fallback=self.summary_api_fallback, summary_routing=summary_status(),
                    retrieval=retrieval_status())

    @contextmanager
    def _io(self, namespace, *, summary_api=None):
        """Own one session's IO, re-resolving across a concurrent swap.

        A swap closes sessions while holding their `_io_lock`, so a request
        that resolved its entry before the swap started would otherwise wake
        up owning the lock of a closed session over replaced files. The entry
        is re-admitted instead. `stale` is only ever set under that same
        `_io_lock`, so reading it here needs no further lock.
        """
        for _ in range(8):
            entry = self.entry(namespace, summary_api=summary_api)
            entry.session._io_lock.acquire()
            if entry.stale:
                entry.session._io_lock.release()
                continue
            try:
                yield entry
            finally:
                entry.session._io_lock.release()
            return
        raise MemoryProxyError('Memory store is being replaced; retry the request', 503)

    @contextmanager
    def quiesce(self):
        """Block admissions and wait out in-flight IO, leaving stores open.

        Enough to take a snapshot (``VACUUM INTO`` is consistent under WAL,
        and quiescing makes the *tree* consistent too, not just each file).
        Not enough to replace files: use :meth:`swap` for that.
        """
        with self.lock:
            if self.closed:
                raise MemoryProxyError('Memory proxy is closing', 503)
            entries = list(self.entries.values())
            held = []
            try:
                for entry in entries:
                    entry.session._io_lock.acquire()
                    held.append(entry)
                yield entries
            finally:
                for entry in reversed(held):
                    entry.session._io_lock.release()

    def swap(self, mutate):
        """Close every session, run ``mutate`` on the files, re-admit lazily.

        The augment counterpart of :meth:`CondenserHandle.swap`, with the same
        invariant: the files move only while nothing holds a store open. The
        difference is that nothing is reopened here -- :meth:`entry` rebuilds a
        session on its next request -- which is also the recovery story when
        ``mutate`` raises. The sessions are already out of the registry, the
        files are untouched in the pre-mutation failures, and the next request
        opens them again, so a failed swap reports its error without wedging
        the proxy. A request that arrives mid-swap waits on :attr:`lock`
        rather than reading a half-swapped tree.
        """
        with self.quiesce() as entries:
            failures = []
            for entry in entries:
                entry.stale = True
                try:
                    entry.session.close()
                except Exception as exc:
                    failures.append(exc)
            self.entries.clear()
            with self._status_lock:
                self._live.clear()
            # The factory caches per-namespace lifecycle and summary state over
            # the same trees. Keeping it would report a conversation that the
            # swap deleted, or call a restored one 'ready' before it reopens.
            forget = getattr(self.factory, 'forget', None)
            if forget is not None:
                forget()
            self.generation += 1
            if failures:
                # Mutating files under a session that would not close is the
                # one thing this must never do.
                raise failures[0]
            try:
                return mutate()
            finally:
                self.last_swap_at = now_iso()

    def flush(self, provider, headers):
        with self._io(self.namespace(provider, headers)) as entry:
            return entry.session.flush()

    def close(self):
        with self.lock:
            self.closed = True
            entries = list(self.entries.values())
        with self._status_lock:
            self._live.clear()
            self._opening.clear()
        failures = []
        for entry in entries:
            try:
                entry.session.close()
            except Exception as exc:
                failures.append(exc)
        close = getattr(self.factory,'close',None)
        if close is not None:
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise failures[0]

    @staticmethod
    def _new_messages(history, incoming, fmt=None):
        if fmt == 'responses':
            # A captured response is one event; its replay can contain several
            # messages and tool calls. Compare the same item boundaries.
            old = [wire.canonical_message(wire.responses.canonical_item(item))
                   for _, message in history for item in wire.responses.input_item(message)]
        else:
            old = [wire.canonical_message(m) for _,m in history]
        values = [wire.canonical_message(m) for m in incoming]
        # Full history, a sliding suffix, or a single new input/tool-result batch.
        for overlap in range(min(len(old),len(values)), 0, -1):
            if old[-overlap:] == values[:overlap]:
                return incoming[overlap:]
        if not old or len(incoming) == 1 or all(wire.tool_result(m) for m in incoming):
            return incoming
        raise MemoryProxyError('History differs from this conversation; use a new conversation ID for a branch', 409)

    @staticmethod
    def _recent(entry, packet):
        wanted = {e['event_id'] for e in packet.recent_events}
        wanted.add(packet.input_event_id)
        start = next((i for i,(event_id,_) in enumerate(entry.messages) if event_id in wanted),len(entry.messages)-1)
        # Never separate a tool result from its assistant call and user request.
        while start > 0 and (entry.messages[start][1]['role'] != 'user' or wire.tool_result(entry.messages[start][1])):
            start -= 1
        # Local text models cannot summarize visual evidence. Keep earlier
        # image-bearing messages intact rather than silently losing screenshots.
        pinned = [m for _,m in entry.messages[:start] if 'responses_content' in m]
        return pinned + [m for _,m in entry.messages[start:]]

    def exchange(self, body, provider, headers, send, summary_send=None, fmt=None):
        """Run synchronously on a worker; send(payload_bytes) bridges async HTTP."""
        # One provider speaks two wire formats. The credential namespace stays
        # keyed on the provider, so a conversation may switch format mid-stream.
        fmt = fmt or wire.format_for(provider)
        try:
            payload = json.loads(body)
            incoming = wire.validate_request(payload, fmt)
        except (ValueError, TypeError, KeyError) as exc:
            raise MemoryProxyError(str(exc)) from exc
        namespace = self.namespace(provider, headers)
        body_sha = hashlib.sha256(body).hexdigest()
        request_key = next((v for k,v in headers.items() if k.lower() == 'x-memory-request-id'),body_sha)
        if not isinstance(request_key,str) or not request_key.strip() or len(request_key)>512:
            raise MemoryProxyError('Invalid x-memory-request-id')
        request_id = 'proxy:' + hashlib.sha256(request_key.encode()).hexdigest()
        summary_api = None
        if self.summary_api_fallback:
            from memory_condense.interfaces.summary_api import ProviderSummaryAPI
            summary_api = ProviderSummaryAPI(provider, payload['model'], summary_send or send, fmt=fmt)
        with self._io(namespace, summary_api=summary_api) as entry:
            session = entry.session
            with session._connect() as db:
                prior = db.execute('SELECT body_sha,state,input_id,failure FROM proxy_requests WHERE request_id=?',(request_id,)).fetchone()
                if prior and prior[0] != body_sha:
                    raise MemoryProxyError('Request ID reused with a different payload',409)
                if prior is None:
                    # Recover a crash between atomic input capture and the
                    # request ledger commit, before any provider reservation.
                    captured = db.execute('SELECT event_id,metadata FROM events WHERE event_id LIKE ? ORDER BY sequence',
                                          (request_id+':input:%',)).fetchall()
                    if captured:
                        if any(json.loads(metadata).get('proxy_request_sha256')!=body_sha for _,metadata in captured):
                            raise MemoryProxyError('Request ID reused with a different captured input',409)
                        prior = (body_sha,'prepared',captured[-1][0],None)
                        db.execute('INSERT INTO proxy_requests(request_id,body_sha,state,input_id) VALUES(?,?,?,?)',(request_id,body_sha,'prepared',prior[2]))
            if prior:
                try:
                    captured = session.event(request_id+':assistant')
                except KeyError:
                    captured = None
                if captured is not None:
                    # Also replay idempotent feedback if shutdown interrupted it.
                    io = captured.metadata['io']
                    response = ChatIO(session).invoke(request_id=request_id, reader=lambda:None,
                        input_event_id=io['input_event_id'], packet_id=io['packet_id'])
                    with session._connect() as db:
                        db.execute("UPDATE proxy_requests SET state='complete',failure=NULL WHERE request_id=?",(request_id,))
                    entry.reload()
                    return WireResponse.restore(response['proxy_wire'])
                if prior[1] in ('calling','failed'):
                    if prior[3]:
                        return WireResponse.restore(json.loads(prior[3]))
                    raise MemoryProxyError('Previous provider outcome is unknown; request was retained and will not be resent',409)
            threaded = bool(payload.get('previous_response_id') or payload.get('conversation'))
            replayed = any(m['role'] == 'assistant' for m in incoming)
            if not prior and not entry.messages and threaded and not replayed:
                # Upstream threading is redundant whenever the turns are also in
                # the body: a replayed history opens the conversation as imported
                # context and the threading is simply dropped. What cannot be
                # picked up is a conversation that exists only in provider
                # storage, recognised by an unanswered dialogue arriving against
                # no local history. Answering its late turn with only its own
                # text would be silent amnesia, so refuse instead.
                raise MemoryProxyError('This conversation has no history here and the request threads one '
                                       'upstream without replaying it; send the prior turns in input, '
                                       'or use a new conversation ID', 409)
            new = self._new_messages(entry.messages, incoming, fmt) if not prior else []
            if not prior and not new:
                raise MemoryProxyError('No new input; reuse the original request ID to replay its response',409)
            with session.capture_exchange():
                if prior:
                    input_id = prior[2]
                else:
                    events = []
                    for i,message in enumerate(new):
                        role = 'tool' if wire.tool_result(message) else message['role']
                        events.append(ChatEvent(f'{request_id}:input:{i}',role,wire.message_text(message),
                                                metadata={'proxy_message':message,'proxy_request_sha256':body_sha}))
                    policy = wire.policy_record(fmt,payload)
                    policies = []
                    if policy != entry.policy:
                        policies = [ChatEvent(request_id+':policy','system',wire.encoded(policy).decode(),metadata={'proxy_policy':policy})]
                    session.ingest_many([*policies,*events])
                    entry.policy = policy
                    entry.messages.extend((e.event_id,m) for e,m in zip(events,new))
                    input_id = events[-1].event_id
                    with session._connect() as db:
                        db.execute('INSERT INTO proxy_requests(request_id,body_sha,state,input_id) VALUES(?,?,?,?)',(request_id,body_sha,'prepared',input_id))
                    # Initial imported context must be indexed before any of it
                    # can be discarded. Ordinary turns do not wait for ingestion.
                    if count_tokens('\n'.join(e.text for e in events)) > self.recent_token_budget:
                        session.flush()
                event = session.event(input_id)
                query = next((wire.message_text(m) for _,m in reversed(entry.messages)
                              if m['role']=='user' and not wire.tool_result(m)),event.text)
                packet = session.recall(query,packet_id=request_id,input_event_id=input_id)
                recent = self._recent(entry,packet)
                inline = wire.use_inline(payload,self.inline_memory,fmt)
                served = wire.prepare_payload(payload,fmt,recent,packet.text,event.text,inline)

                # Pre-send guard: a served body that lost a tool pairing is
                # never sent; the client's original goes through verbatim
                # (passthrough is the sanctioned degraded mode). `served` is
                # the same on every attempt, so the verdict is decided once.
                problems = guard_served_body(fmt, served)
                sent = body if problems else wire.encoded(served)

                def tagged(value):
                    """Compose an attempt outcome with the guard's verdict.

                    Both facts matter and neither may overwrite the other: a
                    passthrough that then got an HTTP 500 is
                    ``invalid_served_body+http_error``, not one or the other.
                    """
                    return f'invalid_served_body+{value}' if problems else value

                def reader():
                    with session._connect() as db:
                        # served_body is what actually went upstream, which on
                        # the passthrough path is the client's own bytes;
                        # served_guard says why the rewrite was not sent.
                        db.execute("UPDATE proxy_requests SET state='calling',served_body=?,served_guard=? "
                                   "WHERE request_id=?",
                                   (sent, wire.encoded(problems).decode() if problems else None, request_id))
                    for attempt in range(self.empty_response_retries+1):
                        started = time.perf_counter()
                        outcome, remote = tagged('unknown'), None
                        try:
                            remote = send(sent)
                            if remote.status >= 400:
                                outcome = tagged('http_error')
                                raise UpstreamFailure(remote)
                            value = wire.assemble_stream(fmt,remote.body) if payload.get('stream') else json.loads(remote.body)
                            message = wire.response_message(fmt,value)
                            finish = wire.finish_reason(fmt,value)
                            if finish not in ('stop','end_turn','tool_calls','function_call','tool_use','stop_sequence','pause_turn','refusal'):
                                outcome = tagged('incomplete_response')
                                raise ValueError('Provider response did not complete: '+str(finish))
                            empty = not wire.text_content(message).strip() and not wire.has_alternative(message)
                            reasoning_only = wire.reasoning_present(fmt,value)
                            if empty:
                                retryable = finish in ('stop','end_turn') and not reasoning_only
                                outcome = tagged('empty_completed' if retryable else 'incomplete_or_reasoning_only')
                                if retryable and attempt < self.empty_response_retries:
                                    time.sleep(attempt+1)
                                    continue
                                raise ValueError('Provider returned no completed answer')
                            outcome = tagged('complete')
                            public = value
                            result = dict(content=wire.message_text(message),finish_reason='stop',elapsed_s=time.perf_counter()-started)
                            parsed = None
                            if inline and not wire.has_alternative(message):
                                parsed = parse_inline_response(dict(result,content=wire.text_content(message),
                                    finish_reason='stop' if finish in ('stop','end_turn') else finish),event.text)
                                public = wire.public_answer(fmt,value,parsed.response['content'])
                                message = wire.response_message(fmt,public)
                                result['content'] = parsed.response['content']
                            result['proxy_message'] = message
                            public_body = (wire.answer_stream(fmt,public) if payload.get('stream') else wire.encoded(public)) if parsed else remote.body
                            public_headers = dict(remote.headers)
                            if parsed:
                                for key in list(public_headers):
                                    if key.lower() in ('content-length','etag','content-md5','content-encoding'):
                                        del public_headers[key]
                            result['proxy_wire'] = WireResponse(remote.status,public_headers,public_body).stored()
                            return replace(parsed,response=result) if parsed else result
                        except (ValueError,KeyError,TypeError) as exc:
                            # Compare against the composed value: after a guard
                            # rejection the outcome is `invalid_served_body+complete`,
                            # and the envelope failure must still be diagnosed.
                            if outcome == tagged('complete'):
                                outcome = tagged('invalid_inline_envelope' if inline else 'invalid_response')
                            raise ValueError('Provider response could not be safely delivered: '+str(exc)) from exc
                        finally:
                            with session._connect() as db:
                                db.execute('INSERT INTO proxy_attempts VALUES(?,?,?,?,?,?,?)',
                                    (request_id,attempt,remote.status if remote else None,outcome,time.perf_counter()-started,
                                     json.dumps(remote.stored()['headers'] if remote else {}),remote.body if remote else None))
                try:
                    result = ChatIO(session).invoke(request_id=request_id,reader=reader,
                                                   input_event_id=input_id,packet_id=packet.packet_id)
                except Exception as exc:
                    if isinstance(exc,UpstreamFailure):
                        failed = exc.response
                    else:
                        failed = WireResponse(502,{'content-type':'application/json'},wire.encoded(
                            {'error':{'type':'memory_upstream_error','message':str(exc)}}))
                    with session._connect() as db:
                        db.execute("UPDATE proxy_requests SET state='failed',failure=? WHERE request_id=?",
                                   (json.dumps(failed.stored()),request_id))
                    entry.reload()
                    return failed
                entry.messages.append((request_id+':assistant',result['proxy_message']))
                with session._connect() as db:
                    db.execute("UPDATE proxy_requests SET state='complete' WHERE request_id=?",(request_id,))
                return WireResponse.restore(result['proxy_wire'])
