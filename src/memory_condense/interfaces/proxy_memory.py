"""Durable ChatSession integration for provider-compatible memory requests."""
from base64 import b64decode, b64encode
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
import time

from memory_condense.application.chat_io import ChatIO
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

    def reload(self):
        self.messages = []
        for event in self.session.events():
            if 'proxy_policy' in event.metadata:
                self.policy = event.metadata['proxy_policy']
            message = event.metadata.get('proxy_message') or event.metadata.get('response', {}).get('proxy_message')
            if message is not None:
                self.messages.append((event.event_id, message))


class MemoryProxy:
    """One namespace per credential/provider/conversation; one IO owner per session.

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

    @staticmethod
    def namespace(provider, headers):
        headers = {k.lower():v for k,v in headers.items()}
        conversation = headers.get('x-memory-conversation-id') or headers.get('x-conversation-id')
        if not conversation or not conversation.strip():
            raise MemoryProxyError('Memory mode requires x-memory-conversation-id')
        if len(conversation) > 512:
            raise MemoryProxyError('Conversation ID is too long')
        # A rotating credential selects a new namespace. Hashes are local only.
        credential = hashlib.sha256(wire.encoded({k:headers[k] for k in
            ('authorization','x-api-key','api-key','openai-api-key') if k in headers})).hexdigest()
        return hashlib.sha256(wire.encoded([provider, credential, conversation.strip()])).hexdigest()

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
                    if 'served_body' not in {r[1] for r in db.execute('PRAGMA table_info(proxy_requests)')}:
                        db.execute('ALTER TABLE proxy_requests ADD COLUMN served_body BLOB')
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
        return dict(active_sessions=len(values), pending_events=pending_events,
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

    def flush(self, provider, headers):
        entry = self.entry(self.namespace(provider, headers))
        with entry.session._io_lock:
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
    def _new_messages(history, incoming):
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
        return [m for _,m in entry.messages[start:]]

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
        entry = self.entry(namespace, summary_api=summary_api)
        session = entry.session
        with session._io_lock:
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
            new = self._new_messages(entry.messages, incoming) if not prior else []
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

                def reader():
                    with session._connect() as db:
                        db.execute("UPDATE proxy_requests SET state='calling',served_body=? WHERE request_id=?",(wire.encoded(served),request_id))
                    for attempt in range(self.empty_response_retries+1):
                        started = time.perf_counter()
                        outcome, remote = 'unknown', None
                        try:
                            # Pre-send guard: a served body that lost a tool
                            # pairing is never sent; the client's original goes
                            # through verbatim (passthrough is the sanctioned
                            # degraded mode) and the ledger records why.
                            problems = guard_served_body(fmt, served)
                            if problems:
                                outcome = 'invalid_served_body'
                            remote = send(body if problems else wire.encoded(served))
                            if remote.status >= 400:
                                outcome = 'http_error'
                                raise UpstreamFailure(remote)
                            value = wire.assemble_stream(fmt,remote.body) if payload.get('stream') else json.loads(remote.body)
                            message = wire.response_message(fmt,value)
                            finish = wire.finish_reason(fmt,value)
                            if finish not in ('stop','end_turn','tool_calls','function_call','tool_use','stop_sequence','pause_turn','refusal'):
                                outcome = 'incomplete_response'
                                raise ValueError('Provider response did not complete: '+str(finish))
                            empty = not wire.text_content(message).strip() and not wire.has_alternative(message)
                            reasoning_only = wire.reasoning_present(fmt,value)
                            if empty:
                                outcome = 'empty_completed' if finish in ('stop','end_turn') and not reasoning_only else 'incomplete_or_reasoning_only'
                                if outcome == 'empty_completed' and attempt < self.empty_response_retries:
                                    time.sleep(attempt+1)
                                    continue
                                raise ValueError('Provider returned no completed answer')
                            outcome = 'complete' if not problems else 'invalid_served_body'
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
                            if outcome == 'complete':
                                outcome = 'invalid_inline_envelope' if inline else 'invalid_response'
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
