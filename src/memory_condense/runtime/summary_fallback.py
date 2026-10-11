"""Keep validated local generations; escalate failed summary jobs once per input."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

import httpx

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.local import extraction_job, raw_extracts, merge_extracts


class SummaryEscalationError(RuntimeError):
    """Keep failed work pending; never disguise an API failure as a good summary."""


def escalation_job(kind, messages, max_tokens, typed_request=None):
    """Make item coverage explicit, including receipt/metadata fragments."""
    prompt, budget = extraction_job(kind, messages, max_tokens, typed_request)
    if kind == 'merge':
        required = list(range(len(typed_request.fragments)))
        coverage = ('The required output indices are '+json.dumps(required)+'. '
            'Return exactly one extract for each index, in that order; never repeat an index. '
            'Every fragment counts, including JSON metadata, recall receipts and feedback. '
            'For a metadata fragment, quote an exact substring from that fragment itself. '
            'Do not replace it with a second quote from a more informative fragment. ')
    else:
        required = [f['label'] for f in json.loads(messages[1]['content'])['fragments']]
        coverage = ('The required output labels are '+json.dumps(required)+'. '
            'Return exactly one atom for each label, in that order. ')
    return [dict(prompt[0], content=prompt[0]['content']+' '+coverage+
                 ('Check coverage and exact substring matches before returning only the indexed lines.'
                  if kind == 'merge' else
                  'Check coverage and exact substring matches before returning only the JSON object.')),
            *prompt[1:]], budget


class LocalFirstSummaries:
    """Conversation-scoped provider, with shared local attention/embedding/models.

    No API is substituted for Qwen attention or BGE vectors. Their model identity
    remains part of the memory index. Only generative raw/merge jobs escalate.
    """
    def __init__(self, local, root, api, *, require_provider=False):
        self.local, self.root, self.api = local, Path(root), api
        self.require_provider = require_provider
        self.metrics = Counter()
        self._lock = threading.RLock()
        self._jobs = {}

    def configure(self, api):
        with self._lock:
            self.api = api

    def embedding(self):
        return self.local.embedding()

    def tactic_router(self):
        return self.local.tactic_router()

    def attention(self, cache):
        return self.local.attention(cache)

    def start_reader(self):
        # Service startup owns the model-start lock. An unavailable summarizer
        # must not repeatedly reload or prevent otherwise healthy local models.
        if self.api is None:
            return self.local.start_reader()
        if getattr(self.local, '_summary_start_failure', None):
            return
        try:
            self.local.start_reader()
        except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            self.local.stop_reader()
            self.local._summary_start_failure = type(exc).__name__
            save(self.root/'local-startup-failure.json', dict(error_type=type(exc).__name__))

    @staticmethod
    def _validate(kind, messages, response, typed_request, *, ground_near_exact=False):
        finish = response.get('generated_finish_reason', response.get('finish_reason'))
        if finish != 'stop':
            raise ValueError('Summary generation did not complete')
        content = response.get('generated_content', response.get('content', ''))
        if not isinstance(content, str):
            raise ValueError('Summary generation returned no text')
        if kind == 'raw':
            value, missing = raw_extracts(json.loads(messages[1]['content'])['fragments'], content,
                                          ground_near_exact=ground_near_exact)
        else:
            value, missing = merge_extracts(typed_request, content)
        if missing or response.get('local_extractive_fallbacks', 0):
            raise ValueError('Summary requires ungenerated replacement extracts')
        return value

    @staticmethod
    def _raw_partial(messages, response):
        """Keep only model-generated, source-exact atoms; identify residual work."""
        fragments = json.loads(messages[1]['content'])['fragments']
        if response.get('local_extractive_fallbacks', 0) and 'generated_content' not in response:
            return {}, list(fragments)
        generated = response.get('generated_content', response.get('content', ''))
        atoms, missing = {}, []
        for fragment in fragments:
            value, replacements = raw_extracts([fragment], generated)
            if replacements:
                missing.append(fragment)
            else:
                atom = value['atoms'][0]
                atoms[atom['label']] = atom
        return atoms, missing

    @staticmethod
    def _raw_messages(messages, fragments):
        payload = json.loads(messages[1]['content'])
        payload['fragments'] = fragments
        return [messages[0], dict(messages[1], content=json.dumps(payload, ensure_ascii=False))]

    def call(self, kind, messages, *, scope, max_tokens=256, typed_request=None, summary_attempt=0):
        if kind not in ('raw', 'merge'):
            return self.local.call(kind, messages, scope=scope, max_tokens=max_tokens,
                                   typed_request=typed_request, summary_attempt=summary_attempt)
        # Validate the job before catching model failures. Bad caller input is
        # not an excuse to send unrelated or malformed data to another model.
        prompt, budget = escalation_job(kind, messages, max_tokens, typed_request)
        with self._lock:
            api = self.api
            if api is None:
                if self.require_provider:
                    raise SummaryEscalationError('A chat request must bind the provider/model before pending summaries can resume')
                return self.local.call(kind, messages, scope=scope, max_tokens=max_tokens,
                    typed_request=typed_request, summary_attempt=summary_attempt)
            identity = dict(format='local-first-source-extracts-v2', kind=kind,
                messages=messages, max_tokens=max_tokens,
                typed_request=asdict(typed_request) if typed_request is not None else None,
                provider=api.identity)
            key = identity_sha256(identity)
            job_lock = self._jobs.setdefault(key, threading.Lock())
        with job_lock:
            directory = self.root/key
            accepted = directory/'accepted.json'
            if accepted.exists():
                try:
                    cached = read(accepted)
                    value = self._validate(kind, messages, cached, typed_request)
                    if json.loads(cached['content']) != value:
                        raise ValueError('Cached summary differs from validated extracts')
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise SummaryEscalationError('Cached API summary needs review; receipt retained at '+str(directory)) from exc
                with self._lock:
                    self.metrics['api_cache_hits'] += 1
                return cached
            # An unknown or invalid paid outcome is retained, never retried by
            # compiler repair loops, concurrent callers, or a process restart.
            if (directory/'reservation.json').exists():
                raise SummaryEscalationError('Prior API summary attempt needs review; receipt retained at '+str(directory))
            started = time.perf_counter()
            local = None
            partial_atoms = None
            api_validation_messages = messages
            local_retry_receipts = []
            residual_labels = []
            failure = getattr(self.local, '_summary_start_failure', None)
            if failure is None:
                try:
                    local = self.local.call(kind, messages, scope=scope, max_tokens=max_tokens,
                                            typed_request=typed_request, summary_attempt=summary_attempt)
                except (httpx.HTTPError, OSError) as exc:
                    failure = type(exc).__name__
                else:
                    try:
                        self._validate(kind, messages, local, typed_request)
                    except ValueError as exc:
                        failure = str(exc)
                    else:
                        with self._lock:
                            self.metrics['local_accepted'] += 1
                        return local
            # A long raw batch may contain several perfectly generated atoms
            # and one malformed/missing label. Preserve the valid exact
            # extracts, retry only the residual atoms locally one at a time,
            # and escalate only what remains. This keeps provider input
            # source-minimal without accepting a synthetic local replacement
            # as a successful model summary.
            if kind == 'raw' and local is not None:
                partial_atoms, missing = self._raw_partial(messages, local)
                if 1 < len(json.loads(messages[1]['content'])['fragments']) and missing:
                    residual = []
                    for fragment in missing:
                        one = self._raw_messages(messages, [fragment])
                        retry = self.local.call(kind, one, scope=scope, max_tokens=max_tokens,
                                                typed_request=None, summary_attempt=summary_attempt)
                        local_retry_receipts.append(retry.get('request_sha256'))
                        try:
                            value = self._validate(kind, one, retry, None)
                        except ValueError:
                            residual.append(fragment)
                        else:
                            atom = value['atoms'][0]
                            partial_atoms[atom['label']] = atom
                    missing = residual
                if not missing:
                    fragments = json.loads(messages[1]['content'])['fragments']
                    value = dict(atoms=[partial_atoms[f['label']] for f in fragments])
                    content = json.dumps(value)
                    receipt = save(directory/'local-repair.json', dict(
                        initial_request_sha256=local.get('request_sha256'),
                        retry_request_sha256=local_retry_receipts,
                        labels=[f['label'] for f in fragments]))
                    result = dict(local, content=content, generated_content=content,
                        initial_generated_content=local.get('generated_content', local.get('content')),
                        generated_finish_reason='stop', finish_reason='stop',
                        request_sha256=receipt.sha256, elapsed_s=time.perf_counter()-started,
                        local_extractive_fallbacks=0, summary_route='local_targeted_repair')
                    with self._lock:
                        self.metrics['local_repaired'] += 1
                    return result
                api_validation_messages = self._raw_messages(messages, missing)
                prompt, budget = escalation_job(kind, api_validation_messages, max_tokens, None)
                residual_labels = [f['label'] for f in missing]
            with self._lock:
                self.metrics['local_failed'] += 1
            request = save(directory/'request.json', dict(**identity, scope=scope,
                api_messages=prompt, api_max_tokens=budget, local_failure=failure,
                local_request_sha256=local.get('request_sha256') if local else None,
                local_retry_request_sha256=local_retry_receipts,
                residual_labels=residual_labels))
            save(directory/'reservation.json', dict(request_sha256=request.sha256))
            with self._lock:
                self.metrics['api_calls'] += 1
            try:
                remote = api(prompt, max_tokens=budget)
                save(directory/'response.json', remote)
                api_source_grounding = False
                try:
                    value = self._validate(kind, api_validation_messages, remote, typed_request)
                except ValueError:
                    if kind != 'raw':
                        raise
                    value = self._validate(kind, api_validation_messages, remote, typed_request,
                                           ground_near_exact=True)
                    api_source_grounding = True
                api_generated = remote['content']
                if partial_atoms is not None:
                    for atom in value['atoms']:
                        partial_atoms[atom['label']] = atom
                    fragments = json.loads(messages[1]['content'])['fragments']
                    value = dict(atoms=[partial_atoms[f['label']] for f in fragments])
                combined = json.dumps(value)
                result = dict(remote, content=combined, generated_content=combined,
                    api_generated_content=api_generated,
                    generated_finish_reason=remote['finish_reason'], request_sha256=request.sha256,
                    elapsed_s=time.perf_counter()-started, local_extractive_fallbacks=0,
                    summary_route='api_escalation', local_failure=failure,
                    api_source_grounding=api_source_grounding)
                save(accepted, result)
            except Exception as exc:
                save(directory/'error.json', dict(error_type=type(exc).__name__, request_sha256=request.sha256))
                with self._lock:
                    self.metrics['api_failed'] += 1
                raise SummaryEscalationError('API summary escalation failed; receipt retained at '+str(directory)) from exc
            with self._lock:
                self.metrics['api_accepted'] += 1
            return result
