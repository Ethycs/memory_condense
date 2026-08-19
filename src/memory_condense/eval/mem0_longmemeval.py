"""LongMemEval ingest, search, packing, attribution, and cleanup adapter."""

from __future__ import annotations

import copy
import re
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.eval.mem0_models import (
    MEM0_ATTRIBUTION_KIND,
    MEM0_CERTIFIED_RENDERING,
    MEM0_CONTEXT_SEPARATOR,
    MEM0_DATE_EXPOSURE_KIND,
    MEM0_ENRICHED_RENDERING,
    MEM0_OFFICIAL_THRESHOLD,
    MEM0_OFFICIAL_TOP_K,
    MEM0_REQUEST_WINDOW_MESSAGES,
    BackendFactory,
    Clock,
    Mem0AdapterError,
    Mem0AdapterStats,
    Mem0AttributionError,
    Mem0Candidate,
    Mem0IngestResult,
    Mem0PackDiagnostic,
    Mem0PoisonedError,
    Mem0PromptBudgetError,
    Mem0ProtocolError,
    Mem0SearchResult,
    MemoryLedger,
    PromptRenderer,
    ScopedMemoryKey,
    SourceRef,
    TokenCounter,
    _Closable,
    _PreparedCorpus,
)
from memory_condense.eval.mem0_protocol import (
    _memory_created_at,
    _memory_id,
    _memory_score,
    _memory_text,
    _merge_refs,
    _official_date_label,
    _prepared_longmemeval_record,
    _prepared_sample,
    _response_rows,
    _safe_label_value,
    _validate_threshold,
)
from memory_condense.eval.mem0_runtime import (
    _OwnedMem0Backend,
    _raise_cleanup_errors,
)
from memory_condense.ingest.loader import BenchmarkSample


class Mem0LongMemEvalAdapter:
    """Fake-testable Mem0 ingestion, retrieval, packing, and cleanup."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        backend: Any | None = None,
        backend_factory: BackendFactory | None = None,
        clock: Clock = time.perf_counter,
        threshold: float = MEM0_OFFICIAL_THRESHOLD,
        top_k: int = MEM0_OFFICIAL_TOP_K,
        vector_client: _Closable | None = None,
        user_scope_factory: Callable[[str], str] | None = None,
        token_counter_identity: str | None = None,
    ) -> None:
        if (backend is None) == (backend_factory is None):
            raise ValueError("Provide exactly one of backend or backend_factory.")
        if not callable(token_counter):
            raise TypeError("token_counter must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive int.")

        self._backend = backend
        self._backend_factory = backend_factory
        self._clock = clock
        self._token_counter = token_counter
        explicit_counter_identity = (
            token_counter_identity.strip()
            if isinstance(token_counter_identity, str)
            else ""
        )
        if token_counter_identity is not None and not explicit_counter_identity:
            raise ValueError("token_counter_identity must be non-empty when set.")
        if token_counter is count_tokens:
            identity = tokenizer_proxy_identity()
            recognized_identity = (
                f"{identity['schema']}:{identity['encoding']}:"
                f"{identity['vocabulary_sha256']}"
            )
            if (
                explicit_counter_identity
                and explicit_counter_identity != recognized_identity
            ):
                raise ValueError(
                    "token_counter_identity disagrees with the recognized "
                    "memory_condense count_tokens identity."
                )
            self._token_counter_identity = recognized_identity
            self._token_counter_identity_verified = True
        elif explicit_counter_identity:
            # A caller can bind a custom counter to an experiment, but this
            # adapter cannot independently verify that declaration.
            self._token_counter_identity = explicit_counter_identity
            self._token_counter_identity_verified = False
        else:
            module = getattr(token_counter, "__module__", "unknown")
            name = getattr(
                token_counter,
                "__qualname__",
                getattr(token_counter, "__name__", type(token_counter).__name__),
            )
            self._token_counter_identity = f"callable:{module}.{name}:unverified"
            self._token_counter_identity_verified = False
        self._threshold = _validate_threshold(threshold)
        self._top_k = top_k
        self._vector_client = vector_client
        self._user_scope_factory = user_scope_factory or self._default_user_scope
        self._stats = Mem0AdapterStats(
            token_counter_identity=self._token_counter_identity,
            token_counter_identity_verified=self._token_counter_identity_verified,
        )
        self._ledger: dict[ScopedMemoryKey, list[SourceRef]] = {}
        self._scopes: list[str] = []
        self._scope_protocol: dict[str, bool] = {}
        self._active_scope: str | None = None
        self._poisoned_reason: str | None = None
        self._closed = False

    @staticmethod
    def _default_user_scope(sample_id: str) -> str:
        safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")
        safe_sample = safe_sample or "sample"
        return f"longmemeval:{safe_sample}:{uuid.uuid4().hex}"

    @property
    def stats(self) -> Mem0AdapterStats:
        return self._stats

    @property
    def ledger(self) -> MemoryLedger:
        return MappingProxyType(
            {key: tuple(rows) for key, rows in self._ledger.items()}
        )

    @property
    def active_user_scope(self) -> str | None:
        return self._active_scope

    @property
    def supports_exact_source_provenance(self) -> bool:
        return False

    def require_exact_source_provenance(self) -> None:
        raise Mem0AttributionError(
            "Mem0 OSS 2.0.18 does not expose exact grounding for inferred "
            "memories. Only request-window attribution is available."
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise Mem0AdapterError("The Mem0 adapter is closed.")

    def _ensure_usable(self) -> None:
        self._ensure_open()
        if self._poisoned_reason is not None:
            raise Mem0PoisonedError(
                "The Mem0 adapter is poisoned after an ambiguous mutation; "
                f"only cleanup is safe ({self._poisoned_reason})."
            )

    def _get_backend(self) -> Any:
        self._ensure_usable()
        if self._backend is None:
            assert self._backend_factory is not None
            self._backend = self._backend_factory()
        return self._backend

    def _runtime_identity_snapshot(self) -> Mapping[str, Any]:
        value = getattr(self._backend, "runtime_identity", {})
        if not isinstance(value, Mapping):
            return MappingProxyType({})
        return MappingProxyType(copy.deepcopy(dict(value)))

    def _backend_is_certified(self) -> bool:
        if not isinstance(self._backend, _OwnedMem0Backend):
            return False
        identity = self._runtime_identity_snapshot()
        return (
            identity.get("certified") is True
            and identity.get("local_owned_state") is True
            and identity.get("on_disk") is True
            and identity.get("protocol")
            == "mem0-oss-2.0.18-certified-local-v1"
        )

    def _tokens(self, text: str) -> int:
        value = self._token_counter(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token_counter must return a non-negative int.")
        return value

    @staticmethod
    def _render_add_input(messages: Sequence[tuple[str, str]]) -> str:
        return "\n".join(f"{role}: {content}" for role, content in messages)

    def _ledger_snapshot(self, scope: str | None = None) -> MemoryLedger:
        return MappingProxyType(
            {
                key: tuple(rows)
                for key, rows in self._ledger.items()
                if scope is None or key[0] == scope
            }
        )

    def _ingest_prepared(self, corpus: _PreparedCorpus) -> Mem0IngestResult:
        self._ensure_usable()
        user_scope = self._user_scope_factory(corpus.sample_id).strip()
        if not user_scope:
            raise ValueError("user_scope_factory returned an empty scope.")
        if user_scope in self._scopes:
            raise ValueError(
                "user_scope_factory returned a duplicate scope; every sample "
                "ingestion must be isolated."
            )
        returned_ids: list[str] = []
        recent_message_refs: deque[SourceRef] = deque(
            maxlen=MEM0_REQUEST_WINDOW_MESSAGES
        )
        backend = self._get_backend()
        self._scopes.append(user_scope)
        self._scope_protocol[user_scope] = corpus.official_longmemeval_protocol
        self._active_scope = user_scope

        for batch in corpus.batches:
            prior_refs: list[SourceRef] = []
            for ref in recent_message_refs:
                if ref not in prior_refs:
                    prior_refs.append(ref)
            request_window = tuple(_merge_refs(prior_refs, (batch.ref,)))
            messages = [
                {"role": role, "content": content}
                for role, content in batch.messages
            ]
            raw_message_tokens = self._tokens(
                self._render_add_input(batch.messages)
            )
            started = self._clock()
            response: Any = None
            operation_error: BaseException | None = None
            try:
                response = backend.add(
                    messages,
                    user_id=user_scope,
                    infer=True,
                )
                rows = _response_rows(response, operation="add")
                response_ids: list[str] = []
                for row_index, row in enumerate(rows):
                    memory_id = _memory_id(row)
                    if memory_id is None:
                        raise Mem0ProtocolError(
                            f"Mem0 add result {row_index} omitted its memory ID."
                        )
                    response_ids.append(memory_id)
            except BaseException as exc:
                operation_error = exc
                self._poisoned_reason = f"add request for {batch.ref.source!r} failed"
                raise
            finally:
                elapsed = max(0.0, self._clock() - started)
                self._stats = replace(
                    self._stats,
                    add_calls=self._stats.add_calls + 1,
                    add_attempted_calls=self._stats.add_attempted_calls + 1,
                    add_completed_calls=(
                        self._stats.add_completed_calls
                        + (0 if operation_error is not None else 1)
                    ),
                    add_failed_calls=(
                        self._stats.add_failed_calls
                        + (1 if operation_error is not None else 0)
                    ),
                    add_latency_s=self._stats.add_latency_s + elapsed,
                    add_raw_message_tokens=(
                        self._stats.add_raw_message_tokens + raw_message_tokens
                    ),
                )

            for memory_id in response_ids:
                returned_ids.append(memory_id)
                key = (user_scope, memory_id)
                self._ledger[key] = _merge_refs(
                    self._ledger.get(key, ()), request_window
                )
            for _role, _content in batch.messages:
                recent_message_refs.append(batch.ref)
            self._stats = replace(
                self._stats,
                add_returned_memories=(
                    self._stats.add_returned_memories + len(response_ids)
                ),
                unique_ledger_memories=len(self._ledger),
            )

        return Mem0IngestResult(
            sample_id=corpus.sample_id,
            user_scope=user_scope,
            batches_added=tuple(batch.ref for batch in corpus.batches),
            returned_memory_ids=tuple(returned_ids),
            ledger=self._ledger_snapshot(user_scope),
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
            date_exposure_kind=MEM0_DATE_EXPOSURE_KIND,
            raw_pair_count=corpus.raw_pair_count,
            skipped_empty_pair_count=corpus.skipped_empty_pair_count,
            official_longmemeval_protocol=corpus.official_longmemeval_protocol,
            comparison_certified=(
                corpus.official_longmemeval_protocol
                and self._backend_is_certified()
                and self._token_counter_identity_verified
            ),
            runtime_identity=self._runtime_identity_snapshot(),
            stats=self._stats,
        )

    def ingest_sample(self, sample: BenchmarkSample) -> Mem0IngestResult:
        """Ingest an already-normalized sample as a non-certifying ablation.

        ``BenchmarkSample`` cannot preserve LongMemEval's empty raw turns. Use
        :meth:`ingest_longmemeval_record` for official comparisons.
        """

        return self._ingest_prepared(_prepared_sample(sample))

    def ingest_longmemeval_record(
        self, record: Mapping[str, Any]
    ) -> Mem0IngestResult:
        """Ingest one raw record with official pairing and empty-pair parity."""

        return self._ingest_prepared(_prepared_longmemeval_record(record))

    ingest = ingest_sample

    def _normalize_pool(
        self, response: Any, *, scope: str
    ) -> tuple[Mem0Candidate, ...]:
        candidates: list[Mem0Candidate] = []
        for rank, row in enumerate(
            _response_rows(response, operation="search"), start=1
        ):
            memory_id = _memory_id(row)
            if memory_id is None:
                raise Mem0AttributionError(
                    f"Mem0 search result {rank} omitted its memory ID."
                )
            key = (scope, memory_id)
            attribution = self._ledger.get(key)
            if not attribution:
                raise Mem0AttributionError(
                    "Mem0 search returned an unaudited memory ID for this "
                    f"scope: {memory_id!r}."
                )
            metadata_value = row.get("metadata", {})
            metadata = (
                dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
            )
            candidates.append(
                Mem0Candidate(
                    rank=rank,
                    memory_id=memory_id,
                    text=_memory_text(row),
                    score=_memory_score(row),
                    created_at=_memory_created_at(row),
                    metadata=MappingProxyType(metadata),
                    request_window_attribution=tuple(attribution),
                    attribution_kind=MEM0_ATTRIBUTION_KIND,
                    raw=copy.deepcopy(row),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _candidate_label(candidate: Mem0Candidate) -> str:
        refs = candidate.request_window_attribution

        def values(name: str) -> str:
            found: list[str] = []
            for ref in refs:
                value = _safe_label_value(getattr(ref, name))
                if value and value not in found:
                    found.append(value)
            return ",".join(found)

        return (
            f"[Memory {candidate.rank} | id={_safe_label_value(candidate.memory_id)} | "
            f"attribution={MEM0_ATTRIBUTION_KIND} | "
            f"source={values('source')} | session={values('session')} | "
            f"batch={values('batch_index')} | date={values('date')}]"
        )

    @classmethod
    def _render_enriched_candidate(cls, candidate: Mem0Candidate) -> str:
        return f"{cls._candidate_label(candidate)}\n{candidate.text}"

    @staticmethod
    def _render_official_context(
        candidates: Sequence[Mem0Candidate],
    ) -> str:
        """Match the official Mem0 benchmark's dated memory-only rendering."""

        grouped: dict[str, list[Mem0Candidate]] = {}
        for candidate in sorted(
            candidates,
            key=lambda item: (item.created_at or "", item.rank),
        ):
            if candidate.created_at is None:
                raise Mem0ProtocolError(
                    "Official Mem0 rendering requires created_at on every "
                    f"returned memory ({candidate.memory_id!r})."
                )
            label = _official_date_label(candidate.created_at)
            grouped.setdefault(label, []).append(candidate)
        sections = [
            f"--- {label} ---\n"
            + "\n".join(f"- {candidate.text}" for candidate in rows)
            for label, rows in grouped.items()
        ]
        return "\n".join(sections)

    @classmethod
    def _render_context(
        cls,
        candidates: Sequence[Mem0Candidate],
        *,
        rendering_mode: str,
    ) -> str:
        if rendering_mode == MEM0_CERTIFIED_RENDERING:
            return cls._render_official_context(candidates)
        if rendering_mode == MEM0_ENRICHED_RENDERING:
            return MEM0_CONTEXT_SEPARATOR.join(
                cls._render_enriched_candidate(candidate)
                for candidate in candidates
            )
        raise ValueError(
            "rendering_mode must be MEM0_CERTIFIED_RENDERING or "
            "MEM0_ENRICHED_RENDERING."
        )

    def _render_prompt(
        self, renderer: PromptRenderer, query: str, context: str
    ) -> str:
        prompt = renderer(query, context)
        if not isinstance(prompt, str):
            raise TypeError("prompt_renderer must return str.")
        return prompt

    def search(
        self,
        query: str,
        *,
        max_prompt_tokens: int,
        prompt_renderer: PromptRenderer,
        prompt_token_overhead: int = 0,
        context_token_budget: int | None = None,
        user_scope: str | None = None,
        threshold: float | None = None,
        rendering_mode: str = MEM0_CERTIFIED_RENDERING,
    ) -> Mem0SearchResult:
        """Search under a complete rendered responder prompt-proxy budget.

        ``max_prompt_tokens`` and ``prompt_token_overhead`` are retained API
        spellings. Both refer to the declared local token-count proxy. Callers
        should include any chat framing allowance in the overhead.
        """

        self._ensure_usable()
        if (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 0
        ):
            raise ValueError("max_prompt_tokens must be a non-negative int.")
        if (
            isinstance(prompt_token_overhead, bool)
            or not isinstance(prompt_token_overhead, int)
            or prompt_token_overhead < 0
        ):
            raise ValueError("prompt_token_overhead must be a non-negative int.")
        if context_token_budget is not None and (
            isinstance(context_token_budget, bool)
            or not isinstance(context_token_budget, int)
            or context_token_budget < 0
        ):
            raise ValueError("context_token_budget must be a non-negative int or null.")
        if not callable(prompt_renderer):
            raise TypeError("prompt_renderer must be callable.")
        if rendering_mode not in {
            MEM0_CERTIFIED_RENDERING,
            MEM0_ENRICHED_RENDERING,
        }:
            raise ValueError(
                "rendering_mode must be MEM0_CERTIFIED_RENDERING or "
                "MEM0_ENRICHED_RENDERING."
            )
        scope = user_scope or self._active_scope
        if not scope or scope not in self._scopes:
            raise ValueError("Search requires a user scope returned by ingest_sample().")
        effective_threshold = _validate_threshold(
            self._threshold if threshold is None else threshold
        )
        official_search_protocol = (
            self._top_k == MEM0_OFFICIAL_TOP_K
            and effective_threshold == MEM0_OFFICIAL_THRESHOLD
        )

        empty_prompt = self._render_prompt(prompt_renderer, query, "")
        empty_prompt_tokens = self._tokens(empty_prompt) + prompt_token_overhead
        if empty_prompt_tokens > max_prompt_tokens:
            raise Mem0PromptBudgetError(
                "The responder prompt without retrieved context already exceeds "
                f"the cap ({empty_prompt_tokens} > {max_prompt_tokens})."
            )

        query_tokens = self._tokens(query)
        backend = self._get_backend()
        started = self._clock()
        try:
            response = backend.search(
                query,
                top_k=self._top_k,
                filters={"user_id": scope},
                threshold=effective_threshold,
                rerank=False,
                explain=False,
            )
        finally:
            elapsed = max(0.0, self._clock() - started)
            self._stats = replace(
                self._stats,
                search_calls=self._stats.search_calls + 1,
                search_latency_s=self._stats.search_latency_s + elapsed,
                search_query_tokens=self._stats.search_query_tokens + query_tokens,
            )

        raw_pool = self._normalize_pool(response, scope=scope)
        raw_tokens = sum(self._tokens(candidate.text) for candidate in raw_pool)
        packed: list[Mem0Candidate] = []
        diagnostics: list[Mem0PackDiagnostic] = []
        context = ""
        context_tokens = 0
        prompt = empty_prompt
        prompt_tokens = empty_prompt_tokens

        for candidate in raw_pool:
            rendered = self._render_context(
                (candidate,), rendering_mode=rendering_mode
            )
            audit_rendered = self._render_enriched_candidate(candidate)
            rendered_tokens = self._tokens(rendered)
            if not candidate.text:
                selected = False
                reason = "empty_memory"
                proposed_context_tokens = context_tokens
                proposed_prompt_tokens = prompt_tokens
                proposed = context
                proposed_prompt = prompt
            else:
                proposed_candidates = [*packed, candidate]
                proposed = self._render_context(
                    proposed_candidates,
                    rendering_mode=rendering_mode,
                )
                proposed_context_tokens = self._tokens(proposed)
                proposed_prompt = self._render_prompt(
                    prompt_renderer, query, proposed
                )
                proposed_prompt_tokens = (
                    self._tokens(proposed_prompt) + prompt_token_overhead
                )
                if (
                    context_token_budget is not None
                    and proposed_context_tokens > context_token_budget
                ):
                    selected = False
                    reason = "context_token_budget"
                elif proposed_prompt_tokens > max_prompt_tokens:
                    selected = False
                    reason = "prompt_token_budget"
                else:
                    selected = True
                    reason = "selected"

            if selected:
                packed.append(candidate)
                context = proposed
                context_tokens = proposed_context_tokens
                prompt = proposed_prompt
                prompt_tokens = proposed_prompt_tokens
            diagnostics.append(
                Mem0PackDiagnostic(
                    candidate=candidate,
                    rendered=rendered,
                    audit_rendered=audit_rendered,
                    rendered_tokens=rendered_tokens,
                    selected=selected,
                    reason=reason,
                    context_tokens_after=context_tokens,
                    prompt_token_proxy_after=prompt_tokens,
                )
            )

        # Re-render and recount the exact strings returned to the caller; BPE
        # boundaries can change across labels, separators, and prompt framing.
        context = self._render_context(packed, rendering_mode=rendering_mode)
        context_tokens = self._tokens(context)
        prompt = self._render_prompt(prompt_renderer, query, context)
        prompt_tokens = self._tokens(prompt) + prompt_token_overhead
        if prompt_tokens > max_prompt_tokens:
            raise Mem0PromptBudgetError(
                "Final prompt recount exceeded the declared cap; the renderer "
                "must be deterministic during one search call."
            )

        self._stats = replace(
            self._stats,
            search_raw_memory_tokens=(
                self._stats.search_raw_memory_tokens + raw_tokens
            ),
            search_context_tokens=(
                self._stats.search_context_tokens + context_tokens
            ),
            search_prompt_token_proxy=(
                self._stats.search_prompt_token_proxy + prompt_tokens
            ),
            search_returned_memories=(
                self._stats.search_returned_memories + len(raw_pool)
            ),
            search_packed_memories=(
                self._stats.search_packed_memories + len(packed)
            ),
        )
        return Mem0SearchResult(
            user_scope=scope,
            query=query,
            context=context,
            context_tokens=context_tokens,
            prompt=prompt,
            prompt_token_proxy=prompt_tokens,
            max_prompt_token_proxy=max_prompt_tokens,
            prompt_token_proxy_overhead=prompt_token_overhead,
            empty_context_prompt_token_proxy=empty_prompt_tokens,
            residual_prompt_token_proxy=max_prompt_tokens - prompt_tokens,
            prompt_token_proxy_budget_compliant=True,
            token_counter_identity=self._token_counter_identity,
            token_counter_identity_verified=self._token_counter_identity_verified,
            packed=tuple(packed),
            raw_pool=raw_pool,
            diagnostics=tuple(diagnostics),
            raw_response=copy.deepcopy(response),
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
            rendering_mode=rendering_mode,
            certified_rendering=(rendering_mode == MEM0_CERTIFIED_RENDERING),
            official_longmemeval_protocol=self._scope_protocol.get(scope, False),
            official_search_protocol=official_search_protocol,
            comparison_certified=(
                rendering_mode == MEM0_CERTIFIED_RENDERING
                and self._scope_protocol.get(scope, False)
                and official_search_protocol
                and self._backend_is_certified()
                and self._token_counter_identity_verified
            ),
            runtime_identity=self._runtime_identity_snapshot(),
            stats=self._stats,
        )

    retrieve = search

    def release_scope(self, user_scope: str) -> None:
        """Delete and forget one completed sample before processing the next."""

        self._ensure_usable()
        if not isinstance(user_scope, str) or user_scope not in self._scopes:
            raise ValueError("release_scope requires a live scope from ingest.")
        backend = self._get_backend()
        try:
            backend.delete_all(user_id=user_scope)
        except BaseException:
            self._poisoned_reason = f"scope release for {user_scope!r} failed"
            raise
        self._scopes.remove(user_scope)
        self._scope_protocol.pop(user_scope, None)
        for key in [key for key in self._ledger if key[0] == user_scope]:
            del self._ledger[key]
        if self._active_scope == user_scope:
            self._active_scope = self._scopes[-1] if self._scopes else None
        self._stats = replace(
            self._stats,
            released_scopes=self._stats.released_scopes + 1,
            unique_ledger_memories=len(self._ledger),
        )

    release_sample = release_scope

    def cleanup(self) -> None:
        """Delete scopes, close owned resources, and clear local attribution."""

        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        backend = self._backend
        if backend is not None:
            delete_all = getattr(backend, "delete_all", None)
            if callable(delete_all):
                for scope in self._scopes:
                    try:
                        delete_all(user_id=scope)
                    except BaseException as exc:
                        errors.append(exc)

            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                try:
                    close_backend()
                except BaseException as exc:
                    errors.append(exc)

        if self._vector_client is not None:
            close_vector = getattr(self._vector_client, "close", None)
            if callable(close_vector):
                try:
                    close_vector()
                except BaseException as exc:
                    errors.append(exc)

        self._ledger.clear()
        self._scopes.clear()
        self._scope_protocol.clear()
        self._active_scope = None
        _raise_cleanup_errors(errors, "Mem0 adapter cleanup failed")

    close = cleanup

    def __enter__(self) -> Mem0LongMemEvalAdapter:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        try:
            self.cleanup()
        except BaseException as cleanup_exc:
            if exc is None:
                raise
            exc.add_note(f"Mem0 cleanup also failed: {cleanup_exc!r}")
        return False


Mem0Adapter = Mem0LongMemEvalAdapter
