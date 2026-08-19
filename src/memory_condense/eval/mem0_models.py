"""Contracts, constants, errors, and immutable results for Mem0 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

MEM0AI_PIN = "2.0.18"
MEM0_API_VERSION = "v1.1"
MEM0_CONTEXT_SEPARATOR = "\n\n"
MEM0_REQUEST_WINDOW_MESSAGES = 10
MEM0_OFFICIAL_TOP_K = 200
MEM0_OFFICIAL_THRESHOLD = 0.1
MEM0_ATTRIBUTION_KIND = "request_window_non_evidence"
MEM0_DATE_EXPOSURE_KIND = "diagnostics_only_not_model_input"
MEM0_PROVIDER_USAGE_STATUS = "unavailable_from_mem0_oss_public_api"
MEM0_CERTIFIED_RENDERING = "official-memory-text-created-at"
MEM0_ENRICHED_RENDERING = "enriched-attribution-noncertifying"
MEM0_BM25_MODEL = "Qdrant/bm25"
MEM0_SPACY_MODEL = "en_core_web_sm"

TokenCounter: TypeAlias = Callable[[str], int]
Clock: TypeAlias = Callable[[], float]
BackendFactory: TypeAlias = Callable[[], Any]
PromptRenderer: TypeAlias = Callable[[str, str], str]
ScopedMemoryKey: TypeAlias = tuple[str, str]
StackPreflight: TypeAlias = Callable[[], "Mem0StackIdentity"]


class Mem0AdapterError(RuntimeError):
    """Base error raised by the optional Mem0 benchmark boundary."""


class Mem0DependencyError(Mem0AdapterError):
    """The exact optional Mem0 distribution is unavailable."""


class Mem0ConfigurationError(Mem0AdapterError, ValueError):
    """A real Mem0 factory was not given a fully isolated frozen config."""


class Mem0ProtocolError(Mem0AdapterError):
    """Mem0 or the benchmark input violated the frozen comparison protocol."""


class Mem0AttributionError(Mem0ProtocolError):
    """A search row cannot be tied to an audited scoped add request window."""


class Mem0PromptBudgetError(Mem0AdapterError, ValueError):
    """The declared final responder prompt cannot fit its token cap."""


class Mem0PoisonedError(Mem0ProtocolError):
    """A possibly mutating operation failed and only cleanup remains safe."""


class _Closable(Protocol):
    def close(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Text-free identity of one official consecutive 1--2 turn add request."""

    sample_id: str
    source: str
    session: str
    session_index: int
    original_session_index: int
    batch_index: int
    date: str
    turn_start: int
    turn_count: int
    roles: tuple[str, ...]

    @property
    def pair(self) -> int:
        """Compatibility spelling for the historical two-turn batch index."""

        return self.batch_index

    @property
    def metadata(self) -> dict[str, str | int]:
        """Text-free audit metadata that is never supplied to Mem0."""

        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "session": self.session,
            "session_index": self.session_index,
            "original_session_index": self.original_session_index,
            "pair": self.batch_index,
            "date": self.date,
            "turn_start": self.turn_start,
            "turn_count": self.turn_count,
            "roles": ",".join(self.roles),
            "date_exposure": MEM0_DATE_EXPOSURE_KIND,
        }


# Kept as an import-compatible name while removing the false implication that
# every official slice contains a user/assistant pair.
SourcePair = SourceRef


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    ref: SourceRef
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PreparedCorpus:
    sample_id: str
    batches: tuple[_PreparedBatch, ...]
    raw_pair_count: int
    skipped_empty_pair_count: int
    official_longmemeval_protocol: bool


@dataclass(frozen=True, slots=True)
class Mem0StackIdentity:
    """Stable, secret-free identity of the certified OSS runtime stack."""

    dependency_versions: Mapping[str, str]
    bm25_model: str
    spacy_model: str
    bm25_operational: bool
    entity_extraction_operational: bool

    @property
    def certified(self) -> bool:
        return self.bm25_operational and self.entity_extraction_operational

    def as_dict(self) -> dict[str, Any]:
        return {
            "dependency_versions": dict(self.dependency_versions),
            "bm25_model": self.bm25_model,
            "spacy_model": self.spacy_model,
            "bm25_operational": self.bm25_operational,
            "entity_extraction_operational": self.entity_extraction_operational,
        }


MemoryLedger: TypeAlias = Mapping[ScopedMemoryKey, tuple[SourceRef, ...]]


@dataclass(frozen=True, slots=True)
class Mem0AdapterStats:
    """Cumulative local proxy accounting; provider usage is unavailable."""

    add_calls: int = 0
    add_attempted_calls: int = 0
    add_completed_calls: int = 0
    add_failed_calls: int = 0
    search_calls: int = 0
    add_latency_s: float = 0.0
    search_latency_s: float = 0.0
    add_raw_message_tokens: int = 0
    search_query_tokens: int = 0
    search_raw_memory_tokens: int = 0
    search_context_tokens: int = 0
    search_prompt_token_proxy: int = 0
    # Compatibility spelling. This is the same caller-supplied local proxy,
    # not an exact provider-token count. Always derived from
    # ``search_prompt_token_proxy`` in ``__post_init__``.
    search_prompt_tokens: int = field(init=False, default=0)
    add_returned_memories: int = 0
    unique_ledger_memories: int = 0
    search_returned_memories: int = 0
    search_packed_memories: int = 0
    released_scopes: int = 0
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_usage_status: str = MEM0_PROVIDER_USAGE_STATUS
    token_counter_identity: str = ""
    token_counter_identity_verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "search_prompt_tokens", self.search_prompt_token_proxy
        )


@dataclass(frozen=True, slots=True)
class Mem0IngestResult:
    """Identity, request-window attribution, and accounting for one sample."""

    sample_id: str
    user_scope: str
    batches_added: tuple[SourceRef, ...]
    returned_memory_ids: tuple[str, ...]
    ledger: MemoryLedger
    attribution_kind: str
    supports_exact_source_provenance: bool
    date_exposure_kind: str
    raw_pair_count: int
    skipped_empty_pair_count: int
    official_longmemeval_protocol: bool
    comparison_certified: bool
    runtime_identity: Mapping[str, Any]
    stats: Mem0AdapterStats

    @property
    def pairs_added(self) -> tuple[SourceRef, ...]:
        """Compatibility alias; batches may be singleton or assistant-first."""

        return self.batches_added


@dataclass(frozen=True, slots=True)
class Mem0Candidate:
    """One rank-preserving, scoped, audited Mem0 search row."""

    rank: int
    memory_id: str
    text: str
    score: float | None
    created_at: str | None
    metadata: Mapping[str, Any]
    request_window_attribution: tuple[SourceRef, ...]
    attribution_kind: str
    raw: Any


@dataclass(frozen=True, slots=True)
class Mem0PackDiagnostic:
    """Why a candidate was included in or excluded from final context."""

    candidate: Mem0Candidate
    rendered: str
    audit_rendered: str
    rendered_tokens: int
    selected: bool
    reason: str
    context_tokens_after: int
    prompt_token_proxy_after: int
    # Compatibility spelling, always derived from prompt_token_proxy_after.
    prompt_tokens_after: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prompt_tokens_after", self.prompt_token_proxy_after
        )


@dataclass(frozen=True, slots=True)
class Mem0SearchResult:
    """Budgeted context plus the complete raw-pool and prompt audit trail.

    ``prompt_token_proxy`` is counted by the caller-supplied counter. It is a
    hard local packing bound, not a claim about provider-token usage.
    """

    user_scope: str
    query: str
    context: str
    context_tokens: int
    prompt: str
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    prompt_token_proxy_overhead: int
    empty_context_prompt_token_proxy: int
    residual_prompt_token_proxy: int
    prompt_token_proxy_budget_compliant: bool
    token_counter_identity: str
    token_counter_identity_verified: bool
    # Compatibility fields below mirror their ``*_proxy`` counterparts and are
    # always derived from them in ``__post_init__``. In particular,
    # prompt_budget_certified certifies only deterministic local packing under
    # the declared counter; it is not provider usage.
    prompt_tokens: int = field(init=False)
    max_prompt_tokens: int = field(init=False)
    prompt_token_overhead: int = field(init=False)
    empty_context_prompt_tokens: int = field(init=False)
    residual_prompt_tokens: int = field(init=False)
    prompt_budget_certified: bool = field(init=False)
    packed: tuple[Mem0Candidate, ...]
    raw_pool: tuple[Mem0Candidate, ...]
    diagnostics: tuple[Mem0PackDiagnostic, ...]
    raw_response: Any
    attribution_kind: str
    supports_exact_source_provenance: bool
    rendering_mode: str
    certified_rendering: bool
    official_longmemeval_protocol: bool
    official_search_protocol: bool
    comparison_certified: bool
    runtime_identity: Mapping[str, Any]
    stats: Mem0AdapterStats

    def __post_init__(self) -> None:
        for alias, canonical in (
            ("prompt_tokens", self.prompt_token_proxy),
            ("max_prompt_tokens", self.max_prompt_token_proxy),
            ("prompt_token_overhead", self.prompt_token_proxy_overhead),
            (
                "empty_context_prompt_tokens",
                self.empty_context_prompt_token_proxy,
            ),
            ("residual_prompt_tokens", self.residual_prompt_token_proxy),
            (
                "prompt_budget_certified",
                self.prompt_token_proxy_budget_compliant,
            ),
        ):
            object.__setattr__(self, alias, canonical)
