"""Independent prompt packing for the locked Mem0 comparison arm.

Stage A of the Mem0 comparison runs the optional Mem0 stack and emits a
sanitized search artifact.  Stage B runs in the frozen memory-condense
environment and calls this module.  The separation is intentional: no prompt
or candidate-selection decision made inside the adapter is trusted here.

The packer admits whole memories in retrieval-rank order.  After each
admission it renders the admitted set in ``created_at`` chronology, rebuilds
the exact two-message :func:`memory_condense.eval.benchmark.build_qa_prompt`,
and recounts the complete chat prompt proxy.  This catches BPE changes at
date headings, bullet separators, numbered-excerpt framing, and the question.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .source_compat import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_qa_prompt,
)
from memory_condense.eval.mem0_adapter import (
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_SPACY_MODEL,
    MEM0AI_PIN,
)

from .protocol import Mem0ComparisonProtocolError


MEM0_PROMPT_PACK_PROTOCOL = "memory-condense-mem0-prompt-pack-v2"
MEM0_RETRIEVAL_ROW_FORMAT = "memory-condense-mem0-retrieval-row-v2"
MEM0_MAX_PROMPT_TOKEN_PROXY = 8_000
MEM0_RUNTIME_PROTOCOL = "mem0-oss-2.0.18-certified-local-v1"
MEM0_PROMPT_CAP_SEMANTICS = (
    "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
)
MEM0_SOURCE_RESPONDER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
MEM0_SOURCE_JUDGE_MODEL = "openai/codex_sdk/gpt-5.6-sol"
# ``recent_window`` is a shared EvalConfig default, but LongMemEval is a
# completed-haystack QA protocol rather than live turn-by-turn replay.  The
# treatment therefore passes ``recent_turns=0`` when assembling its context.
# Keep both values explicit so a report cannot mistake the configured replay
# default for content that actually entered the provider request.
MEM0_CONFIGURED_RECENT_WINDOW = 4
MEM0_EFFECTIVE_RECENT_WINDOW = 0
MEM0_RECENT_WINDOW_SEMANTICS = (
    "longmemeval_completed_haystack_has_no_live_recent_tail_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Mem0PromptProtocolError(Mem0ComparisonProtocolError):
    """A search artifact cannot enter the locked responder prompt."""


@dataclass(frozen=True, slots=True)
class PromptMemory:
    """Sanitized memory fields that are sufficient for prompt reconstruction."""

    rank: int
    memory_id: str
    text: str
    score: float | None
    created_at: str
    attribution_kind: str

    def as_dict(self) -> dict[str, Any]:
        """Return the exact JSON-safe representation used by pool hashes."""

        return {
            "rank": self.rank,
            "memory_id": self.memory_id,
            "text": self.text,
            "score": self.score,
            "created_at": self.created_at,
            "attribution_kind": self.attribution_kind,
        }


@dataclass(frozen=True, slots=True)
class PromptPackDiagnostic:
    """One retrieval-rank admission decision under the complete prompt cap."""

    rank: int
    memory_id: str
    selected: bool
    reason: str
    rendered: str
    rendered_tokens: int
    proposed_prompt_token_proxy: int
    prompt_token_proxy_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "memory_id": self.memory_id,
            "selected": self.selected,
            "reason": self.reason,
            "rendered": self.rendered,
            "rendered_tokens": self.rendered_tokens,
            "proposed_prompt_token_proxy": self.proposed_prompt_token_proxy,
            "prompt_token_proxy_after": self.prompt_token_proxy_after,
        }


@dataclass(frozen=True, slots=True)
class PackedMem0Prompt:
    """Exact provider input and content-addressed retrieval audit fields."""

    protocol: str
    query: str
    context: str
    context_tokens: int
    messages: tuple[dict[str, str], dict[str, str]]
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    residual_prompt_token_proxy: int
    responder_output_token_reserve: int
    request_token_proxy: int
    raw_pool: tuple[PromptMemory, ...]
    packed_pool: tuple[PromptMemory, ...]
    diagnostics: tuple[PromptPackDiagnostic, ...]
    raw_memory_tokens: int
    packed_memory_tokens: int
    raw_pool_sha256: str
    packed_pool_sha256: str
    context_sha256: str
    messages_sha256: str
    rendering_mode: str
    official_longmemeval_protocol: bool
    official_search_protocol: bool
    independently_verified: bool
    adapter_comparison_certified: bool
    prompt_token_proxy_identity: Mapping[str, str | int]
    source_evaluation_identity: Mapping[str, Any]
    source_evaluation_identity_sha256: str
    configured_recent_window: int
    effective_recent_window: int
    recent_window_semantics: str
    attribution_kind: str
    supports_exact_source_provenance: bool

    @property
    def raw_memory_count(self) -> int:
        return len(self.raw_pool)

    @property
    def packed_memory_count(self) -> int:
        return len(self.packed_pool)

    @property
    def packed(self) -> tuple[PromptMemory, ...]:
        """Compatibility alias used by the two-stage shard orchestrator."""

        return self.packed_pool

    @property
    def max_prompt_tokens(self) -> int:
        """Compatibility alias; this is a local input-token proxy cap."""

        return self.max_prompt_token_proxy

    @property
    def residual_prompt_tokens(self) -> int:
        return self.residual_prompt_token_proxy

    def provider_messages(self) -> list[dict[str, str]]:
        """Return the exact mutable two-message input expected by responders."""

        return [dict(message) for message in self.messages]

    def to_retrieval_row(
        self,
        *,
        question_id: str,
        search_latency_s: float,
    ) -> dict[str, Any]:
        """Build a self-hashed, JSON-safe row for the campaign artifact."""

        if not isinstance(question_id, str) or not question_id.strip():
            raise Mem0PromptProtocolError("question_id must be a non-empty string")
        if (
            isinstance(search_latency_s, bool)
            or not isinstance(search_latency_s, (int, float))
            or not math.isfinite(float(search_latency_s))
            or float(search_latency_s) < 0.0
        ):
            raise Mem0PromptProtocolError(
                "search_latency_s must be a finite non-negative number"
            )

        raw_pool = [candidate.as_dict() for candidate in self.raw_pool]
        packed_pool = [candidate.as_dict() for candidate in self.packed_pool]
        row: dict[str, Any] = {
            "format": MEM0_RETRIEVAL_ROW_FORMAT,
            "prompt_pack_protocol": self.protocol,
            "question_id": question_id,
            "query": self.query,
            "context": self.context,
            "context_sha256": self.context_sha256,
            "context_tokens": self.context_tokens,
            "messages": self.provider_messages(),
            "messages_sha256": self.messages_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
            "max_prompt_token_proxy": self.max_prompt_token_proxy,
            "residual_prompt_token_proxy": self.residual_prompt_token_proxy,
            "responder_output_token_reserve": self.responder_output_token_reserve,
            "request_token_proxy": self.request_token_proxy,
            "raw_memory_count": self.raw_memory_count,
            "raw_memory_tokens": self.raw_memory_tokens,
            "raw_pool": raw_pool,
            "raw_pool_sha256": self.raw_pool_sha256,
            "packed_memory_count": self.packed_memory_count,
            "packed_memory_tokens": self.packed_memory_tokens,
            "packed_pool": packed_pool,
            "packed_pool_sha256": self.packed_pool_sha256,
            "search_latency_s": float(search_latency_s),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "rendering_mode": self.rendering_mode,
            "official_longmemeval_protocol": self.official_longmemeval_protocol,
            "official_search_protocol": self.official_search_protocol,
            "independently_verified": self.independently_verified,
            "adapter_comparison_certified": self.adapter_comparison_certified,
            "prompt_token_proxy_identity": dict(
                self.prompt_token_proxy_identity
            ),
            "source_evaluation_identity": _json_copy(
                self.source_evaluation_identity
            ),
            "source_evaluation_identity_sha256": (
                self.source_evaluation_identity_sha256
            ),
            "configured_recent_window": self.configured_recent_window,
            "effective_recent_window": self.effective_recent_window,
            "recent_window_semantics": self.recent_window_semantics,
            "provenance": {
                "kind": self.attribution_kind,
                "supports_exact_source_provenance": (
                    self.supports_exact_source_provenance
                ),
            },
        }
        row["retrieval_row_sha256"] = _canonical_sha256(row)
        return row


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_copy(value: Any) -> Any:
    """Copy a previously validated JSON value without coercing any field."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required(value: Any, name: str, *, label: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise Mem0PromptProtocolError(f"{label} is missing {name!r}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise Mem0PromptProtocolError(
            f"{label} is missing {name!r}"
        ) from exc


def _require_true(value: Any, *, label: str) -> None:
    if value is not True:
        raise Mem0PromptProtocolError(f"{label} must be true")


def _require_false(value: Any, *, label: str) -> None:
    if value is not False:
        raise Mem0PromptProtocolError(f"{label} must be false")


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Mem0PromptProtocolError(f"{label} must be a lowercase SHA-256")


def _validate_runtime_identity(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise Mem0PromptProtocolError("runtime_identity must be a mapping")
    _require_true(value.get("certified"), label="runtime_identity.certified")
    _require_true(
        value.get("local_owned_state"),
        label="runtime_identity.local_owned_state",
    )
    _require_true(value.get("on_disk"), label="runtime_identity.on_disk")
    if value.get("protocol") != MEM0_RUNTIME_PROTOCOL:
        raise Mem0PromptProtocolError(
            "runtime_identity.protocol is not the pinned owned Mem0 protocol"
        )
    _require_sha256(
        value.get("stable_config_sha256"),
        label="runtime_identity.stable_config_sha256",
    )
    _require_sha256(
        value.get("effective_config_sha256"),
        label="runtime_identity.effective_config_sha256",
    )

    stack = value.get("stack")
    if not isinstance(stack, Mapping):
        raise Mem0PromptProtocolError("runtime_identity.stack must be a mapping")
    versions = stack.get("dependency_versions")
    if not isinstance(versions, Mapping) or versions.get("mem0ai") != MEM0AI_PIN:
        raise Mem0PromptProtocolError(
            f"runtime identity must pin mem0ai=={MEM0AI_PIN}"
        )
    if stack.get("bm25_model") != MEM0_BM25_MODEL:
        raise Mem0PromptProtocolError("runtime identity has the wrong BM25 model")
    if stack.get("spacy_model") != MEM0_SPACY_MODEL:
        raise Mem0PromptProtocolError("runtime identity has the wrong spaCy model")
    _require_true(
        stack.get("bm25_operational"),
        label="runtime_identity.stack.bm25_operational",
    )
    _require_true(
        stack.get("entity_extraction_operational"),
        label="runtime_identity.stack.entity_extraction_operational",
    )


def validate_source_evaluation_identity(
    value: Any,
) -> dict[str, Any]:
    """Validate and copy the exact frozen source-evaluation contract.

    The prompt cap and output reserve are not implicit defaults.  Every pack
    is bound to the source validation policy that selected the responder,
    judge, retry behavior, proxy vocabulary, and 100-question claim shape.
    """

    if not isinstance(value, Mapping):
        raise Mem0PromptProtocolError(
            "source_evaluation_identity must be a mapping"
        )
    expected_keys = {
        "responder_model",
        "judge_model",
        "use_judge",
        "provider_retries",
        "max_provider_calls_per_shard",
        "max_prompt_tokens",
        "prompt_cap_semantics",
        "prompt_token_proxy_identity",
        "responder_output_token_reserve",
        "recent_window",
        "accuracy_target",
        "min_target_questions",
        "stress_context_tokens",
        "stress_questions",
        "stress_question_offset",
        "max_samples",
        "sample_offsets",
    }
    if set(value) != expected_keys:
        missing = sorted(expected_keys - set(value))
        extra = sorted(set(value) - expected_keys)
        raise Mem0PromptProtocolError(
            "source_evaluation_identity fields mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    exact_values = {
        "responder_model": MEM0_SOURCE_RESPONDER_MODEL,
        "judge_model": MEM0_SOURCE_JUDGE_MODEL,
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": MEM0_MAX_PROMPT_TOKEN_PROXY,
        "prompt_cap_semantics": MEM0_PROMPT_CAP_SEMANTICS,
        "responder_output_token_reserve": (
            BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "recent_window": MEM0_CONFIGURED_RECENT_WINDOW,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }
    for field, expected in exact_values.items():
        actual = value.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise Mem0PromptProtocolError(
                f"source_evaluation_identity.{field} does not match the "
                "frozen validation policy"
            )
    expected_proxy = tokenizer_proxy_identity()
    raw_proxy = value.get("prompt_token_proxy_identity")
    if not isinstance(raw_proxy, Mapping) or dict(raw_proxy) != expected_proxy:
        raise Mem0PromptProtocolError(
            "source_evaluation_identity.prompt_token_proxy_identity does not "
            "match this frozen root environment"
        )
    return _json_copy(value)


def _official_date_label(value: str) -> str:
    """Match the adapter's fail-closed UTC date rendering independently."""

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%A, %B %d, %Y")
    except ValueError:
        prefix = candidate[:10]
        try:
            parsed = datetime.strptime(prefix, "%Y-%m-%d")
        except ValueError as exc:
            raise Mem0PromptProtocolError(
                f"candidate created_at is invalid: {value!r}"
            ) from exc
        return parsed.strftime("%A, %B %d, %Y")


def _normalize_candidate(value: Any, expected_rank: int) -> PromptMemory:
    label = f"raw_pool[{expected_rank - 1}]"
    rank = _required(value, "rank", label=label)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
        raise Mem0PromptProtocolError(
            f"{label}.rank must be the retrieval position {expected_rank}"
        )
    memory_id = _required(value, "memory_id", label=label)
    if (
        not isinstance(memory_id, str)
        or not memory_id
        or memory_id != memory_id.strip()
    ):
        raise Mem0PromptProtocolError(
            f"{label}.memory_id must be a normalized non-empty string"
        )
    text = _required(value, "text", label=label)
    if not isinstance(text, str):
        raise Mem0PromptProtocolError(f"{label}.text must be a string")
    created_at = _required(value, "created_at", label=label)
    if (
        not isinstance(created_at, str)
        or not created_at
        or created_at != created_at.strip()
    ):
        raise Mem0PromptProtocolError(
            f"{label}.created_at must be a normalized non-empty string"
        )
    _official_date_label(created_at)

    raw_score = _required(value, "score", label=label)
    if raw_score is None:
        score = None
    elif (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        raise Mem0PromptProtocolError(f"{label}.score must be finite or null")
    else:
        score = float(raw_score)

    attribution = _required(value, "attribution_kind", label=label)
    if attribution != MEM0_ATTRIBUTION_KIND:
        raise Mem0PromptProtocolError(
            f"{label}.attribution_kind is not request-window attribution"
        )
    return PromptMemory(
        rank=rank,
        memory_id=memory_id,
        text=text,
        score=score,
        created_at=created_at,
        attribution_kind=attribution,
    )


def render_official_created_at_context(
    candidates: Sequence[PromptMemory],
) -> str:
    """Render memory text under chronological UTC-date headings.

    Admission order is deliberately not render order.  The official Mem0
    answerer takes a retrieval-rank slice and then sorts that slice by the
    returned ``created_at`` string before rendering its human-readable dates.
    Stable rank is the deterministic tie-breaker for reconstructed artifacts.
    """

    ordered = sorted(candidates, key=lambda item: (item.created_at, item.rank))
    lines: list[str] = []
    current_date: str | None = None
    for candidate in ordered:
        date_label = _official_date_label(candidate.created_at)
        if date_label != current_date:
            if lines:
                # The official formatter prefixes each new date heading with
                # a newline, which becomes one blank line after ``join`` and
                # its final ``strip``.  Spell it structurally here.
                lines.append("")
            lines.append(f"--- {date_label} ---")
            current_date = date_label
        lines.append(f"- {candidate.text}")
    return "\n".join(lines)


def _qa_messages(
    question: str,
    candidates: Sequence[PromptMemory],
) -> tuple[str, tuple[dict[str, str], dict[str, str]], int]:
    context = render_official_created_at_context(candidates)
    built = build_qa_prompt(question, [context] if context else [])
    if len(built) != 2 or [message.get("role") for message in built] != [
        "system",
        "user",
    ]:
        raise Mem0PromptProtocolError(
            "build_qa_prompt no longer produces the locked two-message request"
        )
    messages = (dict(built[0]), dict(built[1]))
    return context, messages, count_chat_prompt_token_proxy(messages)


def pack_mem0_prompt(
    question: str,
    search_result: Any,
    *,
    evaluation_identity: Mapping[str, Any],
    max_prompt_tokens: int | None = None,
) -> PackedMem0Prompt:
    """Independently reconstruct one budgeted Mem0 responder request.

    Required search-result fields are ``query``, ``raw_pool``,
    ``official_longmemeval_protocol``, ``official_search_protocol``,
    ``rendering_mode``, ``certified_rendering``, ``comparison_certified``,
    ``runtime_identity``, ``attribution_kind``, and
    ``supports_exact_source_provenance``.  Required candidate fields are
    ``rank``, ``memory_id``, ``text``, ``score``, ``created_at``, and
    ``attribution_kind``.  Mappings and immutable attribute objects are both
    accepted so a sanitized Stage-A artifact can be reconstructed exactly.
    ``evaluation_identity`` is the exact mapping emitted by the verified
    source-validation preflight; it is mandatory and content-addressed.
    """

    source_identity = validate_source_evaluation_identity(evaluation_identity)
    policy_prompt_cap = int(source_identity["max_prompt_tokens"])
    if max_prompt_tokens is None:
        max_prompt_tokens = policy_prompt_cap
    elif max_prompt_tokens != policy_prompt_cap:
        raise Mem0PromptProtocolError(
            "max_prompt_tokens disagrees with source_evaluation_identity"
        )
    if not isinstance(question, str) or not question.strip():
        raise Mem0PromptProtocolError("question must be a non-empty string")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
        or max_prompt_tokens > MEM0_MAX_PROMPT_TOKEN_PROXY
    ):
        raise Mem0PromptProtocolError(
            f"max_prompt_tokens must be within 1..{MEM0_MAX_PROMPT_TOKEN_PROXY}"
        )
    if _required(search_result, "query", label="search_result") != question:
        raise Mem0PromptProtocolError(
            "search_result.query must exactly match the responder question"
        )
    _require_true(
        _required(
            search_result,
            "official_longmemeval_protocol",
            label="search_result",
        ),
        label="search_result.official_longmemeval_protocol",
    )
    _require_true(
        _required(search_result, "official_search_protocol", label="search_result"),
        label="search_result.official_search_protocol",
    )
    if (
        _required(search_result, "rendering_mode", label="search_result")
        != MEM0_CERTIFIED_RENDERING
    ):
        raise Mem0PromptProtocolError(
            "search_result.rendering_mode is not official memory/date rendering"
        )
    _require_true(
        _required(search_result, "certified_rendering", label="search_result"),
        label="search_result.certified_rendering",
    )
    # This bit is an input assertion, not our proof.  Every other gate below
    # is checked even when the adapter asserted comparison certification.
    _require_true(
        _required(search_result, "comparison_certified", label="search_result"),
        label="search_result.comparison_certified",
    )
    if (
        _required(search_result, "attribution_kind", label="search_result")
        != MEM0_ATTRIBUTION_KIND
    ):
        raise Mem0PromptProtocolError(
            "search_result.attribution_kind is not the frozen Mem0 kind"
        )
    _require_false(
        _required(
            search_result,
            "supports_exact_source_provenance",
            label="search_result",
        ),
        label="search_result.supports_exact_source_provenance",
    )
    _validate_runtime_identity(
        _required(search_result, "runtime_identity", label="search_result")
    )

    raw_value = _required(search_result, "raw_pool", label="search_result")
    if not isinstance(raw_value, Sequence) or isinstance(
        raw_value, (str, bytes, bytearray)
    ):
        raise Mem0PromptProtocolError("search_result.raw_pool must be a sequence")
    raw_pool = tuple(
        _normalize_candidate(candidate, rank)
        for rank, candidate in enumerate(raw_value, start=1)
    )
    memory_ids = [candidate.memory_id for candidate in raw_pool]
    if len(memory_ids) != len(set(memory_ids)):
        raise Mem0PromptProtocolError("search_result.raw_pool repeats a memory_id")

    empty_context, empty_messages, empty_proxy = _qa_messages(question, ())
    if empty_context:
        raise Mem0PromptProtocolError("empty Mem0 context unexpectedly rendered text")
    if empty_proxy > max_prompt_tokens:
        raise Mem0PromptProtocolError(
            "the exact QA prompt without memories exceeds max_prompt_tokens"
        )

    packed: list[PromptMemory] = []
    diagnostics: list[PromptPackDiagnostic] = []
    prompt_proxy_after = empty_proxy
    for candidate in raw_pool:
        singleton = render_official_created_at_context((candidate,))
        if not candidate.text:
            diagnostics.append(
                PromptPackDiagnostic(
                    rank=candidate.rank,
                    memory_id=candidate.memory_id,
                    selected=False,
                    reason="empty_memory",
                    rendered=singleton,
                    rendered_tokens=count_tokens(singleton),
                    proposed_prompt_token_proxy=prompt_proxy_after,
                    prompt_token_proxy_after=prompt_proxy_after,
                )
            )
            continue

        _proposed_context, _proposed_messages, proposed_proxy = _qa_messages(
            question, (*packed, candidate)
        )
        selected = proposed_proxy <= max_prompt_tokens
        if selected:
            packed.append(candidate)
            prompt_proxy_after = proposed_proxy
        diagnostics.append(
            PromptPackDiagnostic(
                rank=candidate.rank,
                memory_id=candidate.memory_id,
                selected=selected,
                reason="selected" if selected else "prompt_token_budget",
                rendered=singleton,
                rendered_tokens=count_tokens(singleton),
                proposed_prompt_token_proxy=proposed_proxy,
                prompt_token_proxy_after=prompt_proxy_after,
            )
        )

    context, messages, prompt_proxy = _qa_messages(question, packed)
    if prompt_proxy != prompt_proxy_after or prompt_proxy > max_prompt_tokens:
        raise Mem0PromptProtocolError(
            "final deterministic QA prompt recount disagrees with admission"
        )
    # Make the parity assertion explicit at the final provider boundary.
    expected_messages = build_qa_prompt(question, [context] if context else [])
    if list(messages) != expected_messages:
        raise Mem0PromptProtocolError(
            "provider messages differ from the frozen build_qa_prompt output"
        )
    if prompt_proxy != count_chat_prompt_token_proxy(expected_messages):
        raise Mem0PromptProtocolError(
            "prompt-token proxy differs from the frozen chat counter"
        )

    raw_payload = [candidate.as_dict() for candidate in raw_pool]
    packed_payload = [candidate.as_dict() for candidate in packed]
    identity = dict(tokenizer_proxy_identity())
    return PackedMem0Prompt(
        protocol=MEM0_PROMPT_PACK_PROTOCOL,
        query=question,
        context=context,
        context_tokens=count_tokens(context),
        messages=messages,
        prompt_token_proxy=prompt_proxy,
        max_prompt_token_proxy=max_prompt_tokens,
        residual_prompt_token_proxy=max_prompt_tokens - prompt_proxy,
        responder_output_token_reserve=int(
            source_identity["responder_output_token_reserve"]
        ),
        request_token_proxy=(
            prompt_proxy
            + int(source_identity["responder_output_token_reserve"])
        ),
        raw_pool=raw_pool,
        packed_pool=tuple(packed),
        diagnostics=tuple(diagnostics),
        raw_memory_tokens=sum(count_tokens(candidate.text) for candidate in raw_pool),
        packed_memory_tokens=sum(
            count_tokens(candidate.text) for candidate in packed
        ),
        raw_pool_sha256=_canonical_sha256(raw_payload),
        packed_pool_sha256=_canonical_sha256(packed_payload),
        context_sha256=_text_sha256(context),
        messages_sha256=_canonical_sha256(expected_messages),
        rendering_mode=MEM0_CERTIFIED_RENDERING,
        official_longmemeval_protocol=True,
        official_search_protocol=True,
        independently_verified=True,
        adapter_comparison_certified=True,
        prompt_token_proxy_identity=identity,
        source_evaluation_identity=source_identity,
        source_evaluation_identity_sha256=_canonical_sha256(source_identity),
        configured_recent_window=int(source_identity["recent_window"]),
        effective_recent_window=MEM0_EFFECTIVE_RECENT_WINDOW,
        recent_window_semantics=MEM0_RECENT_WINDOW_SEMANTICS,
        attribution_kind=MEM0_ATTRIBUTION_KIND,
        supports_exact_source_provenance=False,
    )


def verify_provider_input_tokens(
    packed: PackedMem0Prompt,
    provider_input_tokens: int,
) -> bool | None:
    """Fail closed when authoritative provider input usage exceeds the cap.

    Providers sometimes report zero for completed requests, so zero is
    recorded as unavailable (``None``), never as proof of compliance.  The
    output-token reserve is intentionally irrelevant to this input-only check.
    """

    if (
        isinstance(provider_input_tokens, bool)
        or not isinstance(provider_input_tokens, int)
        or provider_input_tokens < 0
    ):
        raise Mem0PromptProtocolError(
            "provider_input_tokens must be a non-negative integer"
        )
    if provider_input_tokens == 0:
        return None
    if provider_input_tokens > packed.max_prompt_token_proxy:
        raise Mem0PromptProtocolError(
            "provider input usage exceeds the locked prompt cap: "
            f"{provider_input_tokens} > {packed.max_prompt_token_proxy}"
        )
    return True


__all__ = [
    "MEM0_MAX_PROMPT_TOKEN_PROXY",
    "MEM0_CONFIGURED_RECENT_WINDOW",
    "MEM0_EFFECTIVE_RECENT_WINDOW",
    "MEM0_PROMPT_CAP_SEMANTICS",
    "MEM0_PROMPT_PACK_PROTOCOL",
    "MEM0_RECENT_WINDOW_SEMANTICS",
    "MEM0_RETRIEVAL_ROW_FORMAT",
    "MEM0_SOURCE_JUDGE_MODEL",
    "MEM0_SOURCE_RESPONDER_MODEL",
    "Mem0PromptProtocolError",
    "PackedMem0Prompt",
    "PromptMemory",
    "PromptPackDiagnostic",
    "pack_mem0_prompt",
    "render_official_created_at_context",
    "validate_source_evaluation_identity",
    "verify_provider_input_tokens",
]
