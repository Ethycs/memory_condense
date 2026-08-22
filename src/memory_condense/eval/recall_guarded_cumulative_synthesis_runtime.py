"""One-load local Qwen runtime for cumulative evidence synthesis.

The runtime deliberately keeps generation and forced-choice evidence scoring
behind one synchronous owner.  The pinned Qwen3-0.6B checkpoint is verified
before it is loaded, then the answerer-owned model and tokenizer are shared
with :class:`CausalChoiceScorer`; no second checkpoint copy is constructed.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.local_qwen import LocalQwenAnswerer
from memory_condense.search.selectors.causal_choice_scorer import (
    QWEN_CHOICE_CHECKPOINT_SHA256,
    QWEN_CHOICE_MODEL_ID,
    QWEN_CHOICE_MODEL_REVISION,
    CausalChoiceEvidence,
    CausalChoiceScoreReport,
    CausalChoiceScorer,
    verify_local_causal_checkpoint,
)


RUNTIME_FORMAT = "memory-condense-recall-guarded-synthesis-runtime-v1"


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_ids(encoded: Any) -> list[int]:
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in (values or ())]


@dataclass(frozen=True, slots=True)
class SynthesisRuntimeIdentity:
    """Content and execution identity for the shared local runtime."""

    format: str
    model_id: str
    model_revision: str
    checkpoint_sha256: str
    runtime: str
    device: str
    dtype: str
    max_position_embeddings: int
    default_max_new_tokens: int
    generation_do_sample: bool = False
    generation_thinking: bool = False
    generation_kv_cache: bool = True
    scoring_kv_cache: bool = False

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SynthesisCompletionReport:
    """Text-free provenance and bounded usage for one completion."""

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    messages_sha256: str
    completion_sha256: str
    input_tokens: int
    output_tokens: int
    max_new_tokens: int
    elapsed_s: float

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SynthesisRuntimeUsage:
    """Cumulative generation and classifier usage for one runtime."""

    completion_calls: int
    completion_input_tokens: int
    completion_output_tokens: int
    completion_elapsed_s: float
    score_calls: int
    score_forward_passes: int
    score_elapsed_s: float

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


class RecallGuardedCumulativeSynthesisRuntime:
    """Verified, deterministic synthesis and density-scoring adapter.

    The adapter is intentionally synchronous.  Generation mutates the
    answerer's per-call ``max_new_tokens`` control, and both operations share
    one model, so a non-reentrant lock prevents overlapping use.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_new_tokens: int = 2048,
        gpu_memory: str = "6GiB",
        cpu_memory: str = "24GiB",
        score_max_candidates: int = 128,
        score_batch_size: int = 8,
        score_query_tokens: int = 192,
        score_candidate_tokens: int = 256,
        score_max_prompt_tokens: int = 768,
        score_max_workspace_tokens: int = 8192,
        score_strict: bool = True,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        root = Path(model_dir)
        checkpoint_sha256 = verify_local_causal_checkpoint(
            root,
            model_id=QWEN_CHOICE_MODEL_ID,
            model_revision=QWEN_CHOICE_MODEL_REVISION,
            expected_checkpoint_sha256=QWEN_CHOICE_CHECKPOINT_SHA256,
        )
        answerer: LocalQwenAnswerer | None = None
        try:
            answerer = LocalQwenAnswerer(
                root,
                max_new_tokens=max_new_tokens,
                gpu_memory=gpu_memory,
                cpu_memory=cpu_memory,
                dtype="float16",
            )
            model = answerer.model
            tokenizer = answerer.tokenizer
            device = str(model.get_input_embeddings().weight.device)
            if not device.casefold().startswith("cuda"):
                raise RuntimeError(
                    "pinned synthesis checkpoint must be resident on CUDA"
                )
            if answerer.dtype_name != "float16":
                raise RuntimeError("pinned synthesis runtime must use float16")
            max_positions = int(
                getattr(model.config, "max_position_embeddings", 0)
            )
            if max_positions < 1:
                raise ValueError(
                    "local synthesis checkpoint has no positive context limit"
                )
            if max_new_tokens > max_positions:
                raise ValueError(
                    "max_new_tokens exceeds the local checkpoint context limit"
                )
            scorer = CausalChoiceScorer(
                model,
                tokenizer,
                torch_module=answerer._torch,
                model_id=QWEN_CHOICE_MODEL_ID,
                model_revision=QWEN_CHOICE_MODEL_REVISION,
                checkpoint_sha256=checkpoint_sha256,
                device=device,
                dtype=answerer.dtype_name,
                max_candidates=score_max_candidates,
                batch_size=score_batch_size,
                query_tokens=score_query_tokens,
                candidate_tokens=score_candidate_tokens,
                max_prompt_tokens=score_max_prompt_tokens,
                max_workspace_tokens=score_max_workspace_tokens,
                require_single_token_labels=True,
                strict=score_strict,
            )
        except BaseException:
            if answerer is not None:
                answerer.close()
            raise

        self._answerer = answerer
        self._scorer = scorer
        self._lock = threading.Lock()
        self._closed = False
        self._default_max_new_tokens = int(max_new_tokens)
        self._completion_calls = 0
        self._completion_input_tokens = 0
        self._completion_output_tokens = 0
        self._completion_elapsed_s = 0.0
        self.last_completion_report: SynthesisCompletionReport | None = None
        self.identity = SynthesisRuntimeIdentity(
            format=RUNTIME_FORMAT,
            model_id=QWEN_CHOICE_MODEL_ID,
            model_revision=QWEN_CHOICE_MODEL_REVISION,
            checkpoint_sha256=checkpoint_sha256,
            runtime=f"{type(model).__module__}.{type(model).__name__}",
            device=device,
            dtype=answerer.dtype_name,
            max_position_embeddings=max_positions,
            default_max_new_tokens=int(max_new_tokens),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("synthesis runtime is closed")

    @staticmethod
    def _validated_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("completion messages must not be empty")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if not role or not isinstance(content, str):
                raise ValueError(
                    f"completion message {index} requires role and string content"
                )
            normalized.append({"role": role, "content": content})
        return normalized

    def _prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        rendered = self._answerer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self._answerer.tokenizer(str(rendered))
        return len(_token_ids(encoded))

    def _output_tokens(self, completion: str) -> int:
        encoded = self._answerer.tokenizer(
            completion,
            add_special_tokens=False,
        )
        return len(_token_ids(encoded))

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Return one deterministic completion and retain its text-free report."""

        normalized = self._validated_messages(messages)
        requested = (
            self._default_max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if requested < 1:
            raise ValueError("max_new_tokens must be positive")

        with self._lock:
            self._require_open()
            input_tokens = self._prompt_tokens(normalized)
            if input_tokens + requested > self.identity.max_position_embeddings:
                raise ValueError(
                    "completion input plus output reserve exceeds the pinned "
                    f"context limit: {input_tokens} + {requested} > "
                    f"{self.identity.max_position_embeddings}"
                )
            previous_limit = self._answerer.max_new_tokens
            started = time.perf_counter()
            try:
                self._answerer.max_new_tokens = requested
                completion = str(self._answerer(normalized))
            finally:
                self._answerer.max_new_tokens = previous_limit
            elapsed = time.perf_counter() - started
            output_tokens = self._output_tokens(completion)
            self._completion_calls += 1
            self._completion_input_tokens += input_tokens
            self._completion_output_tokens += output_tokens
            self._completion_elapsed_s += elapsed
            self.last_completion_report = SynthesisCompletionReport(
                model_id=self.identity.model_id,
                model_revision=self.identity.model_revision,
                checkpoint_sha256=self.identity.checkpoint_sha256,
                messages_sha256=_sha256_json(normalized),
                completion_sha256=hashlib.sha256(
                    completion.encode("utf-8")
                ).hexdigest(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                max_new_tokens=requested,
                elapsed_s=elapsed,
            )
            return completion

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        return self.complete(messages, max_new_tokens=max_new_tokens)

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult] | Mapping[str, str],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> Mapping[str, CausalChoiceEvidence]:
        """Score candidates with the same verified model used for synthesis."""

        with self._lock:
            self._require_open()
            return self._scorer.score_candidates(
                query,
                candidates,
                source_timestamps=source_timestamps,
            )

    @property
    def last_score_report(self) -> CausalChoiceScoreReport | None:
        return self._scorer.last_report

    @property
    def usage(self) -> SynthesisRuntimeUsage:
        return SynthesisRuntimeUsage(
            completion_calls=self._completion_calls,
            completion_input_tokens=self._completion_input_tokens,
            completion_output_tokens=self._completion_output_tokens,
            completion_elapsed_s=self._completion_elapsed_s,
            score_calls=int(self._scorer.calls),
            score_forward_passes=int(self._scorer.forward_passes),
            score_elapsed_s=float(self._scorer.elapsed_s),
        )

    def close(self) -> None:
        """Release both owners of the single model and clear CUDA state."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._scorer.close()
            finally:
                self._answerer.close()

    def __enter__(self) -> "RecallGuardedCumulativeSynthesisRuntime":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "RUNTIME_FORMAT",
    "RecallGuardedCumulativeSynthesisRuntime",
    "SynthesisCompletionReport",
    "SynthesisRuntimeIdentity",
    "SynthesisRuntimeUsage",
]
