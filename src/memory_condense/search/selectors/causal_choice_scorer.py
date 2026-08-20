"""Generation-free answer-value evidence from a local causal checkpoint.

The scorer compares the full conditional log-likelihood of two fixed choice
sequences.  It never calls ``generate`` and explicitly disables the K/V cache.
Returned scores are transient read-time controls; neither logits nor model
state are written into a ``RetrievalResult`` or the durable memory store.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import truncate_to_tokens
from memory_condense.domain.integrity import file_sha256 as _file_sha256
from memory_condense.domain.ranking import (
    round_robin_unique,
    source_rows_with_fallback,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.coverage_models import ReportDumpMixin


QWEN_CHOICE_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN_CHOICE_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SMOLLM_CHOICE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOLLM_CHOICE_MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"

_CAUSAL_CHECKPOINT_MANIFEST_FORMAT = (
    "memory-condense-local-causal-checkpoint-v1"
)
_CAUSAL_METADATA_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)

QWEN_CHOICE_FILE_SHA256: dict[str, str] = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": (
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
    ),
    "tokenizer_config.json": (
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    ),
    "tokenizer.json": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": (
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    ),
}
SMOLLM_CHOICE_FILE_SHA256: dict[str, str] = {
    "config.json": "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee",
    "generation_config.json": (
        "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013"
    ),
    "tokenizer_config.json": (
        "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"
    ),
    "tokenizer.json": (
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
    ),
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": (
        "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"
    ),
    "model.safetensors": (
        "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
    ),
}


def _causal_checkpoint_manifest_sha256(
    file_hashes: Mapping[str, str],
    *,
    model_id: str,
    model_revision: str,
) -> str:
    """Content-address all files that can affect local causal scoring."""

    payload = {
        "format": _CAUSAL_CHECKPOINT_MANIFEST_FORMAT,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "files": {
            str(name): str(digest).casefold()
            for name, digest in sorted(file_hashes.items())
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


QWEN_CHOICE_CHECKPOINT_SHA256 = _causal_checkpoint_manifest_sha256(
    QWEN_CHOICE_FILE_SHA256,
    model_id=QWEN_CHOICE_MODEL_ID,
    model_revision=QWEN_CHOICE_MODEL_REVISION,
)
SMOLLM_CHOICE_CHECKPOINT_SHA256 = _causal_checkpoint_manifest_sha256(
    SMOLLM_CHOICE_FILE_SHA256,
    model_id=SMOLLM_CHOICE_MODEL_ID,
    model_revision=SMOLLM_CHOICE_MODEL_REVISION,
)

# Backward-compatible import names.  These values now bind the full behavioral
# checkpoint manifest, not only the safetensors payload.  The aliases can be
# removed once older evaluation scripts no longer import them.
QWEN_CHOICE_WEIGHTS_SHA256 = QWEN_CHOICE_CHECKPOINT_SHA256
SMOLLM_CHOICE_WEIGHTS_SHA256 = SMOLLM_CHOICE_CHECKPOINT_SHA256


_SYSTEM_PROMPT = (
    "You are a strict evidence classifier. Use only what the memory explicitly "
    "states; do not infer an unstated answer."
)
_USER_PROMPT = """Question:
{query}

Memory author conversation role: {candidate_role}
In the Memory section only, first-person pronouns (I, me, my, mine, we, us,
our, ours) refer to that memory author. For example, "I" in an
assistant-authored memory refers to the assistant, not the user.

Memory:
{candidate}

Choose exactly one label.
A = The memory directly proves a member of the answer set and explicitly identifies its answer value.
B = The memory does not; it is only related, indirect, generic, or null.
Label:"""


@dataclass(frozen=True, slots=True)
class CausalChoiceEvidence(ReportDumpMixin):
    """One candidate's normalized forced-choice evidence."""

    candidate_id: str
    role: str
    inspected: bool
    answerability: float
    value_evidence_logit: float
    direct_log_likelihood: float
    indirect_log_likelihood: float


@dataclass(frozen=True, slots=True)
class CausalChoiceScoreReport(ReportDumpMixin):
    """Text-free provenance and bounds for one scoring call."""

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    runtime: str
    device: str
    dtype: str
    input_candidates: int
    inspected_candidates: int
    output_candidates: int
    choice_sequence_tokens: tuple[int, int]
    workspace_tokens: int
    total_sequence_tokens: int
    forward_passes: int
    elapsed_s: float
    retained_transformer_state_bytes: int = 0
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class CausalChoiceCompanionReport(ReportDumpMixin):
    """Text-free diagnostics for query-conditioned source hydration."""

    input_sources: int
    input_candidates: int
    inspected_candidates: int
    selected_sources: int
    preferred_evidence_role: str | None
    selected_chunk_ids: Mapping[str, str]
    # The A/B choice is the shared, explicitly uncalibrated membership and
    # answerability signal.  Expose only the selected row's scalar so an
    # upstream source refresher can refuse to replace routed evidence with an
    # uninspected or B-labelled fallback.  Candidate text and model state are
    # never retained in this report.
    selected_membership_scores: Mapping[str, float]
    score_report: Mapping[str, Any] | None
    retained_transformer_state_bytes: int = 0
    fallback_reason: str = ""


def _weight_paths(root: Path) -> list[Path]:
    single = root / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        return []
    names = sorted({str(value) for value in weight_map.values()})
    return [root / name for name in names]


def _checkpoint_paths(root: Path) -> tuple[list[Path], list[str]]:
    """Resolve the complete local checkpoint manifest without network I/O."""

    resolved_root = root.resolve()
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    weights = _weight_paths(root)
    if not weights:
        missing.append("model.safetensors[.index.json]")
    else:
        missing.extend(path.name for path in weights if not path.is_file())
    if not (root / "tokenizer.json").is_file() and not (
        root / "vocab.json"
    ).is_file():
        missing.append("tokenizer.json/vocab.json")

    paths = [
        root / name
        for name in _CAUSAL_METADATA_NAMES
        if (root / name).is_file()
    ]
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        paths.append(index_path)
    paths.extend(weights)
    unique: dict[str, Path] = {}
    for path in paths:
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"checkpoint manifest path escapes model directory: {path}"
            ) from exc
        unique[relative] = path
    return [unique[name] for name in sorted(unique)], missing


def verify_local_causal_checkpoint(
    model_dir: str | Path,
    *,
    model_id: str = "local-causal-choice",
    model_revision: str = "",
    expected_checkpoint_sha256: str = "",
    expected_weights_sha256: str = "",
) -> str:
    """Return a digest of the exact model, tokenizer, and weight manifest.

    ``expected_weights_sha256`` is retained as a compatibility keyword, but
    it now denotes this full manifest digest.  This closes the prior gap where
    altered tokenizer/config files could share an apparently identical model
    identity merely because ``model.safetensors`` was unchanged.
    """

    root = Path(model_dir)
    paths, missing = _checkpoint_paths(root)
    if missing:
        raise FileNotFoundError(
            f"incomplete local causal checkpoint under {root}: "
            f"missing {', '.join(missing)}"
        )

    if expected_checkpoint_sha256 and expected_weights_sha256:
        if expected_checkpoint_sha256.casefold() != expected_weights_sha256.casefold():
            raise ValueError("conflicting expected causal checkpoint SHA-256 values")
    file_hashes = {
        path.resolve().relative_to(root.resolve()).as_posix(): _file_sha256(path)
        for path in paths
    }
    actual = _causal_checkpoint_manifest_sha256(
        file_hashes,
        model_id=model_id,
        model_revision=model_revision,
    )
    expected = str(
        expected_checkpoint_sha256 or expected_weights_sha256
    ).strip().casefold()
    if expected and not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"unexpected local causal checkpoint sha256: {actual}; "
            f"expected {expected}"
        )
    return actual


def _as_token_ids(encoded: Any) -> list[int]:
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in (values or ())]


class CausalChoiceScorer:
    """Score explicit answer-value evidence with one bounded causal forward."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        torch_module: Any | None = None,
        model_id: str = "local-causal-choice",
        model_revision: str = "",
        checkpoint_sha256: str = "",
        device: str = "cpu",
        dtype: str = "float32",
        max_candidates: int = 128,
        batch_size: int = 8,
        query_tokens: int = 192,
        candidate_tokens: int = 128,
        max_prompt_tokens: int = 512,
        max_workspace_tokens: int = 8192,
        direct_choice: str = " A",
        indirect_choice: str = " B",
        require_single_token_labels: bool = False,
        strict: bool = False,
    ) -> None:
        if min(
            max_candidates,
            batch_size,
            query_tokens,
            candidate_tokens,
            max_prompt_tokens,
            max_workspace_tokens,
        ) < 1:
            raise ValueError("causal choice bounds must be positive")
        if torch_module is None:
            import torch as torch_module

        self._torch = torch_module
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.device = str(device)
        self.dtype_name = str(dtype)
        self.max_candidates = int(max_candidates)
        self.requested_batch_size = int(batch_size)
        self.query_tokens = int(query_tokens)
        self.candidate_tokens = int(candidate_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_workspace_tokens = int(max_workspace_tokens)
        self.strict = bool(strict)
        self._choices = (str(direct_choice), str(indirect_choice))
        self._choice_ids = tuple(
            tuple(self._encode(choice, add_special_tokens=False))
            for choice in self._choices
        )
        if any(not ids for ids in self._choice_ids):
            raise ValueError("forced-choice labels must each encode to tokens")
        if self._choice_ids[0] == self._choice_ids[1]:
            raise ValueError("forced-choice labels must encode distinctly")
        if len(self._choice_ids[0]) != len(self._choice_ids[1]):
            raise ValueError(
                "forced-choice sequences must have equal token length so "
                "sequence likelihoods are comparable"
            )
        if require_single_token_labels and any(
            len(ids) != 1 for ids in self._choice_ids
        ):
            raise ValueError("forced-choice labels must each be one token")
        choice_width = len(self._choice_ids[0])
        per_candidate_workspace = (
            self.max_prompt_tokens
            if choice_width == 1
            else 2 * (self.max_prompt_tokens + choice_width)
        )
        if per_candidate_workspace > self.max_workspace_tokens:
            raise ValueError(
                "causal choice workspace cannot hold one candidate prompt"
            )
        self.batch_size = min(
            self.requested_batch_size,
            max(1, self.max_workspace_tokens // per_candidate_workspace),
        )
        eval_model = getattr(self.model, "eval", None)
        if callable(eval_model):
            eval_model()
        self.calls = 0
        self.forward_passes = 0
        self.elapsed_s = 0.0
        self.last_report: CausalChoiceScoreReport | None = None
        self.last_source_companion_report: CausalChoiceCompanionReport | None = None

    @classmethod
    def from_local_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        model_id: str,
        model_revision: str = "",
        expected_weights_sha256: str = "",
        device: str = "cuda",
        dtype: str = "auto",
        **kwargs: Any,
    ) -> "CausalChoiceScorer":
        """Load an exact local checkpoint without network access."""

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from memory_condense.modeling.qwen_dtype import resolve_local_qwen_dtype

        root = Path(model_dir)
        checkpoint_sha256 = verify_local_causal_checkpoint(
            root,
            model_id=model_id,
            model_revision=model_revision,
            expected_weights_sha256=expected_weights_sha256,
        )
        target = str(device)
        if target.casefold().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested causal-choice device is unavailable: {target}")
        torch_dtype, dtype_name = resolve_local_qwen_dtype(
            torch,
            dtype,
            device=target,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch_dtype,
            device_map=target,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        return cls(
            model,
            tokenizer,
            torch_module=torch,
            model_id=model_id,
            model_revision=model_revision,
            checkpoint_sha256=checkpoint_sha256,
            device=target,
            dtype=dtype_name,
            **kwargs,
        )

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
        )
        return _as_token_ids(encoded)

    def _render_prompt(
        self,
        query: str,
        candidate: str,
        candidate_role: str = "",
    ) -> str:
        # Render the role as a quoted scalar so malformed external role text
        # cannot create a new prompt section.  Empty roles (the legacy mapping
        # input) remain explicit rather than being silently treated as user.
        rendered_role = json.dumps(
            str(candidate_role).strip() or "unknown",
            ensure_ascii=False,
        )
        user = _USER_PROMPT.format(
            query=truncate_to_tokens(query, self.query_tokens),
            candidate_role=rendered_role,
            candidate=truncate_to_tokens(candidate, self.candidate_tokens),
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return f"{_SYSTEM_PROMPT}\n\n{user}"
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            parameters = inspect.signature(apply_template).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        if any(
            parameter.name == "enable_thinking"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            kwargs["enable_thinking"] = False
        return str(apply_template(messages, **kwargs))

    def _prompt_ids(
        self,
        query: str,
        candidate: str,
        candidate_role: str = "",
    ) -> list[int]:
        prompt = self._render_prompt(query, candidate, candidate_role)
        ids = self._encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("forced-choice prompt encoded to no tokens")
        if len(ids) <= self.max_prompt_tokens:
            return ids
        # Preserve both the classifier instruction and the final label cue.
        prefix = self.max_prompt_tokens // 2
        return [*ids[:prefix], *ids[-(self.max_prompt_tokens - prefix) :]]

    @staticmethod
    def _candidate_rows(
        candidates: Sequence[RetrievalResult] | Mapping[str, str],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> list[tuple[str, str, str]]:
        if isinstance(candidates, Mapping):
            return [
                (str(candidate_id), str(text), "")
                for candidate_id, text in candidates.items()
            ]
        rows: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        timestamps = source_timestamps or {}
        for result in candidates:
            candidate_id = result.chunk.chunk_id
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            role = result.turn.role if result.turn is not None else ""
            source_id = result.durable_source_id
            timestamp = str(timestamps.get(source_id, "")).strip()
            text = result.chunk.text
            if timestamp:
                text = f"Source timestamp: {timestamp}\n{text}"
            rows.append((candidate_id, text, str(role)))
        return rows

    def _score_bounded(
        self,
        query: str,
        rows: Sequence[tuple[str, str, str]],
    ) -> tuple[dict[str, CausalChoiceEvidence], int, int, int]:
        torch = self._torch
        model = self.model
        if model is None:
            raise RuntimeError("causal choice scorer is closed")
        input_device = model.get_input_embeddings().weight.device
        evidence: dict[str, CausalChoiceEvidence] = {}
        forward_passes = 0
        peak_workspace_tokens = 0
        total_sequence_tokens = 0
        single_token_choices = len(self._choice_ids[0]) == 1
        for start in range(0, len(rows), self.batch_size):
            microbatch = rows[start : start + self.batch_size]
            sequences: list[list[int]] = []
            prompt_lengths: list[int] = []
            choice_indices: list[int] = []
            for _candidate_id, text, role in microbatch:
                prompt_ids = self._prompt_ids(query, text, role)
                if single_token_choices:
                    # Both labels are predicted from the same final prompt
                    # state. Do not duplicate the prompt or append either
                    # label merely to read that state.
                    sequences.append(prompt_ids)
                    prompt_lengths.append(len(prompt_ids))
                else:
                    for choice_index, choice_ids in enumerate(self._choice_ids):
                        sequences.append([*prompt_ids, *choice_ids])
                        prompt_lengths.append(len(prompt_ids))
                        choice_indices.append(choice_index)

            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
            width = max(len(sequence) for sequence in sequences)
            workspace_tokens = len(sequences) * width
            if workspace_tokens > self.max_workspace_tokens:
                raise RuntimeError(
                    "causal choice microbatch exceeded its workspace bound"
                )
            peak_workspace_tokens = max(
                peak_workspace_tokens,
                workspace_tokens,
            )
            total_sequence_tokens += sum(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(sequences), width),
                int(pad_token_id or 0),
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                input_ids[row_index, : len(sequence)] = torch.tensor(
                    sequence,
                    dtype=torch.long,
                )
                attention_mask[row_index, : len(sequence)] = 1

            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                if single_token_choices:
                    row_indices = torch.arange(
                        len(sequences),
                        device=input_device,
                    )
                    positions = torch.tensor(
                        [length - 1 for length in prompt_lengths],
                        dtype=torch.long,
                        device=input_device,
                    )
                    # Cast only one vocabulary row per candidate, not the
                    # complete [batch, sequence, vocabulary] tensor.
                    final_logits = outputs.logits[
                        row_indices,
                        positions,
                        :,
                    ].float()
                    normalizers = torch.logsumexp(final_logits, dim=-1)
                    direct = (
                        final_logits[:, self._choice_ids[0][0]] - normalizers
                    )
                    indirect = (
                        final_logits[:, self._choice_ids[1][0]] - normalizers
                    )
                    likelihood_tensor = torch.stack(
                        (direct, indirect),
                        dim=1,
                    ).reshape(-1)
                else:
                    likelihood_tensors = []
                    for row_index, (prompt_length, choice_index) in enumerate(
                        zip(prompt_lengths, choice_indices, strict=True)
                    ):
                        choice_ids = self._choice_ids[choice_index]
                        positions = torch.arange(
                            prompt_length - 1,
                            prompt_length + len(choice_ids) - 1,
                            device=input_device,
                        )
                        targets = torch.tensor(
                            choice_ids,
                            dtype=torch.long,
                            device=input_device,
                        )
                        # Full-sequence choices still gather only the label
                        # positions before the FP32 softmax.
                        choice_logits = outputs.logits[
                            row_index,
                            positions,
                            :,
                        ].float()
                        likelihood_tensors.append(
                            torch.log_softmax(choice_logits, dim=-1)[
                                torch.arange(
                                    len(choice_ids),
                                    device=input_device,
                                ),
                                targets,
                            ].sum()
                        )
                    likelihood_tensor = torch.stack(likelihood_tensors)
                # Exactly one device-to-host synchronization per microbatch.
                likelihoods = likelihood_tensor.detach().cpu().tolist()
            forward_passes += 1

            for candidate_index, (candidate_id, _text, role) in enumerate(
                microbatch
            ):
                direct = float(likelihoods[2 * candidate_index])
                indirect = float(likelihoods[2 * candidate_index + 1])
                logit = direct - indirect
                probability = (
                    1.0 / (1.0 + math.exp(-logit))
                    if logit >= 0.0
                    else math.exp(logit) / (1.0 + math.exp(logit))
                )
                evidence[candidate_id] = CausalChoiceEvidence(
                    candidate_id=candidate_id,
                    role=role,
                    inspected=True,
                    answerability=float(probability),
                    value_evidence_logit=float(logit),
                    direct_log_likelihood=direct,
                    indirect_log_likelihood=indirect,
                )
            del outputs, likelihood_tensor, input_ids, attention_mask
        return (
            evidence,
            forward_passes,
            peak_workspace_tokens,
            total_sequence_tokens,
        )

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult] | Mapping[str, str],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> Mapping[str, CausalChoiceEvidence]:
        """Return one transient evidence row per unique candidate."""

        started = time.perf_counter()
        rows = self._candidate_rows(
            candidates,
            source_timestamps=source_timestamps,
        )
        bounded = rows[: self.max_candidates]
        evidence = {
            candidate_id: CausalChoiceEvidence(
                candidate_id=candidate_id,
                role=role,
                inspected=False,
                answerability=0.5,
                value_evidence_logit=0.0,
                direct_log_likelihood=0.0,
                indirect_log_likelihood=0.0,
            )
            for candidate_id, _text, role in rows
        }
        fallback_reason = (
            "candidate_bound: "
            f"inspected {len(bounded)} of {len(rows)} candidates"
            if len(bounded) < len(rows)
            else ""
        )
        forward_passes = 0
        peak_workspace_tokens = 0
        total_sequence_tokens = 0
        if bounded:
            try:
                (
                    scored,
                    forward_passes,
                    peak_workspace_tokens,
                    total_sequence_tokens,
                ) = self._score_bounded(query, bounded)
                evidence.update(scored)
            except Exception as exc:
                if self.strict:
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        self.calls += 1
        self.forward_passes += forward_passes
        self.elapsed_s += elapsed
        self.last_report = CausalChoiceScoreReport(
            model_id=self.model_id,
            model_revision=self.model_revision,
            checkpoint_sha256=self.checkpoint_sha256,
            runtime=f"{type(self.model).__module__}.{type(self.model).__name__}",
            device=self.device,
            dtype=self.dtype_name,
            input_candidates=len(rows),
            inspected_candidates=len(bounded) if forward_passes else 0,
            output_candidates=len(evidence),
            choice_sequence_tokens=(
                len(self._choice_ids[0]),
                len(self._choice_ids[1]),
            ),
            workspace_tokens=peak_workspace_tokens,
            total_sequence_tokens=total_sequence_tokens,
            forward_passes=forward_passes,
            elapsed_s=elapsed,
            fallback_reason=fallback_reason,
        )
        return evidence

    def score(
        self,
        query: str,
        candidates: Sequence[RetrievalResult] | Mapping[str, str],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> Mapping[str, CausalChoiceEvidence]:
        """Backward-compatible alias for :meth:`score_candidates`."""

        return self.score_candidates(
            query,
            candidates,
            source_timestamps=source_timestamps,
        )

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> Mapping[str, RetrievalResult]:
        """Choose one raw payload per source with a query-derived role prior."""

        from memory_condense.search.selectors.coverage_selector import (
            _surface_value_evidence,
            compile_set_program,
        )

        source_rows, fallback = source_rows_with_fallback(candidates_by_source)
        flattened = round_robin_unique(
            [candidates for _source_id, candidates in source_rows]
        )

        selected = dict(fallback)
        fallback_reason = ""
        score_report: dict[str, Any] | None = None
        evidence: Mapping[str, CausalChoiceEvidence] = {}
        if flattened:
            try:
                proposed = self.score_candidates(
                    query,
                    flattened,
                    source_timestamps=source_timestamps,
                )
                if not isinstance(proposed, Mapping):
                    raise TypeError("causal choice scorer did not return a mapping")
                evidence = proposed
            except Exception as exc:
                if self.strict:
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"

        if self.last_report is not None:
            score_report = self.last_report.model_dump()
            fallback_reason = fallback_reason or self.last_report.fallback_reason
        program = compile_set_program(query)
        preferred_role = program.preferred_evidence_role
        for source_id, candidates in source_rows:
            eligible: list[tuple[int, RetrievalResult, CausalChoiceEvidence]] = []
            for local_rank, result in enumerate(candidates, start=1):
                row = evidence.get(result.chunk.chunk_id)
                if not isinstance(row, CausalChoiceEvidence) or not row.inspected:
                    continue
                eligible.append((local_rank, result, row))
            if not eligible:
                continue
            _rank, winner, _row = max(
                eligible,
                key=lambda item: (
                    1
                    if preferred_role is None
                    else int(
                        item[1].turn is not None
                        and item[1].turn.role.casefold() == preferred_role
                    ),
                    0.70 * item[2].answerability
                    + 0.30 * _surface_value_evidence(
                        item[1].chunk.text,
                        None,
                    ),
                    -item[0],
                ),
            )
            # ``winner`` is selected only from this source's supplied raw rows.
            selected[source_id] = winner

        selected_membership_scores: dict[str, float] = {}
        for source_id, result in selected.items():
            row = evidence.get(result.chunk.chunk_id)
            if isinstance(row, CausalChoiceEvidence) and row.inspected:
                selected_membership_scores[source_id] = float(row.answerability)

        self.last_source_companion_report = CausalChoiceCompanionReport(
            input_sources=len(source_rows),
            input_candidates=sum(len(candidates) for _source_id, candidates in source_rows),
            inspected_candidates=sum(
                int(row.inspected) for row in evidence.values()
            ),
            selected_sources=len(selected),
            preferred_evidence_role=preferred_role,
            selected_chunk_ids={
                source_id: result.chunk.chunk_id
                for source_id, result in selected.items()
            },
            selected_membership_scores=selected_membership_scores,
            score_report=score_report,
            fallback_reason=fallback_reason,
        )
        return selected

    def close(self) -> None:
        """Release the local checkpoint and allocator cache."""

        if getattr(self, "model", None) is not None:
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer = None
        self.last_report = None
        self.last_source_companion_report = None
        gc.collect()
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
