"""Local, lazy Transformers adapter for Liquid LFM2 checkpoints."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LFM_EXTRACT_MODEL_ID = "LiquidAI/LFM2-350M-Extract"
DEFAULT_LFM_EXTRACT_REVISION = "d99a6f06ea16a2f83998789389a64b66d40c4198"
DEFAULT_LFM_TRANSCRIPT_MODEL_ID = "LiquidAI/LFM2-2.6B-Transcript"
DEFAULT_LFM_TRANSCRIPT_REVISION = "1b607be3f244de841a55c9fe426713dd950ab281"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lfm_checkpoint_identity(
    model_dir: str | Path,
    *,
    model_id: str = DEFAULT_LFM_EXTRACT_MODEL_ID,
    revision: str = DEFAULT_LFM_EXTRACT_REVISION,
) -> dict[str, Any]:
    """Bind the local weights and prompt-critical tokenizer files."""

    root = Path(model_dir).resolve()
    required = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ]
    single_weights = root / "model.safetensors"
    index_path = root / "model.safetensors.index.json"
    if single_weights.is_file():
        required.append(single_weights.name)
    elif index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = sorted(set(index["weight_map"].values()))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors checkpoint index") from exc
        if not weight_files or any(Path(name).name != name for name in weight_files):
            raise ValueError("checkpoint index contains an invalid shard path")
        required.extend([index_path.name, *weight_files])
    else:
        raise FileNotFoundError("LFM checkpoint has no safetensors weights")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"LFM checkpoint files are missing: {missing}")
    files = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": _sha256_file(root / name),
        }
        for name in required
    }
    return {
        "provider": "local_transformers",
        "model_id": str(model_id),
        "revision": str(revision),
        "checkpoint_files": files,
    }


def lfm_extract_checkpoint_identity(
    model_dir: str | Path,
    *,
    model_id: str = DEFAULT_LFM_EXTRACT_MODEL_ID,
    revision: str = DEFAULT_LFM_EXTRACT_REVISION,
) -> dict[str, Any]:
    """Backward-compatible spelling for the generic LFM identity helper."""

    return lfm_checkpoint_identity(
        model_dir,
        model_id=model_id,
        revision=revision,
    )


@dataclass(frozen=True, slots=True)
class LFMExtractMetrics:
    load_seconds: float
    generation_seconds: float
    input_tokens: int
    output_tokens: int
    peak_cuda_bytes: int | None


class LFMCompletion:
    """A resident single-turn, greedy completion callable.

    Loading is delayed until the first request.  Core extraction code sees
    only ``complete(system_prompt, user_prompt) -> str`` and therefore remains
    independent of Transformers and this particular checkpoint.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 512,
        model_id: str = DEFAULT_LFM_EXTRACT_MODEL_ID,
        revision: str = DEFAULT_LFM_EXTRACT_REVISION,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device_name = str(device)
        self.dtype_name = str(dtype).casefold()
        if self.dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.max_new_tokens = int(max_new_tokens)
        self.model_id = str(model_id)
        self.revision = str(revision)
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._windows_dll_handle: Any | None = None
        self._windows_cudnn_handles: list[Any] = []
        self._identity: dict[str, Any] | None = None
        self._load_seconds = 0.0
        self.last_metrics: LFMExtractMetrics | None = None

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            identity = lfm_checkpoint_identity(
                self.model_dir,
                model_id=self.model_id,
                revision=self.revision,
            )
            identity.update(
                {
                    "runtime": "transformers",
                    "device": self.device_name,
                    "dtype": self.dtype_name,
                    "decoding": "greedy",
                    "max_new_tokens": self.max_new_tokens,
                }
            )
            self._identity = identity
        return json.loads(json.dumps(self._identity))

    def load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        # Invoking a Pixi/Conda interpreter directly does not prepend its DLL
        # directory as activation would.  Keep the handle alive for the model
        # lifetime so CUDA can resolve cuDNN on Windows without mutating PATH.
        if os.name == "nt" and self._windows_dll_handle is None:
            dll_directory = Path(sys.prefix) / "Library" / "bin"
            if dll_directory.is_dir():
                self._windows_dll_handle = os.add_dll_directory(
                    str(dll_directory)
                )
                # Some cuDNN 9 builds discover component DLLs with their own
                # loader rather than Python's dependency loader.  Absolute
                # preloads keep direct (non-activated) Pixi invocation viable.
                for name in (
                    "nvrtc-builtins64_126.dll",
                    "nvrtc64_120_0.dll",
                    "cudnn64_9.dll",
                    "cudnn_graph64_9.dll",
                    "cudnn_ops64_9.dll",
                    "cudnn_cnn64_9.dll",
                    "cudnn_adv64_9.dll",
                    "cudnn_heuristic64_9.dll",
                    "cudnn_engines_precompiled64_9.dll",
                    "cudnn_engines_runtime_compiled64_9.dll",
                ):
                    path = dll_directory / name
                    if path.is_file():
                        self._windows_cudnn_handles.append(ctypes.WinDLL(str(path)))
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("LFM Extract requires torch and transformers") from exc

        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch reports no CUDA device")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype_name]
        if device.type == "cpu" and dtype == torch.float16:
            raise ValueError("float16 CPU inference is unsupported; use float32")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="eager",
        )
        model.to(device)
        model.requires_grad_(False)
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._load_seconds = time.perf_counter() - started

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        generation_seconds = time.perf_counter() - started
        output_ids = generated[0, input_ids.shape[1] :]
        completion = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        peak = (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda"
            else None
        )
        self.last_metrics = LFMExtractMetrics(
            load_seconds=self._load_seconds,
            generation_seconds=generation_seconds,
            input_tokens=int(input_ids.shape[1]),
            output_tokens=int(output_ids.shape[0]),
            peak_cuda_bytes=peak,
        )
        return completion

    def metrics_dict(self) -> dict[str, Any] | None:
        return None if self.last_metrics is None else asdict(self.last_metrics)

    def close(self) -> None:
        model = self._model
        torch = self._torch
        self._model = None
        self._tokenizer = None
        self._torch = None
        if model is not None:
            del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        handle = self._windows_dll_handle
        self._windows_dll_handle = None
        if handle is not None:
            handle.close()
        self._windows_cudnn_handles.clear()

    def __enter__(self) -> LFMCompletion:
        self.load()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


LFMExtractCompletion = LFMCompletion


__all__ = [
    "DEFAULT_LFM_EXTRACT_MODEL_ID",
    "DEFAULT_LFM_EXTRACT_REVISION",
    "DEFAULT_LFM_TRANSCRIPT_MODEL_ID",
    "DEFAULT_LFM_TRANSCRIPT_REVISION",
    "LFMCompletion",
    "LFMExtractCompletion",
    "LFMExtractMetrics",
    "lfm_checkpoint_identity",
    "lfm_extract_checkpoint_identity",
]
