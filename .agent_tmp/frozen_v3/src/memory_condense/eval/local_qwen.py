"""Local full-checkpoint Qwen responder for no-API benchmark calibration."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Sequence


_DTYPE_NAMES = {"auto", "bfloat16", "float16", "float32"}


def resolve_local_qwen_dtype(
    torch_module: Any,
    requested: str = "auto",
    *,
    device: str | None = None,
) -> tuple[Any, str]:
    """Choose a generation dtype that the active GPU executes natively.

    Turing and older CUDA devices do not have native BF16 tensor cores. Using
    BF16 there is valid but dramatically slower than FP16, so ``auto`` keeps
    BF16 on Ampere+ and uses FP16 on pre-Ampere CUDA. CPU-only loading keeps
    the checkpoint's compact BF16 footprint.
    """

    name = str(requested).strip().casefold()
    if name not in _DTYPE_NAMES:
        raise ValueError(
            f"unsupported local Qwen dtype {requested!r}; "
            f"expected one of {sorted(_DTYPE_NAMES)}"
        )
    if name == "auto":
        target = str(device or "cuda").casefold()
        use_cuda = target.startswith("cuda") and torch_module.cuda.is_available()
        if use_cuda:
            try:
                major, _minor = (
                    torch_module.cuda.get_device_capability()
                    if device is None
                    else torch_module.cuda.get_device_capability(device)
                )
            except Exception:
                major = 0
            name = "bfloat16" if major >= 8 else "float16"
        else:
            name = "bfloat16"
    return getattr(torch_module, name), name


def strip_qwen_thinking(text: str) -> str:
    """Remove an accidental Qwen thinking prelude from a short QA answer."""
    value = text.strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    return re.sub(r"^<think>.*$", "", value, flags=re.DOTALL).strip()


class LocalQwenAnswerer:
    """Load Qwen3 once and answer OpenAI-style chat prompts locally.

    Generation K/V exists only for one answer and is released by Transformers
    afterward. Nothing from the model is written into the memory database.
    CPU offload is explicit because the BF16 8B checkpoint is larger than an
    8 GiB GPU; ``max_memory`` keeps coexistence with the embedding model safe.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_new_tokens: int = 64,
        gpu_memory: str = "4GiB",
        cpu_memory: str = "24GiB",
        dtype: str = "auto",
        stop_strings: Sequence[str] | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        root = Path(model_dir)
        if not (
            (root / "model.safetensors.index.json").exists()
            or (root / "model.safetensors").exists()
        ):
            raise FileNotFoundError(
                f"full local checkpoint weights not found under {root}"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        torch_dtype, self.dtype_name = resolve_local_qwen_dtype(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        max_memory: dict[Any, str] = {"cpu": cpu_memory}
        if torch.cuda.is_available():
            max_memory[0] = gpu_memory
        self.model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            max_memory=max_memory if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.stop_strings = tuple(value for value in (stop_strings or ()) if value)
        self.calls = 0
        self.elapsed_s = 0.0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        started = time.perf_counter()
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        generation_options: dict[str, Any] = {}
        if self.stop_strings:
            generation_options.update(
                stop_strings=list(self.stop_strings),
                tokenizer=self.tokenizer,
            )
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_options,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.calls += 1
        self.elapsed_s += time.perf_counter() - started
        return strip_qwen_thinking(answer)

    def close(self) -> None:
        """Release model references and the CUDA allocator cache."""
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        if getattr(self, "_torch", None) is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
