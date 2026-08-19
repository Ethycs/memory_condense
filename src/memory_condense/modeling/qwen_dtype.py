"""Qwen generation-dtype selection, shared by every local-Qwen runtime.

Lives in ``modeling`` because it is a model-runtime concern: ``search``
selectors and the ``eval`` harness both need it, and the only prior home
(``eval.local_qwen``) forced the one upward ``search -> eval`` import in the
codebase. ``eval.local_qwen`` re-exports it, so existing import and
monkeypatch surfaces are unchanged.
"""

from __future__ import annotations

from typing import Any

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
