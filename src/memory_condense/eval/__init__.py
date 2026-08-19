"""Evaluation pipeline for memory_condense.

The small public facade is lazy so importing a provenance or schema helper
does not initialize responder/provider integrations.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any


_EXPORTS = {
    "replay_conversation": (
        "memory_condense.eval.runner",
        "replay_conversation",
    ),
    "run_eval": ("memory_condense.eval.runner", "run_eval"),
    "run_sweep": ("memory_condense.eval.sweep", "run_sweep"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> _Any:
    """Load and cache an evaluation facade object on first access."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(_import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy facade objects to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
