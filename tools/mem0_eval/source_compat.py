"""Source-epoch imports for the independently frozen Mem0 tool.

Validation v3 keeps core modules at the package root.  The active v4 layout
moved those same public contracts into responsibility-focused subpackages.
The bootstrap verifies the complete selected source tree before this module
is imported; this module then selects exactly one on-disk layout and never
falls back after an import failure.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any


def _source_layout() -> str:
    spec = importlib.util.find_spec("memory_condense")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if len(locations) != 1:
        raise ImportError("memory_condense must resolve to one frozen package root")
    package = Path(locations[0]).resolve(strict=True)
    v3 = all(
        (package / relative).is_file()
        for relative in ("_tokenizer.py", "embedding.py", "loader.py")
    )
    v4 = all(
        (package / relative).is_file()
        for relative in (
            "domain/_tokenizer.py",
            "modeling/embedding.py",
            "ingest/loader.py",
        )
    )
    if v3 == v4:
        raise ImportError(
            "frozen memory_condense source has an ambiguous or unsupported layout"
        )
    return "v3" if v3 else "v4"


SOURCE_LAYOUT = _source_layout()


_CONTRACTS = {
    "count_chat_prompt_token_proxy": ("_tokenizer", "domain._tokenizer"),
    "count_tokens": ("_tokenizer", "domain._tokenizer"),
    "tokenizer_proxy_identity": ("_tokenizer", "domain._tokenizer"),
    "BGE_M3_CHECKPOINT_SHA256": ("embedding", "modeling.embedding"),
    "DEFAULT_MODEL_DIM": ("embedding", "modeling.embedding"),
    "DEFAULT_MODEL_NAME": ("embedding", "modeling.embedding"),
    "DEFAULT_MODEL_REVISION": ("embedding", "modeling.embedding"),
    "verify_bge_m3_checkpoint": ("embedding", "modeling.embedding"),
    "BenchmarkSample": ("loader", "ingest.loader"),
    "parse_longmemeval": ("loader", "ingest.loader"),
    # The frozen v3 adapter kept runtime/protocol helpers in one module; v4
    # split them by responsibility.  Resumable execution resolves these exact
    # private seams through the already authenticated source layout instead of
    # probing/falling back after an import error.
    "_OwnedMem0Backend": (
        "eval.mem0_adapter",
        "eval.mem0_runtime",
    ),
    "_assert_mem0_state_binding": (
        "eval.mem0_adapter",
        "eval.mem0_runtime",
    ),
    "_assert_no_live_mem0_telemetry": (
        "eval.mem0_adapter",
        "eval.mem0_runtime",
    ),
    "_memory_id": ("eval.mem0_adapter", "eval.mem0_protocol"),
    "_redacted_stable_config": (
        "eval.mem0_adapter",
        "eval.mem0_runtime",
    ),
    "_remove_owned_state": (
        "eval.mem0_adapter",
        "eval.mem0_runtime",
    ),
    "_response_rows": ("eval.mem0_adapter", "eval.mem0_protocol"),
    "_sha256_json": ("eval.mem0_adapter", "eval.mem0_runtime"),
}


def __getattr__(name: str) -> Any:
    modules = _CONTRACTS.get(name)
    if modules is None:
        raise AttributeError(name)
    relative = modules[0] if SOURCE_LAYOUT == "v3" else modules[1]
    value = getattr(importlib.import_module(f"memory_condense.{relative}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BGE_M3_CHECKPOINT_SHA256",
    "BenchmarkSample",
    "DEFAULT_MODEL_DIM",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REVISION",
    "SOURCE_LAYOUT",
    "count_chat_prompt_token_proxy",
    "count_tokens",
    "parse_longmemeval",
    "tokenizer_proxy_identity",
    "verify_bge_m3_checkpoint",
    "_OwnedMem0Backend",
    "_assert_mem0_state_binding",
    "_assert_no_live_mem0_telemetry",
    "_memory_id",
    "_redacted_stable_config",
    "_remove_owned_state",
    "_response_rows",
    "_sha256_json",
]
