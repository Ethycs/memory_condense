"""Compile a query-free contextual-card sidecar with a local Liquid model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.modeling.lfm_extract import (
    DEFAULT_LFM_EXTRACT_MODEL_ID,
    DEFAULT_LFM_EXTRACT_REVISION,
    LFMCompletion,
)
from memory_condense.search.context_cards import (
    CONTEXT_CARD_SCHEMA,
    ContextCardPolicy,
    ContextMemory,
    ContextWindowUnavailableError,
    build_context_window,
    context_card_to_dict,
    make_context_card_request,
    materialize_context_card,
)


FORMAT = "memory-condense-context-card-shadow-assay-v1"


def _smoke_fixture() -> tuple[list[ContextMemory], tuple[str, ...]]:
    memories = [
        ContextMemory(
            "atlas-001",
            "project-atlas",
            1,
            "user",
            "Mira owns the Atlas deployment. Its codename is Cobalt.",
            "2026-09-01T09:00:00-07:00",
        ),
        ContextMemory(
            "garden-001",
            "garden-notes",
            1,
            "user",
            "The south bed contains rosemary and thyme.",
            "2026-09-01T10:00:00-07:00",
        ),
        ContextMemory(
            "atlas-002",
            "project-atlas",
            2,
            "assistant",
            "The Cobalt launch is scheduled for Monday at 09:00.",
            "2026-09-01T09:01:00-07:00",
        ),
        ContextMemory(
            "garden-002",
            "garden-notes",
            2,
            "user",
            "Water the rosemary on Friday, but leave the thyme alone.",
            "2026-09-02T10:00:00-07:00",
        ),
        ContextMemory(
            "atlas-003",
            "project-atlas",
            3,
            "user",
            "Actually, move it to Tuesday at 14:00 and keep the same codename.",
            "2026-09-02T11:00:00-07:00",
        ),
        ContextMemory(
            "atlas-004",
            "project-atlas",
            4,
            "user",
            "Mira will notify the launch team after the rollback drill is complete.",
            "2026-09-02T11:02:00-07:00",
        ),
    ]
    return memories, ("garden-002", "atlas-003", "atlas-004")


def _read_input(path: Path) -> tuple[list[ContextMemory], tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "memories",
        "target_memory_ids",
    }:
        raise ValueError("input requires exactly memories and target_memory_ids")
    if not isinstance(payload["memories"], list) or not isinstance(
        payload["target_memory_ids"], list
    ):
        raise ValueError("input memories and target_memory_ids must be arrays")
    memories = [ContextMemory(**row) for row in payload["memories"]]
    targets = tuple(str(value) for value in payload["target_memory_ids"])
    if not targets:
        raise ValueError("at least one target_memory_id is required")
    return memories, targets


def _publish_no_clobber(path: Path, payload: dict[str, Any]) -> tuple[str, Path]:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(path.name + ".sha256")
    path_created = False
    sidecar_created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        path_created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        sidecar_created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(f"{digest}  {path.name}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path_created and path.exists():
            path.unlink()
        if sidecar_created and sidecar.exists():
            sidecar.unlink()
        raise
    return digest, sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_LFM_EXTRACT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_LFM_EXTRACT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-memories", type=int, default=16)
    parser.add_argument("--max-window-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    memories, targets = (
        _smoke_fixture() if args.smoke else _read_input(args.input.resolve())
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        targets = targets[: args.limit]
    policy = ContextCardPolicy(
        previous_memories=args.previous_memories,
        max_window_tokens=args.max_window_tokens,
    )
    runtime = LFMCompletion(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        revision=args.model_revision,
    )
    rows: list[dict[str, Any]] = []
    provider_calls = 0
    started = time.perf_counter()
    generator_identity = runtime.identity
    try:
        for target_id in targets:
            try:
                window = build_context_window(memories, target_id, policy=policy)
            except ContextWindowUnavailableError as exc:
                target = next(memory for memory in memories if memory.memory_id == target_id)
                rows.append(
                    {
                        "target_memory_id": target_id,
                        "source_id": target.source_id,
                        "ordinal": target.ordinal,
                        "status": "unavailable",
                        "unavailable_reason": str(exc),
                    }
                )
                continue
            request = make_context_card_request(window, policy=policy)
            raw_completion = runtime.complete(
                request.system_prompt,
                request.user_prompt,
            )
            provider_calls += 1
            row: dict[str, Any] = {
                "target_memory_id": target_id,
                "source_id": window.target.source_id,
                "ordinal": window.target.ordinal,
                "window_memory_ids": [memory.memory_id for memory in window.memories],
                "window_token_count": window.token_count,
                "window_receipt_sha256": window.receipt_sha256,
                "prompt_sha256": request.prompt_sha256,
                "raw_completion": raw_completion,
                "completion_metrics": runtime.metrics_dict(),
            }
            try:
                card = materialize_context_card(
                    raw_completion,
                    request,
                    generator_identity=generator_identity,
                )
            except ValueError as exc:
                row.update({"status": "invalid", "validation_error": str(exc)})
            else:
                row.update({"status": "card", "card": context_card_to_dict(card)})
            rows.append(row)
    finally:
        runtime.close()
    elapsed = time.perf_counter() - started
    valid = sum(row["status"] == "card" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    unavailable = sum(row["status"] == "unavailable" for row in rows)
    payload = {
        "format": FORMAT,
        "card_schema": CONTEXT_CARD_SCHEMA,
        "policy": policy.identity_payload(),
        "generator_identity": generator_identity,
        "input_identity_sha256": identity_sha256(
            {
                "memories": [memory.identity_payload() for memory in memories],
                "target_memory_ids": list(targets),
            }
        ),
        "provider_calls": provider_calls,
        "valid_cards": valid,
        "invalid_cards": invalid,
        "unavailable_cards": unavailable,
        "elapsed_seconds": elapsed,
        "rows": rows,
    }
    digest, sidecar = _publish_no_clobber(args.output.resolve(), payload)
    print(
        canonical_json(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "sha256_file": str(sidecar),
                "provider_calls": provider_calls,
                "valid_cards": valid,
                "invalid_cards": invalid,
                "unavailable_cards": unavailable,
                "elapsed_seconds": elapsed,
            }
        )
    )
    return 0 if valid == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
