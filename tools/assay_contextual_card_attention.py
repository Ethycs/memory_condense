"""Replay precomputed contextual cards through the Qwen attention linker."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tools")]

from tools.assay_contextual_cards import (
    FORMAT as COMPILE_FORMAT,
    _publish_no_clobber,
    _read_input,
    _smoke_fixture,
)
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.context_card_retrieval import (
    attention_search_context_cards,
    lexical_route_context_card_sources,
)
from memory_condense.search.context_cards import (
    CONTEXT_CARD_SCHEMA,
    ContextCardPolicy,
    build_context_window,
    context_card_to_dict,
    make_context_card_request,
    materialize_context_card,
)


FORMAT = "memory-condense-context-card-attention-assay-v1"
SMOKE_QUERIES = (
    (
        "Which plant should be watered on Friday, and which should be left alone?",
        "garden-002",
    ),
    ("When is the Cobalt launch now scheduled?", "atlas-003"),
    (
        "Who will notify the launch team after the rollback drill?",
        "atlas-004",
    ),
)


def _read_canonical(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ValueError("card artifact must not be a symlink")
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8")
    payload = json.loads(raw)
    if raw != canonical_json(payload) + "\n":
        raise ValueError("card artifact is not canonical JSON")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError("card artifact SHA-256 sidecar is missing or invalid")
    return payload, digest


def _policy(payload: dict[str, Any]) -> ContextCardPolicy:
    expected = {
        "schema",
        "previous_memories",
        "max_window_tokens",
        "max_facts",
        "max_fact_tokens",
        "max_entities",
        "max_topics",
        "max_citations_per_fact",
        "max_quote_chars",
        "max_card_tokens",
    }
    if set(payload) != expected or payload["schema"] != CONTEXT_CARD_SCHEMA:
        raise ValueError("card artifact policy is incompatible")
    return ContextCardPolicy(**{key: value for key, value in payload.items() if key != "schema"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", action="append")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--attention-layer", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--source-limit", type=int, default=1)
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if isinstance(args.repeat, bool) or args.repeat < 1:
        raise ValueError("--repeat must be positive")
    memories, _targets = (
        _smoke_fixture() if args.smoke else _read_input(args.input.resolve())
    )
    if args.cards.is_symlink():
        raise ValueError("card artifact must not be a symlink")
    artifact_path = args.cards.resolve()
    artifact, artifact_sha256 = _read_canonical(artifact_path)
    if (
        artifact.get("format") != COMPILE_FORMAT
        or artifact.get("card_schema") != CONTEXT_CARD_SCHEMA
    ):
        raise ValueError("card artifact format is incompatible")
    policy = _policy(artifact["policy"])
    expected_input_identity = identity_sha256(
        {
            "memories": [memory.identity_payload() for memory in memories],
            "target_memory_ids": list(_targets),
        }
    )
    if artifact.get("input_identity_sha256") != expected_input_identity:
        raise ValueError("card artifact input population does not match this assay")
    artifact_rows = artifact.get("rows")
    if not isinstance(artifact_rows, list) or tuple(
        row.get("target_memory_id") for row in artifact_rows if isinstance(row, dict)
    ) != tuple(_targets):
        raise ValueError("card artifact rows do not match the target population")
    generator_identity = artifact["generator_identity"]
    cards = []
    for row in artifact_rows:
        if row.get("status") != "card":
            continue
        window = build_context_window(
            memories, row["target_memory_id"], policy=policy
        )
        request = make_context_card_request(window, policy=policy)
        if (
            row["window_receipt_sha256"] != window.receipt_sha256
            or row["prompt_sha256"] != request.prompt_sha256
        ):
            raise ValueError("card row no longer matches its raw input window")
        card = materialize_context_card(
            row["raw_completion"],
            request,
            generator_identity=generator_identity,
        )
        if context_card_to_dict(card) != row["card"]:
            raise ValueError("card row does not replay to its sealed materialization")
        cards.append(card)
    if args.query:
        queries = tuple((query, None) for query in args.query)
    elif args.smoke:
        queries = SMOKE_QUERIES
    else:
        raise ValueError("--query is required outside the smoke fixture")
    queries = queries * args.repeat
    all_sources = tuple(dict.fromkeys(memory.source_id for memory in memories))
    started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(
        args.qwen_model_dir.resolve(),
        layers=args.layers,
        device="cuda",
        dtype=args.dtype,
    )
    linker = QwenMemoryLinker(
        encoder,
        layer=args.attention_layer,
        max_candidates=8,
        max_workspace_tokens=1024,
    )
    loaded_seconds = time.perf_counter() - started
    qwen_checkpoint_sha256 = encoder.checkpoint_sha256
    rows = []
    correct = 0
    try:
        for query, expected_target in queries:
            query_started = time.perf_counter()
            routes = (
                ()
                if args.attention_only
                else lexical_route_context_card_sources(
                    query,
                    cards,
                    max_sources=args.source_limit,
                    eligible_source_ids=all_sources,
                )
            )
            source_order = (
                all_sources
                if args.attention_only
                else tuple(route.source_id for route in routes)
            )
            result = attention_search_context_cards(
                query,
                cards,
                memories,
                linker=linker,
                source_order=source_order,
                eligible_target_memory_ids=tuple(
                    target_id
                    for target_id in _targets
                    if next(
                        memory.source_id
                        for memory in memories
                        if memory.memory_id == target_id
                    )
                    in set(source_order)
                ),
                source_scope_complete=(
                    args.attention_only or set(source_order) == set(all_sources)
                ),
                group_size=8,
                beam_per_group=2,
                top_k=1,
            )
            elapsed = time.perf_counter() - query_started
            selected = result.selected_card_ids[0] if result.selected_card_ids else None
            selected_card = next(
                (card for card in cards if card.card_id == selected), None
            )
            selected_target = (
                None if selected_card is None else selected_card.target_memory_id
            )
            is_correct = (
                None if expected_target is None else selected_target == expected_target
            )
            correct += int(is_correct is True)
            inspection = result.inspection
            rows.append(
                {
                    "query": query,
                    "query_sha256": identity_sha256(query),
                    "expected_target_memory_id": expected_target,
                    "source_routes": [
                        {
                            "source_id": route.source_id,
                            "score": route.score,
                            "matched_terms": list(route.matched_terms),
                        }
                        for route in routes
                    ],
                    "selected_card_id": selected,
                    "selected_target_memory_id": selected_target,
                    "correct": is_correct,
                    "hydrated_memory_ids": [
                        evidence.memory_id for evidence in result.evidence
                    ],
                    "omitted_memory_ids": list(result.omitted_memory_ids),
                    "requires_raw_fallback": result.requires_raw_fallback,
                    "attention_overflow": result.attention_overflow,
                    "source_scope_complete": result.source_scope_complete,
                    "covered_source_ids": list(result.covered_source_ids),
                    "uncovered_source_ids": list(result.uncovered_source_ids),
                    "covered_target_memory_ids": list(
                        result.covered_target_memory_ids
                    ),
                    "uncovered_target_memory_ids": list(
                        result.uncovered_target_memory_ids
                    ),
                    "elapsed_seconds": elapsed,
                    "passes": 0 if inspection is None else inspection.passes,
                    "candidate_inspections": (
                        0
                        if inspection is None
                        else inspection.total_candidate_inspections
                    ),
                    "max_workspace_tokens": (
                        0 if inspection is None else inspection.max_workspace_tokens
                    ),
                }
            )
    finally:
        del linker
        del encoder
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    scored = sum(expected is not None for _query, expected in queries)
    payload = {
        "format": FORMAT,
        "card_artifact_sha256": artifact_sha256,
        "card_schema": CONTEXT_CARD_SCHEMA,
        "card_count": len(cards),
        "qwen_model_dir": str(args.qwen_model_dir.resolve()),
        "qwen_checkpoint_sha256": qwen_checkpoint_sha256,
        "layers": args.layers,
        "attention_layer": args.attention_layer,
        "dtype": args.dtype,
        "attention_only": args.attention_only,
        "source_limit": args.source_limit,
        "load_seconds": loaded_seconds,
        "query_count": len(rows),
        "correct": correct if scored else None,
        "accuracy": correct / scored if scored else None,
        "rows": rows,
    }
    digest, sidecar = _publish_no_clobber(args.output.resolve(), payload)
    print(
        canonical_json(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "sha256_file": str(sidecar),
                "cards": len(cards),
                "queries": len(rows),
                "accuracy": payload["accuracy"],
                "load_seconds": loaded_seconds,
                "query_seconds": [row["elapsed_seconds"] for row in rows],
            }
        )
    )
    return 0 if payload["accuracy"] in {None, 1.0} else 2


if __name__ == "__main__":
    raise SystemExit(main())
