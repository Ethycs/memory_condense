"""Experimental live per-head memory backed by the Qwen3 prefix encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from memory_condense.associations.cav_memory import CAVBank
from memory_condense.associations.qwen_live_memory import QwenLiveHeadMemory
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder


def run_smoke_benchmark(
    memory: QwenLiveHeadMemory,
    dataset_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    benchmark_cav_weight = float(payload.get("cav_weight", 0.10))
    for episode in payload["memories"]:
        memory.write(
            episode["id"],
            episode["text"],
            importance=float(episode.get("importance", 0.0)),
            metadata={"source": str(dataset_path)},
        )

    associations = [tuple(pair) for pair in payload.get("associations", [])]
    diagnostics = memory.graph.calibrate_heads(associations, keep=4)
    diagnostics["directed_edge_count"] = memory.graph.edge_count
    diagnostics["entry_layer"] = memory.layer
    diagnostics["association_layer"] = memory.association_layer

    arms = {
        "residual": {"mode": "residual", "cav_weight": 0.0},
        "cav_residual": {
            "mode": "residual",
            "cav_weight": benchmark_cav_weight,
            "cav_mode": "positive",
        },
        "associative_cav_residual_qk": {
            "mode": "associative",
            "cav_weight": benchmark_cav_weight,
            "seed_k": 2,
            "hops": 1,
        },
        "direct_qk": {"hops": 1, "cav_weight": 0.0},
        "cav_qk": {"hops": 1, "cav_weight": 0.25, "cav_mode": "positive"},
        "recursive_cav_qk_ov": {
            "hops": 2,
            "cav_weight": 0.25,
            "cav_mode": "positive",
        },
    }
    results: dict[str, Any] = {}
    for arm, options in arms.items():
        rows: list[dict[str, Any]] = []
        recall_at_1 = 0
        recall_at_3 = 0
        for query in payload["queries"]:
            arm_options = dict(options)
            mode = arm_options.pop("mode", "head")
            if mode == "residual":
                result = memory.retrieve_residual(
                    query["text"], top_k=3, **arm_options
                )
            elif mode == "associative":
                result = memory.retrieve_associative(
                    query["text"], top_k=3, **arm_options
                )
            else:
                result = memory.retrieve(query["text"], top_k=3, **arm_options)
            ids = [hit.episode_id for hit in result.hits]
            recall_at_1 += int(query["answer_id"] in ids[:1])
            recall_at_3 += int(query["answer_id"] in ids[:3])
            rows.append(
                {
                    "query": query["text"],
                    "answer_id": query["answer_id"],
                    "retrieved": ids,
                    "hops": result.hop_episode_ids,
                    "cav_signature": result.query_cav_signature,
                }
            )
        count = len(payload["queries"])
        results[arm] = {
            "recall_at_1": recall_at_1 / count,
            "recall_at_3": recall_at_3 / count,
            "rows": rows,
        }
    results["_diagnostics"] = diagnostics
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cav-report", type=Path, required=True)
    parser.add_argument("--cav-vectors", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--association-layer", type=int)
    parser.add_argument("--concept", action="append", default=["binding_constraint"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = Qwen3PrefixEncoder(
        args.model_dir,
        layers=7,
        device="cuda",
        dtype="bfloat16",
    )
    bank = CAVBank.load(
        args.cav_report,
        args.cav_vectors,
        layer=args.layer,
        concepts=args.concept,
        device=encoder.device,
    )
    memory = QwenLiveHeadMemory(
        encoder,
        layer=args.layer,
        association_layer=args.association_layer,
        cav_bank=bank,
    )
    results = run_smoke_benchmark(memory, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for arm, result in results.items():
        if arm.startswith("_"):
            print(f"{arm}: {json.dumps(result)}")
            continue
        print(
            f"{arm}: recall@1={result['recall_at_1']:.3f}, "
            f"recall@3={result['recall_at_3']:.3f}"
        )
    print(f"result: {args.output}")
    return 0
