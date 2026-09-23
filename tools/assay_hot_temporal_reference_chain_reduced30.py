#!/usr/bin/env python3
"""Construct/replay a reference-only successor over the sealed r9 failure30.

No corpus, answer artifact, reference answer, judgment, or provider is opened.
The shared reduced30 lifecycle can consume the resulting selection unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_reduced30_construction as harness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256, require_sha256
from tools.matched_eval.hot_temporal_reference_chain import (
    FORMAT,
    TemporalReferenceChainError,
    compose_temporal_reference_chain,
    split_packet,
)


DEFAULT_PARENT = Path(
    "eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-"
    "reduced30-20260908-r9/selection.json"
)
DEFAULT_PARENT_SHA256 = "41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28"
DEFAULT_OUTPUT = Path(
    "eval_results/longmemeval-1m-hot-temporal-reference-chain-reduced30-20260908-r1"
)


def _implementation_identity() -> dict[str, Any]:
    """Bind the new code and its repository dependencies, including the lock."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_temporal_reference_chain_reduced30.py",
        "tools/matched_eval/hot_temporal_reference_chain.py",
        "tools/assay_hot_reduced30_construction.py",
        "tools/assay_hot_v3_typed_operator_full100.py",
        "tools/matched_eval/typed_operator_spec.py",
        "tools/_routed_repair_routing.py",
        "tools/matched_eval/artifacts.py",
        "tools/matched_eval/contracts.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/domain/text_numbers.py",
        "pixi.lock",
    )
    files = [{"path": path, "sha256": file_sha256(root / path)} for path in paths]
    body = {"format": FORMAT + "-implementation", "files": files}
    return {**body, "sha256": identity_sha256(body)}


def build_selection(parent_path: Path, expected_parent_sha256: str) -> dict[str, Any]:
    require_sha256(expected_parent_sha256, "expected parent selection")
    parent = read_sealed_json(parent_path)
    if parent.sha256 != expected_parent_sha256:
        raise TemporalReferenceChainError("parent selection hash changed")
    harness.validate_selection(parent.payload)
    output = []
    status_counts: Counter[str] = Counter()
    changed_ordinals = []
    for row in parent.payload["questions"]:
        path, arm = harness.find_provider_arm(row["source_row"], row["telemetry"]["arm_path"])
        _, _, question = split_packet(arm)
        if quote_sha256(question) != row["prompt_question_sha256"]:
            raise TemporalReferenceChainError("parent dated question binding changed")
        if "temporal_reference_chain" in arm:
            raise TemporalReferenceChainError("refusing to stack the same policy twice")
        successor = compose_temporal_reference_chain(arm)
        audit = successor["temporal_reference_chain"]
        status_counts[audit["status"]] += 1
        if audit["status"] == "selected":
            changed_ordinals.append(row["global_ordinal"])
        source_body = {
            "format": FORMAT + "-source-row",
            "ordinal": row["global_ordinal"],
            "question_id": row["question_id"],
            "prompt_question_sha256": row["prompt_question_sha256"],
            "parent_selection_row_receipt_sha256": row["row_receipt_sha256"],
            "parent_arm_path": path,
            "arms": {"a3_protected_union": successor},
        }
        output.append((row["global_ordinal"], {
            **source_body, "row_receipt_sha256": identity_sha256(source_body),
        }))
    result = harness._artifact(
        module=sys.modules[__name__], rows=output, arm_path="arms.a3_protected_union",
        input_binding={
            "input_mode": "sealed_packet_temporal_reference_index",
            "parent_selection_sha256": parent.sha256,
            "implementation": _implementation_identity(),
        },
    )
    result.pop("receipt_sha256")
    result["temporal_reference_summary"] = {
        "status_counts": dict(sorted(status_counts.items())),
        "changed_global_ordinals": changed_ordinals,
        "unchanged_prompt_count": harness.QUESTION_COUNT - len(changed_ordinals),
        "parent_evidence_bytes_preserved_count": harness.QUESTION_COUNT,
        "provider_calls": 0,
    }
    result["receipt_sha256"] = identity_sha256(result)
    harness.validate_selection(result)
    return result


def verify_selection(selection_path: Path, parent_path: Path,
                     expected_parent_sha256: str) -> dict[str, Any]:
    """Replay every transformation; a recomputed outer seal alone is insufficient."""
    selection = read_sealed_json(selection_path)
    expected = build_selection(parent_path, expected_parent_sha256)
    if selection.payload != expected:
        raise TemporalReferenceChainError("successor differs from exact parent/policy replay")
    return {"verified": True, "selection_sha256": selection.sha256,
            "new_provider_calls": 0, **copy.deepcopy(expected["temporal_reference_summary"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("construct", "verify"))
    parser.add_argument("--parent-selection", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--expected-parent-sha256", default=DEFAULT_PARENT_SHA256)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    path = args.output_root / "selection.json"
    if args.command == "construct":
        payload = build_selection(args.parent_selection, args.expected_parent_sha256)
        artifact, created = publish_sealed_json(path, payload)
        result = {"created": created, "selection_sha256": artifact.sha256,
                  "new_provider_calls": 0, **payload["temporal_reference_summary"]}
    else:
        result = verify_selection(path, args.parent_selection, args.expected_parent_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
