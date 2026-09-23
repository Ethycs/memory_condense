"""Construct and replay source-grouped packets with conventional retrieval fixed."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_reduced30_construction as harness
from tools import assay_hot_v7_spine_episode_fact_reserved_full100 as parent_renderer
from tools.assay_hot_temporal_reference_chain_reduced30 import DEFAULT_PARENT, DEFAULT_PARENT_SHA256
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.hot_source_grouped_packet import FORMAT, compose_source_grouped_packet


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = ["tools/assay_hot_source_grouped_reduced30.py", "tools/matched_eval/hot_source_grouped_packet.py",
             "src/memory_condense/search/packing/source_grouped_packet.py",
             "tools/matched_eval/hot_temporal_reference_chain.py", "tools/assay_hot_reduced30_construction.py"]
    body = {"files": [{"path": path, "sha256": file_sha256(root / path)} for path in files],
            "parent_implementation": parent_renderer._implementation_identity()}
    return {**body, "sha256": identity_sha256(body)}


def build_selection(parent_path=DEFAULT_PARENT, expected_parent_sha256=DEFAULT_PARENT_SHA256):
    parent = read_sealed_json(parent_path)
    if parent.sha256 != expected_parent_sha256:
        raise ValueError("parent selection hash changed")
    harness.validate_selection(parent.payload)
    rows, timings, audits = [], [], []
    for row in parent.payload["questions"]:
        _, arm = harness.find_provider_arm(row["source_row"], row["telemetry"]["arm_path"])
        started = time.perf_counter()
        successor = compose_source_grouped_packet(arm)
        timings.append(time.perf_counter() - started)
        audits.append(successor["source_grouped_packet"])
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"],
                "question_id": row["question_id"], "prompt_question_sha256": row["prompt_question_sha256"],
                "parent_selection_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": successor}}
        rows.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    result = harness._artifact(module=sys.modules[__name__], rows=rows, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "sealed_conventional_packet_rendering", "parent_selection_sha256": parent.sha256,
                       "implementation": implementation_identity()})
    result.pop("receipt_sha256")
    result["source_grouped_summary"] = {
        "status_counts": dict(Counter(a["status"] for a in audits)),
        "raw_evidence_preserved_count": sum(a["raw_evidence_preserved"] for a in audits),
        "system_prompt_preserved_count": sum(a["system_prompt_preserved"] for a in audits),
        "context_token_delta_total": sum(a["context_token_delta"] for a in audits),
        "provider_calls": 0, "retrieval_changed": False,
    }
    result["receipt_sha256"] = identity_sha256(result)
    harness.validate_selection(result)
    return result, timings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("construct", "verify"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload, timings = build_selection()
    if args.command == "construct":
        artifact, created = publish_sealed_json(args.output_root / "selection.json", payload)
        publish_sealed_json(args.output_root / "construction-runtime.json", {
            "selection_sha256": artifact.sha256, "per_question_seconds": timings,
            "measurement": "authenticated adaptation including legacy rendering, full audit copying and token accounting; excludes corpus retrieval and providers",
        })
    else:
        artifact = read_sealed_json(args.output_root / "selection.json")
        if artifact.payload != payload:
            raise ValueError("selection differs from exact parent/policy replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, **payload["source_grouped_summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
