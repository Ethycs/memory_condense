"""Prepare a separate full100 date comparison from authenticated seed controls.

This prepares prompts only. It never sends answers or judges and never changes
the existing semantic-seed campaign or its compilation scheduler.
"""
import argparse
import hashlib
from pathlib import Path

from tools import evaluate_spine_as_of as evaluation
from tools import run_spine_as_of_full100 as runner
from tools import run_spine_semantic_seed_full100_v3 as source_runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


SOURCE = Path("eval_results/full1m-spine-semantic-seeds-full100-20260910-r3")
SOURCE_SHA = "87f5d9ed7314a5341479f2ef2d8357ecb62ff95be6f2e7ecb7c28b7315bfefe3"
DIAGNOSTIC = Path("eval_results/full1m-spine-as-of-development50-20260910-r1")
DIAGNOSTIC_SHA = "abe09db4fa11b862b0d5347e99d49f34ea0dc67eae435be841b19982334da9e4"


def source_binding(offset):
    if type(offset) is not int or offset not in runner.OFFSETS:
        raise ValueError("a complete locked namespace offset is required")
    source = read_sealed_json(SOURCE / "protocol.json")
    if source.sha256 != SOURCE_SHA:
        raise ValueError("frozen source campaign changed")
    binding = read_sealed_json(SOURCE / "prepared" / f"offset-{offset:03d}.json")
    p = binding.payload
    control_root = Path(p["root"])
    control = evaluation.seed_evaluation.load_preflight(control_root)
    if (p["protocol_sha256"] != SOURCE_SHA or p["offset"] != offset or
            p["preflight_sha256"] != control.sha256 or control.payload["shard_offset"] != offset or
            p["answer_call_cap"] != 50 or p["maximum_logical_judgments"] != 20 or
            len(control.payload["calls"]) != 50 or p["raw_token_proxy"] != control.payload["raw_token_proxy"] or
            p["raw_token_proxy"] < 1_000_000):
        raise ValueError("incomplete or changed frozen control memory")
    return binding, control_root, control


def protocol_payload():
    source = read_sealed_json(SOURCE / "protocol.json")
    diagnostic = read_sealed_json(DIAGNOSTIC / "audit.json")
    if source.sha256 != SOURCE_SHA or diagnostic.sha256 != DIAGNOSTIC_SHA:
        raise ValueError("source campaign or date diagnostic changed")
    p = dict(source.payload)
    p.pop("seed_policy")
    p.update({"format": "memory-condense-as-of-full100-protocol-v1",
        "predecessor_protocol_sha256": source.sha256,
        "source_campaign_root": str(SOURCE.resolve()), "date_diagnostic_sha256": diagnostic.sha256,
        "memory_arms": list(evaluation.MEMORY_ARMS), "routes": evaluation.ROUTES,
        "reader_policies": evaluation.reader_policies(), "short_controls": evaluation.SHORT_CONTROLS,
        "as_of_policy": evaluation.AS_OF_POLICY,
        "control_prompts_must_reproduce_frozen_semantic_seeds": True,
        "historical_predictions_reused": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                           for name in runner.PROTOCOL_IMPLEMENTATION}})
    return p


def freeze(root):
    present = [offset for offset in runner.OFFSETS if (SOURCE / "prepared" / f"offset-{offset:03d}.json").exists()]
    controls = [source_binding(offset)[1] for offset in present]
    source_runner.require_unstarted(controls)
    protocol, _ = publish_sealed_json(root / "protocol.json", protocol_payload())
    print({"protocol_sha256": protocol.sha256, "answer_call_cap": 500,
           "maximum_logical_judgments": 200, "new_provider_calls": 0}, flush=True)
    return protocol


def prepare_namespace(root, offset):
    protocol = read_sealed_json(root / "protocol.json")
    if protocol.payload != protocol_payload():
        raise ValueError("as-of full100 protocol changed")
    binding, control_root, control = source_binding(offset)
    target = root / "namespaces" / f"offset-{offset:03d}"
    if (target / "preflight.json").exists():
        raise ValueError("prepared namespace already exists; preserve it and inspect before any successor")
    evaluation.prepare(target, control_root)
    preflight = evaluation.load_preflight(target)
    if preflight.payload["control_preflight_sha256"] != control.sha256:
        raise ValueError("prepared date comparison changed its control binding")
    verified = []
    if offset < 50:
        for call in preflight.payload["calls"]:
            if call["arm"] != "as_of":
                continue
            ordinal = call["question"]["ordinal"]
            saved = read_sealed_json(DIAGNOSTIC / f"offset-{offset:03d}" / "candidate-prompts" / "as_of" / f"{ordinal:03d}.json")
            if (saved.payload["evaluation_preflight_sha256"] != control.sha256 or
                    saved.payload["question"] != call["question"] or saved.payload["messages"] != call["messages"]):
                raise ValueError("fresh date preparation differs from the sealed diagnostic")
            verified.append(saved.sha256)
    receipt, _ = publish_sealed_json(root / "prepared" / f"offset-{offset:03d}.json", {
        "protocol_sha256": protocol.sha256, "offset": offset, "root": str(target.resolve()),
        "preflight_sha256": preflight.sha256, "raw_token_proxy": preflight.payload["raw_token_proxy"],
        "answer_call_cap": 50, "maximum_logical_judgments": 20,
        "source_prepared_sha256": binding.sha256, "control_preflight_sha256": control.sha256,
        "reproduced_diagnostic_prompt_shas": verified, "new_provider_calls": 0})
    print({"prepared_sha256": receipt.sha256, "offset": offset, "answer_call_cap": 50,
           "reproduced_diagnostic_prompts": len(verified), "new_provider_calls": 0}, flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("protocol", "namespace"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offset", type=int)
    args = parser.parse_args()
    if args.phase == "protocol":
        freeze(args.output_root)
    else:
        prepare_namespace(args.output_root, args.offset)
