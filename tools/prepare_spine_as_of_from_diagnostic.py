"""Transfer authenticated live-diagnostic prompts into unstarted preflights.

No model, raw store, answer, or query-vector cache is read here. The unchanged
timed evaluator still recomputes every memory prompt before its API call.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from tools import audit_spine_as_of as diagnostic_tool
from tools import prepare_spine_as_of_full100 as preparation
from tools import run_spine_as_of_full100 as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


evaluation = runner.evaluation
TEMPLATE_SHA = "9987cab305f6d77bc1b99e527eb97845e0842713778c945e513d9ed77759141b"


def make_preflight(template, control_root, control, candidates):
    p = deepcopy(template.payload)
    for name in runner.NAMESPACE_FIELDS - {"calls", "control_preflight_root", "control_preflight_sha256"}:
        p[name] = control.payload[name]
    p["control_preflight_root"], p["control_preflight_sha256"] = str(control_root.resolve()), control.sha256
    original = {c["question"]["ordinal"]: c for c in control.payload["calls"] if c["arm"] == "semantic_seeds"}
    offset = p["shard_offset"]
    if set(original) != set(range(offset, offset + 10)) or set(candidates) != set(original):
        raise ValueError("ten complete control and diagnostic questions are required")
    expected_implementation = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in diagnostic_tool.IMPLEMENTATION}
    calls = []
    for i, ordinal in enumerate(range(offset, offset + 10)):
        question = original[ordinal]["question"]
        saved = candidates[ordinal].payload
        if (saved["arm"] != "as_of" or saved["question"] != question or
                saved["evaluation_preflight_sha256"] != control.sha256 or
                saved["messages_sha256"] != identity_sha256(saved["messages"]) or
                saved["implementation"] != expected_implementation or
                saved["query_vectors_computed"] is not True or saved["new_provider_calls"] != 0):
            raise ValueError("date diagnostic prompt, query, or implementation binding changed")
        messages = {"short_api": evaluation.answer_messages(question),
            "semantic_seeds": original[ordinal]["messages"], "as_of": saved["messages"]}
        messages.update({arm + "_api": messages[arm] for arm in evaluation.MEMORY_ARMS})
        for arm in evaluation.call_arm_order(i):
            calls.append({"call_index": len(calls), "question": question, "arm": arm,
                "messages": messages[arm], "messages_sha256": identity_sha256(messages[arm])})
    evaluation.validate_matched_calls(calls)
    evaluation.validate_control_calls(control, calls)
    p["calls"] = calls
    return p


def prepare(root):
    protocol = read_sealed_json(root / "protocol.json")
    if protocol.payload != preparation.protocol_payload():
        raise ValueError("frozen date-comparison protocol changed")
    template = evaluation.load_preflight(root / "namespaces/offset-000")
    if template.sha256 != TEMPLATE_SHA:
        raise ValueError("the independently prepared live template changed")
    aggregate = read_sealed_json(preparation.DIAGNOSTIC / "audit.json")
    if aggregate.sha256 != preparation.DIAGNOSTIC_SHA:
        raise ValueError("sealed development50 diagnostic changed")
    audits = {row["offset"]: row for row in aggregate.payload["bindings"]}
    pending = []
    for offset in range(0, 50, 10):
        binding, control_root, control = preparation.source_binding(offset)
        namespace = preparation.DIAGNOSTIC / f"offset-{offset:03d}"
        audit = read_sealed_json(namespace / "audit.json")
        if audit.sha256 != audits[offset]["audit_sha256"]:
            raise ValueError("namespace diagnostic no longer belongs to development50")
        candidates = {}
        for row in audit.payload["rows"]:
            if row["arm"] != "as_of":
                continue
            ordinal = row["ordinal"]
            saved = read_sealed_json(namespace / "candidate-prompts/as_of" / f"{ordinal:03d}.json")
            if ordinal in candidates or saved.sha256 != row["candidate_prompt_sha256"]:
                raise ValueError("duplicate or changed diagnostic prompt")
            candidates[ordinal] = saved
        payload = make_preflight(template, control_root, control, candidates)
        pending.append((offset, binding, control, payload, [candidates[i].sha256 for i in sorted(candidates)]))
    if pending[0][3] != template.payload:
        raise ValueError("diagnostic transfer does not reproduce the independent live preparation")
    targets = [root / "namespaces" / f"offset-{offset:03d}" for offset, *_ in pending]
    for target in targets:
        if list((target / "journal").glob("*")) or (target / "answers.json").exists() or (target / "judge-preflight.json").exists():
            raise ValueError("started evaluation cannot receive transferred preparation")
    bindings = []
    for offset, source, control, payload, candidates in pending:
        target = root / "namespaces" / f"offset-{offset:03d}"
        preflight, _ = publish_sealed_json(target / "preflight.json", payload)
        if evaluation.load_preflight(target).sha256 != preflight.sha256:
            raise ValueError("published preflight failed the live evaluator's validation")
        if offset:
            prepared, _ = publish_sealed_json(root / "prepared" / f"offset-{offset:03d}.json", {
                "protocol_sha256": protocol.sha256, "offset": offset, "root": str(target.resolve()),
                "preflight_sha256": preflight.sha256, "raw_token_proxy": payload["raw_token_proxy"],
                "answer_call_cap": 50, "maximum_logical_judgments": 20,
                "source_prepared_sha256": source.sha256, "control_preflight_sha256": control.sha256,
                "reproduced_diagnostic_prompt_shas": candidates,
                "preparation_source": "authenticated earlier live diagnostic prompts",
                "query_vectors_computed_in_this_preparation": False, "new_provider_calls": 0,
                "transfer_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        else:
            prepared = read_sealed_json(root / "prepared/offset-000.json")
        bindings.append({"offset": offset, "prepared_sha256": prepared.sha256, "preflight_sha256": preflight.sha256})
        print({"offset": offset, "preflight_sha256": preflight.sha256,
               "prepared_sha256": prepared.sha256, "new_provider_calls": 0}, flush=True)
    artifact, _ = publish_sealed_json(root / "preparation-first50.json", {
        "format": "memory-condense-as-of-development50-preparation-v1", "protocol_sha256": protocol.sha256,
        "bindings": bindings, "answer_call_count": 250, "new_provider_calls": 0,
        "independent_live_template_reproduced": True, "query_vectors_computed_in_this_transfer": False,
        "cached_query_vectors_for_serving": False, "live_recomputation_required_at_serving": True,
        "predictions_loaded": False, "gold_loaded": False,
        "transfer_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print({"preparation_sha256": artifact.sha256, "prepared_requests": 250}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    prepare(parser.parse_args().output_root)
