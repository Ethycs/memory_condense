"""Verify complete source admission including support-only escaping of embedded quotes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from tools import verify_spine_admission_method_v10 as previous
from tools import verify_spine_admission_method_v7 as source
from tools import spine_compaction_transport as compaction_transport
from tools import repair_spine_summary_budget_v4 as compact
from tools import recover_spine_summary_budget_v2 as recovery
from tools.admit_spine_corpus_v7 import admit
from tools.verify_spine_admission_method import expected_repair_row
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_quote_json_repair_v4 import repair_support_list_closures
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.spine_transport_lineage_v3 import verify_transport_lineage


ADMISSION_IMPLEMENTATION = ("tools/admit_spine_corpus_v7.py",
    "src/memory_condense/search/spine_source_admission.py",
    "src/memory_condense/search/spine_quote_json_repair.py",
    "src/memory_condense/search/spine_quote_json_repair_v2.py",
    "src/memory_condense/search/spine_quote_json_repair_v3.py",
    "src/memory_condense/search/spine_quote_json_repair_v4.py")
IMPLEMENTATION = (*previous.IMPLEMENTATION, *ADMISSION_IMPLEMENTATION,
    "tools/verify_spine_admission_method_v11.py")
SYNTAX_RULE = "support delimiter repair and escaping inside curly-quoted support only; preserve original response and every summary byte"


def normalize_method(method):
    return {**method, "format": "memory-condense-conditional-source-admission-method-v11",
        "syntax_repair": SYNTAX_RULE,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def conditional_method(policy):
    normalized = policy
    if policy.get("format") == "memory-condense-spine-source-binding-policy-v10":
        if (policy.get("syntax_repair") != SYNTAX_RULE or
                policy.get("invalid_slot_recovery") != "preserve valid original summaries; at most two summary-only attempts per invalid slot" or
                policy.get("implementation") != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ADMISSION_IMPLEMENTATION}):
            raise ValueError("unrecognized support escaping admission rule")
        normalized = {k: v for k, v in policy.items() if k != "invalid_slot_recovery"}
        normalized.update(format="memory-condense-spine-source-binding-policy-v8",
            syntax_repair=source.SYNTAX_RULE,
            implementation={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source.ADMISSION_IMPLEMENTATION})
    return normalize_method(previous.conditional_method(normalized))


def expected_population(requests, destination):
    rows = []
    for request in requests:
        p = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
            prompt_population=[p["messages"]], model=p["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            response = runtime.run().logical_completions[0]
        finally:
            runtime.close()
        parsed, _ = repair_support_list_closures(response)
        for atom in json.loads(parsed)["atoms"]:
            if count_tokens(atom["summary"]) > 128:
                rows.append(expected_repair_row({"label": atom["label"]}, request, parsed))
    return rows


def verify_extended_content(atoms_path, repair_root):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None or not re.fullmatch(r"offset-\d{3}", atoms_path.parent.name):
        raise ValueError("unknown source-atom artifact path")
    limit, offset = int(match[1]), int(atoms_path.parent.name[7:])
    root = atoms_path.parent.parent
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    policies = [read_sealed_json(path) for path in atoms_path.parent.glob(f"source-binding-policy-*-prefix-{limit:04d}.json")]
    policies = [p for p in policies if p.sha256 == data["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    policy = policies[0]
    method = conditional_method(policy.payload)
    if policy.payload["format"] != "memory-condense-spine-source-binding-policy-v10":
        raise ValueError("support escaping requires the current source-admission policy")
    corpus, namespace, requests, execution = prepare(root, offset, limit)
    if (limit != namespace["request_count"] or data["complete_namespace"] is not True or
            data["corpus_preflight_sha256"] != corpus.sha256 or
            policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
            policy.payload["execution_preflight_sha256"] != execution.sha256):
        raise ValueError("terminal-session recovery must admit the complete bound namespace")
    repair_sha = policy.payload["summary_budget_repairs_sha256"]
    if (repair_root is None) != (repair_sha is None):
        raise ValueError("supply exactly the summary repair root bound by this admission")
    expected = expected_population(requests, atoms_path.parent)
    batches = compact.batches_for(expected) if expected else []
    attempts = recovered = 0
    if repair_root is None:
        if expected:
            raise ValueError("source admission omitted required compactions or slot recovery")
    else:
        preflight = read_sealed_json(repair_root / "preflight.json")
        repairs = read_sealed_json(repair_root / "repairs.json")
        p = preflight.payload
        if (not expected or repairs.sha256 != repair_sha or repairs.payload["preflight_sha256"] != preflight.sha256 or
                p["rows"] != expected or p["corpus_preflight_sha256"] != corpus.sha256 or
                p["execution_preflight_sha256"] != execution.sha256 or p["model"] != compact.MODEL or
                p["gateway"] != compact.GATEWAY or p["raw_qwen_inputs"] is not False or p["retries"] != 0):
            raise ValueError("summary repair changed its complete attributed population or protocol")
        if p["format"] == "memory-condense-spine-summary-budget-repair-v4":
            if (p["batches"] != batches or p["maximum_provider_calls"] != len(batches) or
                    repairs.payload["successful_batch_count"] != len(batches)):
                raise ValueError("summary compaction batch population changed")
            replay = compact.run(repair_root, False)
        elif p["format"] == "memory-condense-spine-summary-budget-recovery-v2":
            if (p["original_batch_count"] != len(batches) or p["recovery_words"] != [48, 24] or
                    p["maximum_recovery_calls"] != 2 * len(p["failures"])):
                raise ValueError("invalid-slot recovery policy or population changed")
            replay = recovery.run(repair_root, False, 0)
            attempts, recovered = repairs.payload["recovery_attempts"], len(p["failures"])
            if (not recovered <= attempts <= 2 * recovered or
                    repairs.payload["recovered_summary_count"] != recovered or
                    repairs.payload["original_batch_count"] != len(batches)):
                raise ValueError("invalid-slot recovery exceeded its attributed attempt allowance")
        else:
            raise ValueError("unknown summary repair format for terminal-session recovery")
        if replay.sha256 != repairs.sha256:
            raise ValueError("summary repair changed during zero-call replay")
    lineage = verify_transport_lineage(root, offset, repair_root)
    if lineage["additional_compaction_attempts"] != 0:
        raise ValueError("raw terminal-session recovery cannot hide an extra compaction transport stage")
    admit(root, offset, limit, repair_root)
    if read_sealed_json(atoms_path).sha256 != atoms.sha256:
        raise ValueError("source atoms changed during verification")
    return {"atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "summary_repair_root": str(repair_root.resolve()) if repair_root else None,
        "summary_budget_repairs_sha256": repair_sha, "complete_namespace": True,
        "raw_span_population_sha256": data["raw_span_population_sha256"], "atom_count": len(data["atoms"]),
        "budget_compacted_atoms": data["budget_compacted_atoms"], "required_compaction_batches": len(batches),
        "compaction_recovery_attempts": attempts, "recovered_summary_count": recovered,
        "compaction_provider_attempts": len(batches) + attempts, "transport_lineage": lineage,
        "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
        "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True,
        "method": method, "method_sha256": identity_sha256(method)}


def verify(atoms_path, repair_root=None, transport_root=None):
    atoms_path = Path(atoms_path)
    atoms = read_sealed_json(atoms_path)
    policies = [read_sealed_json(path) for path in atoms_path.parent.glob("source-binding-policy-*.json")]
    policies = [p for p in policies if p.sha256 == atoms.payload["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    if policies[0].payload["format"] != "memory-condense-spine-source-binding-policy-v10":
        prior = previous.verify(atoms_path, repair_root, transport_root)
        payload = {**prior.payload, "predecessor_verification_sha256": prior.sha256}
    else:
        declared = compaction_transport.declared_stage(repair_root)
        if ((declared is None) != (transport_root is None) or
                declared is not None and declared != Path(transport_root).resolve()):
            raise ValueError("the declared compaction transport stage must be supplied exactly")
        payload = verify_extended_content(atoms_path, repair_root)
        extra = compaction_transport.verify(transport_root, repair_root, payload) if transport_root else None
        payload.update(compaction_only_transport=extra,
            compaction_transport_root=str(Path(transport_root).resolve()) if transport_root else None,
            compaction_provider_attempts=payload["compaction_provider_attempts"] +
                (extra["additional_compaction_attempts"] if extra else 0))
    method = normalize_method(payload["method"])
    limit = int(re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)[1])
    result, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v11-prefix-{limit:04d}.json"), {
        **payload, "format": "memory-condense-conditional-source-admission-verification-v11",
        "method": method, "method_sha256": identity_sha256(method)})
    print({"verification_sha256": result.sha256, "method_sha256": result.payload["method_sha256"],
        "complete_namespace": result.payload["complete_namespace"], "new_provider_calls": 0}, flush=True)
    return result


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v11-prefix-{match[1]}.json"))
    p = receipt.payload
    if (p["atoms_sha256"] != atoms.sha256 or p["source_binding_policy_sha256"] != atoms.payload["source_binding_policy_sha256"] or
            not p["complete_namespace"] or not atoms.payload["complete_namespace"]):
        raise ValueError("method verification is bound to a different or partial memory")
    replay = verify(atoms_path, Path(p["summary_repair_root"]) if p["summary_repair_root"] else None,
        Path(p["compaction_transport_root"]) if p["compaction_transport_root"] else None)
    if replay.sha256 != receipt.sha256:
        raise ValueError("conditional admission verification changed")
    return p["method_sha256"], replay.sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--summary-repair-root", type=Path)
    parser.add_argument("--compaction-transport-root", type=Path)
    args = parser.parse_args()
    verify(args.atoms, args.summary_repair_root, args.compaction_transport_root)
