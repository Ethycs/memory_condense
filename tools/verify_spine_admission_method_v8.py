"""Verify complete admission with bounded recovery of v4 compaction batches."""
import argparse
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from tools import verify_spine_admission_method_v7 as previous
from tools import recover_spine_summary_budget_v2 as recovery
from tools.finish_spine_compaction_batches import IMPLEMENTATION as COMPLETION_IMPLEMENTATION
from tools.admit_spine_corpus_v6 import admit
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.repair_spine_summary_budget_v4 import batches_for
from tools.spine_transport_lineage_v2 import verify_transport_lineage


ADMISSION_IMPLEMENTATION = tuple(
    "tools/admit_spine_corpus_v6.py" if name == "tools/admit_spine_corpus_v5.py" else name
    for name in previous.ADMISSION_IMPLEMENTATION)
RECOVERY_RULE = "preserve valid original summaries; at most two summary-only attempts per invalid slot"
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION, *ADMISSION_IMPLEMENTATION, *recovery.IMPLEMENTATION,
    *COMPLETION_IMPLEMENTATION, "tools/verify_spine_admission_method_v8.py")))


def conditional_method(policy):
    normalized = policy
    if policy.get("format") == "memory-condense-spine-source-binding-policy-v9":
        if (policy.get("invalid_slot_recovery") != RECOVERY_RULE or
                policy.get("implementation") != previous.base.base.legacy.hashes(ADMISSION_IMPLEMENTATION)):
            raise ValueError("unrecognized completed-slot recovery admission rule")
        normalized = {k: v for k, v in policy.items() if k != "invalid_slot_recovery"}
        normalized["format"] = "memory-condense-spine-source-binding-policy-v8"
        normalized["implementation"] = previous.base.base.legacy.hashes(previous.ADMISSION_IMPLEMENTATION)
    method = previous.conditional_method(normalized)
    method["format"] = "memory-condense-conditional-source-admission-method-v8"
    method["compaction"]["original_batch_completion"] = (
        "finish only unstarted original summary batches; preserve completed outputs; never retry unknown requests")
    method["implementation"] = previous.base.base.legacy.hashes(IMPLEMENTATION)
    return method


def verify(atoms_path, repair_root=None):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None or not re.fullmatch(r"offset-\d{3}", atoms_path.parent.name):
        raise ValueError("unknown source-atom artifact path")
    limit, offset = int(match[1]), int(atoms_path.parent.name[7:])
    root = atoms_path.parent.parent
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    policies = [read_sealed_json(p) for p in atoms_path.parent.glob(f"source-binding-policy-*-prefix-{limit:04d}.json")]
    policies = [p for p in policies if p.sha256 == data["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    policy = policies[0]
    method = conditional_method(policy.payload)
    repair_preflight = read_sealed_json(repair_root / "preflight.json") if repair_root else None
    recovered_v4 = (repair_preflight is not None and
        repair_preflight.payload.get("format") == "memory-condense-spine-summary-budget-recovery-v2")
    if not recovered_v4:
        if policy.payload["format"] == "memory-condense-spine-source-binding-policy-v9":
            raise ValueError("completed-slot admission requires its authenticated recovery root")
        prior = previous.verify(atoms_path, repair_root)
        payload = {**prior.payload, "predecessor_verification_sha256": prior.sha256}
    else:
        if policy.payload["format"] != "memory-condense-spine-source-binding-policy-v9":
            raise ValueError("v4 summary recovery requires the current exact source admission policy")
        corpus, namespace, requests, execution = prepare(root, offset, limit)
        if (limit != namespace["request_count"] or not data["complete_namespace"] or
                data["corpus_preflight_sha256"] != corpus.sha256 or
                policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
                policy.payload["execution_preflight_sha256"] != execution.sha256):
            raise ValueError("summary recovery namespace binding changed")
        repairs = read_sealed_json(repair_root / "repairs.json")
        p = repair_preflight.payload
        expected = previous.expected_population(requests, atoms_path.parent)
        batches = batches_for(expected)
        if (repairs.sha256 != policy.payload["summary_budget_repairs_sha256"] or
                repairs.payload["preflight_sha256"] != repair_preflight.sha256 or
                p["corpus_preflight_sha256"] != corpus.sha256 or
                p["execution_preflight_sha256"] != execution.sha256 or
                p["rows"] != expected or p["original_batch_count"] != len(batches) or
                p["model"] != recovery.parent.MODEL or p["gateway"] != recovery.parent.GATEWAY or
                p["raw_qwen_inputs"] is not False or p["retries"] != 0 or
                p["recovery_words"] != [48, 24] or p["maximum_recovery_calls"] != 2 * len(p["failures"])):
            raise ValueError("summary recovery changed the complete attributed population or attempt bounds")
        # Reconstruct original jobs, authenticated batch outputs, invalid slots
        # and all individual recoveries before accepting their resulting text.
        replay = recovery.run(repair_root, False, 0)
        if replay.sha256 != repairs.sha256:
            raise ValueError("summary recovery changed during zero-call replay")
        attempts, failed = repairs.payload["recovery_attempts"], len(p["failures"])
        if (not failed <= attempts <= 2 * failed or
                repairs.payload["recovered_summary_count"] != failed or
                repairs.payload["original_batch_count"] != len(batches)):
            raise ValueError("completed-invalid-slot recovery exceeded its attributed attempts")
        lineage = verify_transport_lineage(root, offset, repair_root)
        extra = lineage["additional_compaction_attempts"]
        if extra not in (0, 1):
            raise ValueError("compaction transport attempt allowance exceeded")
        admit(root, offset, limit, repair_root)
        if read_sealed_json(atoms_path).sha256 != atoms.sha256:
            raise ValueError("source atoms changed during verification")
        payload = {"atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
            "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
            "summary_repair_root": str(repair_root.resolve()), "summary_budget_repairs_sha256": repairs.sha256,
            "complete_namespace": data["complete_namespace"], "raw_span_population_sha256": data["raw_span_population_sha256"],
            "atom_count": len(data["atoms"]), "budget_compacted_atoms": data["budget_compacted_atoms"],
            "required_compaction_batches": len(batches), "compaction_recovery_attempts": attempts,
            "compaction_provider_attempts": len(batches) + attempts + extra,
            "recovered_summary_count": failed, "transport_lineage": lineage,
            "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
            "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True,
            "extended_support_delimiter_rule_needed": any(
                bool(r.get("support_json_syntax_repair", {}).get("quote_edits"))
                for r in data["admission_audits"] if r.get("support_json_syntax_repair"))}
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v8-prefix-{limit:04d}.json"), {
        **payload, "format": "memory-condense-conditional-source-admission-verification-v8",
        "method": method, "method_sha256": identity_sha256(method)})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "complete_namespace": data["complete_namespace"],
        "compaction_provider_attempts": payload["compaction_provider_attempts"], "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v8-prefix-{match[1]}.json"))
    p = receipt.payload
    if (p["atoms_sha256"] != atoms.sha256 or
            p["source_binding_policy_sha256"] != atoms.payload["source_binding_policy_sha256"] or
            not p["complete_namespace"] or not atoms.payload["complete_namespace"]):
        raise ValueError("method verification is bound to a different or partial memory")
    replay = verify(atoms_path, Path(p["summary_repair_root"]) if p["summary_repair_root"] else None)
    if replay.sha256 != receipt.sha256:
        raise ValueError("conditional admission verification changed")
    return p["method_sha256"], replay.sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--summary-repair-root", type=Path)
    args = parser.parse_args()
    verify(args.atoms, args.summary_repair_root)
