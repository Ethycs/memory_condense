"""Separate an atom admission method from its namespace-specific receipts."""
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from tools.matched_eval.artifacts import read_sealed_json


def method_sha256(policy):
    return identity_sha256({k: v for k, v in policy.items() if k not in {
        "corpus_preflight_sha256", "execution_preflight_sha256", "summary_budget_repairs_sha256"}})


def load_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    policies = [read_sealed_json(p) for p in sorted(atoms_path.parent.glob(
        f"source-binding-policy-*-prefix-{match.group(1)}.json"))]
    policies = [p for p in policies if p.sha256 == atoms.payload["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("source admission policy is missing or ambiguous")
    policy = policies[0]
    execution = read_sealed_json(atoms_path.parent / f"execution-prefix-{match.group(1)}.json")
    if (policy.payload["execution_preflight_sha256"] != execution.sha256 or
            policy.payload["corpus_preflight_sha256"] != atoms.payload["corpus_preflight_sha256"] or
            execution.payload["corpus_preflight_sha256"] != atoms.payload["corpus_preflight_sha256"]):
        raise ValueError("admission execution or corpus binding changed")
    if atoms.payload["budget_compacted_atoms"] and not policy.payload["summary_budget_repairs_sha256"]:
        raise ValueError("compacted atoms lack their repair binding")
    return method_sha256(policy.payload), policy.sha256
