"""Verify complete source admission including a separate compaction-only reissue."""
import argparse
import hashlib
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from tools import verify_spine_admission_method_v8 as previous
from tools import spine_compaction_transport as transport
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


IMPLEMENTATION = (*previous.IMPLEMENTATION, "tools/spine_compaction_transport.py",
    "tools/verify_spine_admission_method_v9.py")


def normalize_method(method):
    method = {**method, "compaction": dict(method["compaction"])}
    method["format"] = "memory-condense-conditional-source-admission-method-v9"
    method["compaction"]["separate_transport_stage"] = (
        "at most one additional compaction attempt; authenticate every original and successor; "
        "preserve successful responses and the original unresolved reservation; same request protocol")
    method["implementation"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}
    return method


def conditional_method(policy):
    return normalize_method(previous.conditional_method(policy))


def verify(atoms_path, repair_root=None, transport_root=None):
    atoms_path = Path(atoms_path)
    declared = transport.declared_stage(repair_root)
    if (declared is None) != (transport_root is None) or (
            declared is not None and declared != Path(transport_root).resolve()):
        raise ValueError("the declared compaction transport stage must be supplied exactly")
    prior = previous.verify(atoms_path, repair_root)
    p = prior.payload
    extra = transport.verify(transport_root, repair_root, p) if transport_root is not None else None
    if extra is not None and p["transport_lineage"]["additional_compaction_attempts"]:
        raise ValueError("two transport recoveries exceed the namespace compaction allowance")
    count = p["compaction_provider_attempts"] + (extra["additional_compaction_attempts"] if extra else 0)
    method = normalize_method(p["method"])
    limit = int(re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)[1])
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v9-prefix-{limit:04d}.json"), {
        **p, "format": "memory-condense-conditional-source-admission-verification-v9",
        "predecessor_verification_sha256": prior.sha256,
        "compaction_transport_root": str(Path(transport_root).resolve()) if transport_root else None,
        "compaction_only_transport": extra, "compaction_provider_attempts": count,
        "method": method, "method_sha256": identity_sha256(method)})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "complete_namespace": p["complete_namespace"], "compaction_provider_attempts": count,
        "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v9-prefix-{match[1]}.json"))
    p = receipt.payload
    if (p["atoms_sha256"] != atoms.sha256 or
            p["source_binding_policy_sha256"] != atoms.payload["source_binding_policy_sha256"] or
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
