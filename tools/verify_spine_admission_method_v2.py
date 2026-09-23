"""Replay source admission, including one explicit bounded transport recovery.

The legacy v3/v4 receipts record whether a namespace needed compaction. They
remain distinct and immutable. This verifier checks the same conditional rule
against every original response before comparing methods across namespaces.
Version 2 counts prior unknown attempts and verifies the preserved transport
lineage. No questions, gold answers, raw Qwen inputs, or new provider calls are allowed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_quote_json_repair import repair_support_list_closures
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.admit_spine_corpus import admit
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.repair_spine_summary_budget import IMPLEMENTATION as REPAIR_IMPLEMENTATION
from tools.build_spine_corpus_hierarchy import GATEWAY, MODEL
from tools.spine_transport_lineage import TRANSPORT_POLICY, verify_transport_lineage


ADMISSION_IMPLEMENTATION = (
    "tools/admit_spine_corpus.py", "src/memory_condense/search/spine_source_admission.py",
    "src/memory_condense/search/spine_quote_json_repair.py",
)
IMPLEMENTATION = tuple(dict.fromkeys((*ADMISSION_IMPLEMENTATION, *REPAIR_IMPLEMENTATION,
    "tools/verify_spine_admission_method.py", "tools/execute_spine_corpus.py",
    "src/memory_condense/eval/fast_completion_runtime.py",
    "tools/stage_spine_transport_recovery.py", "tools/spine_transport_lineage.py",
    "tools/execute_spine_transport_recovery.py",
    "tools/probe_spine_gateway_readiness.py",
    "tools/verify_spine_admission_method_v2.py")))


def hashes(names):
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def conditional_method(policy):
    """Recognize only the exact two receipts emitted by the pinned admission code."""
    has_repairs = policy.get("summary_budget_repairs_sha256") is not None
    version = "v4" if has_repairs else "v3"
    expected = {
        "format": "memory-condense-spine-source-binding-policy-" + version,
        "corpus_preflight_sha256": policy["corpus_preflight_sha256"],
        "execution_preflight_sha256": policy["execution_preflight_sha256"],
        "summary_use": "routing only; never answer evidence",
        "mandatory_source_binding": "complete exact input fragment",
        "generated_quote_checks": "retain unchanged as diagnostics; not an entailment certificate",
        "summary_text_changes_allowed": "only authenticated over-budget summary compactions" if has_repairs else False,
        "summary_budget_repairs_sha256": policy["summary_budget_repairs_sha256"],
        "summary_entailment_verified": False,
        "syntax_repair": "missing or misplaced support-string terminators only; preserve original response and every summary byte",
        "implementation": hashes(ADMISSION_IMPLEMENTATION),
    }
    if policy != expected:
        raise ValueError("unrecognized source admission rules or implementation")
    return {
        "format": "memory-condense-conditional-source-admission-method-v2",
        "summary_use": expected["summary_use"], "mandatory_source_binding": expected["mandatory_source_binding"],
        "generated_quote_checks": expected["generated_quote_checks"], "syntax_repair": expected["syntax_repair"],
        "summary_entailment_verified": False,
        "summary_change_rule": "preserve every original summary at or below 128 tokens; compact every over-budget summary through the authenticated summary-only protocol",
        "compaction": {"model": MODEL, "gateway": GATEWAY, "max_output_tokens_per_summary": 128,
            "maximum_jobs_per_batch": 8, "maximum_provider_calls_per_namespace": 2,
            "maximum_successful_batches_per_namespace": 1,
            "maximum_additional_transport_attempts": 1, "retries": 0,
            "raw_qwen_inputs": False},
        "transport_policy": TRANSPORT_POLICY,
        "implementation": hashes(IMPLEMENTATION),
    }


def expected_repair_row(row, request, response):
    """Reconstruct the Qwen input from its attributed Terra summary, never raw text."""
    parsed, _ = repair_support_list_closures(response)
    index = int(row["label"][1:])
    atom = json.loads(parsed)["atoms"][index]
    span = request.payload["raw_spans"][index]
    if atom["label"] != row["label"]:
        raise ValueError("compaction label changed")
    job = SpineSummaryRequest("user_spine" if span["role"] == "user" else "attached_context",
        (SpineSummaryFragment(span["role"], span["created_at"], atom["summary"]),), max_output_tokens=128)
    return {"raw_request_sha256": request.sha256, "label": atom["label"],
        "original_summary_sha256": quote_sha256(atom["summary"]), "role": span["role"],
        "job": json.loads(json.dumps(asdict(job)))}


def verify(atoms_path, repair_root=None):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None or not re.fullmatch(r"offset-\d{3}", atoms_path.parent.name):
        raise ValueError("unknown source-atom artifact path")
    limit, offset = int(match[1]), int(atoms_path.parent.name[7:])
    root = atoms_path.parent.parent
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    corpus, _, requests, execution = prepare(root, offset, limit)
    policies = [read_sealed_json(p) for p in atoms_path.parent.glob(f"source-binding-policy-*-prefix-{limit:04d}.json")]
    policies = [p for p in policies if p.sha256 == data["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    policy = policies[0]
    if (data["corpus_preflight_sha256"] != corpus.sha256 or policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
            policy.payload["execution_preflight_sha256"] != execution.sha256):
        raise ValueError("source admission namespace binding changed")
    method = conditional_method(policy.payload)
    repair_sha = policy.payload["summary_budget_repairs_sha256"]
    if (repair_root is not None) != (repair_sha is not None):
        raise ValueError("supply exactly the summary-repair root bound by this admission")
    if repair_root is not None:
        repairs = read_sealed_json(repair_root / "repairs.json")
        preflight = read_sealed_json(repair_root / "preflight.json")
        p = preflight.payload
        if (repairs.sha256 != repair_sha or repairs.payload["preflight_sha256"] != preflight.sha256 or
                p["corpus_preflight_sha256"] != corpus.sha256 or p["execution_preflight_sha256"] != execution.sha256 or
                p["model"] != MODEL or p["gateway"] != GATEWAY or p["raw_qwen_inputs"] is not False or
                p["maximum_provider_calls"] != 1 or p["retries"] != 0):
            raise ValueError("summary compaction protocol or execution binding changed")
        by_sha = {request.sha256: request for request in requests}
        responses = {}
        for row in p["rows"]:
            request = by_sha[row["raw_request_sha256"]]
            if request.sha256 not in responses:
                q = request.payload
                runtime = FastCompletionRuntime(checkpoint_dir=atoms_path.parent / "raw-checkpoints" / request.sha256,
                    prompt_population=[q["messages"]], model=q["model"], client=None,
                    max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
                    benchmark_provenance={"raw_request_sha256": request.sha256})
                try:
                    responses[request.sha256] = runtime.run().logical_completions[0]
                finally:
                    runtime.close()
            if row != expected_repair_row(row, request, responses[request.sha256]):
                raise ValueError("Qwen compaction input is not the attributed original summary")
    lineage = verify_transport_lineage(root, offset, repair_root)
    # Replay is the proof, not a relabeled receipt. The original admission tool
    # authenticates every completion, applies only exact over-budget repairs,
    # checks complete raw coverage, and refuses to overwrite any changed atom,
    # quote diagnostic, or summary byte. Compaction also replays with no client.
    admit(root, offset, limit, repair_root)
    if read_sealed_json(atoms_path).sha256 != atoms.sha256:
        raise ValueError("source atoms changed during verification")
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v2-prefix-{limit:04d}.json"), {
        "format": "memory-condense-conditional-source-admission-verification-v2",
        "atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "summary_repair_root": str(repair_root.resolve()) if repair_root else None,
        "summary_budget_repairs_sha256": repair_sha, "complete_namespace": data["complete_namespace"],
        "raw_span_population_sha256": data["raw_span_population_sha256"],
        "atom_count": len(data["atoms"]), "budget_compacted_atoms": data["budget_compacted_atoms"],
        "method": method, "method_sha256": identity_sha256(method),
        "transport_lineage": lineage,
        "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
        "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "complete_namespace": data["complete_namespace"], "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    """Used by full100 gates; replay the proof instead of trusting its flags."""
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v2-prefix-{match[1]}.json"))
    p = receipt.payload
    if (p["atoms_sha256"] != atoms.sha256 or p["source_binding_policy_sha256"] != atoms.payload["source_binding_policy_sha256"] or
            not p["complete_namespace"] or not atoms.payload["complete_namespace"]):
        raise ValueError("method verification is bound to a different or partial memory")
    repair_root = Path(p["summary_repair_root"]) if p["summary_repair_root"] else None
    replay = verify(atoms_path, repair_root)
    if replay.sha256 != receipt.sha256:
        raise ValueError("conditional admission verification changed")
    return p["method_sha256"], replay.sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--summary-repair-root", type=Path)
    args = parser.parse_args()
    verify(args.atoms, args.summary_repair_root)
