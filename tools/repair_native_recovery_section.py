"""Subdivide rejected summaries in one explicit transport-recovery response."""
from collections import defaultdict
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import messages, pack, restore, admit
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_resegmentation import subdivide, reconcile_sections
from tools import recover_native_spine_transport as recovery
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized


def implementation():
    return {**recovery.implementation(), **{name: digest(name) for name in (
        "tools/repair_native_recovery_section.py", "src/memory_condense/search/native_spine_repair.py",
        "src/memory_condense/search/native_spine_resegmentation.py")}}


def load(source, ordinal):
    source = Path(source)
    result = recovery.execute(source, False)
    plan = read_sealed_json(source/"preflight.json")
    p = plan.payload
    candidates = [r for r in result.payload["rows"] if r["ordinal"] == ordinal]
    if type(ordinal) is not int or len(candidates) != 1 or candidates[0]["status"] != "invalid_summary":
        raise ValueError("section repair requires an invalid explicit recovery response")
    original_root = Path(p["source_root"])
    original_plan = read_sealed_json(original_root/"preflight.json")
    request = read_sealed_json(original_root/original_plan.payload["requests"][ordinal]["path"])
    validation = read_sealed_json(source/candidates[0]["path"])
    if validation.sha256 != candidates[0]["sha256"]:
        raise ValueError("recovery validation changed")
    runtime = FastCompletionRuntime(checkpoint_dir=source/"checkpoints"/request.sha256,
        prompt_population=[request.payload["messages"]], model=recovery.MODEL, client=None,
        max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"], max_concurrency=1, retries=0,
        benchmark_provenance={"native_transport_recovery_sha256": plan.sha256,
                              "original_source_request_sha256": request.sha256})
    try:
        response = runtime.run().logical_completions[0]
    finally:
        runtime.close()
    if quote_sha256(response) != validation.payload["response_sha256"]:
        raise ValueError("recovery response changed")
    fragments = restore(request.payload)
    valid, bad = partition(response, fragments)
    pieces, owners = [], []
    for i in bad:
        parts = subdivide(fragments[i])
        pieces.extend(parts)
        owners.extend([i]*len(parts))
    snapshot = {"recovery_result_sha256": result.sha256, "recovery_preflight_sha256": plan.sha256,
        "source_preflight_sha256": original_plan.sha256, "source_request_sha256": request.sha256,
        "sources_sha256": p["sources_sha256"], "ordinal": ordinal,
        "recovery_validation_sha256": validation.sha256, "recovery_response_sha256": quote_sha256(response),
        "pieces": [f.pointer() for f in pieces], "owners": owners,
        "unchanged_valid_atom_indices": list(valid), "replaced_original_atom_indices": list(bad)}
    return fragments, response, tuple(pieces), owners, snapshot


def prepare(source, root, ordinal):
    root = Path(root)
    _, _, pieces, _, snapshot = load(source, ordinal)
    saved, _ = publish_sealed_json(root/"source-snapshot.json", snapshot)
    bindings = []
    for i, group in enumerate(pack(pieces, max_atoms=8)):
        job, _ = publish_sealed_json(root/"requests"/f"{i:06d}.json", {
            "source_snapshot_sha256": saved.sha256, "messages": messages(group),
            "pointers": [f.pointer() for f in group]})
        bindings.append({"path": str(job.path.relative_to(root)), "sha256": job.sha256})
    plan, _ = publish_sealed_json(root/"preflight.json", {
        "source_root": str(Path(source).resolve()), "ordinal": ordinal, "source_snapshot_sha256": saved.sha256,
        "source_preflight_sha256": snapshot["source_preflight_sha256"], "sources_sha256": snapshot["sources_sha256"],
        "jobs": bindings, "maximum_new_provider_calls": len(bindings), "model": recovery.MODEL,
        "gateway": recovery.runner.compiler.GATEWAY, "max_prompt_tokens": 7000, "max_new_tokens": 2048,
        "automatic_retries": 0, "raw_inputs_to_qwen": False, "raw_text_changed": False,
        "valid_recovery_summaries_unchanged": True, "implementation": implementation(),
    })
    print({"recovery_section_preflight_sha256": plan.sha256, "new_sections": len(pieces),
           "maximum_new_provider_calls": len(bindings)}, flush=True)
    return plan


def execute(root, enable_provider=False):
    root = Path(root)
    with _phase_lock(root, "native-recovery-section-repair"):
        plan = read_sealed_json(root/"preflight.json")
        p = plan.payload
        if (p["implementation"] != implementation() or p["model"] != recovery.MODEL
                or p["gateway"] != recovery.runner.compiler.GATEWAY
                or p["max_prompt_tokens"] != 7000 or p["max_new_tokens"] != 2048):
            raise ValueError("recovery section repair policy changed")
        fragments, response, pieces, owners, snapshot = load(p["source_root"], p["ordinal"])
        saved = read_sealed_json(root/"source-snapshot.json")
        groups = tuple(pack(pieces, max_atoms=8))
        if (saved.sha256 != p["source_snapshot_sha256"] or saved.payload != snapshot
                or len(groups) != len(p["jobs"]) or len(groups) != p["maximum_new_provider_calls"]):
            raise ValueError("recovery section repair input scope changed")
        replacements, cursor, calls, hits = defaultdict(list), 0, 0, 0
        for binding, group in zip(p["jobs"], groups, strict=True):
            job = read_sealed_json(root/binding["path"])
            if (job.sha256 != binding["sha256"] or job.payload["source_snapshot_sha256"] != saved.sha256
                    or restore(job.payload) != group or job.payload["messages"] != messages(group)):
                raise ValueError("recovery section repair request changed")
            def factory(client):
                return FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/job.sha256,
                    prompt_population=[job.payload["messages"]], model=recovery.MODEL, client=client,
                    max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
                    benchmark_provenance={"native_recovery_section_request_sha256": job.sha256})
            runtime = factory(None)
            try:
                remaining = 1-len(_authenticated_records(runtime))
            finally:
                runtime.close()
            batch, new, replay, _ = _run_exactly_authorized(runtime_factory=factory,
                authorized_provider_calls=remaining, enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", p["gateway"]).with_options(timeout=240, max_retries=0))
            atoms = admit(batch.logical_completions[0], group)
            for i, atom in enumerate(atoms):
                replacements[owners[cursor+i]].append(atom)
            cursor += len(group)
            calls += new
            hits += replay
        atoms = list(reconcile_sections(response, fragments, replacements))
        result, _ = publish_sealed_json(root/"result.json", {
            "recovery_section_preflight_sha256": plan.sha256, "source_preflight_sha256": snapshot["source_preflight_sha256"],
            "source_request_sha256": snapshot["source_request_sha256"], "ordinal": p["ordinal"],
            "status": "accepted", "summaries": atoms, "original_atom_count": len(fragments),
            "unchanged_valid_atom_indices": snapshot["unchanged_valid_atom_indices"],
            "replaced_original_atom_indices": snapshot["replaced_original_atom_indices"],
            "original_journal_unchanged": True, "raw_text_changed": False,
            "complete_original_raw_coverage": True, "full100_target_passed": False,
        })
        print({"recovery_section_result_sha256": result.sha256, "new_calls": calls, "replay_hits": hits}, flush=True)
        return result
