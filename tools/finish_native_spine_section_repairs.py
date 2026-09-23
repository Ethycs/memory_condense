"""Refine only unresolved source sections, retaining every accepted prior atom."""
import argparse
import hashlib
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import messages, pack, restore
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_resegmentation import subdivide
from tools import resegment_native_spine_repairs as stage
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized


def implementation():
    return {**stage.implementation(), "tools/finish_native_spine_section_repairs.py":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def expand(pieces, owners, accepted):
    if len(pieces) != len(owners) or any(type(i) is not int or not 0 <= i < len(pieces) for i in accepted):
        raise ValueError("accepted section population changed")
    expanded, expanded_owners, ready, needed = [], [], {}, []
    for index, (piece, owner) in enumerate(zip(pieces, owners, strict=True)):
        if index in accepted:
            if accepted[index]["pointer"] != piece.pointer():
                raise ValueError("accepted raw section changed")
            ready[len(expanded)] = accepted[index]
            expanded.append(piece)
            expanded_owners.append(owner)
        else:
            parts = subdivide(piece)
            needed.extend(range(len(expanded), len(expanded)+len(parts)))
            expanded.extend(parts)
            expanded_owners.extend(owner for _ in parts)
    return tuple(expanded), tuple(expanded_owners), ready, tuple(needed)


def load(source):
    result = stage.execute(source, False)
    p = read_sealed_json(source / "preflight.json")
    _, originals, kept, pieces, owners, _ = stage.inputs(Path(p.payload["prior_root"]))
    accepted, cursor = {}, 0
    for binding in p.payload["jobs"]:
        job = read_sealed_json(source / binding["path"])
        if job.sha256 != binding["sha256"]:
            raise ValueError("prior section request changed")
        group = restore(job.payload)
        runtime = FastCompletionRuntime(
            checkpoint_dir=source / "checkpoints" / job.sha256, prompt_population=[job.payload["messages"]],
            model=stage.prior.MODEL, client=None, max_prompt_tokens=p.payload["max_prompt_tokens"],
            max_new_tokens=p.payload["max_new_tokens"], max_concurrency=1, retries=0,
            benchmark_provenance={"native_resegmentation_request_sha256": job.sha256},
        )
        try:
            batch = runtime.run()
        finally:
            runtime.close()
        try:
            valid, _ = partition(batch.logical_completions[0], group)
            accepted.update({cursor+i: a for i, a in valid.items()})
        except (ValueError, TypeError, KeyError):
            pass
        cursor += len(group)
    if (len(accepted) != result.payload["accepted_new_sections"]
            or sorted(set(range(len(pieces)))-set(accepted)) != result.payload["invalid_new_section_positions"]):
        raise ValueError("previous accepted section population changed")
    expanded, owners, ready, needed = expand(pieces, owners, accepted)
    snapshot = {"source_preflight_sha256": p.sha256, "source_result_sha256": result.sha256,
                "pieces": [f.pointer() for f in expanded], "owners": [list(o) for o in owners],
                "ready": [{"position": i, "atom": a} for i, a in sorted(ready.items())],
                "needed_positions": list(needed)}
    return originals, kept, expanded, owners, ready, needed, snapshot


def prepare(source, root):
    _, _, pieces, _, ready, needed, payload = load(source)
    snapshot, _ = publish_sealed_json(root / "source-snapshot.json", payload)
    jobs = []
    for index, group in enumerate(pack((pieces[i] for i in needed), max_atoms=8)):
        job, _ = publish_sealed_json(root / "requests" / f"{index:06}.json", {
            "messages": messages(group), "pointers": [f.pointer() for f in group],
            "source_snapshot_sha256": snapshot.sha256,
        })
        jobs.append({"path": str(job.path.relative_to(root)), "sha256": job.sha256})
    plan, _ = publish_sealed_json(root / "preflight.json", {
        "source_root": str(source.resolve()), "source_snapshot_sha256": snapshot.sha256,
        "jobs": jobs, "model": stage.prior.MODEL, "gateway": stage.prior.compiler.GATEWAY,
        "maximum_new_provider_calls": len(jobs), "concurrency": 1, "retries": 0,
        "max_prompt_tokens": 7000, "max_new_tokens": 2048, "raw_inputs_to_qwen": False,
        "implementation": implementation(),
    })
    print({"refinement_preflight_sha256": plan.sha256, "kept_sections": len(ready),
           "new_sections": len(needed), "maximum_new_provider_calls": len(jobs)}, flush=True)


def execute(root, enable_provider=False):
    with _phase_lock(root, "native-section-refinement"):
        plan = read_sealed_json(root / "preflight.json")
        p = plan.payload
        if (p["implementation"] != implementation() or p["model"] != stage.prior.MODEL
                or p["gateway"] != stage.prior.compiler.GATEWAY or p["raw_inputs_to_qwen"] is not False):
            raise ValueError("section refinement policy changed")
        originals, kept, pieces, owners, ready, needed, payload = load(Path(p["source_root"]))
        snapshot = read_sealed_json(root / "source-snapshot.json")
        if snapshot.sha256 != p["source_snapshot_sha256"] or snapshot.payload != payload:
            raise ValueError("section refinement input changed")
        jobs, observed = [], []
        for binding in p["jobs"]:
            job = read_sealed_json(root / binding["path"])
            if job.sha256 != binding["sha256"] or job.payload["source_snapshot_sha256"] != snapshot.sha256:
                raise ValueError("section refinement request changed")
            group = restore(job.payload)
            observed.extend(group)
            jobs.append((job, group))
        if tuple(observed) != tuple(pieces[i] for i in needed) or len(jobs) != p["maximum_new_provider_calls"]:
            raise ValueError("section refinement would send changed raw content")
        accepted = dict(ready)
        cursor = calls = hits = 0
        for job, group in jobs:
            def factory(client):
                return FastCompletionRuntime(
                    checkpoint_dir=root / "checkpoints" / job.sha256,
                    prompt_population=[job.payload["messages"]], model=p["model"], client=client,
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    max_concurrency=1, retries=0,
                    benchmark_provenance={"native_refinement_request_sha256": job.sha256},
                )
            audit = factory(None)
            try:
                remaining = 1-len(_authenticated_records(audit))
            finally:
                audit.close()
            batch, new, replay, _ = _run_exactly_authorized(
                runtime_factory=factory, authorized_provider_calls=remaining, enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", p["gateway"]).with_options(timeout=240, max_retries=0),
            )
            try:
                valid, _ = partition(batch.logical_completions[0], group)
                accepted.update({needed[cursor+i]: a for i, a in valid.items()})
            except (ValueError, TypeError, KeyError):
                pass
            cursor += len(group)
            calls += new
            hits += replay
        complete, pending = stage.collect(originals, kept, pieces, owners, accepted)
        batches = []
        parent = read_sealed_json(Path(p["source_root"]) / "source-snapshot.json")
        for n, summaries in complete.items():
            row = originals[n]
            result, _ = publish_sealed_json(root / "admitted-batches" / f"{n:06}.json", {
                "refinement_preflight_sha256": plan.sha256,
                "source_preflight_sha256": parent.payload["source_preflight_sha256"],
                "source_request_sha256": row[0].sha256, "source_validation_sha256": row[1].sha256,
                "ordinal": n, "summaries": summaries, "original_atom_count": len(row[2]),
                "unchanged_valid_atom_indices": list(row[4]), "replaced_original_atom_indices": list(row[5]),
                "complete_original_raw_coverage": True, "raw_text_changed": False,
                "summary_entailment_verified": False,
            })
            batches.append({"path": str(result.path.relative_to(root)), "sha256": result.sha256})
        result, _ = publish_sealed_json(root / "result.json", {
            "refinement_preflight_sha256": plan.sha256, "admitted_original_batches": batches,
            "unresolved_original_batches": pending, "kept_prior_sections": len(ready),
            "accepted_new_sections": len(accepted)-len(ready), "complete_repair_snapshot": not pending,
            "full_source_compilation_complete": False, "full100_target_passed": False,
        })
        print({"refinement_result_sha256": result.sha256, "complete_original_batches": len(batches),
               "pending_original_batches": pending, "new_calls": calls, "replay_hits": hits}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.source_root, args.output_root)
    else:
        execute(args.output_root, args.enable_provider)
