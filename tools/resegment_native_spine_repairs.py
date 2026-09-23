"""Recover dense-list summaries by splitting exact failed source fragments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, messages, pack, restore
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_resegmentation import reconcile_sections, subdivide
from tools import repair_native_spine_batches as prior
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import (
    _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized,
)


FILES = ("tools/resegment_native_spine_repairs.py", "src/memory_condense/search/native_spine_resegmentation.py")


def implementation():
    return {**prior.implementation(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                       for name in FILES}}


def inputs(prior_root):
    plan = read_sealed_json(prior_root / "preflight.json")
    p = plan.payload
    if p["implementation"] != prior.implementation() or p["model"] != prior.MODEL:
        raise ValueError("original repair implementation or model changed")
    source = Path(p["source_root"])
    parent = read_sealed_json(source / "preflight.json")
    if parent.sha256 != p["source_preflight_sha256"]:
        raise ValueError("full source compilation changed")
    originals, snapshots = {}, {}
    for binding in p["snapshots"]:
        snapshot = read_sealed_json(prior_root / binding["path"])
        if snapshot.sha256 != binding["sha256"]:
            raise ValueError("failed-batch snapshot changed")
        n = snapshot.payload["ordinal"]
        row = prior.original(source, parent, n)
        if (row[0].sha256 != snapshot.payload["source_request_sha256"]
                or row[1].sha256 != snapshot.payload["source_validation_sha256"]
                or list(row[5]) != snapshot.payload["repair_atom_indices"]
                or list(row[4]) != snapshot.payload["unchanged_atom_indices"]):
            raise ValueError("failed original atoms changed")
        originals[n], snapshots[n] = row, snapshot
    kept, pieces, owners, proofs = {}, [], [], []
    expected = {(n, i) for n, row in originals.items() for i in row[5]}
    for binding in p["jobs"]:
        job = read_sealed_json(prior_root / binding["path"])
        j = job.payload
        pair = (j["ordinal"], j["atom_index"])
        if job.sha256 != binding["sha256"] or pair not in expected:
            raise ValueError("prior repair job population changed")
        expected.remove(pair)
        fragment = originals[pair[0]][2][pair[1]]
        if (j["original_snapshot_sha256"] != snapshots[pair[0]].sha256
                or j["pointer"] != fragment.pointer()
                or j["messages"] != prior.repair_messages(fragment)):
            raise ValueError("prior repair source changed")
        runtime = FastCompletionRuntime(
            checkpoint_dir=prior_root / "checkpoints" / job.sha256,
            prompt_population=[j["messages"]], model=prior.MODEL, client=None,
            max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
            max_concurrency=1, retries=0,
            benchmark_provenance={"native_repair_request_sha256": job.sha256},
        )
        try:
            batch = runtime.run()
        finally:
            runtime.close()
        response = batch.logical_completions[0]
        proofs.append({"job_sha256": job.sha256,
                       "response_journal_sha256": batch.unique_records[0].response_journal_sha256})
        try:
            kept[pair] = admit(response, (fragment,))
        except (ValueError, TypeError, KeyError):
            split = subdivide(fragment)
            pieces.extend(split)
            owners.extend(pair for _ in split)
    if expected:
        raise ValueError("prior repair population is incomplete")
    snapshot = {
        "prior_preflight_sha256": plan.sha256, "source_preflight_sha256": parent.sha256,
        "source_root": str(source), "prior_response_receipts": proofs,
        "kept_repairs": [{"ordinal": n, "atom_index": i, "summaries": list(atoms)}
                         for (n, i), atoms in sorted(kept.items())],
        "replacement_sections": [{"ordinal": n, "atom_index": i, "pointer": f.pointer()}
                                 for (n, i), f in zip(owners, pieces, strict=True)],
    }
    return plan, originals, kept, tuple(pieces), tuple(owners), snapshot


def prepare(prior_root, root):
    _, _, kept, pieces, _, payload = inputs(prior_root)
    snapshot, _ = publish_sealed_json(root / "source-snapshot.json", payload)
    jobs = []
    for index, group in enumerate(pack(pieces, max_atoms=8)):
        request, _ = publish_sealed_json(root / "requests" / f"{index:06}.json", {
            "ordinal": index, "source_snapshot_sha256": snapshot.sha256,
            "messages": messages(group), "pointers": [f.pointer() for f in group],
        })
        jobs.append({"path": str(request.path.relative_to(root)), "sha256": request.sha256})
    plan, _ = publish_sealed_json(root / "preflight.json", {
        "format": "native-spine-exact-resegmentation-v1", "prior_root": str(prior_root.resolve()),
        "source_snapshot_sha256": snapshot.sha256, "jobs": jobs, "model": prior.MODEL,
        "gateway": prior.compiler.GATEWAY, "maximum_new_provider_calls": len(jobs),
        "replacement_section_count": len(pieces), "kept_prior_repairs": len(kept),
        "raw_inputs_to_qwen": False, "max_prompt_tokens": 7000, "max_new_tokens": 2048,
        "concurrency": 1, "retries": 0, "implementation": implementation(),
        "source_text_or_valid_summary_changes_allowed": False,
    })
    print({"resegmentation_preflight_sha256": plan.sha256, "kept_repairs": len(kept),
           "replacement_sections": len(pieces), "maximum_new_provider_calls": len(jobs)}, flush=True)
    return plan


def collect(originals, kept, pieces, owners, accepted):
    """Admit every complete original batch independently of any other failure."""
    if (len(pieces) != len(owners) or any(type(pos) is not int or not 0 <= pos < len(pieces)
            or atom["pointer"] != pieces[pos].pointer() for pos, atom in accepted.items())):
        raise ValueError("accepted subsections escaped their prepared source population")
    replacements = defaultdict(dict)
    for (n, i), atoms in kept.items():
        replacements[n][i] = atoms
    positions = defaultdict(list)
    for pos, pair in enumerate(owners):
        positions[pair].append(pos)
    for (n, i), indices in positions.items():
        if all(pos in accepted for pos in indices):
            replacements[n][i] = tuple(accepted[pos] for pos in indices)
    complete, pending = {}, []
    for n, row in originals.items():
        missing = set(row[5]) - set(replacements[n])
        if missing:
            pending.append({"ordinal": n, "unresolved_original_atom_indices": sorted(missing)})
        else:
            complete[n] = reconcile_sections(row[3], row[2], replacements[n])
    return complete, pending


def execute(root, enable_provider=False):
    with _phase_lock(root, "native-resegmentation"):
        plan = read_sealed_json(root / "preflight.json")
        p = plan.payload
        if (p["implementation"] != implementation() or p["model"] != prior.MODEL
                or p["gateway"] != prior.compiler.GATEWAY or p["raw_inputs_to_qwen"] is not False):
            raise ValueError("subdivision policy changed")
        _, originals, kept, pieces, owners, payload = inputs(Path(p["prior_root"]))
        snapshot = read_sealed_json(root / "source-snapshot.json")
        if snapshot.sha256 != p["source_snapshot_sha256"] or snapshot.payload != payload:
            raise ValueError("subdivision source or prior outputs changed")
        loaded, restored = [], []
        for binding in p["jobs"]:
            job = read_sealed_json(root / binding["path"])
            if job.sha256 != binding["sha256"] or job.payload["source_snapshot_sha256"] != snapshot.sha256:
                raise ValueError("subdivision request changed")
            group = restore(job.payload)
            loaded.append((job, group))
            restored.extend(group)
        if tuple(restored) != pieces or len(loaded) != p["maximum_new_provider_calls"]:
            raise ValueError("subdivision request population changed")
        accepted, invalid = {}, []
        cursor = calls = hits = 0
        for job, group in loaded:
            def factory(client):
                return FastCompletionRuntime(
                    checkpoint_dir=root / "checkpoints" / job.sha256,
                    prompt_population=[job.payload["messages"]], model=prior.MODEL, client=client,
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    max_concurrency=1, retries=0,
                    benchmark_provenance={"native_resegmentation_request_sha256": job.sha256},
                )
            audit = factory(None)
            try:
                remaining = 1-len(_authenticated_records(audit))
            finally:
                audit.close()
            batch, new, replay, _ = _run_exactly_authorized(
                runtime_factory=factory, authorized_provider_calls=remaining,
                enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", prior.compiler.GATEWAY).with_options(
                    timeout=240, max_retries=0),
            )
            try:
                valid, bad = partition(batch.logical_completions[0], group)
                accepted.update({cursor+i: atom for i, atom in valid.items()})
                invalid.extend(cursor+i for i in bad)
            except (ValueError, TypeError, KeyError):
                invalid.extend(range(cursor, cursor+len(group)))
            cursor += len(group)
            calls += new
            hits += replay
        complete, pending = collect(originals, kept, pieces, owners, accepted)
        batches = []
        for n, summaries in complete.items():
            row = originals[n]
            result, _ = publish_sealed_json(root / "admitted-batches" / f"{n:06}.json", {
                "resegmentation_preflight_sha256": plan.sha256,
                "source_preflight_sha256": payload["source_preflight_sha256"],
                "source_request_sha256": row[0].sha256, "source_validation_sha256": row[1].sha256,
                "ordinal": n, "summaries": summaries, "original_atom_count": len(row[2]),
                "unchanged_valid_atom_indices": list(row[4]), "replaced_original_atom_indices": list(row[5]),
                "complete_original_raw_coverage": True, "raw_text_changed": False,
                "summary_entailment_verified": False,
            })
            batches.append({"path": str(result.path.relative_to(root)), "sha256": result.sha256})
        result, _ = publish_sealed_json(root / "result.json", {
            "resegmentation_preflight_sha256": plan.sha256,
            "admitted_original_batches": batches, "unresolved_original_batches": pending,
            "accepted_new_sections": len(accepted), "invalid_new_section_positions": invalid,
            "kept_prior_repairs": len(kept), "complete_repair_snapshot": not pending,
            "full_source_compilation_complete": False, "full100_target_passed": False,
        })
        print({"resegmentation_result_sha256": result.sha256, "complete_original_batches": len(batches),
               "pending_original_batches": pending, "new_calls": calls, "replay_hits": hits}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.prior_root, args.output_root)
    else:
        execute(args.output_root, args.enable_provider)
