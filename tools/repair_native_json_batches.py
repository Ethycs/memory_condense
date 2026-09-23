"""Recover malformed native batch JSON with fresh, separately recorded raw-section calls."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, messages, pack, restore
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_summary import BodyFragment
from tools import finish_native_spine_section_repairs as refinement
from tools import repair_native_spine_batches as original_repair
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import (
    _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized,
)


MODEL = original_repair.MODEL
GATEWAY = original_repair.compiler.GATEWAY


def implementation():
    return {**refinement.implementation(), "tools/repair_native_json_batches.py": digest(__file__)}


def malformed_original(source, preflight, ordinal):
    p = preflight.payload
    binding = p["requests"][ordinal]
    request = read_sealed_json(source / binding["path"])
    if request.sha256 != binding["sha256"] or request.payload["ordinal"] != ordinal:
        raise ValueError("original raw request changed")
    fragments = restore(request.payload)
    key = identity_sha256({"request_sha256": request.sha256, "model": MODEL})
    validated = read_sealed_json(source / "validated" / f"{key}.json")
    v = validated.payload
    if (v["preflight_sha256"] != preflight.sha256 or v["request_sha256"] != request.sha256
            or v["model"] != MODEL or v["status"] != "invalid_summary"):
        raise ValueError("only terminally rejected original batches may be repaired")
    runtime = FastCompletionRuntime(
        checkpoint_dir=source / "checkpoints" / key, prompt_population=[request.payload["messages"]],
        model=MODEL, client=None, max_prompt_tokens=p["max_prompt_tokens"],
        max_new_tokens=p["max_new_tokens"], max_concurrency=1, retries=0,
        benchmark_provenance={"native_compile_request_sha256": request.sha256},
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    response = batch.logical_completions[0]
    if quote_sha256(response) != v["response_sha256"]:
        raise ValueError("original response changed")
    try:
        partition(response, fragments)
    except json.JSONDecodeError:
        # No parsed atom list was admitted from this source response. Preserve
        # its bytes and request a new, explicitly attributed summary population.
        return request, validated, fragments, response, {}, tuple(range(len(fragments)))
    raise ValueError("JSON recovery requires a terminal malformed-JSON response")


def collect(originals, pieces, owners, accepted):
    """Validate new coverage without parsing or rewriting the malformed response."""
    if (len(pieces) != len(owners) or any(type(pos) is not int or not 0 <= pos < len(pieces)
            or atom["pointer"] != pieces[pos].pointer() for pos, atom in accepted.items())):
        raise ValueError("JSON repair sections escaped their prepared source population")
    positions = defaultdict(list)
    for pos, (ordinal, index) in enumerate(owners):
        if ordinal not in originals or not 0 <= index < len(originals[ordinal][2]):
            raise ValueError("JSON repair owner escaped its original fragment population")
        positions[ordinal, index].append(pos)
    complete, pending = {}, []
    for ordinal, row in originals.items():
        missing = [i for i in range(len(row[2])) if not positions[ordinal, i]
                   or any(pos not in accepted for pos in positions[ordinal, i])]
        if missing:
            pending.append({"ordinal": ordinal, "unresolved_original_atom_indices": missing})
            continue
        atoms = []
        for index, original in enumerate(row[2]):
            cursor = original.start_char
            for pos in positions[ordinal, index]:
                atom = accepted[pos]
                pointer = atom["pointer"]
                end = pointer["end_char"]
                if type(end) is not int or not cursor < end <= original.end_char:
                    raise ValueError("JSON repair changed original fragment coverage")
                piece = BodyFragment(original.body_sha256, original.turn_ordinal, original.role,
                    cursor, end, original.turn_text_sha256,
                    original.text[cursor-original.start_char:end-original.start_char])
                if piece.pointer() != pointer:
                    raise ValueError("JSON repair raw pointer changed")
                validated = admit(canonical_json({"atoms": [{"label": "T0", "summary": atom["summary"]}]}),
                                  (piece,))[0]
                if validated != atom:
                    raise ValueError("JSON repair contains unsupported fields")
                atoms.append(atom)
                cursor = end
            if cursor != original.end_char:
                raise ValueError("JSON repair dropped an original fragment suffix")
        complete[ordinal] = tuple(atoms)
    return complete, pending


def inputs(source, ordinals, previous=None):
    parent = read_sealed_json(source/"preflight.json")
    p = parent.payload
    if (p["implementation"] != original_repair.compiler.implementation() or p["models"] != [MODEL]
            or p["gateway"] != GATEWAY or p["raw_inputs_to_qwen"] is not False
            or not ordinals or len(set(ordinals)) != len(ordinals)
            or any(type(n) is not int or not 0 <= n < len(p["requests"]) for n in ordinals)):
        raise ValueError("direct repair requires explicit rejected ordinals from the unchanged compiler")
    ordinals = tuple(sorted(ordinals))
    originals = {n: malformed_original(source, parent, n) for n in ordinals}
    previous_sha = None
    if previous is None:
        pieces = tuple(row[2][i] for row in originals.values() for i in row[5])
        owners = tuple((n, i) for n, row in originals.items() for i in row[5])
        accepted = {}
    else:
        prior = read_sealed_json(previous/"preflight.json")
        if (prior.payload["source_preflight_sha256"] != parent.sha256
                or tuple(prior.payload["ordinals"]) != ordinals
                or Path(prior.payload["source_root"]).resolve() != source.resolve()):
            raise ValueError("refinement must retain the same rejected original population")
        result, state = _execute(previous, False)
        previous_sha = result.sha256
        pieces, owners, accepted = state
        if len(accepted) == len(pieces):
            raise ValueError("a complete repair needs no further subdivision")
    if previous is None:
        ready, needed = {}, tuple(range(len(pieces)))
    else:
        pieces, owners, ready, needed = refinement.expand(pieces, owners, accepted)
    snapshot = {
        "source_preflight_sha256": parent.sha256, "previous_result_sha256": previous_sha,
        "originals": [{"ordinal": n, "request_sha256": r[0].sha256, "validation_sha256": r[1].sha256,
            "response_sha256": quote_sha256(r[3]), "valid_atom_indices": list(r[4]),
            "invalid_atom_indices": list(r[5])} for n, r in originals.items()],
        "pieces": [f.pointer() for f in pieces], "owners": [list(pair) for pair in owners],
        "ready": [{"position": i, "atom": atom} for i, atom in sorted(ready.items())],
        "needed_positions": list(needed),
    }
    return parent, originals, pieces, owners, ready, needed, snapshot


def prepare(source, root, ordinals, *, previous=None):
    source, root = Path(source), Path(root)
    previous = None if previous is None else Path(previous)
    if previous is not None and previous.resolve() == root.resolve():
        raise ValueError("refinement must use a distinct output root")
    parent, originals, pieces, _, ready, needed, payload = inputs(source, ordinals, previous)
    snapshot, _ = publish_sealed_json(root/"source-snapshot.json", payload)
    jobs = []
    for ordinal, group in enumerate(pack((pieces[i] for i in needed), max_atoms=8)):
        job, _ = publish_sealed_json(root/"requests"/f"{ordinal:06d}.json", {
            "source_snapshot_sha256": snapshot.sha256, "messages": messages(group),
            "pointers": [f.pointer() for f in group],
        })
        jobs.append({"path": str(job.path.relative_to(root)), "sha256": job.sha256})
    plan, _ = publish_sealed_json(root/"preflight.json", {
        "format": "native-spine-malformed-json-repair-v1", "source_root": str(source.resolve()),
        "source_preflight_sha256": parent.sha256, "sources_sha256": parent.payload["sources_sha256"],
        "previous_root": str(previous.resolve()) if previous is not None else None,
        "ordinals": sorted(originals), "source_snapshot_sha256": snapshot.sha256, "jobs": jobs,
        "model": MODEL, "gateway": GATEWAY, "max_prompt_tokens": 7000, "max_new_tokens": 2048,
        "concurrency": 1, "retries": 0, "maximum_new_provider_calls": len(jobs),
        "original_valid_summaries_unchanged": True, "raw_text_changed": False,
        "raw_inputs_to_qwen": False, "implementation": implementation(),
    })
    print({"json_repair_preflight_sha256": plan.sha256, "original_batches": len(originals),
        "unchanged_valid_summaries": sum(len(r[4]) for r in originals.values()),
        "kept_prior_sections": len(ready), "new_sections": len(needed),
        "maximum_new_provider_calls": len(jobs)}, flush=True)
    return plan


def _execute(root, enable_provider):
    root = Path(root)
    with _phase_lock(root, "native-json-batch-repair"):
        plan = read_sealed_json(root/"preflight.json")
        p = plan.payload
        if (p.get("format") != "native-spine-malformed-json-repair-v1"
                or p["implementation"] != implementation() or p["model"] != MODEL or p["gateway"] != GATEWAY
                or p["raw_inputs_to_qwen"] is not False or p["raw_text_changed"] is not False
                or p["original_valid_summaries_unchanged"] is not True):
            raise ValueError("direct repair policy changed")
        parent, originals, pieces, owners, ready, needed, payload = inputs(Path(p["source_root"]),
            p["ordinals"], Path(p["previous_root"]) if p["previous_root"] else None)
        snapshot = read_sealed_json(root/"source-snapshot.json")
        if (parent.sha256 != p["source_preflight_sha256"] or snapshot.sha256 != p["source_snapshot_sha256"]
                or snapshot.payload != payload):
            raise ValueError("direct repair source population or saved accepted sections changed")
        groups = tuple(pack((pieces[i] for i in needed), max_atoms=8))
        if len(p["jobs"]) != len(groups) or len(groups) != p["maximum_new_provider_calls"]:
            raise ValueError("direct repair request population changed")
        jobs = []
        for binding, group in zip(p["jobs"], groups, strict=True):
            job = read_sealed_json(root/binding["path"])
            if (job.sha256 != binding["sha256"] or job.payload["source_snapshot_sha256"] != snapshot.sha256
                    or restore(job.payload) != group or job.payload["messages"] != messages(group)):
                raise ValueError("direct repair would send changed or additional source text")
            jobs.append((job, group))
        accepted, cursor, calls, hits = dict(ready), 0, 0, 0
        for job, group in jobs:
            def factory(client):
                return FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/job.sha256,
                    prompt_population=[job.payload["messages"]], model=MODEL, client=client,
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    max_concurrency=1, retries=0,
                    benchmark_provenance={"native_json_batch_repair_sha256": job.sha256})
            audit = factory(None)
            try:
                remaining = 1-len(_authenticated_records(audit))
            finally:
                audit.close()
            batch, new, replay, _ = _run_exactly_authorized(runtime_factory=factory,
                authorized_provider_calls=remaining, enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY).with_options(
                    timeout=240, max_retries=0))
            try:
                valid, _ = partition(batch.logical_completions[0], group)
                accepted.update({needed[cursor+i]: atom for i, atom in valid.items()})
            except (ValueError, TypeError, KeyError):
                # Structurally unattributable responses admit none of this
                # request's pieces; accepted earlier requests remain intact.
                pass
            cursor += len(group)
            calls += new
            hits += replay
            print({"direct_repair_sections_processed": cursor, "prepared_sections": len(needed),
                   "accepted_sections": len(accepted), "new_calls": calls}, flush=True)
        complete, pending = collect(originals, pieces, owners, accepted)
        batches = []
        for ordinal, atoms in complete.items():
            row = originals[ordinal]
            artifact, _ = publish_sealed_json(root/"admitted-batches"/f"{ordinal:06d}.json", {
                "json_repair_preflight_sha256": plan.sha256, "source_preflight_sha256": parent.sha256,
                "source_request_sha256": row[0].sha256, "source_validation_sha256": row[1].sha256,
                "ordinal": ordinal, "summaries": atoms, "original_atom_count": len(row[2]),
                "unchanged_valid_atom_indices": list(row[4]), "replaced_original_atom_indices": list(row[5]),
                "complete_original_raw_coverage": True, "raw_text_changed": False,
                "summary_entailment_verified": False,
            })
            batches.append({"path": str(artifact.path.relative_to(root)), "sha256": artifact.sha256})
        result, _ = publish_sealed_json(root/"result.json", {
            "json_repair_preflight_sha256": plan.sha256, "source_preflight_sha256": parent.sha256,
            "admitted_original_batches": batches, "unresolved_original_batches": pending,
            "accepted_section_positions": sorted(accepted),
            "invalid_section_positions": sorted(set(range(len(pieces)))-set(accepted)),
            "kept_prior_sections": len(ready), "accepted_new_sections": len(accepted)-len(ready),
            "complete_repair_snapshot": not pending, "full_source_compilation_complete": False,
            "full100_target_passed": False,
        })
        print({"direct_repair_result_sha256": result.sha256, "complete_original_batches": len(batches),
               "pending_original_batches": len(pending), "new_calls": calls, "replay_hits": hits}, flush=True)
        return result, (pieces, owners, accepted)


def execute(root, enable_provider=False):
    return _execute(root, enable_provider)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.source_root, args.output_root, args.batch, previous=args.previous_root)
    else:
        execute(args.output_root, args.enable_provider)
