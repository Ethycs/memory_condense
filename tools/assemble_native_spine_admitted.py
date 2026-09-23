"""Assemble complete native bodies from original receipts and verified repairs."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, restore
from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_repair import partition
from memory_condense.search.native_spine_resegmentation import reconcile_sections
from tools import finish_native_spine_section_repairs as refinement
from tools import run_native_spine_batches as runner
from tools.assemble_native_spine_summaries import FILES as STORAGE_FILES, SummaryBodies, digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


MODEL = refinement.stage.prior.MODEL


def implementation():
    return {**refinement.implementation(), **{name: digest(name) for name in (
        *STORAGE_FILES, "tools/run_native_spine_batches.py", "tools/assemble_native_spine_admitted.py",
    )}}


class AdmittedSummaryBodies(SummaryBodies):
    """Summary-only cache with its own repair-aware producer identity."""

    def __init__(self, root):
        self.manifest = read_sealed_json(Path(root) / "summary-bodies.json")
        p = self.manifest.payload
        snapshot = read_sealed_json(Path(root) / "admission-snapshot.json")
        if (p["format"] != "native-spine-admitted-body-store-v1"
                or p["implementation"] != implementation()
                or snapshot.sha256 != p["admission_snapshot_sha256"]):
            raise ValueError("admitted summary storage implementation or snapshot changed")
        path = (Path(root) / "summary-bodies.sqlite").resolve()
        if digest(path) != p["database_sha256"]:
            raise ValueError("admitted summary database changed")
        self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def verified_repairs(roots, preflight, model):
    overrides, receipts = {}, []
    for root in roots:
        root = Path(root)
        # Reconstruct the complete chain from normal, authenticated responses.
        # Provider access is deliberately unavailable throughout admission.
        result = refinement.execute(root, False)
        plan = read_sealed_json(root / "preflight.json")
        if plan.payload["model"] != model:
            raise ValueError("repair used a different raw summarizer")
        receipts.append({"root": str(root.resolve()), "result_sha256": result.sha256})
        for binding in result.payload["admitted_original_batches"]:
            path = (root / binding["path"]).resolve()
            path.relative_to((root / "admitted-batches").resolve())
            row = read_sealed_json(path)
            p = row.payload
            ordinal = p["ordinal"]
            if (row.sha256 != binding["sha256"] or type(ordinal) is not int
                    or not 0 <= ordinal < len(preflight.payload["requests"])
                    or p["source_preflight_sha256"] != preflight.sha256
                    or p["refinement_preflight_sha256"] != plan.sha256
                    or p["complete_original_raw_coverage"] is not True
                    or p["raw_text_changed"] is not False):
                raise ValueError("repair does not bind the original compilation")
            if ordinal in overrides:
                raise ValueError("duplicate repaired batch; choose one repair lineage")
            overrides[ordinal] = row
    return overrides, receipts


def repaired_atoms(response, fragments, replacement):
    """Recheck coverage and preserve valid original strings at the boundary."""
    valid, bad = partition(response, fragments)
    atoms, cursor, groups = replacement["summaries"], 0, {}
    if (replacement["original_atom_count"] != len(fragments)
            or replacement["unchanged_valid_atom_indices"] != list(valid)
            or replacement["replaced_original_atom_indices"] != list(bad)):
        raise ValueError("repair changed its original atom population")
    for i, fragment in enumerate(fragments):
        if i in valid:
            if cursor >= len(atoms) or atoms[cursor] != valid[i]:
                raise ValueError("repair changed an originally valid summary")
            cursor += 1
            continue
        parts = []
        while cursor < len(atoms):
            atom = atoms[cursor]
            parts.append(atom)
            cursor += 1
            if atom["pointer"]["end_char"] >= fragment.end_char:
                break
        groups[i] = parts
    if cursor != len(atoms) or list(reconcile_sections(response, fragments, groups)) != atoms:
        raise ValueError("repair changed the original raw coverage")
    return atoms


def replay_atoms(root, preflight, model, request, validation, repair=None):
    p, v = preflight.payload, validation.payload
    key = identity_sha256({"request_sha256": request.sha256, "model": model})
    runtime = FastCompletionRuntime(
        checkpoint_dir=root / "checkpoints" / key,
        prompt_population=[request.payload["messages"]], model=model, client=None,
        max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
        max_concurrency=1, retries=0,
        benchmark_provenance={"native_compile_request_sha256": request.sha256},
    )
    try:
        response = runtime.run().logical_completions[0]
    finally:
        runtime.close()
    if quote_sha256(response) != v["response_sha256"]:
        raise ValueError("original response changed before admission")
    fragments = restore(request.payload)
    if repair is not None:
        return repaired_atoms(response, fragments, repair.payload)
    atoms = admit(response, fragments)
    if list(atoms) != v["summaries"]:
        raise ValueError("original summaries differ from their authenticated response")
    return atoms


def admission_snapshot(root, preflight, bindings, model, overrides, repair_receipts):
    rows = []
    for ordinal, binding in enumerate(bindings):
        key = identity_sha256({"request_sha256": binding["sha256"], "model": model})
        path = root / "validated" / f"{key}.json"
        row = {"ordinal": ordinal, "request_sha256": binding["sha256"], "status": "pending"}
        repair = overrides.get(ordinal)
        if path.exists():
            validation = read_sealed_json(path)
            v = validation.payload
            if (v["preflight_sha256"] != preflight.sha256 or v["request_sha256"] != binding["sha256"]
                    or v["model"] != model or v["status"] not in {"accepted", "invalid_summary"}):
                raise ValueError("original validation does not match its request")
            row.update(status=v["status"], validation_path=str(path.resolve()),
                       validation_sha256=validation.sha256)
            if repair is not None:
                r = repair.payload
                if (v["status"] != "invalid_summary" or r["source_request_sha256"] != binding["sha256"]
                        or r["source_validation_sha256"] != validation.sha256):
                    raise ValueError("repair cannot replace this original receipt")
                row.update(status="repaired", repair_path=str(repair.path.resolve()), repair_sha256=repair.sha256)
        elif repair is not None:
            raise ValueError("repair lacks its original terminal receipt")
        rows.append(row)
    return {"compiler_preflight_sha256": preflight.sha256, "model": model,
            "repair_results": repair_receipts, "rows": rows,
            "implementation": implementation(), "new_model_calls": 0}


def assemble(input_root, output_root, model=MODEL, repair_roots=(), allow_partial=False):
    input_root, output_root = Path(input_root), Path(output_root)
    with _phase_lock(output_root, "native-admitted-body-assembly"):
        preflight, bindings = runner.load_requests(input_root, model)
        p = preflight.payload
        overrides, receipts = verified_repairs(repair_roots, preflight, model)
        sources_root = Path(p["sources_root"])
        sources = read_sealed_json(sources_root / "sources.json")
        raw_path = (sources_root / sources.payload["body_bank_path"]).resolve()
        raw_path.relative_to(sources_root.resolve())
        if (sources.sha256 != p["sources_sha256"] or digest(raw_path) != p["body_bank_sha256"]
                or (p["mode"] == "full" and p["body_count"] != sources.payload["body_count"])):
            raise ValueError("complete original source bank changed")
        if (output_root / "summary-bodies.json").exists():
            with closing(AdmittedSummaryBodies(output_root)) as store:
                saved = read_sealed_json(output_root / "admission-snapshot.json").payload
                if (saved["compiler_preflight_sha256"] != preflight.sha256 or saved["model"] != model
                        or saved["repair_results"] != receipts or saved["implementation"] != implementation()
                        or len(saved["rows"]) != len(bindings)):
                    raise ValueError("admission snapshot belongs to different inputs")
                if not allow_partial and not store.manifest.payload["all_prepared_bodies_admitted"]:
                    raise ValueError("full admission cannot reuse a partial snapshot")
                for ordinal, (row, binding) in enumerate(zip(saved["rows"], bindings, strict=True)):
                    if row["ordinal"] != ordinal or row["request_sha256"] != binding["sha256"]:
                        raise ValueError("snapshot request population changed")
                    if row["status"] != "pending":
                        key = identity_sha256({"request_sha256": binding["sha256"], "model": model})
                        original = read_sealed_json(input_root / "validated" / f"{key}.json")
                        if original.sha256 != row["validation_sha256"]:
                            raise ValueError("snapshot original receipt changed")
                    if row["status"] == "repaired" and overrides[ordinal].sha256 != row["repair_sha256"]:
                        raise ValueError("snapshot repair receipt changed")
                return store.manifest
        payload = admission_snapshot(input_root, preflight, bindings, model, overrides, receipts)
        ready = sum(row["status"] in {"accepted", "repaired"} for row in payload["rows"])
        if not allow_partial and ready != len(bindings):
            raise ValueError("full admission requires every original batch or a verified repair")
        snapshot, _ = publish_sealed_json(output_root / "admission-snapshot.json", payload)
        partial, target = output_root / "summary-bodies.sqlite.partial", output_root / "summary-bodies.sqlite"
        if target.exists():
            raise ValueError("an unfinished admitted summary store already exists")
        with partial.open("xb"):
            pass
        body_count = atom_count = covered_fragments = seen_bodies = 0
        current_sha, current, original_count, missing = None, [], 0, False
        with closing(sqlite3.connect(raw_path.as_uri() + "?mode=ro", uri=True)) as raw, closing(sqlite3.connect(partial)) as database:
            database.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, summaries_json TEXT NOT NULL, summary_sha256 TEXT NOT NULL)")

            def flush():
                nonlocal body_count, atom_count, covered_fragments, seen_bodies
                if current_sha is None:
                    return
                seen_bodies += 1
                if missing:
                    return
                found = raw.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (current_sha,)).fetchone()
                if found is None or validate_body_summaries(json.loads(found[0]), current) != current_sha:
                    raise ValueError("admitted body does not cover its complete original raw source")
                database.execute("INSERT INTO bodies VALUES (?,?,?)",
                                 (current_sha, canonical_json(current), identity_sha256(current)))
                body_count += 1
                atom_count += len(current)
                covered_fragments += original_count

            for binding, row in zip(bindings, payload["rows"], strict=True):
                request = read_sealed_json(input_root / binding["path"])
                if request.sha256 != binding["sha256"]:
                    raise ValueError("prepared request changed during body assembly")
                available = row["status"] in {"accepted", "repaired"}
                grouped = {}
                if available:
                    validation = read_sealed_json(row["validation_path"])
                    if validation.sha256 != row["validation_sha256"]:
                        raise ValueError("validation changed during body assembly")
                    atoms = replay_atoms(input_root, preflight, model, request, validation, overrides.get(row["ordinal"]))
                    for atom in atoms:
                        grouped.setdefault(atom["pointer"]["body_sha256"], []).append(atom)
                counts = {}
                for pointer in request.payload["pointers"]:
                    sha = pointer["body_sha256"]
                    counts[sha] = counts.get(sha, 0) + 1
                if available and list(grouped) != list(counts):
                    raise ValueError("admitted batch changed its source bodies")
                for sha, count in counts.items():
                    if current_sha is not None and current_sha != sha:
                        flush()
                        current, original_count, missing = [], 0, False
                    current_sha = sha
                    original_count += count
                    missing = missing or not available
                    current.extend(grouped.get(sha, ()))
            flush()
            if seen_bodies != p["body_count"]:
                raise ValueError("prepared body order or population changed")
            all_admitted = ready == len(bindings)
            if all_admitted and (body_count != p["body_count"] or covered_fragments != p["fragment_count"]):
                raise ValueError("complete admission has missing original source coverage")
            database.commit()
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("admitted summary database integrity failed")
        partial.rename(target)
        result, _ = publish_sealed_json(output_root / "summary-bodies.json", {
            "format": "native-spine-admitted-body-store-v1", "sources_sha256": sources.sha256,
            "compiler_preflight_sha256": preflight.sha256, "admission_snapshot_sha256": snapshot.sha256,
            "model": model, "database_sha256": digest(target), "body_count": body_count,
            "atom_count": atom_count, "original_fragments_covered": covered_fragments,
            "additional_sections": atom_count-covered_fragments,
            "prepared_body_count": p["body_count"], "prepared_fragment_count": p["fragment_count"],
            "admitted_batches": ready, "repaired_batches": len(overrides),
            "pending_batches": sum(row["status"] == "pending" for row in payload["rows"]),
            "unrepaired_batches": sum(row["status"] == "invalid_summary" for row in payload["rows"]),
            "all_prepared_bodies_admitted": all_admitted,
            "complete_source_compilation": p["mode"] == "full" and all_admitted,
            "hierarchies_compiled": False, "full100_target_passed": False,
            "summary_entailment_verified": False, "new_model_calls": 0,
            "implementation": implementation(),
        })
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--repair-root", action="append", type=Path, default=[])
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = assemble(args.input_root, args.output_root, args.model, args.repair_root, args.allow_partial)
    print({"summary_bodies_sha256": result.sha256, **{k: v for k, v in result.payload.items()
                                                   if k != "implementation"}}, flush=True)
