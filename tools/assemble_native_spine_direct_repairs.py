"""Admit complete bodies from original outputs and either verified repair lineage.

The read-only body-store wire format is unchanged. This producer has a distinct
implementation binding and never attributes its execution to the old assembler.
"""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.native_spine_memory import validate_body_summaries
from tools import assemble_native_spine_admitted as base
from tools import repair_native_spine_sections as direct
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


def implementation():
    return {**base.implementation(), **direct.implementation(),
            "tools/assemble_native_spine_direct_repairs.py": base.digest(__file__)}


class DirectRepairSummaryBodies(base.AdmittedSummaryBodies):
    def __init__(self, root):
        super().__init__(root)
        try:
            p = self.manifest.payload
            snapshot = read_sealed_json(Path(root)/"admission-snapshot.json")
            if (p["producer_format"] != "native-spine-direct-repair-body-assembly-v1"
                    or p["producer_implementation"] != implementation()
                    or snapshot.payload["producer_implementation"] != implementation()):
                raise ValueError("direct-repair body producer changed")
        except Exception:
            self.close()
            raise


def verified_repairs(preflight, legacy_roots, direct_roots):
    overrides, receipts = base.verified_repairs(legacy_roots, preflight, base.MODEL)
    for root in map(Path, direct_roots):
        result = direct.execute(root, False)
        plan = read_sealed_json(root/"preflight.json")
        if (result.payload["source_preflight_sha256"] != preflight.sha256
                or result.payload["direct_repair_preflight_sha256"] != plan.sha256
                or plan.payload["model"] != base.MODEL):
            raise ValueError("direct repair belongs to another original compilation")
        receipts.append({"root": str(root.resolve()), "result_sha256": result.sha256,
                         "producer": "native-spine-direct-section-repair-v1"})
        for binding in result.payload["admitted_original_batches"]:
            path = (root/binding["path"]).resolve()
            path.relative_to((root/"admitted-batches").resolve())
            row = read_sealed_json(path)
            p = row.payload
            ordinal = p["ordinal"]
            if (row.sha256 != binding["sha256"] or p["direct_repair_preflight_sha256"] != plan.sha256
                    or p["source_preflight_sha256"] != preflight.sha256
                    or type(ordinal) is not int or not 0 <= ordinal < len(preflight.payload["requests"])
                    or p["complete_original_raw_coverage"] is not True or p["raw_text_changed"] is not False):
                raise ValueError("direct repair raw coverage binding changed")
            if ordinal in overrides:
                raise ValueError("duplicate repaired batch; select one final lineage")
            overrides[ordinal] = row
    return overrides, receipts


def body_groups(input_root, preflight, bindings, rows, overrides):
    """Stream ordered bodies, excluding the whole body if any fragment is missing."""
    current_sha, atoms, original_count, missing = None, [], 0, False
    seen = set()
    for binding, row in zip(bindings, rows, strict=True):
        request = read_sealed_json(input_root/binding["path"])
        if request.sha256 != binding["sha256"]:
            raise ValueError("original prepared request changed")
        available = row["status"] in {"accepted", "repaired"}
        grouped, counts = {}, {}
        if available:
            validation = read_sealed_json(row["validation_path"])
            if validation.sha256 != row["validation_sha256"]:
                raise ValueError("original validation changed during admission")
            admitted = base.replay_atoms(input_root, preflight, base.MODEL, request, validation,
                                         overrides.get(row["ordinal"]))
            for atom in admitted:
                grouped.setdefault(atom["pointer"]["body_sha256"], []).append(atom)
        for pointer in request.payload["pointers"]:
            sha = pointer["body_sha256"]
            counts[sha] = counts.get(sha, 0)+1
        if available and list(grouped) != list(counts):
            raise ValueError("admission changed the request's source body population")
        for sha, count in counts.items():
            if current_sha is not None and current_sha != sha:
                yield current_sha, atoms, original_count, missing
                seen.add(current_sha)
                atoms, original_count, missing = [], 0, False
            if sha in seen:
                raise ValueError("prepared body fragments are no longer contiguous")
            current_sha = sha
            original_count += count
            missing = missing or not available
            atoms.extend(grouped.get(sha, ()))
    if current_sha is not None:
        yield current_sha, atoms, original_count, missing


def assemble(input_root, output_root, *, legacy_roots=(), direct_roots=(), allow_partial=False):
    input_root, output_root = Path(input_root), Path(output_root)
    with _phase_lock(output_root, "native-direct-repair-body-assembly"):
        preflight, bindings = base.runner.load_requests(input_root, base.MODEL)
        p = preflight.payload
        overrides, receipts = verified_repairs(preflight, legacy_roots, direct_roots)
        source_root = Path(p["sources_root"])
        sources = read_sealed_json(source_root/"sources.json")
        raw_path = (source_root/sources.payload["body_bank_path"]).resolve()
        raw_path.relative_to(source_root.resolve())
        if (sources.sha256 != p["sources_sha256"] or base.digest(raw_path) != p["body_bank_sha256"]
                or (p["mode"] == "full" and p["body_count"] != sources.payload["body_count"])):
            raise ValueError("complete original raw source bank changed")
        saved_path = output_root/"admission-snapshot.json"
        if saved_path.exists():
            snapshot = read_sealed_json(saved_path)
            saved = snapshot.payload
            if (saved["compiler_preflight_sha256"] != preflight.sha256 or saved["model"] != base.MODEL
                    or saved["repair_results"] != receipts or saved["producer_implementation"] != implementation()
                    or saved["implementation"] != base.implementation() or len(saved["rows"]) != len(bindings)):
                raise ValueError("direct-repair admission snapshot input changed")
            for ordinal, (row, binding) in enumerate(zip(saved["rows"], bindings, strict=True)):
                if row["ordinal"] != ordinal or row["request_sha256"] != binding["sha256"]:
                    raise ValueError("admission snapshot request population changed")
                if row["status"] != "pending" and read_sealed_json(row["validation_path"]).sha256 != row["validation_sha256"]:
                    raise ValueError("admission snapshot original receipt changed")
                if row["status"] == "repaired" and overrides[ordinal].sha256 != row["repair_sha256"]:
                    raise ValueError("admission snapshot repair receipt changed")
        else:
            saved = base.admission_snapshot(input_root, preflight, bindings, base.MODEL, overrides, receipts)
            saved["producer_implementation"] = implementation()
            if not allow_partial and any(r["status"] not in {"accepted", "repaired"} for r in saved["rows"]):
                raise ValueError("full admission requires every original batch or a verified repair")
            snapshot, _ = publish_sealed_json(saved_path, saved)
        ready = sum(r["status"] in {"accepted", "repaired"} for r in saved["rows"])
        if not allow_partial and ready != len(bindings):
            raise ValueError("full admission cannot reuse a partial snapshot")
        if (output_root/"summary-bodies.json").exists():
            with closing(DirectRepairSummaryBodies(output_root)) as store:
                return store.manifest
        target, partial = output_root/"summary-bodies.sqlite", output_root/"summary-bodies.sqlite.partial"
        if target.exists():
            raise ValueError("unfinished direct-repair body store requires explicit recovery")
        with partial.open("xb"):
            pass
        body_count = atom_count = fragment_count = seen_bodies = 0
        with closing(sqlite3.connect(raw_path.as_uri()+"?mode=ro", uri=True)) as raw, closing(sqlite3.connect(partial)) as db:
            db.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, summaries_json TEXT NOT NULL, summary_sha256 TEXT NOT NULL)")
            for sha, atoms, count, missing in body_groups(input_root, preflight, bindings, saved["rows"], overrides):
                seen_bodies += 1
                if missing:
                    continue
                row = raw.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (sha,)).fetchone()
                if row is None or validate_body_summaries(json.loads(row[0]), atoms) != sha:
                    raise ValueError("admitted summaries fail complete original body coverage")
                db.execute("INSERT INTO bodies VALUES (?,?,?)", (sha, canonical_json(atoms), identity_sha256(atoms)))
                body_count += 1
                atom_count += len(atoms)
                fragment_count += count
            all_admitted = ready == len(bindings)
            if (seen_bodies != p["body_count"] or (all_admitted and
                    (body_count != p["body_count"] or fragment_count != p["fragment_count"]))):
                raise ValueError("admission changed the complete prepared body population")
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("direct-repair body database integrity failed")
        partial.rename(target)
        result, _ = publish_sealed_json(output_root/"summary-bodies.json", {
            "format": "native-spine-admitted-body-store-v1", "sources_sha256": sources.sha256,
            "compiler_preflight_sha256": preflight.sha256, "admission_snapshot_sha256": snapshot.sha256,
            "model": base.MODEL, "database_sha256": base.digest(target), "body_count": body_count,
            "atom_count": atom_count, "original_fragments_covered": fragment_count,
            "additional_sections": atom_count-fragment_count, "prepared_body_count": p["body_count"],
            "prepared_fragment_count": p["fragment_count"], "admitted_batches": ready,
            "repaired_batches": len(overrides), "pending_batches": sum(r["status"] == "pending" for r in saved["rows"]),
            "unrepaired_batches": sum(r["status"] == "invalid_summary" for r in saved["rows"]),
            "all_prepared_bodies_admitted": all_admitted, "complete_source_compilation": p["mode"] == "full" and all_admitted,
            "hierarchies_compiled": False, "full100_target_passed": False, "summary_entailment_verified": False,
            "new_model_calls": 0, "implementation": base.implementation(),
            "producer_format": "native-spine-direct-repair-body-assembly-v1", "producer_implementation": implementation(),
        })
        return result
