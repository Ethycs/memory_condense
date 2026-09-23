"""Admit exact native bodies with separately accounted transport recoveries."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.native_spine_memory import validate_body_summaries
from tools import assemble_native_spine_direct_repairs as direct
from tools import recover_native_spine_transport as recovery
from tools import repair_native_recovery_section as recovery_sections
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock

base = direct.base
READY = {"accepted", "repaired", "recovered"}
FORMAT = "native-spine-recovered-body-assembly-v1"


def implementation():
    return {**direct.implementation(), **recovery_sections.implementation(),
            "tools/assemble_native_spine_recovered.py": base.digest(__file__)}


class RecoveredSummaryBodies(base.AdmittedSummaryBodies):
    def __init__(self, root):
        super().__init__(root)
        try:
            snapshot = read_sealed_json(Path(root)/"admission-snapshot.json")
            if (self.manifest.payload["producer_format"] != FORMAT
                    or self.manifest.payload["producer_implementation"] != implementation()
                    or snapshot.payload["producer_implementation"] != implementation()):
                raise ValueError("recovered body producer changed")
        except Exception:
            self.close()
            raise


def verified_recoveries(preflight, roots, section_roots=()):
    admitted, receipts = {}, []
    for root in map(Path, roots):
        result = recovery.execute(root, False)
        plan = read_sealed_json(root/"preflight.json")
        if result.payload["source_preflight_sha256"] != preflight.sha256:
            raise ValueError("recovery belongs to another source compilation")
        receipts.append({"root": str(root.resolve()), "result_sha256": result.sha256})
        for binding in result.payload["rows"]:
            if binding["status"] != "accepted":
                continue
            path = (root/binding["path"]).resolve()
            path.relative_to((root/"validated").resolve())
            row = read_sealed_json(path)
            p = row.payload
            ordinal = p["ordinal"]
            if (row.sha256 != binding["sha256"] or p["recovery_preflight_sha256"] != plan.sha256
                    or type(ordinal) is not int or not 0 <= ordinal < len(preflight.payload["requests"])
                    or p["source_request_sha256"] != preflight.payload["requests"][ordinal]["sha256"]
                    or p["status"] != "accepted" or p["original_journal_unchanged"] is not True
                    or p["raw_text_changed"] is not False):
                raise ValueError("recovery admission source binding changed")
            if ordinal in admitted:
                raise ValueError("duplicate transport recovery; select one explicit lineage")
            admitted[ordinal] = row
    for root in map(Path, section_roots):
        result = recovery_sections.execute(root, False)
        p = result.payload
        ordinal = p["ordinal"]
        if (p["source_preflight_sha256"] != preflight.sha256 or type(ordinal) is not int
                or not 0 <= ordinal < len(preflight.payload["requests"])
                or p["source_request_sha256"] != preflight.payload["requests"][ordinal]["sha256"]
                or p["complete_original_raw_coverage"] is not True):
            raise ValueError("repaired recovery belongs to another original source")
        if ordinal in admitted:
            raise ValueError("duplicate transport recovery; select one explicit lineage")
        admitted[ordinal] = result
        receipts.append({"root": str(root.resolve()), "result_sha256": result.sha256, "kind": "recovery_section_repair"})
    return admitted, receipts


def body_groups(input_root, preflight, bindings, rows, repairs, recovered):
    current, atoms, count, missing, seen = None, [], 0, False, set()
    for binding, row in zip(bindings, rows, strict=True):
        request = read_sealed_json(input_root/binding["path"])
        if request.sha256 != binding["sha256"]:
            raise ValueError("original prepared request changed")
        ready = row["status"] in READY
        groups, counts = {}, {}
        if ready:
            validation = read_sealed_json(row["validation_path"])
            if validation.sha256 != row["validation_sha256"]:
                raise ValueError("body admission validation changed")
            if row["status"] == "recovered":
                if recovered[row["ordinal"]].sha256 != validation.sha256:
                    raise ValueError("recovered body admission changed")
                values = validation.payload["summaries"]
                if ("recovery_section_preflight_sha256" not in validation.payload
                        and [a["pointer"] for a in values] != request.payload["pointers"]):
                    raise ValueError("recovery changed original source coverage")
            else:
                values = base.replay_atoms(input_root, preflight, base.MODEL, request, validation,
                                           repairs.get(row["ordinal"]))
            for atom in values:
                groups.setdefault(atom["pointer"]["body_sha256"], []).append(atom)
        for pointer in request.payload["pointers"]:
            sha = pointer["body_sha256"]
            counts[sha] = counts.get(sha, 0)+1
        if ready and list(groups) != list(counts):
            raise ValueError("recovery changed request body order")
        for sha, n in counts.items():
            if current is not None and sha != current:
                yield current, atoms, count, missing
                seen.add(current)
                atoms, count, missing = [], 0, False
            if sha in seen:
                raise ValueError("original body fragments are no longer contiguous")
            current = sha
            atoms.extend(groups.get(sha, ()))
            count += n
            missing = missing or not ready
    if current is not None:
        yield current, atoms, count, missing


def assemble(input_root, output_root, *, legacy_roots=(), direct_roots=(), recovery_roots=(),
             recovery_section_roots=(), allow_partial=False):
    input_root, output_root = Path(input_root), Path(output_root)
    with _phase_lock(output_root, "native-recovered-body-assembly"):
        preflight, bindings = base.runner.load_requests(input_root, base.MODEL)
        p = preflight.payload
        repairs, repair_receipts = direct.verified_repairs(preflight, legacy_roots, direct_roots)
        recovered, recovery_receipts = verified_recoveries(preflight, recovery_roots, recovery_section_roots)
        if repairs.keys() & recovered.keys():
            raise ValueError("a recovery cannot also replace a validated summary repair")
        source_root = Path(p["sources_root"])
        sources = read_sealed_json(source_root/"sources.json")
        raw_path = (source_root/sources.payload["body_bank_path"]).resolve()
        raw_path.relative_to(source_root.resolve())
        if (sources.sha256 != p["sources_sha256"] or base.digest(raw_path) != p["body_bank_sha256"]
                or p["body_count"] != sources.payload["body_count"]):
            raise ValueError("recovered admission raw source bank changed")
        saved_path = output_root/"admission-snapshot.json"
        if saved_path.exists():
            snapshot = read_sealed_json(saved_path)
            saved = snapshot.payload
            if (saved["compiler_preflight_sha256"] != preflight.sha256 or saved["model"] != base.MODEL
                    or saved["repair_results"] != repair_receipts or saved["recovery_results"] != recovery_receipts
                    or saved["producer_implementation"] != implementation()
                    or saved["implementation"] != base.implementation() or len(saved["rows"]) != len(bindings)):
                raise ValueError("recovered admission snapshot changed")
            for ordinal, (row, binding) in enumerate(zip(saved["rows"], bindings, strict=True)):
                if row["ordinal"] != ordinal or row["request_sha256"] != binding["sha256"]:
                    raise ValueError("recovered snapshot request population changed")
                if row["status"] != "pending" and read_sealed_json(row["validation_path"]).sha256 != row["validation_sha256"]:
                    raise ValueError("recovered snapshot validation changed")
                if row["status"] == "repaired" and repairs[ordinal].sha256 != row["repair_sha256"]:
                    raise ValueError("recovered snapshot repair changed")
                if row["status"] == "recovered" and recovered[ordinal].sha256 != row["validation_sha256"]:
                    raise ValueError("recovered snapshot transport receipt changed")
        else:
            saved = base.admission_snapshot(input_root, preflight, bindings, base.MODEL, repairs, repair_receipts)
            saved.update(producer_implementation=implementation(), recovery_results=recovery_receipts)
            for ordinal, result in recovered.items():
                row = saved["rows"][ordinal]
                if row["status"] != "pending":
                    raise ValueError("transport recovery cannot replace an original validated response")
                row.update(status="recovered", validation_path=str(result.path.resolve()), validation_sha256=result.sha256)
            if not allow_partial and any(r["status"] not in READY for r in saved["rows"]):
                raise ValueError("full admission requires every original batch or explicit repair/recovery")
            snapshot, _ = publish_sealed_json(saved_path, saved)
        ready = sum(r["status"] in READY for r in saved["rows"])
        if not allow_partial and ready != len(bindings):
            raise ValueError("full admission cannot reuse a partial snapshot")
        if (output_root/"summary-bodies.json").exists():
            with closing(RecoveredSummaryBodies(output_root)) as store:
                return store.manifest
        target, partial = output_root/"summary-bodies.sqlite", output_root/"summary-bodies.sqlite.partial"
        if target.exists():
            raise ValueError("unfinished recovered store requires explicit recovery")
        with partial.open("xb"):
            pass
        body_count = atom_count = fragment_count = seen_bodies = 0
        with closing(sqlite3.connect(raw_path.as_uri()+"?mode=ro", uri=True)) as raw, closing(sqlite3.connect(partial)) as db:
            db.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, summaries_json TEXT NOT NULL, summary_sha256 TEXT NOT NULL)")
            for sha, atoms, count, missing in body_groups(input_root, preflight, bindings, saved["rows"], repairs, recovered):
                seen_bodies += 1
                if missing:
                    continue
                row = raw.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (sha,)).fetchone()
                if row is None or validate_body_summaries(json.loads(row[0]), atoms) != sha:
                    raise ValueError("recovered body fails complete exact raw coverage")
                db.execute("INSERT INTO bodies VALUES (?,?,?)", (sha, canonical_json(atoms), identity_sha256(atoms)))
                body_count += 1
                atom_count += len(atoms)
                fragment_count += count
            all_admitted = ready == len(bindings)
            if (seen_bodies != p["body_count"] or (all_admitted and
                    (body_count != p["body_count"] or fragment_count != p["fragment_count"]))):
                raise ValueError("recovered store changed prepared body coverage")
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("recovered body database integrity failed")
        partial.rename(target)
        return publish_sealed_json(output_root/"summary-bodies.json", {
            "format": "native-spine-admitted-body-store-v1", "sources_sha256": sources.sha256,
            "compiler_preflight_sha256": preflight.sha256, "admission_snapshot_sha256": snapshot.sha256,
            "model": base.MODEL, "database_sha256": base.digest(target), "body_count": body_count,
            "atom_count": atom_count, "original_fragments_covered": fragment_count,
            "additional_sections": atom_count-fragment_count, "prepared_body_count": p["body_count"],
            "prepared_fragment_count": p["fragment_count"], "admitted_batches": ready,
            "repaired_batches": len(repairs), "recovered_batches": len(recovered),
            "recovered_section_repaired_batches": sum("recovery_section_preflight_sha256" in r.payload for r in recovered.values()),
            "pending_batches": sum(r["status"] == "pending" for r in saved["rows"]),
            "unrepaired_batches": sum(r["status"] == "invalid_summary" for r in saved["rows"]),
            "all_prepared_bodies_admitted": all_admitted, "complete_source_compilation": p["mode"] == "full" and all_admitted,
            "hierarchies_compiled": False, "full100_target_passed": False, "summary_entailment_verified": False,
            "new_model_calls": 0, "implementation": base.implementation(),
            "producer_format": FORMAT, "producer_implementation": implementation(),
        })[0]
