"""Continue native exchanges from authenticated partial journals and explicit recovery."""
import argparse
from dataclasses import asdict
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.native_spine_merges import neutral_key
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools import compile_native_spine_exchanges as original
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


from tools import native_recovered_merge_seed as recovery_seed
from tools import compile_reused_native_spine_exchanges as reused
from tools import recover_native_summary_lengths as recovery

FORMAT = "native-spine-recovered-exchanges-v1"


def implementation():
    return {**reused.implementation(), "tools/native_recovered_merge_seed.py": digest(recovery_seed.__file__),
            "tools/recover_native_summary_lengths.py": digest(recovery.__file__),
            "tools/compile_recovered_native_spine_exchanges.py": digest(__file__)}


def _execute(root, backend, budget=0, *, recovery_root=None):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("native exchange compilation allows at most 128 new local jobs per invocation")
    root = Path(root).resolve()
    with _phase_lock(root, "native-exchange-compilation"):
        inputs = read_sealed_json(root / "inputs.json")
        p = inputs.payload
        if (p["implementation"] != original.implementation() or p["raw_text_included"] is not False
                or p["question_or_gold_inputs"] is not False):
            raise ValueError("native exchange source inputs changed")
        if recovery_root is None:
            recovery_root = Path(read_sealed_json(root/"preflight.json").payload["recovery"]["root"])
        seed, receipt = recovery_seed.load(recovery_root, backend, inputs.sha256)
        preflight, _ = publish_sealed_json(root / "preflight.json", {
            "inputs_sha256": inputs.sha256, "backend": backend.identity,
            "backend_sha256": backend.identity_sha256, "max_channel_tokens": 128,
            "max_prompt_tokens": 2048, "maximum_new_local_jobs_per_invocation": 128,
            "maximum_recovery_attempts": 2, "automatic_retries": 0,
            "raw_inputs_to_qwen": False, "timestamp_metadata_in_model_inputs": False,
            "implementation": original.implementation(),
            "producer_format": FORMAT, "producer_implementation": implementation(),
            "recovery": receipt, "reused_merge_keys": len(seed),
            "reused_merge_cache_sha256": identity_sha256(seed),
        })
        groups, population = {}, []
        for binding in p["bodies"]:
            path = (root / binding["path"]).resolve()
            path.relative_to((root / "bodies").resolve())
            body = read_sealed_json(path)
            b = body.payload
            if (body.sha256 != binding["sha256"] or b["summary_body_store_sha256"] != p["summary_body_store_sha256"]
                    or b["source"]["body_sha256"] != binding["body_sha256"] or b["raw_text_included"] is not False
                    or binding["body_sha256"] in groups):
                raise ValueError("native summary body input changed")
            atoms = tuple(SectionSummary.from_dict(a) for a in b["atoms"])
            groups[binding["body_sha256"]] = (body, atoms)
            population.extend(s.receipt_sha256 for a in atoms for s in a.spans)
        if len(groups) != p["body_count"] or len(population) != p["atom_count"] or len(set(population)) != len(population):
            raise ValueError("native exchange atomic population changed")
        journal = original.NeutralJournal(root, preflight, backend, budget)
        journal.cache.values.update(seed)
        journal.replay()
        summarize = ReusingSpineSummarizer(journal.cache)
        done, exchange_count = {}, 0
        while len(done) < len(groups):
            pending = {}
            for sha, (body, atoms) in groups.items():
                if sha in done:
                    continue
                try:
                    exchanges = compile_user_spine_exchanges(atoms, summarize=summarize,
                        summarizer_identity=preflight.sha256, max_channel_tokens=128, max_prompt_tokens=2048)
                except PendingMerge as missing:
                    pending.setdefault(neutral_key(missing.request), missing.request)
                    continue
                expected = tuple(s for atom in atoms for s in atom.spans)
                if tuple(s for e in exchanges for s in e.section.spans) != expected:
                    raise ValueError("native exchange compilation changed exact raw coverage")
                artifact, _ = publish_sealed_json(root / "exchanges" / f"{sha}.json", {
                    "preflight_sha256": preflight.sha256, "body_input_sha256": body.sha256,
                    "body_sha256": sha, "exchanges": [asdict(e) for e in exchanges],
                    "raw_span_population_sha256": identity_sha256([s.receipt_sha256 for s in expected]),
                    "raw_inputs_to_qwen": False,
                })
                done[sha] = {"path": str(artifact.path.relative_to(root)), "sha256": artifact.sha256}
                exchange_count += len(exchanges)
            print({"complete_body_exchanges": len(done), "pending_merge_jobs": len(pending)}, flush=True)
            before = len(journal.cache.values)
            if pending and not journal.resolve(pending) and len(journal.cache.values) == before:
                break
            if not pending:
                break
        payload = {"preflight_sha256": preflight.sha256, "body_count": len(done),
                   "prepared_body_count": len(groups), "exchange_count": exchange_count,
                   "compiled_bodies": [done[sha] for sha in sorted(done)],
                   "complete_available_body_exchanges": len(done) == len(groups),
                   "complete_source_compilation": p["complete_source_compilation"],
                   "raw_span_population_sha256": identity_sha256(population),
                   "raw_inputs_to_qwen": False, "remote_provider_calls": 0,
                   "hierarchies_compiled": False, "full100_target_passed": False}
        filename = "result.json" if len(done) == len(groups) else f"partial-{identity_sha256(payload)}.json"
        result, _ = publish_sealed_json(root / filename, payload)
        print({"result_sha256": result.sha256, "complete_body_exchanges": len(done),
               "exchanges": exchange_count, "new_local_jobs": journal.jobs, "new_local_batches": journal.calls}, flush=True)
        return result, dict(journal.cache.values)


def execute(root, backend, budget=0, *, recovery_root=None):
    return _execute(root, backend, budget, recovery_root=recovery_root)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()
    backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
        Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
    execute(args.root, backend, args.budget, recovery_root=args.recovery_root)
