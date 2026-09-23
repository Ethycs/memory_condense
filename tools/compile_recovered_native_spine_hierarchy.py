"""Compile expanded native trees with authenticated exchange and parent reuse."""
import argparse
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.section_summary import SectionSummary
from tools import compile_native_spine_exchanges as exchange_compiler
from tools import compile_native_spine_parent_budgets as parent
from tools import compile_recovered_native_spine_exchanges as reused_exchanges
from tools import prepare_recovered_native_spine_attention as attention_admission
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_hierarchy import FrozenAttention
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = "native-spine-recovered-parent-producer-v1"


def implementation():
    return {**parent.implementation(), **attention_admission.implementation(),
            "tools/compile_recovered_native_spine_hierarchy.py": digest(__file__)}


def merge_values(target, values):
    for key, value in values.items():
        if key in target and target[key] != value:
            raise ValueError("conflicting accepted native parent merge")
        target[key] = value


def reusable_parents(roots, backend, sources_sha256, attention_method_sha256, *, chain=()):
    values, receipts, seen = {}, [], set()
    for source in roots:
        source = Path(source).resolve()
        if source in seen or source in chain:
            raise ValueError("duplicate or cyclic native parent reuse source")
        seen.add(source)
        plan = read_sealed_json(source/"preflight.json")
        result = read_sealed_json(source/"result.json")
        p, r = plan.payload, result.payload
        inputs = read_sealed_json(Path(p["exchange_root"])/"inputs.json")
        if (p["format"] != "native-spine-parent-budgeted-hierarchy-v1"
                or p["implementation"] != parent.implementation()
                or p["backend_sha256"] != backend.identity_sha256
                or p["attention_method_sha256"] != attention_method_sha256
                or (p["max_exchange_channel_tokens"], p["max_parent_channel_tokens"]) != (128, 512)
                or p["raw_inputs_to_qwen"] is not False
                or inputs.payload["sources_sha256"] != sources_sha256
                or r["preflight_sha256"] != plan.sha256
                or r["complete_available_body_hierarchies"] is not True):
            raise ValueError("native parent reuse source or method changed")
        ancestors = {}
        if p.get("producer_format") == FORMAT:
            if p["producer_implementation"] != implementation():
                raise ValueError("expanded parent reuse producer changed")
            ancestors, ancestor_receipts = reusable_parents([Path(b["root"]) for b in p["reuse_parent_roots"]],
                backend, sources_sha256, attention_method_sha256, chain=(*chain, source))
            if ancestor_receipts != p["reuse_parent_roots"]:
                raise ValueError("native parent reuse ancestry changed")
        elif "producer_format" in p:
            raise ValueError("unsupported native parent reuse producer")
        journal = exchange_compiler.NeutralJournal(source, plan, backend, 0)
        journal.cache.values.update(ancestors)
        journal.replay()
        merge_values(values, journal.cache.values)
        receipts.append({"root": str(source), "preflight_sha256": plan.sha256,
                         "result_sha256": result.sha256, "accepted_merge_count": len(journal.cache.values),
                         "merge_cache_sha256": identity_sha256(journal.cache.values)})
    return values, receipts


def load_groups(exchange_root, scorer, backend):
    # Prove the exchange result by zero-call replay before seeding its merge cache.
    result, seed = reused_exchanges._execute(exchange_root, backend, 0)
    plan = read_sealed_json(exchange_root / "preflight.json")
    inputs = read_sealed_json(exchange_root / "inputs.json")
    if (not result.payload["complete_available_body_exchanges"]
            or inputs.sha256 != plan.payload["inputs_sha256"]
            or result.sha256 != scorer.preflight.payload["exchange_result_sha256"]):
        raise ValueError("hierarchy requires matching completed native exchanges and attention")
    atom_bindings = {b["body_sha256"]: b for b in inputs.payload["bodies"]}
    attention_bodies = {b["body_sha256"]: b for b in scorer.preflight.payload["bodies"]}
    groups = {}
    for binding in result.payload["compiled_bodies"]:
        body = read_sealed_json(exchange_root / binding["path"])
        b = body.payload
        sha = b["body_sha256"]
        if (body.sha256 != binding["sha256"] or b["preflight_sha256"] != plan.sha256
                or sha in groups or attention_bodies[sha]["exchanges_sha256"] != body.sha256):
            raise ValueError("hierarchy exchange source changed")
        atom_input = read_sealed_json(exchange_root / atom_bindings[sha]["path"])
        if atom_input.sha256 != atom_bindings[sha]["sha256"] or atom_input.sha256 != b["body_input_sha256"]:
            raise ValueError("hierarchy atomic fallback source changed")
        atoms = tuple(SectionSummary.from_dict(a) for a in atom_input.payload["atoms"])
        exchanges = tuple(UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e["section"])))
                          for e in b["exchanges"])
        if tuple(s for a in atoms for s in a.spans) != tuple(s for e in exchanges for s in e.section.spans):
            raise ValueError("native exchanges differ from the original atomic partition")
        groups[sha] = (body, atom_input, atoms, exchanges)
    if set(groups) != set(atom_bindings) or set(groups) != set(attention_bodies):
        raise ValueError("native hierarchy source population incomplete")
    return result, groups, seed


def execute(root, exchange_root, attention_root, backend, budget=0, *, reuse_roots=None):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("expanded hierarchy invocation allows at most 128 new local jobs")
    root, exchange_root, attention_root = Path(root), Path(exchange_root), Path(attention_root)
    with _phase_lock(root, "native-expanded-parent-compilation"):
        admission = attention_admission.validate_admission(attention_root)
        if Path(admission.payload["exchange_root"]).resolve() != exchange_root.resolve():
            raise ValueError("expanded parent attention belongs to different exchanges")
        scorer = FrozenAttention(attention_root)
        exchanges, groups, seed = load_groups(exchange_root, scorer, backend)
        if reuse_roots is None:
            saved = read_sealed_json(root/"preflight.json") if (root/"preflight.json").exists() else None
            reuse_roots = [Path(r["root"]) for r in saved.payload["reuse_parent_roots"]] if saved else []
        sources = read_sealed_json(exchange_root/"inputs.json").payload["sources_sha256"]
        inherited, receipts = reusable_parents(reuse_roots, backend, sources, scorer.method.sha256,
                                                chain=(root.resolve(),))
        merge_values(seed, inherited)
        preflight, _ = publish_sealed_json(root/"preflight.json", {
            "format": "native-spine-parent-budgeted-hierarchy-v1",
            "exchange_root": str(exchange_root.resolve()), "attention_root": str(attention_root.resolve()),
            "exchange_result_sha256": exchanges.sha256, "attention_result_sha256": scorer.result.sha256,
            "attention_method_sha256": scorer.method.sha256,
            "backend": backend.identity, "backend_sha256": backend.identity_sha256,
            "prepared_body_count": len(groups), "leaf_token_cap": 512, "max_leaf_exchanges": 2,
            "max_exchange_channel_tokens": 128, "max_parent_channel_tokens": 512,
            "window_exchange_cap": 8, "max_prompt_tokens": 2048,
            "maximum_new_local_jobs_per_invocation": 128, "maximum_recovery_attempts": 2,
            "automatic_retries": 0, "raw_inputs_to_qwen": False,
            "timestamp_metadata_in_model_inputs": False, "original_atomic_addresses_preserved": True,
            "merge_reuse_policy": "same exact neutral request only; no cross-budget key substitution",
            "implementation": parent.implementation(),
            "producer_format": FORMAT, "producer_implementation": implementation(),
            "attention_admission_sha256": admission.sha256,
            "reuse_parent_roots": receipts, "inherited_parent_merge_keys": len(inherited),
        })
        journal = exchange_compiler.NeutralJournal(root, preflight, backend, budget)
        journal.cache.values.update(seed)
        journal.replay()
        done = parent.compile_groups(root, preflight, groups, scorer, journal)
        payload = {
            "preflight_sha256": preflight.sha256, "body_count": len(done), "prepared_body_count": len(groups),
            "compiled_bodies": [done[sha] for sha in sorted(done)],
            "leaf_count": sum(row["leaf_count"] for row in done.values()),
            "parent_count": sum(row["parent_count"] for row in done.values()),
            "atomic_count": sum(row["atomic_count"] for row in done.values()),
            "complete_available_body_hierarchies": len(done) == len(groups),
            "complete_source_compilation": exchanges.payload["complete_source_compilation"],
            "complete_native_hierarchies": len(done) == len(groups) and exchanges.payload["complete_source_compilation"],
            "original_atomic_addresses_preserved": True, "raw_inputs_to_qwen": False,
            "new_attention_passes": 0, "remote_provider_calls": 0, "full100_target_passed": False,
            "producer_format": FORMAT,
        }
        filename = "result.json" if len(done) == len(groups) else f"partial-{identity_sha256(payload)}.json"
        result, _ = publish_sealed_json(root/filename, payload)
        print({"expanded_hierarchy_result_sha256": result.sha256, "complete_body_hierarchies": len(done),
               "new_local_jobs": journal.jobs, "new_local_batches": journal.calls,
               "inherited_parent_merge_keys": len(inherited)}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, action="append")
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()
    backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
        Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
    execute(args.root, args.exchange_root, args.attention_root, backend, args.budget, reuse_roots=args.reuse_root)
