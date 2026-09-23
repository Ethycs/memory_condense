"""Build native summary hierarchies from cached attention, retaining atomic addresses."""
import argparse
from dataclasses import asdict
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.surprise_models import AttentionHeadSurpriseReceipt, ScoredSurpriseSequence
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange, build_user_spine_hierarchy
from memory_condense.search.native_spine_merges import neutral_key
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools import compile_native_spine_attention as attention
from tools import compile_native_spine_exchanges as exchange_compiler
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


def implementation():
    return {**attention.implementation(), "tools/compile_native_spine_hierarchy.py": digest(__file__)}


class FrozenAttention:
    """No encoder or generation capability: missing or changed cached signals fail."""
    max_spans = 8
    span_token_cap = 128

    def __init__(self, root):
        self.preflight = read_sealed_json(root / "preflight.json")
        self.result = read_sealed_json(root / "result.json")
        p, r = self.preflight.payload, self.result.payload
        self.cache_root = Path(p["cache_root"])
        self.method = read_sealed_json(self.cache_root / "method.json")
        if (p["implementation"] != attention.implementation()
                or p["cache_method_sha256"] != self.method.sha256
                or r["preflight_sha256"] != self.preflight.sha256
                or r["cache_method_sha256"] != self.method.sha256
                or r["all_prepared_attention_complete"] is not True
                or self.method.payload["implementation"] != {name: digest(name) for name in attention.FILES}
                or p["raw_inputs_to_qwen"] is not False or r["raw_inputs_to_qwen"] is not False):
            raise ValueError("complete native attention binding changed")
        self.receipts = {row["key"]: row for row in r["receipts"]}
        if (len(self.receipts) != len(r["receipts"]) or set(self.receipts) != set(p["jobs"])
                or len(self.receipts) != r["window_count"]):
            raise ValueError("native attention cache population changed")
        self.values = {}

    def score_sequence(self, texts):
        texts = tuple(texts)
        key = identity_sha256({"preflight_sha256": self.method.sha256, "texts": list(texts)})
        if key not in self.receipts or self.preflight.payload["jobs"].get(key) != list(texts):
            raise ValueError("hierarchy requested an unprepared native attention window")
        if key not in self.values:
            row = read_sealed_json(self.cache_root / "attention" / f"{key}.json")
            if row.sha256 != self.receipts[key]["artifact_sha256"] or row.payload["preflight_sha256"] != self.method.sha256:
                raise ValueError("native attention cache receipt changed")
            p = row.payload
            signal = ScoredSurpriseSequence(p["scores"], p["similarities"], AttentionHeadSurpriseReceipt(**p["receipt"]))
            if signal.receipt.receipt_sha256 != self.receipts[key]["signal_receipt_sha256"]:
                raise ValueError("native attention signal identity changed")
            for field in ("model_id", "model_revision", "checkpoint_sha256", "device", "dtype",
                          "prefix_layers", "attention_layer", "head_vote_k", "max_input_spans",
                          "span_token_cap", "linker_max_candidates", "linker_max_workspace_tokens", "owned_runtime_binding"):
                if getattr(signal.receipt, field) != self.method.payload[field]:
                    raise ValueError("native attention signal differs from its pinned method")
            signal.validate_inputs(texts)
            self.values[key] = signal
        return self.values[key]


def load_groups(exchange_root, scorer, backend):
    # Prove the exchange result by zero-call replay before seeding its merge cache.
    result = exchange_compiler.execute(exchange_root, backend, 0)
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
    seed = exchange_compiler.NeutralJournal(exchange_root, plan, backend, 0)
    seed.replay()
    return result, groups, seed.cache.values


def compile_groups(root, preflight, groups, scorer, journal):
    """Publish completed trees after every bounded wave; incomplete trees stay absent."""
    summarize = ReusingSpineSummarizer(journal.cache)
    done = {}
    while len(done) < len(groups):
        pending = {}
        for sha, (body, atom_input, atoms, exchanges) in groups.items():
            if sha in done:
                continue
            try:
                hierarchy = build_user_spine_hierarchy(exchanges, scorer=scorer, summarize=summarize,
                    summarizer_identity=preflight.sha256, leaf_token_cap=512, max_leaf_exchanges=2,
                    max_channel_tokens=128, window_exchange_cap=8, max_prompt_tokens=2048)
            except PendingMerge as missing:
                pending.setdefault(neutral_key(missing.request), missing.request)
                continue
            index, atomic_index = hierarchy.summary_index(), SectionSummaryIndex(atoms)
            by_id = {s.section_id: s for s in index.sections}
            expected = tuple(s for atom in atoms for s in atom.spans)
            if (len(hierarchy.root_section_ids) != 1
                    or by_id[hierarchy.root_section_ids[0]].spans != expected):
                raise ValueError("native hierarchy failed complete original raw coverage")
            leaves = tuple(s for s in index.sections if not s.child_section_ids)
            artifact, _ = publish_sealed_json(root / "hierarchies" / f"{sha}.json", {
                "preflight_sha256": preflight.sha256, "body_sha256": sha,
                "exchanges_sha256": body.sha256, "atomic_input_sha256": atom_input.sha256,
                "index_json": index.to_json(), "atomic_index_json": atomic_index.to_json(),
                "root_section_ids": list(hierarchy.root_section_ids),
                "leaf_count": len(leaves), "parent_count": len(index.sections)-len(leaves),
                "atomic_count": len(atoms), "exchange_count": len(exchanges),
                "raw_span_population_sha256": identity_sha256([s.receipt_sha256 for s in expected]),
                "splits": [asdict(s) for s in hierarchy.splits],
                "attention_receipts": [w.signal.receipt_sha256 for w in hierarchy.windows],
                "oversized_exchange_ids": list(hierarchy.oversized_exchange_ids),
                "original_atomic_addresses_preserved": True, "raw_inputs_to_qwen": False,
            })
            done[sha] = {"path": str(artifact.path.relative_to(root)), "sha256": artifact.sha256,
                         "leaf_count": len(leaves), "parent_count": len(index.sections)-len(leaves),
                         "atomic_count": len(atoms)}
        print({"complete_body_hierarchies": len(done), "pending_merge_jobs": len(pending)}, flush=True)
        if not pending:
            break
        before = len(journal.cache.values)
        resolved = journal.resolve(pending)
        if not resolved and len(journal.cache.values) == before:
            break
    return done


def execute(root, exchange_root, attention_root, backend, budget=0):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("native hierarchy invocation allows at most 128 new local jobs")
    with _phase_lock(root, "native-hierarchy-compilation"):
        scorer = FrozenAttention(attention_root)
        exchanges, groups, seed = load_groups(exchange_root, scorer, backend)
        preflight, _ = publish_sealed_json(root / "preflight.json", {
            "exchange_root": str(exchange_root.resolve()), "attention_root": str(attention_root.resolve()),
            "exchange_result_sha256": exchanges.sha256, "attention_result_sha256": scorer.result.sha256,
            "attention_method_sha256": scorer.method.sha256,
            "backend": backend.identity, "backend_sha256": backend.identity_sha256,
            "prepared_body_count": len(groups), "leaf_token_cap": 512, "max_leaf_exchanges": 2,
            "max_channel_tokens": 128, "window_exchange_cap": 8, "max_prompt_tokens": 2048,
            "maximum_new_local_jobs_per_invocation": 128, "maximum_recovery_attempts": 2,
            "automatic_retries": 0, "raw_inputs_to_qwen": False,
            "timestamp_metadata_in_model_inputs": False, "original_atomic_addresses_preserved": True,
            "implementation": implementation(),
        })
        journal = exchange_compiler.NeutralJournal(root, preflight, backend, budget)
        journal.cache.values.update(seed)
        journal.replay()
        done = compile_groups(root, preflight, groups, scorer, journal)
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
        }
        name = "result.json" if len(done) == len(groups) else f"partial-{identity_sha256(payload)}.json"
        result, _ = publish_sealed_json(root / name, payload)
        print({"hierarchy_result_sha256": result.sha256, "complete_body_hierarchies": len(done),
               "leaves": payload["leaf_count"], "parents": payload["parent_count"],
               "new_local_jobs": journal.jobs, "new_local_batches": journal.calls}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--probe-root", type=Path, default=Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"))
    parser.add_argument("--dependency-root", type=Path, default=Path(".cache/local-qwen-runtime/site-packages"))
    parser.add_argument("--model-root", type=Path, default=Path("../../.cache/models/Qwen3-8B"))
    args = parser.parse_args()
    backend = NativeQwenBackend(args.probe_root, args.dependency_root, args.model_root)
    execute(args.output_root, args.exchange_root, args.attention_root, backend, args.budget)
