"""Cache local Qwen attention over native user summaries for hierarchy cuts."""
import argparse
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.modeling.qwen_prefix import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, expected_prefix_checkpoint_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from tools.assemble_native_spine_summaries import digest
from tools.build_spine_corpus_hierarchy import ScalarAttentionCache
from tools.compile_native_spine_exchanges import implementation as exchange_implementation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


FILES = ("tools/compile_native_spine_attention.py", "tools/build_spine_corpus_hierarchy.py",
         "src/memory_condense/modeling/qwen_prefix.py", "src/memory_condense/associations/qwen_memory_linker.py",
         "src/memory_condense/search/episodes/qwen_episode_signal.py",
         "src/memory_condense/search/episodes/surprise_models.py", "src/memory_condense/domain/_tokenizer.py")


def implementation():
    return {**exchange_implementation(), **{name: digest(name) for name in FILES}}


def user_windows(exchanges):
    rows = tuple(exchanges)
    if not rows or any(type(e) is not UserSpineExchange for e in rows):
        raise TypeError("attention preparation requires typed, complete source exchanges")
    if len({e.section.source_id for e in rows}) != 1:
        raise ValueError("attention windows cannot cross source occurrences")
    SectionSummaryIndex(tuple(e.section for e in rows))
    texts = tuple(e.user_spine if e.user_spine is not None else "Unowned prelude." for e in rows)
    if any(count_tokens(t) > 128 for t in texts):
        raise ValueError("a user summary would be truncated by attention")
    windows, start = [], 0
    while start < len(rows):
        end = min(len(rows), start+8)
        windows.append({"start_exchange": start, "end_exchange": end, "texts": list(texts[start:end])})
        if end == len(rows):
            break
        start = end-1
    return windows


def cache_method(root):
    # This identity is independent of a corpus/snapshot or occurrence timestamp.
    return publish_sealed_json(root / "method.json", {
        "model_id": DEFAULT_MODEL_ID, "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": expected_prefix_checkpoint_sha256(6),
        "device": "cuda", "dtype": "float16", "prefix_layers": 6, "attention_layer": 5,
        "head_vote_k": 4, "max_input_spans": 8, "span_token_cap": 128,
        "linker_max_candidates": 8, "linker_max_workspace_tokens": 4096,
        "owned_runtime_binding": True, "raw_inputs_to_qwen": False,
        "implementation": {name: digest(name) for name in FILES},
    })[0]


def prepare(exchange_root, root, cache_root):
    with _phase_lock(root, "native-attention-preparation"):
        result = read_sealed_json(exchange_root / "result.json")
        plan = read_sealed_json(exchange_root / "preflight.json")
        p = result.payload
        if (p["preflight_sha256"] != plan.sha256 or plan.payload["implementation"] != exchange_implementation()
                or p["complete_available_body_exchanges"] is not True or p["raw_inputs_to_qwen"] is not False):
            raise ValueError("attention requires the completed bound exchange population")
        method = cache_method(cache_root)
        bodies, jobs, atom_population = [], {}, []
        for binding in p["compiled_bodies"]:
            artifact = read_sealed_json(exchange_root / binding["path"])
            a = artifact.payload
            if artifact.sha256 != binding["sha256"] or a["preflight_sha256"] != plan.sha256:
                raise ValueError("native exchanges changed before attention preparation")
            exchanges = tuple(UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e["section"])))
                              for e in a["exchanges"])
            spans = [s.receipt_sha256 for e in exchanges for s in e.section.spans]
            if identity_sha256(spans) != a["raw_span_population_sha256"]:
                raise ValueError("exchange raw span population changed")
            atom_population.extend(spans)
            windows = []
            for window in user_windows(exchanges):
                key = identity_sha256({"preflight_sha256": method.sha256, "texts": window["texts"]})
                jobs[key] = window["texts"]
                windows.append({"start_exchange": window["start_exchange"], "end_exchange": window["end_exchange"], "key": key})
            bodies.append({"body_sha256": a["body_sha256"], "exchanges_sha256": artifact.sha256,
                           "exchange_count": len(exchanges), "windows": windows})
        if (len(bodies) != p["body_count"] or len({b["body_sha256"] for b in bodies}) != len(bodies)
                or identity_sha256(atom_population) != p["raw_span_population_sha256"]):
            raise ValueError("attention preparation changed the complete exchange population")
        artifact, _ = publish_sealed_json(root / "preflight.json", {
            "exchange_result_sha256": result.sha256, "cache_method_sha256": method.sha256,
            "cache_root": str(cache_root.resolve()), "bodies": bodies, "jobs": jobs,
            "unique_summary_windows": len(jobs), "complete_source_compilation": p["complete_source_compilation"],
            "raw_inputs_to_qwen": False, "query_or_gold_inputs": False,
            "implementation": implementation(),
        })
        print({"attention_preflight_sha256": artifact.sha256, "bodies": len(bodies),
               "unique_summary_windows": len(jobs)}, flush=True)
        return artifact


def execute(root):
    preflight = read_sealed_json(root / "preflight.json")
    with _phase_lock(root, "native-attention-compilation"), _phase_lock(Path(preflight.payload["cache_root"]), "native-attention-cache"):
        if read_sealed_json(root / "preflight.json").sha256 != preflight.sha256:
            raise ValueError("native attention preflight changed before execution")
        p = preflight.payload
        if p["implementation"] != implementation() or p["raw_inputs_to_qwen"] is not False:
            raise ValueError("native attention implementation changed")
        cache_root = Path(p["cache_root"])
        method = cache_method(cache_root)
        if method.sha256 != p["cache_method_sha256"]:
            raise ValueError("native attention cache method changed")
        cache = ScalarAttentionCache(cache_root, method)
        receipts, new_windows = [], 0
        checked_fields = ("model_id", "model_revision", "checkpoint_sha256", "device", "dtype",
                          "prefix_layers", "attention_layer", "head_vote_k", "max_input_spans",
                          "span_token_cap", "linker_max_candidates", "linker_max_workspace_tokens", "owned_runtime_binding")
        for key, texts in sorted(p["jobs"].items()):
            if (key != identity_sha256({"preflight_sha256": method.sha256, "texts": texts})
                    or not 1 <= len(texts) <= 8 or any(count_tokens(t) > 128 for t in texts)):
                raise ValueError("native attention input population changed")
            existed = (cache_root / "attention" / f"{key}.json").exists()
            signal = cache.score_sequence(tuple(texts))
            if any(getattr(signal.receipt, field) != method.payload[field] for field in checked_fields):
                raise ValueError("native attention receipt differs from the pinned local method")
            artifact = read_sealed_json(cache_root / "attention" / f"{key}.json")
            receipts.append({"key": key, "artifact_sha256": artifact.sha256,
                             "signal_receipt_sha256": signal.receipt.receipt_sha256})
            new_windows += not existed
            if len(receipts) % 100 == 0:
                print({"completed_attention_windows": len(receipts), "new_local_windows": new_windows}, flush=True)
        if len(receipts) != p["unique_summary_windows"]:
            raise ValueError("native attention window population incomplete")
        result, _ = publish_sealed_json(root / "result.json", {
            "preflight_sha256": preflight.sha256, "cache_method_sha256": method.sha256,
            "body_count": len(p["bodies"]), "window_count": len(receipts), "receipts": receipts,
            "all_prepared_attention_complete": True, "complete_source_compilation": p["complete_source_compilation"],
            "raw_inputs_to_qwen": False, "remote_provider_calls": 0,
            "hierarchies_compiled": False, "full100_target_passed": False,
        })
        print({"attention_result_sha256": result.sha256, "windows": len(receipts),
               "new_local_windows": new_windows}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.exchange_root, args.output_root, args.cache_root)
    else:
        execute(args.output_root)
