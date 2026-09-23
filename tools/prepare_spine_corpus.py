"""Prepare complete-source, question-independent raw summary batches for full100.

This command makes no model calls. Terra is the only raw summarizer. Qwen's
later hierarchy stage receives the resulting summaries, never these requests.
The full source population is selected by authenticated namespace bindings,
without selecting evidence using questions, labels, or previous predictions.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sqlite3

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import _atoms
from memory_condense.search.spine_batch_summary import RawSummaryFragment, pack_summary_batches, batch_messages
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256


SOURCE_PARENT = Path("eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards")
PROBES = Path("eval_results/longmemeval-1m-hot-retrieval-full100-validation-20260905/probes.json")
PROBE_SHA = "75af9c3faa307a995c134dd9b7b44fd9e94b91d5d4f0a7f8e44ac5fcba9ecfc0"
IMPLEMENTATION = ("tools/prepare_spine_corpus.py", "src/memory_condense/search/spine_batch_summary.py",
                  "src/memory_condense/search/episodes/attention_hierarchy.py",
                  "src/memory_condense/search/section_summary.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(main_root, output_root):
    probes = read_sealed_json(PROBES)
    require(probes.sha256 == PROBE_SHA, "source namespace manifest changed")
    # Drop the question plane immediately. Only source_bindings enter ingest.
    bindings = probes.payload["source_bindings"]
    sources_sha = identity_sha256(bindings)
    del probes
    namespaces = []
    for binding in bindings:
        shard = (main_root / SOURCE_PARENT / f"offset-{binding['shard_offset']:03d}").resolve()
        source = read_sealed_json(shard / "source-current-selection.json")
        require(source.sha256 == binding["source_selection_sha256"], "source selection changed")
        database = (shard / "source-current" / source.payload["selected_store_entry"] / "store/memory.db").resolve()
        database.relative_to(shard / "source-current")
        require(hashlib.sha256(database.read_bytes()).hexdigest() == binding["database_sha256"], "source database changed")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT turn_id,source_id,role,text,created_at,ordinal FROM turns ORDER BY ordinal,turn_id").fetchall()
        require(len(rows) == binding["turn_count"], "namespace turn coverage changed")
        grouped = {}
        for turn_id, source_id, role, text, created_at, ordinal in rows:
            grouped.setdefault(source_id, []).append(Turn(turn_id=turn_id, source_id=source_id,
                role=role, text=text, created_at=created_at))
        fragments = []
        for turns in grouped.values():
            for turn in turns:
                fragments.extend(RawSummaryFragment(span, text) for span, text in _atoms(turn, 2048))
        batches = pack_summary_batches(fragments, max_atoms=10, max_prompt_tokens=7000)
        require(tuple(f for batch in batches for f in batch) == tuple(fragments), "raw fragment partition changed")
        require({f.span.turn_id for f in fragments} == {r[0] for r in rows}, "some raw turns have no atoms")
        requests = []
        for index, batch in enumerate(batches):
            payload = {"format": "memory-condense-spine-corpus-raw-request-v1",
                "namespace_database_sha256": binding["database_sha256"], "batch_index": index,
                "source_id": batch[0].span.source_id, "model": "codex_sdk/gpt-5.6-terra",
                "messages": batch_messages(batch), "raw_spans": [f.span.identity_payload() for f in batch],
                "max_completion_tokens": 3072, "max_prompt_tokens": 7000, "retries": 0,
                "qwen_raw_access": False, "gold_loaded": False}
            request, _ = publish_sealed_json(output_root / f"offset-{binding['shard_offset']:03d}" / "requests" / f"{index:04d}.json", payload)
            requests.append({"path": str(request.path.relative_to(output_root)), "sha256": request.sha256,
                             "atom_count": len(batch), "prompt_token_proxy": count_chat_prompt_token_proxy(payload["messages"])})
        namespace = {"shard_offset": binding["shard_offset"], "database_sha256": binding["database_sha256"],
            "source_selection_sha256": source.sha256, "turn_count": len(rows), "source_count": len(grouped),
            "atom_count": len(fragments), "raw_token_proxy": sum(f.span.token_count for f in fragments),
            "requests": requests, "request_count": len(requests),
            "span_population_sha256": identity_sha256([f.span.receipt_sha256 for f in fragments])}
        namespaces.append(namespace)
        print({k: v for k, v in namespace.items() if k != "requests"}, flush=True)
    manifest = {"format": "memory-condense-full100-spine-corpus-preflight-v1", "gold_loaded": False,
        "provider_calls": 0, "source_binding_population_sha256": sources_sha,
        "source_selection_policy": "all authenticated source namespaces, every turn in transcript order",
        "model": "codex_sdk/gpt-5.6-terra", "gateway": "https://central-dev.zt:4000/v1",
        "qwen_input_policy": "subsequent compiled summaries only; raw requests are Terra-only",
        "namespace_count": len(namespaces), "namespaces": namespaces,
        "raw_request_count": sum(n["request_count"] for n in namespaces),
        "raw_atom_count": sum(n["atom_count"] for n in namespaces),
        "raw_token_proxy": sum(n["raw_token_proxy"] for n in namespaces),
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION},
        "summary_entailment_certified": False, "hierarchy_constructed": False, "target_gate_passed": False}
    result, _ = publish_sealed_json(output_root / "preflight.json", manifest)
    print({"preflight_sha256": result.sha256, "raw_requests": manifest["raw_request_count"],
           "raw_atoms": manifest["raw_atom_count"], "raw_tokens": manifest["raw_token_proxy"], "provider_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=Path("../.."))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.main_root, args.output_root)
