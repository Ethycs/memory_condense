"""Freeze local BGE addresses, then reorder the existing conventional packet.

All selected raw evidence and the original prompt format remain intact. This is
a presentation-order test over a fixed candidate set, not corpus-wide retrieval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from tools import assay_hot_reduced30_construction as harness
from tools import assay_hot_v7_spine_episode_fact_reserved_full100 as legacy
from tools.assay_hot_temporal_reference_chain_reduced30 import DEFAULT_PARENT, DEFAULT_PARENT_SHA256
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256
from tools.matched_eval.hot_temporal_reference_chain import _global_rows, split_packet

FORMAT = "memory-condense-hot-dense-packet-order-v1"


def _inputs():
    parent = read_sealed_json(DEFAULT_PARENT)
    if parent.sha256 != DEFAULT_PARENT_SHA256:
        raise ValueError("frozen conventional parent changed")
    harness.validate_selection(parent.payload)
    items = []
    for row in parent.payload["questions"]:
        _, arm = harness.find_provider_arm(row["source_row"], row["telemetry"]["arm_path"])
        prefix, context, question = split_packet(arm)
        globals_ = _global_rows(arm, context)
        query = re.sub(r"^\[Question asked at [^\]]+\]\s*", "", question)
        items.append((row, arm, prefix, context, question, query, globals_))
    return parent, items


def _implementation():
    root = Path(__file__).resolve().parents[1]
    paths = ["tools/assay_hot_dense_order_reduced30.py", "tools/matched_eval/hot_temporal_reference_chain.py",
             "tools/assay_hot_reduced30_construction.py", "src/memory_condense/search/hot_retrieval.py",
             "src/memory_condense/modeling/embedding.py"]
    body = {"files": [{"path": p, "sha256": file_sha256(root / p)} for p in paths],
            "parent_implementation": legacy._implementation_identity()}
    return {**body, "sha256": identity_sha256(body)}


def compile_addresses(root: Path):
    parent, items = _inputs()
    texts = sorted({row["text"] for item in items for row in item[-1]}, key=quote_sha256)
    text_ids = [quote_sha256(text) for text in texts]
    queries = [item[5] for item in items]
    root.mkdir(parents=True, exist_ok=False)
    preflight, _ = publish_sealed_json(root / "address-preflight.json", {
        "format": FORMAT + "-address-preflight", "parent_selection_sha256": parent.sha256,
        "implementation": _implementation(), "raw_text_sha256s": text_ids,
        "query_sha256s": [quote_sha256(q) for q in queries],
        "policy": "pinned BGE-M3 cosine; descending individual global-excerpt score; exact-ID tie break; keep every excerpt and the original E/F/advisory tail",
        "provider_calls": 0, "gold_loaded": False,
    })
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    service = EmbeddingService(device="cuda", batch_size=8, verify_checkpoint=True)
    started = time.perf_counter()
    matrices, query_vectors, query_times = [], [], []
    try:
        for start in range(0, len(texts), 64):
            matrices.append(service.embed_queries(texts[start:start + 64]))
            print(json.dumps({"embedded_raw_excerpts": min(start + 64, len(texts)), "total": len(texts)}), flush=True)
        compilation_seconds = time.perf_counter() - started
        for query in queries:
            before = time.perf_counter()
            query_vectors.append(service.embed_query(query))
            query_times.append(time.perf_counter() - before)
        identity = {**service.execution_identity, "model_id": service.model_name,
                    "model_revision": service.model_revision, "checkpoint_sha256": service.checkpoint_sha256}
    finally:
        service.close()
    raw = np.concatenate(matrices).astype(np.float32)
    query = np.asarray(query_vectors, dtype=np.float32)
    for matrix in (raw, query):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or (norms <= 0).any():
            raise ValueError("invalid embedding matrix")
        matrix /= norms
    np.save(root / "raw-vectors.npy", raw, allow_pickle=False)
    np.save(root / "query-vectors.npy", query, allow_pickle=False)
    artifact, _ = publish_sealed_json(root / "addresses.json", {
        "format": FORMAT + "-addresses", "preflight_sha256": preflight.sha256,
        "parent_selection_sha256": parent.sha256, "implementation": _implementation(),
        "raw_text_sha256s": text_ids, "query_sha256s": [quote_sha256(q) for q in queries],
        "raw_vectors_sha256": file_sha256(root / "raw-vectors.npy"),
        "query_vectors_sha256": file_sha256(root / "query-vectors.npy"), "embedding_identity": identity,
        "provider_calls": 0, "gold_loaded": False,
    })
    publish_sealed_json(root / "address-runtime.json", {
        "addresses_sha256": artifact.sha256, "raw_compilation_including_model_load_seconds": compilation_seconds,
        "query_encode_seconds": query_times,
        "scope": "selected-excerpt address preparation, not full-corpus ingestion or end-to-end retrieval",
    })
    print(json.dumps({"addresses_sha256": artifact.sha256, "raw_excerpt_count": len(texts)}), flush=True)


def build_selection(root: Path):
    parent, items = _inputs()
    artifact = read_sealed_json(root / "addresses.json")
    address = artifact.payload
    if address["parent_selection_sha256"] != parent.sha256 or address["implementation"] != _implementation():
        raise ValueError("address input or implementation changed")
    for name in ("raw", "query"):
        if file_sha256(root / f"{name}-vectors.npy") != address[f"{name}_vectors_sha256"]:
            raise ValueError("compiled vector file changed")
    raw = np.load(root / "raw-vectors.npy", allow_pickle=False)
    query = np.load(root / "query-vectors.npy", allow_pickle=False)
    by_hash = dict(zip(address["raw_text_sha256s"], raw, strict=True))
    rows = []
    changed = 0
    for ordinal, (row, arm, prefix, context, question, query_text, globals_) in enumerate(items):
        if quote_sha256(query_text) != address["query_sha256s"][ordinal]:
            raise ValueError("query-address binding changed")
        by_id = {g["evidence_id"]: g for g in globals_}
        ids = sorted(by_id)
        matrix = np.asarray([by_hash[by_id[i]["raw_text_sha256"]] for i in ids], dtype=np.float32)
        with ExactDenseAddressIndex(ids, matrix) as index:
            ranked = index.search(query[ordinal], limit=len(ids))
        def block(g):
            return f"<{g['citation']}>\n[{g['created_at']} | {g['role']}] {g['text']}"
        original_global = "\n\n".join(block(g) for g in globals_)
        if not context.startswith(original_global):
            raise ValueError("global rendering changed")
        tail = context[len(original_global):]
        next_context = "\n\n".join(block(by_id[hit.chunk_id]) for hit in ranked) + tail
        successor = copy.deepcopy(arm)
        messages = [copy.deepcopy(arm["provider_messages"][0]), {"role": "user", "content": prefix + next_context + "\n\nQuestion: " + question + "\nShort answer:"}]
        if count_tokens(next_context) > 10000 or count_chat_prompt_token_proxy(messages) + 256 > 11000:
            raise ValueError("order-only candidate exceeds unchanged hard caps")
        encoded = canonical_json_bytes({"messages": messages})
        successor.update(provider_messages=messages, provider_payload_sha256=hashlib.sha256(encoded).hexdigest(),
                         provider_payload_utf8_bytes=len(encoded), context_token_proxy=count_tokens(next_context),
                         prompt_token_proxy=count_chat_prompt_token_proxy(messages),
                         prompt_workspace_token_proxy=count_chat_prompt_token_proxy(messages) + 256)
        changed += messages != arm["provider_messages"]
        audit = {"format": FORMAT, "parent_provider_payload_sha256": arm["provider_payload_sha256"],
                 "addresses_sha256": artifact.sha256, "raw_excerpt_membership_preserved": True,
                 "original_tail_sha256": quote_sha256(tail), "tail_preserved": True,
                 "system_and_framing_preserved": True, "provider_calls": 0, "frontier_closed": False,
                 "ordered_rows": [{"citation": by_id[hit.chunk_id]["citation"], "evidence_id": hit.chunk_id,
                                    "raw_text_sha256": by_id[hit.chunk_id]["raw_text_sha256"], "cosine": hit.score} for hit in ranked]}
        successor["dense_packet_order"] = {**audit, "receipt_sha256": identity_sha256(audit)}
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"], "question_id": row["question_id"],
                "prompt_question_sha256": row["prompt_question_sha256"], "parent_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": successor}}
        rows.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    result = harness._artifact(module=sys.modules[__name__], rows=rows, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "conventional_dense_presentation_order", "parent_selection_sha256": parent.sha256,
                       "addresses_sha256": artifact.sha256, "implementation": _implementation()})
    assert_gold_blind(result, path="dense_packet_order.selection")
    harness.validate_selection(result)
    return result, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "construct", "verify"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compile":
        compile_addresses(args.output_root)
        return 0
    result, changed = build_selection(args.output_root)
    path = args.output_root / "selection.json"
    if args.command == "construct":
        artifact, created = publish_sealed_json(path, result)
    else:
        artifact = read_sealed_json(path)
        if artifact.payload != result:
            raise ValueError("selection differs from exact frozen parent/address replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, "changed_prompts": changed, "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
