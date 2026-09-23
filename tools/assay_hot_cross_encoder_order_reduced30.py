"""Conventional MiniLM reranking control over the fixed raw fast packet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens, truncate_to_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.search.selectors.cross_encoder_selector import (
    MS_MARCO_MODEL_ID, MS_MARCO_MODEL_REVISION, verify_ms_marco_checkpoint,
)
from tools import assay_hot_dense_order_reduced30 as dense
from tools import assay_hot_reduced30_construction as harness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256

FORMAT = "memory-condense-hot-cross-encoder-packet-order-v1"
MODEL_DIR = Path("F:/Keytone/Documents/GitHub/memory_condense/.cache/models/ms-marco-MiniLM-L6-v2")


def implementation():
    root = Path(__file__).resolve().parents[1]
    body = {"assay_sha256": file_sha256(Path(__file__)), "parent_adapter": dense._implementation(),
            "selector_sha256": file_sha256(root / "src/memory_condense/search/selectors/cross_encoder_selector.py")}
    return {**body, "sha256": identity_sha256(body)}


def compile_scores(root: Path):
    from sentence_transformers import CrossEncoder

    parent, items = dense._inputs()
    root.mkdir(parents=True, exist_ok=False)
    weights = verify_ms_marco_checkpoint(MODEL_DIR)
    model_files = {name: file_sha256(MODEL_DIR / name) for name in (
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")}
    preflight, _ = publish_sealed_json(root / "score-preflight.json", {
        "format": FORMAT + "-preflight", "parent_selection_sha256": parent.sha256,
        "implementation": implementation(), "model_id": MS_MARCO_MODEL_ID,
        "model_revision": MS_MARCO_MODEL_REVISION, "weights_sha256": weights, "model_files": model_files,
        "query_token_proxy_cap": 128, "passage_token_proxy_cap": 320, "pair_model_token_cap": 512,
        "batch_size": 16, "device": "cuda", "candidate_scope": "all already-selected global excerpts",
        "policy": "descending relevance logit; stable input-order ties; no deletion; E/F tail unchanged",
        "provider_calls": 0, "gold_loaded": False,
    })
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder = CrossEncoder(str(MODEL_DIR), device="cuda", local_files_only=True,
                           trust_remote_code=False, max_length=512, model_kwargs={"use_safetensors": True})
    rows, timings = [], []
    try:
        for row, _arm, _prefix, _context, _question, query, globals_ in items:
            pairs = [(truncate_to_tokens(query, 128), truncate_to_tokens(g["text"], 320)) for g in globals_]
            started = time.perf_counter()
            values = np.asarray(encoder.predict(pairs, batch_size=16, show_progress_bar=False,
                                                convert_to_numpy=True), dtype=float).reshape(-1)
            timings.append(time.perf_counter() - started)
            if len(values) != len(globals_) or not np.isfinite(values).all():
                raise ValueError("invalid cross-encoder scores")
            rows.append({"global_ordinal": row["global_ordinal"], "query_sha256": quote_sha256(query),
                         "scores": [{"evidence_id": g["evidence_id"], "raw_text_sha256": g["raw_text_sha256"],
                                     "scored_text_sha256": quote_sha256(pair[1]), "score": float(score)}
                                    for g, pair, score in zip(globals_, pairs, values, strict=True)]})
            print(json.dumps({"scored_global_ordinal": row["global_ordinal"], "candidates": len(globals_)}), flush=True)
    finally:
        del encoder
    artifact, _ = publish_sealed_json(root / "scores.json", {
        "format": FORMAT + "-scores", "preflight_sha256": preflight.sha256,
        "parent_selection_sha256": parent.sha256, "implementation": implementation(),
        "questions": rows, "provider_calls": 0, "gold_loaded": False,
    })
    publish_sealed_json(root / "score-runtime.json", {"scores_sha256": artifact.sha256,
        "per_question_seconds": timings, "scope": "reranking only, excluding model load, retrieval and answer generation"})
    print(json.dumps({"scores_sha256": artifact.sha256}), flush=True)


def build_selection(root: Path):
    parent, items = dense._inputs()
    artifact = read_sealed_json(root / "scores.json")
    scores = artifact.payload
    if scores["parent_selection_sha256"] != parent.sha256 or scores["implementation"] != implementation():
        raise ValueError("frozen score input changed")
    output = []
    for item, scored in zip(items, scores["questions"], strict=True):
        row, arm, prefix, context, question, query, globals_ = item
        if scored["global_ordinal"] != row["global_ordinal"] or scored["query_sha256"] != quote_sha256(query):
            raise ValueError("score question binding changed")
        for raw, score in zip(globals_, scored["scores"], strict=True):
            if (raw["evidence_id"] != score["evidence_id"] or raw["raw_text_sha256"] != score["raw_text_sha256"]
                or quote_sha256(truncate_to_tokens(raw["text"], 320)) != score["scored_text_sha256"]
                or not np.isfinite(score["score"])):
                raise ValueError("score raw-evidence binding changed")
        order = sorted(range(len(globals_)), key=lambda i: (-scored["scores"][i]["score"], i))
        blocks = [f"<{g['citation']}>\n[{g['created_at']} | {g['role']}] {g['text']}" for g in globals_]
        original_global = "\n\n".join(blocks)
        if not context.startswith(original_global):
            raise ValueError("original global raw rendering changed")
        tail = context[len(original_global):]
        next_context = "\n\n".join(blocks[i] for i in order) + tail
        messages = [copy.deepcopy(arm["provider_messages"][0]), {"role": "user", "content": prefix + next_context + "\n\nQuestion: " + question + "\nShort answer:"}]
        if count_tokens(next_context) > 10000 or count_chat_prompt_token_proxy(messages) + 256 > 11000:
            raise ValueError("order-only packet exceeds unchanged budget")
        successor = copy.deepcopy(arm)
        encoded = canonical_json_bytes({"messages": messages})
        successor.update(provider_messages=messages, provider_payload_sha256=hashlib.sha256(encoded).hexdigest(),
            provider_payload_utf8_bytes=len(encoded), context_token_proxy=count_tokens(next_context),
            prompt_token_proxy=count_chat_prompt_token_proxy(messages),
            prompt_workspace_token_proxy=count_chat_prompt_token_proxy(messages) + 256)
        audit = {"format": FORMAT, "parent_provider_payload_sha256": arm["provider_payload_sha256"],
                 "scores_sha256": artifact.sha256, "ordered_citations": [globals_[i]["citation"] for i in order],
                 "raw_membership_and_tail_preserved": True, "tail_sha256": quote_sha256(tail),
                 "system_and_framing_preserved": True, "frontier_closed": False, "provider_calls": 0}
        successor["cross_encoder_packet_order"] = {**audit, "receipt_sha256": identity_sha256(audit)}
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"], "question_id": row["question_id"],
                "prompt_question_sha256": row["prompt_question_sha256"], "parent_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": successor}}
        output.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    result = harness._artifact(module=sys.modules[__name__], rows=output, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "conventional_cross_encoder_presentation_order", "parent_selection_sha256": parent.sha256,
                       "scores_sha256": artifact.sha256, "implementation": implementation()})
    assert_gold_blind(result, path="cross_encoder_order.selection")
    harness.validate_selection(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "construct", "verify"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compile":
        compile_scores(args.output_root)
        return 0
    payload = build_selection(args.output_root)
    path = args.output_root / "selection.json"
    if args.command == "construct":
        artifact, created = publish_sealed_json(path, payload)
    else:
        artifact = read_sealed_json(path)
        if artifact.payload != payload:
            raise ValueError("selection differs from exact parent/score replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
