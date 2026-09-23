"""Replace derived fact hints with conventional, verbatim retrieval highlights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.search.packing.evidence_highlights import HighlightCandidate, render_evidence_highlights
from tools import assay_hot_raw_packet_reduced30 as raw
from tools import assay_hot_reduced30_construction as harness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256

FORMAT = "memory-condense-hot-verbatim-highlight-packet-v1"
SCORE_ROOT = Path("eval_results/longmemeval-fast-cross-encoder-order-reduced30-20260908-r1")
SCORES_SHA256 = "2c6e51f0a0159b894f3933da511a7b0f836bbfc75924335bfd3b737b6e38033c"


def build_selection():
    parent, items = raw.base._inputs()
    scores = read_sealed_json(SCORE_ROOT / "scores.json")
    if scores.sha256 != SCORES_SHA256 or scores.payload["parent_selection_sha256"] != parent.sha256:
        raise ValueError("frozen conventional score binding changed")
    output = []
    for item, score_row in zip(items, scores.payload["questions"], strict=True):
        row, arm, prefix, context, question, query, globals_ = item
        if score_row["global_ordinal"] != row["global_ordinal"] or score_row["query_sha256"] != quote_sha256(query):
            raise ValueError("highlight question binding changed")
        by_id = {g["evidence_id"]: g for g in globals_}
        if set(by_id) != {s["evidence_id"] for s in score_row["scores"]}:
            raise ValueError("highlight candidate population changed")
        ranked = sorted(score_row["scores"], key=lambda s: -s["score"])
        candidates = []
        for score in ranked:
            g = by_id[score["evidence_id"]]
            if g["raw_text_sha256"] != score["raw_text_sha256"]:
                raise ValueError("highlight text binding changed")
            candidates.append(HighlightCandidate(g["citation"], g["role"], g["created_at"], g["text"]))
        highlights = render_evidence_highlights(candidates)
        successor = raw.compose(arm, prefix, context, question, globals_)
        # Use the exact prefix authenticated by the raw ablation; apparent
        # delimiters in raw data must never shorten it.
        user = successor["provider_messages"][1]["content"]
        raw_context = user[len(prefix):].rpartition("\n\nQuestion: ")[0]
        next_context = raw_context + ("\n\n" + highlights.text if highlights.text else "")
        messages = [dict(arm["provider_messages"][0]), {"role": "user", "content": prefix + next_context + "\n\nQuestion: " + question + "\nShort answer:"}]
        if count_tokens(next_context) > 10000 or count_chat_prompt_token_proxy(messages) + 256 > 11000:
            raise ValueError("highlight packet exceeds unchanged caps")
        payload = canonical_json_bytes({"messages": messages})
        old_context_tokens = successor["context_token_proxy"]
        old_workspace_tokens = successor["prompt_workspace_token_proxy"]
        successor.update(provider_messages=messages, provider_payload_sha256=hashlib.sha256(payload).hexdigest(),
            provider_payload_utf8_bytes=len(payload), context_token_proxy=count_tokens(next_context),
            prompt_token_proxy=count_chat_prompt_token_proxy(messages),
            prompt_workspace_token_proxy=count_chat_prompt_token_proxy(messages) + 256)
        if "overlay_context_token_delta" in successor:
            successor["overlay_context_token_delta"] += successor["context_token_proxy"] - old_context_tokens
        if "overlay_workspace_token_delta" in successor:
            successor["overlay_workspace_token_delta"] += successor["prompt_workspace_token_proxy"] - old_workspace_tokens
        audit = {"format": FORMAT, "scores_sha256": scores.sha256, "bindings": list(highlights.bindings),
                 "highlight_token_count": count_tokens(highlights.text), "raw_membership_preserved": True,
                 "provider_calls": 0, "frontier_closed": False}
        successor["verbatim_highlights"] = {**audit, "receipt_sha256": identity_sha256(audit)}
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"], "question_id": row["question_id"],
                "prompt_question_sha256": row["prompt_question_sha256"], "parent_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": successor}}
        output.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    project = Path(__file__).resolve().parents[1]
    result = harness._artifact(module=sys.modules[__name__], rows=output, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "verbatim_conventional_highlights", "parent_selection_sha256": parent.sha256,
                       "scores_sha256": scores.sha256, "implementation": {
                           "assay_sha256": file_sha256(Path(__file__)), "raw_adapter_sha256": file_sha256(Path(raw.__file__)),
                           "renderer_sha256": file_sha256(project / "src/memory_condense/search/packing/evidence_highlights.py"),
                           "parent_adapter": raw.base._implementation()}})
    assert_gold_blind(result, path="highlight_packet.selection")
    harness.validate_selection(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("construct", "verify"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_selection()
    path = args.output_root / "selection.json"
    if args.command == "construct":
        artifact, created = publish_sealed_json(path, payload)
    else:
        artifact = read_sealed_json(path)
        if artifact.payload != payload:
            raise ValueError("highlight packet differs from frozen replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
