"""Ablate derived F/advisory hints while retaining all conventional raw evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_dense_order_reduced30 as base
from tools import assay_hot_reduced30_construction as harness
from tools import assay_hot_v7_spine_episode_fact_reserved_full100 as renderer
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256

FORMAT = "memory-condense-hot-raw-packet-ablation-v1"


def compose(arm, prefix, context, question, globals_):
    rows = [{**g, "chunk_id": g["evidence_id"]} for g in globals_]
    episodes = arm.get("episode_manifests", [])
    raw = renderer._render_context(rows, episodes, [])
    if not context.startswith(raw):
        raise ValueError("exact raw parent context changed")
    tail = context[len(raw):]
    if tail and not tail.startswith("\n\n<FACTS "):
        raise ValueError("unrecognized derived tail; refusing to remove possible raw evidence")
    # Numeric completion rows must already be hydrated as raw evidence. The
    # parent conservation receipt and exact original rendering remain bound.
    visible_labels = renderer._provider_label_index(rows, episodes)
    for binding in arm.get("numeric_slot_completion", {}).get("provider_bindings", []):
        for kind in ("anchor", "backing"):
            evidence_id = binding.get(f"{kind}_evidence_id")
            label = binding.get(f"{kind}_provider_label")
            if not evidence_id or not label or visible_labels.get(evidence_id) != label:
                raise ValueError("numeric completion hint lacks retained raw backing")
    messages = [copy.deepcopy(arm["provider_messages"][0]), {"role": "user", "content": prefix + raw + "\n\nQuestion: " + question + "\nShort answer:"}]
    result = copy.deepcopy(arm)
    removed = {}
    for key in ("fact_ledger", "fact_advisory", "numeric_slot_completion", "fact_budget_reservation"):
        if key in result:
            removed[key] = result.pop(key)
    result["rendered_fact_ids"] = []
    if "rendered_partitions" in result:
        result["rendered_partitions"]["fact_count"] = 0
    if "provider_provenance_manifest" in result:
        result["provider_provenance_manifest"] = renderer._provider_provenance_manifest(
            arm["global_citation_manifest"], episodes, [])
    payload = canonical_json_bytes({"messages": messages})
    result.update(provider_messages=messages, provider_payload_sha256=hashlib.sha256(payload).hexdigest(),
        provider_payload_utf8_bytes=len(payload), context_token_proxy=count_tokens(raw),
        prompt_token_proxy=count_chat_prompt_token_proxy(messages),
        prompt_workspace_token_proxy=count_chat_prompt_token_proxy(messages) + 256)
    for metric, value in (("overlay_context_token_delta", "context_token_proxy"), ("overlay_workspace_token_delta", "prompt_workspace_token_proxy")):
        if metric in result:
            result[metric] += result[value] - arm[value]
    audit = {"format": FORMAT, "parent_provider_payload_sha256": arm["provider_payload_sha256"],
             "raw_prefix_sha256": quote_sha256(raw), "removed_derived_tail_sha256": quote_sha256(tail),
             "removed_derived_tail_tokens": count_tokens(tail), "all_raw_preserved": True,
             "system_and_framing_preserved": True, "removed_hints_parent_audit": removed,
             "provider_calls": 0, "frontier_closed": False}
    result["raw_packet_ablation"] = {**audit, "receipt_sha256": identity_sha256(audit)}
    return result


def build_selection():
    parent, items = base._inputs()
    rows = []
    for row, arm, prefix, context, question, _query, globals_ in items:
        successor = compose(arm, prefix, context, question, globals_)
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"], "question_id": row["question_id"],
                "prompt_question_sha256": row["prompt_question_sha256"], "parent_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": successor}}
        rows.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    result = harness._artifact(module=sys.modules[__name__], rows=rows, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "raw_packet_without_derived_tail", "parent_selection_sha256": parent.sha256,
                       "implementation": {"assay_sha256": file_sha256(Path(__file__)), "parent_adapter": base._implementation()}})
    assert_gold_blind(result, path="raw_packet.selection")
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
            raise ValueError("raw-only packet differs from exact parent replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
