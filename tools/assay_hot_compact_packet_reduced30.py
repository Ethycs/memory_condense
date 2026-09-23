"""Conventional bounded admission of complete raw G/E evidence units.

Unlike the presentation ablations this deliberately omits lower-ranked units.
Every admitted exchange includes its exact G-reference dependencies. Omission
is recorded explicitly and never grants factual completeness or absence.
"""

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
from tools import assay_hot_cross_encoder_order_reduced30 as cross
from tools import assay_hot_reduced30_construction as harness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, canonical_json_bytes, identity_sha256

FORMAT = "memory-condense-hot-compact-conventional-packet-v1"
MAX_CONTEXT_TOKENS = 4000
MAX_PRIMARY_UNITS = 24
SCORE_ROOT = Path("eval_results/longmemeval-fast-cross-encoder-order-reduced30-20260908-r1")
SCORE_SHA256 = "2c6e51f0a0159b894f3933da511a7b0f836bbfc75924335bfd3b737b6e38033c"


def units_for(arm, context, globals_):
    renderer = cross.dense.legacy
    rows = [{**g, "chunk_id": g["evidence_id"]} for g in globals_]
    units = [{"label": g["citation"], "block": f"<{g['citation']}>\n[{g['created_at']} | {g['role']}] {g['text']}",
              "dependencies": [], "source_id": g["source_id"], "raw_ids": [g["evidence_id"]],
              "score_text": g["text"]} for g in globals_]
    by_id = {g["evidence_id"]: g for g in globals_}
    episodes = arm.get("episode_manifests", [])
    previous = renderer._render_context(rows, [], [])
    for i, episode in enumerate(episodes, 1):
        through = renderer._render_context(rows, episodes[:i], [])
        block = through[len(previous) + 2:]
        if not block.startswith(f"<E{i}>\n"):
            raise ValueError("episode unit boundary changed")
        deps = [by_id[ref["global_evidence_id"]]["citation"] for ref in episode["global_refs"]]
        score_text = block
        for ref in episode["global_refs"]:
            g = by_id[ref["global_evidence_id"]]
            score_text = score_text.replace(f"<REF {g['citation']}>", f"[{g['role']}] {g['text']}")
        units.append({"label": f"E{i}", "block": block, "dependencies": list(dict.fromkeys(deps)),
                      "source_id": episode["source_id"], "raw_ids": [r["chunk_id"] for r in episode["raw_rows"]],
                      "score_text": score_text})
        previous = through
    if not context.startswith(previous):
        raise ValueError("complete raw packet differs from its parent manifest")
    return units


def admit_units(units, scores, *, max_context_tokens=MAX_CONTEXT_TOKENS, max_primary_units=MAX_PRIMARY_UNITS):
    """Budget whole units and their dependency closure, using conservative costs."""
    labels = [u["label"] for u in units]
    by_label = {u["label"]: u for u in units}
    if len(labels) != len(by_label) or set(labels) != set(scores):
        raise ValueError("unit score population mismatch")
    for unit in units:
        if any(dep not in by_label or by_label[dep]["dependencies"] for dep in unit["dependencies"]):
            raise ValueError("unit has a foreign or nested dependency")
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("nonfinite unit score")
    costs = {u["label"]: count_tokens(u["block"]) + 2 for u in units}
    order = sorted(labels, key=lambda label: -scores[label])
    selected, primaries, spent = set(), [], 0
    for label in order:
        if label in selected:
            continue
        if len(primaries) == max_primary_units:
            break
        additions = ({label} | set(by_label[label]["dependencies"])) - selected
        cost = sum(costs[a] for a in additions)
        if spent + cost > max_context_tokens:
            continue
        selected.update(additions)
        primaries.append(label)
        spent += cost
    context = "\n\n".join(u["block"] for u in units if u["label"] in selected)
    if not selected or count_tokens(context) > max_context_tokens:
        raise ValueError("no nonempty whole-unit packet fits the exact context cap")
    return context, {"primary_labels": primaries, "selected_labels": [s for s in labels if s in selected],
                     "omitted_labels": [s for s in labels if s not in selected], "conservative_token_cost": spent,
                     "context_token_proxy": count_tokens(context), "frontier_closed": False}


def implementation():
    return {"assay_sha256": file_sha256(Path(__file__)), "cross_encoder_adapter": cross.implementation()}


def compile_scores(root):
    from sentence_transformers import CrossEncoder

    parent, items = cross.dense._inputs()
    previous = read_sealed_json(SCORE_ROOT / "scores.json")
    if previous.sha256 != SCORE_SHA256 or previous.payload["parent_selection_sha256"] != parent.sha256:
        raise ValueError("frozen global scores changed")
    root.mkdir(parents=True, exist_ok=False)
    preflight, _ = publish_sealed_json(root / "score-preflight.json", {
        "format": FORMAT + "-preflight", "parent_selection_sha256": parent.sha256, "global_scores_sha256": previous.sha256,
        "implementation": implementation(), "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_primary_units": MAX_PRIMARY_UNITS, "query_tokens": 128, "passage_tokens": 320,
        "model_max_length": 512, "batch_size": 16, "provider_calls": 0, "gold_loaded": False,
        "policy": "rank complete G/E units; admit dependency-closed units under the context cap; retain original display order; remove derived hints",
    })
    cross.verify_ms_marco_checkpoint(cross.MODEL_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder = CrossEncoder(str(cross.MODEL_DIR), device="cuda", local_files_only=True,
        trust_remote_code=False, max_length=512, model_kwargs={"use_safetensors": True})
    output, times = [], []
    try:
        for item, old in zip(items, previous.payload["questions"], strict=True):
            row, arm, _prefix, context, _question, query, globals_ = item
            if old["global_ordinal"] != row["global_ordinal"] or old["query_sha256"] != quote_sha256(query):
                raise ValueError("global score query changed")
            units = units_for(arm, context, globals_)
            old_by_id = {s["evidence_id"]: s for s in old["scores"]}
            scores = {}
            for g in globals_:
                if old_by_id[g["evidence_id"]]["raw_text_sha256"] != g["raw_text_sha256"]:
                    raise ValueError("global score raw binding changed")
                scores[g["citation"]] = old_by_id[g["evidence_id"]]["score"]
            episodes = [u for u in units if u["label"].startswith("E")]
            started = time.perf_counter()
            values = np.asarray(encoder.predict([(truncate_to_tokens(query, 128), truncate_to_tokens(u["score_text"], 320)) for u in episodes],
                batch_size=16, show_progress_bar=False, convert_to_numpy=True), dtype=float).reshape(-1) if episodes else []
            times.append(time.perf_counter() - started)
            scores.update({u["label"]: float(score) for u, score in zip(episodes, values, strict=True)})
            output.append({"global_ordinal": row["global_ordinal"], "query_sha256": quote_sha256(query),
                           "unit_population_sha256": identity_sha256(units), "scores": scores})
    finally:
        del encoder
    artifact, _ = publish_sealed_json(root / "scores.json", {"format": FORMAT + "-scores",
        "preflight_sha256": preflight.sha256, "parent_selection_sha256": parent.sha256,
        "implementation": implementation(), "questions": output, "provider_calls": 0, "gold_loaded": False})
    publish_sealed_json(root / "score-runtime.json", {"scores_sha256": artifact.sha256,
        "episode_scoring_seconds": times, "excludes": "global scoring, model load, corpus retrieval and providers"})
    print(json.dumps({"scores_sha256": artifact.sha256}), flush=True)


def build_selection(root):
    parent, items = cross.dense._inputs()
    artifact = read_sealed_json(root / "scores.json")
    if artifact.payload["parent_selection_sha256"] != parent.sha256 or artifact.payload["implementation"] != implementation():
        raise ValueError("compact score binding changed")
    output = []
    for item, scored in zip(items, artifact.payload["questions"], strict=True):
        row, arm, prefix, context, question, query, globals_ = item
        units = units_for(arm, context, globals_)
        if (scored["global_ordinal"] != row["global_ordinal"] or scored["query_sha256"] != quote_sha256(query)
            or scored["unit_population_sha256"] != identity_sha256(units)):
            raise ValueError("compact unit identity changed")
        compact, admission = admit_units(units, scored["scores"])
        messages = [copy.deepcopy(arm["provider_messages"][0]), {"role": "user", "content": prefix + compact + "\n\nQuestion: " + question + "\nShort answer:"}]
        encoded = canonical_json_bytes({"messages": messages})
        selected = set(admission["selected_labels"])
        selected_g = [g["evidence_id"] for g in globals_ if g["citation"] in selected]
        omitted_g = [g["evidence_id"] for g in globals_ if g["citation"] not in selected]
        audit = {"format": FORMAT, **admission, "scores_sha256": artifact.sha256,
                 "unit_population_sha256": identity_sha256(units), "parent_provider_payload_sha256": arm["provider_payload_sha256"],
                 "admitted_unit_bindings": [{"label": u["label"], "source_id": u["source_id"], "block_sha256": quote_sha256(u["block"]),
                                             "raw_ids": u["raw_ids"], "dependencies": u["dependencies"]} for u in units if u["label"] in selected],
                 "parent_raw_membership_preserved": not admission["omitted_labels"], "provider_calls": 0}
        candidate = {"mode": "compact_conventional_raw_units", "provider_messages": messages,
            "provider_payload_sha256": hashlib.sha256(encoded).hexdigest(), "provider_payload_utf8_bytes": len(encoded),
            "context_token_proxy": count_tokens(compact), "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "prompt_workspace_token_proxy": count_chat_prompt_token_proxy(messages) + 256,
            "rendered_parent_evidence_ids": selected_g, "global_raw_selected_evidence_ids": selected_g,
            "global_raw_omitted_evidence_ids": omitted_g, "parent_all_rendered": not omitted_g, "parent_rows_protected": False,
            "rendered_fact_ids": [],
            "packed_chunk_ids": list(dict.fromkeys(r for u in units if u["label"] in selected for r in u["raw_ids"])),
            "rendered_raw_chunk_ids": list(dict.fromkeys(r for u in units if u["label"] in selected and u["label"].startswith("E") for r in u["raw_ids"])),
            "compact_admission": {**audit, "receipt_sha256": identity_sha256(audit)}}
        if candidate["prompt_workspace_token_proxy"] > 5500:
            raise ValueError("compact workspace cap exceeded")
        body = {"format": FORMAT + "-source-row", "ordinal": row["global_ordinal"], "question_id": row["question_id"],
                "prompt_question_sha256": row["prompt_question_sha256"], "parent_row_receipt_sha256": row["row_receipt_sha256"],
                "arms": {"a3_protected_union": candidate}}
        output.append((row["global_ordinal"], {**body, "row_receipt_sha256": identity_sha256(body)}))
    result = harness._artifact(module=sys.modules[__name__], rows=output, arm_path="arms.a3_protected_union",
        input_binding={"input_mode": "compact_conventional_admission", "parent_selection_sha256": parent.sha256,
                       "scores_sha256": artifact.sha256, "implementation": implementation()})
    assert_gold_blind(result, path="compact_packet.selection")
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
            raise ValueError("compact packet differs from frozen parent/score replay")
        created = False
    print(json.dumps({"selection_sha256": artifact.sha256, "created": created, "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
