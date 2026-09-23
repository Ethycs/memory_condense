"""Audit post-question transcript dates in the fifty prepared evidence packets.

This reads already prepared development evidence and its exact raw stores. It
does not route, load references or predictions, or measure answer accuracy.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.summary_time_prior import _ASKED, mention_window
from tools import evaluate_spine_semantic_seeds as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


EXCERPT = re.compile(r"^<S(\d+)\.(\d+) \[([^|\n]+) \| (user|assistant|system|tool)\]>\n"
    r"(.*?)(?=\n<S\d+\.\d+ \[|\n</SECTION S\d+>)", re.M | re.S)


def audit(output):
    rows, bindings = [], []
    for offset in range(0, 50, 10):
        root = Path(f"eval_results/full1m-spine-semantic-seeds-joint-offset{offset:03d}-20260910-r1")
        frozen = evaluation.load_preflight(root)
        p = frozen.payload
        manifest = read_sealed_json(Path(p["index_root"]) / "index.json")
        raw = read_sealed_json(Path(p["index_root"]) / "raw-turns.json")
        if manifest.sha256 != p["index_manifest_sha256"] or raw.sha256 != manifest.payload["raw_turns_sha256"]:
            raise ValueError("prepared packet raw store binding changed")
        by_date_role, source_dates = defaultdict(list), defaultdict(set)
        for turn in raw.payload["turns"]:
            if quote_sha256(turn["text"]) != turn["text_sha256"]:
                raise ValueError("raw transcript text changed")
            by_date_role[(turn["created_at"], turn["role"])].append(turn)
            source_dates[turn["source_id"]].add(datetime.fromisoformat(turn["created_at"]).date())
        bindings.append({"offset": offset, "evaluation_preflight_sha256": frozen.sha256,
            "index_manifest_sha256": manifest.sha256, "raw_turns_sha256": raw.sha256,
            "source_count": len(source_dates), "sources_with_multiple_mention_days": sum(len(d) > 1 for d in source_dates.values())})
        for call in p["calls"]:
            if call["arm"] not in evaluation.MEMORY_ARMS:
                continue
            question = call["question"]
            match = _ASKED.match(question["prompt_question"])
            if match is None or _ASKED.sub("", question["prompt_question"]).strip() != question["retrieval_query"].strip():
                raise ValueError("missing or unbound question date")
            asked = datetime.strptime(match.group(1), "%Y/%m/%d").date()
            excerpts = []
            body = call["messages"][1]["content"]
            for match in EXCERPT.finditer(body):
                section, ordinal, stamp, role, text = match.groups()
                matches = [turn for turn in by_date_role[(stamp, role)] if text in turn["text"]]
                if not matches:
                    raise ValueError("framed excerpt has no exact raw transcript at its stated date and role")
                excerpts.append({"label": f"S{section}.{ordinal}", "created_at": stamp, "role": role,
                    "text_sha256": quote_sha256(text), "raw_text_tokens": count_tokens(text),
                    "matching_raw_turn_ids": sorted(turn["turn_id"] for turn in matches),
                    "after_question_day": datetime.fromisoformat(stamp).date() > asked})
            if not excerpts:
                raise ValueError("expected a nonempty complete prepared memory packet")
            markers = re.findall(r"^<S\d+\.\d+ \[", body, re.M)
            if len(markers) != len(excerpts) or len({e["label"] for e in excerpts}) != len(excerpts):
                raise ValueError("ambiguous or incomplete evidence framing")
            future = [e for e in excerpts if e["after_question_day"]]
            rows.append({"ordinal": question["ordinal"], "arm": call["arm"],
                "question_sha256": question["prompt_question_sha256"], "asked_day": asked.isoformat(),
                "messages_sha256": call["messages_sha256"], "existing_mention_window_active": mention_window(question["prompt_question"]) is not None,
                "excerpt_count": len(excerpts), "future_excerpt_count": len(future),
                "raw_text_tokens": sum(e["raw_text_tokens"] for e in excerpts),
                "future_raw_text_tokens": sum(e["raw_text_tokens"] for e in future), "excerpts": excerpts})
    totals = {}
    for arm in evaluation.MEMORY_ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if sorted(row["ordinal"] for row in selected) != list(range(50)):
            raise ValueError("all fifty development questions are required")
        totals[arm] = {"questions": 50, "questions_with_future_excerpts": sum(r["future_excerpt_count"] > 0 for r in selected),
            **{key: sum(r[key] for r in selected) for key in ("excerpt_count", "future_excerpt_count", "raw_text_tokens", "future_raw_text_tokens")}}
    result, _ = publish_sealed_json(output, {"format": "memory-condense-development50-future-evidence-audit-v1",
        "bindings": bindings, "rows": rows, "totals": totals,
        "same_day_excerpts_retained": True, "scope": "transcript mention date; no inference about event dates",
        "new_provider_calls": 0, "query_vectors_computed": False, "gold_loaded": False, "predictions_loaded": False,
        "router_changed": False, "answer_accuracy_measured": False, "full100_target_eligible": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print({"future_evidence_audit_sha256": result.sha256, "totals": totals, "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
