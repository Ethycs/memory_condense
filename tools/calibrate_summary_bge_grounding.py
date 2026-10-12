"""Calibrate the BGE summary-validator gate from sealed fuzzy failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from memory_condense.runtime.artifacts import save
from memory_condense.runtime.embedding import FastEmbedBGE
from memory_condense.runtime.semantic_grounding import (
    BGESourceGrounder,
    _CANDIDATE_ALGORITHM,
    semantic_candidate_spans,
)


# These provider outputs were inspected against their assigned source fragments.
# Pinning receipt/item identities prevents a later unrelated failure from being
# silently treated as a positive calibration example.
SUPPORTED_FAILURES = {
    "b4225c6468ddb71f64b01ec8239b255c6982310207cb33b5998336e7f972aa5f": ("T1", "T2", "T5"),
    "5c0dc920a39694ff2b91a1606822d6ea68501e3a0cdff184452b3bfb1959b0c0": ("T6",),
    "d1242ca2443b59293cc95ec7f980e3642dcad3cfbd7dd0dbf1ba9318272b4073": ("0",),
    "18cb76fab65419cc3c5156c6865c4f76f496936c2b4603417fc07f83ebff49c6": ("T1", "T2", "T4"),
    "ebabd29141b61e5730e7dc9cf72a4541bb17876da602832950e8a61cf192dd65": ("T4",),
    "00e68422ac6364bf61d084dcd79b594b8fa740a1180d520d579c39766a8c87bf": ("T0", "T1"),
    "fd34f8d6c11ebb958fd82f53270498c3331bfb9e63fb3d217ed982b0e78493c7": ("T0",),
    "17c257321c4708f85b717becda2e3deb28bea9e01c30f5962e3c5e22f0799856": ("0",),
    "fdc0742cbbd34a23dc3a288de2c9869a8c7e398214103b01ae3de5864fcb620a": ("T0", "T1", "T2", "T6"),
    "a2233826b86e183602b2edf3a992b45bda21f596d03378c43d1abacfe114394a": ("T6",),
}


def _examples(eval_root: Path) -> list[dict]:
    by_receipt = {}
    for root in sorted(eval_root.glob("longmemeval-s-native-release-sol-*")):
        for error in root.glob("runtime-runs/answer-*/summary-fallback/*/error.json"):
            directory = error.parent
            if directory.name not in SUPPORTED_FAILURES or not (directory / "response.json").exists():
                continue
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            content = json.loads((directory / "response.json").read_text(encoding="utf-8"))["content"]
            items = []
            if request["kind"] == "raw":
                fragments = json.loads(request["api_messages"][1]["content"])["fragments"]
                rows = json.loads(content)["atoms"]
                for fragment, row in zip(fragments, rows, strict=True):
                    if row.get("label") == fragment.get("label") and isinstance(row.get("summary"), str):
                        items.append(dict(item=str(fragment["label"]), proposal=row["summary"],
                                          source=fragment["fragment"]))
            elif request["kind"] == "merge":
                sources = [fragment["summary"] for fragment in request["typed_request"]["fragments"]]
                for line in content.splitlines():
                    if "\t" in line and line.split("\t", 1)[0].isdigit():
                        index = int(line.split("\t", 1)[0])
                        items.append(dict(item=str(index), proposal=line.split("\t", 1)[1],
                                          source=sources[index]))
            expected = SUPPORTED_FAILURES[directory.name]
            if tuple(item["item"] for item in items) != expected:
                raise ValueError(f"Sealed failure population changed for {directory.name}")
            by_receipt[directory.name] = items
    missing = sorted(set(SUPPORTED_FAILURES) - set(by_receipt))
    if missing:
        raise FileNotFoundError("Missing sealed validator receipts: " + ", ".join(missing))
    return [dict(receipt=receipt, **item) for receipt in SUPPORTED_FAILURES
            for item in by_receipt[receipt]]


def calibrate(eval_root: Path, encoder) -> dict:
    rows = _examples(eval_root)
    candidate_sets = [semantic_candidate_spans(row["source"], row["proposal"]) for row in rows]
    if any(not candidates for candidates in candidate_sets):
        raise ValueError("A supported failure produced no exact candidate spans")
    texts = tuple(dict.fromkeys(text for row, candidates in zip(rows, candidate_sets, strict=True)
                                for text in (row["proposal"], *candidates)))
    vectors = np.asarray(encoder.embed_queries(texts), dtype=np.float32)
    if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
        raise ValueError("BGE calibration vectors are invalid")
    by_text = dict(zip(texts, vectors, strict=True))
    positives, negatives, details = [], [], []
    for index, (row, candidates) in enumerate(zip(rows, candidate_sets, strict=True)):
        proposal = by_text[row["proposal"]]
        own = [(float(np.dot(proposal, by_text[text])), text) for text in candidates]
        positive_score, positive_text = max(own)
        foreign = [(float(np.dot(proposal, by_text[text])), other_index, text)
                   for other_index, (other, other_candidates) in enumerate(zip(
                       rows, candidate_sets, strict=True))
                   if other_index != index and other["source"] != row["source"]
                   for text in other_candidates]
        negative_score, negative_index, negative_text = max(foreign)
        positives.append(positive_score)
        negatives.append(negative_score)
        details.append({
            "receipt": row["receipt"], "item": row["item"],
            "positive_score": positive_score,
            "positive_text_sha256": hashlib.sha256(positive_text.encode()).hexdigest(),
            "hard_negative_score": negative_score,
            "hard_negative_receipt": rows[negative_index]["receipt"],
            "hard_negative_text_sha256": hashlib.sha256(negative_text.encode()).hexdigest(),
        })
    thresholds = sorted(set(positives + negatives))
    scored = []
    for threshold in thresholds:
        tp = sum(score >= threshold for score in positives)
        fp = sum(score >= threshold for score in negatives)
        scored.append((tp / len(positives) - fp / len(negatives), -fp, tp, threshold))
    _youden, negative_fp, true_positive, selected = max(scored)
    reviewed = math.floor(min(positives) * 100) / 100
    policy = {
        "format": "memory-condense-bge-summary-validator-calibration-v1",
        "model_id": encoder.model_name,
        "model_revision": encoder.model_revision,
        "candidate_algorithm": _CANDIDATE_ALGORITHM,
        "pre_review_cutoff": round(selected, 3),
        "reviewed_cutoff": reviewed,
    }
    grounder = BGESourceGrounder(encoder, policy=policy)
    pre_recovered = sum(grounder(row["source"], row["proposal"]) is not None for row in rows)
    reviewed_recovered = sum(grounder(row["source"], row["proposal"], reviewed=True) is not None
                             for row in rows)
    return dict(**policy, calibration={
        "positive_count": len(positives), "hard_negative_count": len(negatives),
        "pre_review_true_positive": true_positive, "false_positive": -negative_fp,
        "positive_min": min(positives), "positive_median": float(np.median(positives)),
        "negative_max": max(negatives), "negative_median": float(np.median(negatives)),
        "production_pre_review_recovered": pre_recovered,
        "production_reviewed_recovered": reviewed_recovered,
    }, details=details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("eval_results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.eval_root, FastEmbedBGE())
    receipt = save(args.output, report)
    print(json.dumps({"path": str(receipt.path), "sha256": receipt.sha256,
                      **report["calibration"]}, sort_keys=True))


if __name__ == "__main__":
    main()
