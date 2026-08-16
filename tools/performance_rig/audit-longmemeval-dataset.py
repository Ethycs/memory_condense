"""Audit a LongMemEval JSON before locking or spending provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "samples", "examples"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError("dataset root is not a record list or supported wrapper")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = _records(json.loads(args.dataset.read_text(encoding="utf-8")))
    ids = [str(row.get("question_id") or "") for row in rows]
    answer_types = Counter(type(row.get("answer")).__name__ for row in rows)
    numeric_answers = [
        {
            "question_id": str(row.get("question_id") or ""),
            "answer": row.get("answer"),
            "answer_type": type(row.get("answer")).__name__,
        }
        for row in rows
        if isinstance(row.get("answer"), (int, float))
        and not isinstance(row.get("answer"), bool)
    ]
    blank_string_answers = [
        str(row.get("question_id") or "")
        for row in rows
        if isinstance(row.get("answer"), str) and not row["answer"].strip()
    ]
    null_answers = [
        str(row.get("question_id") or "")
        for row in rows
        if row.get("answer") is None
    ]
    categories = Counter(str(row.get("question_type") or "") for row in rows)
    report = {
        "format": "memory-condense-longmemeval-dataset-audit-v1",
        "path": str(args.dataset.resolve()),
        "sha256": _sha256(args.dataset),
        "bytes": args.dataset.stat().st_size,
        "records": len(rows),
        "unique_question_ids": len(set(ids)),
        "blank_question_ids": sum(not question_id for question_id in ids),
        "duplicate_question_ids": len(ids) - len(set(ids)),
        "categories": dict(sorted(categories.items())),
        "answer_types": dict(sorted(answer_types.items())),
        "numeric_answer_count": len(numeric_answers),
        "numeric_answers": numeric_answers,
        "blank_string_answer_count": len(blank_string_answers),
        "blank_string_answer_ids": blank_string_answers,
        "null_answer_count": len(null_answers),
        "null_answer_ids": null_answers,
        "abstention_id_count": sum(question_id.endswith("_abs") for question_id in ids),
        "records_with_original_answer": sum(
            "original_answer" in row for row in rows
        ),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
