from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tools import assay_hot_retrieval_1m as assay
from tools import evaluate_hot_retrieval_answers as evaluator


def _selection() -> dict[str, object]:
    rows = []
    for ordinal, question_id in enumerate(evaluator.ORIGINAL_ORDERED_QUESTION_IDS):
        messages = [
            {"role": "system", "content": "answer from evidence"},
            {"role": "user", "content": f"evidence and question {ordinal}"},
        ]
        payload = assay._canonical_json_bytes({"messages": messages})  # noqa: SLF001
        rows.append(
            {
                "ordinal": ordinal,
                "question_id": question_id,
                "arms": {
                    "a3_protected_union": {
                        "raw_evidence_only": True,
                        "provider_messages": messages,
                        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "provider_payload_utf8_bytes": len(payload),
                        "prompt_workspace_token_proxy": 100,
                    }
                },
            }
        )
    return {
        "format": assay.SELECTION_FORMAT,
        "questions": rows,
        "gold_fields_present": False,
        "provider_calls": 0,
    }


def test_bound_selection_verifies_provider_payload_without_gold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    selection = _selection()
    monkeypatch.setattr(
        evaluator.assay,
        "_load_selection",
        lambda _root: (selection, "a" * 64),
    )

    observed, digest = evaluator._load_bound_selection(  # noqa: SLF001
        tmp_path,
        "a" * 64,
    )

    assert observed is selection
    assert digest == "a" * 64
    assert selection["gold_fields_present"] is False


def test_bound_selection_rejects_digest_or_payload_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    selection = _selection()
    monkeypatch.setattr(
        evaluator.assay,
        "_load_selection",
        lambda _root: (selection, "a" * 64),
    )
    with pytest.raises(ValueError, match="selection digest changed"):
        evaluator._load_bound_selection(tmp_path, "b" * 64)  # noqa: SLF001

    selection["questions"][0]["arms"]["a3_protected_union"][  # type: ignore[index]
        "provider_payload_utf8_bytes"
    ] += 1
    with pytest.raises(ValueError, match="provider payload binding changed"):
        evaluator._load_bound_selection(tmp_path, "a" * 64)  # noqa: SLF001


def test_answer_cli_has_no_dataset_surface_but_judge_requires_one(tmp_path) -> None:
    parser = evaluator._parser()  # noqa: SLF001
    answer = parser.parse_args(
        ["--output-root", str(tmp_path), "answer"]
    )
    assert answer.command == "answer"
    assert not hasattr(answer, "dataset")

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-root", str(tmp_path), "judge"])
    judge = parser.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "judge",
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )
    assert judge.command == "judge"
    assert judge.dataset == tmp_path / "gold.json"


def test_completion_records_must_match_unique_usage() -> None:
    record = SimpleNamespace(messages_sha256="a")
    batch = SimpleNamespace(
        unique_records=(record,),
        usage=SimpleNamespace(unique_calls=1),
    )
    assert evaluator._record_by_messages_sha(batch) == {"a": record}  # noqa: SLF001

    batch.usage.unique_calls = 2
    with pytest.raises(RuntimeError, match="record population changed"):
        evaluator._record_by_messages_sha(batch)  # noqa: SLF001
