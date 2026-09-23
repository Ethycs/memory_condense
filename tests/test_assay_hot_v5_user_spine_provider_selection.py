from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._retrieval_qa_prompt import build_qa_prompt
from tools import assay_hot_v4_user_envelope_provider_selection as source_assay
from tools import assay_hot_v5_operation_aware_provider_selection as operation_a
from tools import assay_hot_v5_user_spine_provider_selection as assay
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_v5_user_spine_prompt import (
    HARD_WORKSPACE_TOKEN_CAP,
    OPERATION_AWARE_SYSTEM_PROMPT,
)


def _evidence(
    evidence_id: str,
    source_id: str,
    role: str,
    text: str,
) -> dict[str, object]:
    rendered = f"[2026-09-06T00:00:00+00:00 | {role}] {text}"
    return {
        "chunk_id": evidence_id,
        "created_at": "2026-09-06T00:00:00+00:00",
        "evidence_id": evidence_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": role,
        "route": "fixture",
        "score": 1.0,
        "source_id": source_id,
        "turn_id": f"turn-{evidence_id}",
    }


def _construction() -> dict[str, object]:
    question = "[Question asked at 2026-09-07] What was recommended?"
    assistant = _evidence("b-assistant", "source-b", "assistant", "Use blue.")
    user = _evidence("b-user", "source-b", "user", "I prefer blue.")
    other = _evidence("a-user", "source-a", "user", "I need a bicycle.")
    packed = [assistant, user, copy.deepcopy(assistant), other]
    source_arm = {
        "packed_chunk_ids": [row["evidence_id"] for row in packed],
        "packed_evidence": packed,
        "provider_messages": build_qa_prompt(
            question, [str(row["rendered_text"]) for row in packed]
        ),
    }
    body = {
        "effective_arm": source_arm,
        "ordinal": 0,
        "prompt_question_sha256": quote_sha256(question),
        "question_id": "q0",
    }
    row = {**body, "row_receipt_sha256": identity_sha256(body)}
    return {"questions": [row]}


def _project(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    return assay._project_selection(  # noqa: SLF001
        _construction(),
        construction_sha256=assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
        runtime_sha256=assay.EXPECTED_SOURCE_RUNTIME_SHA256,
        replay_sha256=assay.EXPECTED_SOURCE_REPLAY_SHA256,
    )


def test_projection_conserves_selected_multiset_then_exact_id_dedups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction = _construction()
    projected = _project(monkeypatch)
    source_row = construction["questions"][0]  # type: ignore[index]
    source_arm = source_row["effective_arm"]  # type: ignore[index]
    row = projected["questions"][0]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]
    selected = source_arm["packed_evidence"]
    selected_ids = [item["evidence_id"] for item in selected]

    assert row["ordinal"] == source_row["ordinal"] == 0
    assert row["question_id"] == source_row["question_id"] == "q0"
    assert row["prompt_question_sha256"] == source_row["prompt_question_sha256"]
    assert arm["selected_population_sha256"] == identity_sha256(selected)
    assert arm["selected_evidence_ids"] == selected_ids
    assert arm["selected_row_sha256s"] == [
        identity_sha256(item) for item in selected
    ]
    assert arm["selected_evidence_count"] == 4
    assert arm["retained_evidence_ids"] == [
        "b-assistant",
        "b-user",
        "a-user",
    ]
    assert arm["dedup_excluded_evidence_count"] == 1
    assert arm["dedup_stage"] == "post_selection_exact_evidence_id"
    assert arm["unique_selected_rows_omitted"] == 0
    assert Counter(arm["rendered_evidence_ids"]) == Counter(
        arm["retained_evidence_ids"]
    )
    assert arm["provider_messages"][0] == {
        "role": "system",
        "content": OPERATION_AWARE_SYSTEM_PROMPT,
    }
    assert arm["provider_messages"][0]["content"] == (
        operation_a.OPERATION_AWARE_SYSTEM_PROMPT
    )
    user_content = arm["provider_messages"][1]["content"]
    assert user_content.index("I prefer blue.") < user_content.index("Use blue.")
    assert user_content.count("Use blue.") == 1
    assert arm["prompt_workspace_token_proxy"] <= HARD_WORKSPACE_TOKEN_CAP
    assert projected["aggregate"] == {
        "dedup_excluded_evidence_count": 1,
        "max_prompt_workspace_token_proxy": arm[
            "prompt_workspace_token_proxy"
        ],
        "retained_evidence_count": 3,
        "selected_evidence_count": 4,
        "selected_populations_sha256": identity_sha256(
            [arm["selected_population_sha256"]]
        ),
        "unique_selected_rows_omitted": 0,
    }
    assert projected["provider_calls"] == 0
    assert projected["gold_fields_present"] is False
    assert_gold_blind(projected)

    assay.hot._atomic_write_json(  # noqa: SLF001
        tmp_path / assay.SELECTION_NAME, projected
    )
    loaded, _digest = assay._load_selection(tmp_path)  # noqa: SLF001
    assert loaded == projected


def test_materialize_uses_only_sealed_r2_triple_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    observed: list[Path] = []

    def load(root: Path) -> tuple[dict[str, object], str, str, str]:
        observed.append(root)
        return (
            _construction(),
            assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
            assay.EXPECTED_SOURCE_RUNTIME_SHA256,
            assay.EXPECTED_SOURCE_REPLAY_SHA256,
        )

    monkeypatch.setattr(source_assay, "_load_source", load)
    source_root = tmp_path / "sealed-source-with-unread-evaluation"
    output_root = tmp_path / "fresh-output"
    digest = assay.materialize(source_root=source_root, output_root=output_root)

    assert observed == [source_root]
    assert digest == hashlib.sha256(
        (output_root / assay.SELECTION_NAME).read_bytes()
    ).hexdigest()
    assert {path.name for path in output_root.iterdir()} == {
        assay.SELECTION_NAME,
        f"{assay.SELECTION_NAME}.sha256",
    }


def test_loader_fails_closed_on_claimed_unique_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected = _project(monkeypatch)
    tampered = copy.deepcopy(projected)
    row = tampered["questions"][0]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]
    arm["unique_selected_rows_omitted"] = 1
    tampered["aggregate"]["unique_selected_rows_omitted"] = 1  # type: ignore[index]
    unsigned = dict(row)
    unsigned.pop("row_receipt_sha256")
    row["row_receipt_sha256"] = identity_sha256(unsigned)
    assay.hot._atomic_write_json(  # noqa: SLF001
        tmp_path / assay.SELECTION_NAME, tampered
    )

    with pytest.raises(ValueError, match="evidence conservation"):
        assay._load_selection(tmp_path)  # noqa: SLF001


def test_source_population_and_question_identity_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    construction = _construction()
    row = construction["questions"][0]  # type: ignore[index]
    row["question_id"] = ""
    unsigned = dict(row)
    unsigned.pop("row_receipt_sha256")
    row["row_receipt_sha256"] = identity_sha256(unsigned)

    with pytest.raises(ValueError):
        assay._project_selection(  # noqa: SLF001
            construction,
            construction_sha256=assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
            runtime_sha256=assay.EXPECTED_SOURCE_RUNTIME_SHA256,
            replay_sha256=assay.EXPECTED_SOURCE_REPLAY_SHA256,
        )
