from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_v3_provider_free_witness as packet_tools
from tools import assay_hot_v4_user_envelope_provider_selection as assay
from tools.matched_eval.contracts import identity_sha256


def _source_row() -> dict[str, object]:
    question = "[Question asked at 2026-02-01] What happened?"
    raw = "The user-led exchange says the launch happened on Tuesday."
    rendered = f"[2026-01-01T00:00:00+00:00 | user] {raw}"
    evidence = {
        "chunk_id": "1" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
        "evidence_id": "1" * 64,
        "raw_text": raw,
        "raw_text_sha256": quote_sha256(raw),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": "user",
        "route": "fixture",
        "score": 1.0,
        "source_id": "source-a",
        "turn_id": "turn-a",
    }
    arm, _audit = packet_tools._pack_ranked_raw_evidence(  # noqa: SLF001
        [evidence],
        prompt_question=question,
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )
    body = {
        "effective_arm": arm,
        "ordinal": 0,
        "prompt_question_sha256": quote_sha256(question),
        "question_id": "q0",
    }
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _selection(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    return assay._project_selection(  # noqa: SLF001
        {"questions": [_source_row()]},
        construction_sha256=assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
        runtime_sha256=assay.EXPECTED_SOURCE_RUNTIME_SHA256,
        replay_sha256=assay.EXPECTED_SOURCE_REPLAY_SHA256,
    )


def test_compact_selection_round_trips_and_rebinds_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(monkeypatch)
    assay.hot._atomic_write_json(tmp_path / assay.SELECTION_NAME, selection)  # noqa: SLF001
    loaded, _digest = assay._load_selection(tmp_path)  # noqa: SLF001
    assert loaded == selection

    tampered = copy.deepcopy(selection)
    row = tampered["questions"][0]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]
    messages = arm["provider_messages"]
    messages[1]["content"] = messages[1]["content"].replace(
        "What happened?", "When was the launch?"
    )
    payload = assay.hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    arm["provider_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    arm["provider_payload_utf8_bytes"] = len(payload)
    arm["prompt_token_proxy"] = assay.hot.count_chat_prompt_token_proxy(messages)
    arm["prompt_workspace_token_proxy"] = (
        arm["prompt_token_proxy"] + assay.hot.RESPONDER_OUTPUT_TOKEN_RESERVE
    )
    unsigned = dict(row)
    unsigned.pop("row_receipt_sha256")
    row["row_receipt_sha256"] = identity_sha256(unsigned)
    second = tmp_path / "tampered"
    assay.hot._atomic_write_json(second / assay.SELECTION_NAME, tampered)  # noqa: SLF001
    with pytest.raises(ValueError, match="question binding"):
        assay._load_selection(second)  # noqa: SLF001


def test_compact_loader_rejects_workspace_arithmetic_even_with_new_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(monkeypatch)
    row = selection["questions"][0]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]
    arm["prompt_workspace_token_proxy"] -= 1
    unsigned = dict(row)
    unsigned.pop("row_receipt_sha256")
    row["row_receipt_sha256"] = identity_sha256(unsigned)
    assay.hot._atomic_write_json(tmp_path / assay.SELECTION_NAME, selection)  # noqa: SLF001
    with pytest.raises(ValueError, match="token boundary"):
        assay._load_selection(tmp_path)  # noqa: SLF001


def test_source_loader_opens_only_the_three_gold_free_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _source_row()
    construction = {
        "format": assay.SOURCE_CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "question_count": 1,
        "questions": [row],
    }
    runtime = {
        "construction_sha256": "a" * 64,
        "format": assay.SOURCE_RUNTIME_FORMAT,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
    }
    replay = {
        "byte_identical": True,
        "construction_sha256": "a" * 64,
        "format": assay.SOURCE_REPLAY_FORMAT,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "question_count": 1,
        "runtime_sha256": "b" * 64,
    }
    artifacts = {
        "construction.json": (construction, "a" * 64),
        "runtime.json": (runtime, "b" * 64),
        "replay.json": (replay, "c" * 64),
    }
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "EXPECTED_SOURCE_CONSTRUCTION_SHA256", "a" * 64)
    monkeypatch.setattr(assay, "EXPECTED_SOURCE_RUNTIME_SHA256", "b" * 64)
    monkeypatch.setattr(assay, "EXPECTED_SOURCE_REPLAY_SHA256", "c" * 64)
    opened: list[str] = []

    def read(path: Path) -> tuple[dict[str, object], str]:
        opened.append(path.name)
        return artifacts[path.name]

    monkeypatch.setattr(assay.hot, "_read_json_artifact", read)
    loaded, *_hashes = assay._load_source(Path("source"))  # noqa: SLF001
    assert loaded is construction
    assert opened == ["construction.json", "runtime.json", "replay.json"]
