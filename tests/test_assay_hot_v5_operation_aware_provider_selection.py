from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._retrieval_qa_prompt import build_qa_prompt
from tools import assay_hot_v4_user_envelope_provider_selection as source_assay
from tools import assay_hot_v5_operation_aware_provider_selection as assay
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.hot_v5_user_spine_prompt import (
    OPERATION_AWARE_SYSTEM_PROMPT,
)


def _source_selection() -> dict[str, object]:
    question = "[Question asked at 2026-09-07] What happened?"
    messages = build_qa_prompt(
        question,
        ["[2026-09-06T12:00:00+00:00 | user] The launch completed."],
    )
    payload = assay.hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    arm = {
        "context_token_proxy": 12,
        "packed_chunk_ids": ["1" * 64],
        "packed_evidence_sha256": "2" * 64,
        "prompt_token_proxy": assay.hot.count_chat_prompt_token_proxy(messages),
        "prompt_workspace_token_proxy": (
            assay.hot.count_chat_prompt_token_proxy(messages)
            + assay.hot.RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_evidence_only": True,
    }
    body = {
        "arms": {"a3_protected_union": arm},
        "format": source_assay.ROW_FORMAT,
        "local_ordinal": 0,
        "ordinal": 0,
        "prompt_question_sha256": quote_sha256(question),
        "question_id": "q0",
        "shard_offset": 0,
        "source_row_receipt_sha256": "3" * 64,
    }
    row = {**body, "row_receipt_sha256": identity_sha256(body)}
    return {
        "format": source_assay.FORMAT,
        "gold_fields_present": False,
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": 1,
        "questions": [row],
        "source_construction_sha256": source_assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256,
        "source_replay_sha256": source_assay.EXPECTED_SOURCE_REPLAY_SHA256,
        "source_runtime_sha256": source_assay.EXPECTED_SOURCE_RUNTIME_SHA256,
        "status": "sealed_gold_free_user_envelope_provider_packets",
    }


def _project(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    source = _source_selection()
    projected = assay._project_selection(  # noqa: SLF001
        source,
        source_selection_sha256=assay.EXPECTED_SOURCE_SELECTION_SHA256,
    )
    return source, projected


def test_projection_changes_only_system_message_and_prompt_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, projected = _project(monkeypatch)
    source_row = source["questions"][0]  # type: ignore[index]
    row = projected["questions"][0]  # type: ignore[index]
    source_arm = source_row["arms"]["a3_protected_union"]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]

    assert arm["provider_messages"][0] == {  # type: ignore[index]
        "role": "system",
        "content": OPERATION_AWARE_SYSTEM_PROMPT,
    }
    messages = arm["provider_messages"]
    source_messages = source_arm["provider_messages"]
    assert messages[1] == source_messages[1]
    assert messages[1]["content"].encode("utf-8") == source_messages[1][
        "content"
    ].encode("utf-8")
    for field in (
        "context_token_proxy",
        "packed_chunk_ids",
        "packed_evidence_sha256",
        "raw_evidence_only",
    ):
        assert arm[field] == source_arm[field]
    for field in (
        "ordinal",
        "local_ordinal",
        "shard_offset",
        "question_id",
        "prompt_question_sha256",
    ):
        assert row[field] == source_row[field]
    assert projected["provider_calls"] == 0
    assert projected["gold_fields_present"] is False

    assay.hot._atomic_write_json(  # noqa: SLF001
        tmp_path / assay.SELECTION_NAME, projected
    )
    loaded, _digest = assay._load_selection(tmp_path)  # noqa: SLF001
    assert loaded == projected


def test_materialize_delegates_to_the_single_source_selection_authenticator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    source = _source_selection()
    observed: list[Path] = []

    def load(root: Path) -> tuple[dict[str, object], str]:
        observed.append(root)
        return source, assay.EXPECTED_SOURCE_SELECTION_SHA256

    monkeypatch.setattr(source_assay, "_load_selection", load)
    source_root = tmp_path / "source-with-unread-gold-files"
    output_root = tmp_path / "fresh-output"
    digest = assay.materialize(source_root=source_root, output_root=output_root)

    assert observed == [source_root]
    assert digest == hashlib.sha256(
        (output_root / assay.SELECTION_NAME).read_bytes()
    ).hexdigest()


def test_loader_rejects_noncanonical_policy_even_with_recomputed_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, projected = _project(monkeypatch)
    tampered = copy.deepcopy(projected)
    row = tampered["questions"][0]  # type: ignore[index]
    arm = row["arms"]["a3_protected_union"]  # type: ignore[index]
    messages = arm["provider_messages"]
    messages[0]["content"] = "A different universal policy."
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
    assay.hot._atomic_write_json(  # noqa: SLF001
        tmp_path / assay.SELECTION_NAME, tampered
    )

    with pytest.raises(ValueError, match="prompt envelope"):
        assay._load_selection(tmp_path)  # noqa: SLF001


def test_source_digest_is_pinned_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_assay,
        "_load_selection",
        lambda _root: (_source_selection(), "f" * 64),
    )
    with pytest.raises(ValueError, match="provider selection digest"):
        assay._load_source_selection(Path("source"))  # noqa: SLF001
