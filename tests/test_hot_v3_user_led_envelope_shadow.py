from __future__ import annotations

import copy
from collections import Counter
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval import hot_v3_user_led_envelope_shadow as subject
from tools.matched_eval import query_guided_scan as cache_tools
from tools.matched_eval.contracts import identity_sha256


NAMESPACE_ID = "a" * 64
DATABASE_SHA256 = "b" * 64
STORE_RECEIPT_SHA256 = "c" * 64


def _cached_row(
    chunk_id: str,
    *,
    turn_id: str,
    source_id: str,
    role: str,
    ordinal: int,
    text: str,
) -> cache_tools.CachedContentRow:
    return cache_tools.CachedContentRow(
        namespace_id=NAMESPACE_ID,
        partition_id=cache_tools._partition(source_id),  # noqa: SLF001
        chunk_id=chunk_id,
        turn_id=turn_id,
        source_id=source_id,
        role=role,
        created_at=f"2026-01-{ordinal + 1:02d}T00:00:00+00:00",
        ordinal=ordinal,
        turn_start_char=0,
        turn_end_char=len(text),
        text=text,
        text_sha256=quote_sha256(text),
        token_count=count_tokens(text),
        sentence_windows=cache_tools._sentence_windows(text),  # noqa: SLF001
    )


def _cache(*rows: cache_tools.CachedContentRow) -> cache_tools.NamespacePartitionCache:
    grouped: dict[str, list[cache_tools.CachedContentRow]] = {}
    for row in sorted(rows, key=lambda value: (value.ordinal, value.chunk_id)):
        grouped.setdefault(row.partition_id, []).append(row)
    frozen = MappingProxyType(
        {partition: tuple(values) for partition, values in sorted(grouped.items())}
    )
    partitions = [
        {
            "content_row_count": len(values),
            "content_rows_sha256": identity_sha256(
                [row.receipt_projection() for row in values]
            ),
            "partition_id": partition,
        }
        for partition, values in frozen.items()
    ]
    body = {
        "cache_immutable": True,
        "content_row_count": len(rows),
        "database_read_passes": 1,
        "format": cache_tools.CACHE_FORMAT,
        "metadata_row_count": 0,
        "namespace_id": NAMESPACE_ID,
        "partitions": partitions,
        "physical_store_row_count": len(rows),
        "source_database_sha256": DATABASE_SHA256,
        "source_store_receipt_sha256": STORE_RECEIPT_SHA256,
    }
    return cache_tools.NamespacePartitionCache(
        namespace_id=NAMESPACE_ID,
        source_database_sha256=DATABASE_SHA256,
        source_store_receipt_sha256=STORE_RECEIPT_SHA256,
        physical_store_row_count=len(rows),
        metadata_row_count=0,
        rows_by_partition=frozen,
        cache_receipt_sha256=identity_sha256(body),
    )


def _parent_row(row: cache_tools.CachedContentRow) -> dict[str, Any]:
    rendered = f"[{row.created_at} | {row.role}] {row.text}"
    return {
        "chunk_id": row.chunk_id,
        "created_at": row.created_at,
        "evidence_id": row.chunk_id,
        "raw_text": row.text,
        "raw_text_sha256": row.text_sha256,
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": row.role,
        "route": "sealed_parent",
        "score": 1.0,
        "source_id": row.source_id,
        "turn_id": row.turn_id,
    }


def _synthetic_parent(
    row: cache_tools.CachedContentRow,
    *,
    logical_id: str,
    excerpt: str,
) -> dict[str, Any]:
    rendered = f"[{row.created_at} | {row.role}] {excerpt}"
    return {
        "backing_chunk_id": row.chunk_id,
        "chunk_id": logical_id,
        "created_at": row.created_at,
        "evidence_id": logical_id,
        "excerpt_occurrence": True,
        "raw_text": excerpt,
        "raw_text_sha256": quote_sha256(excerpt),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": row.role,
        "route": "sealed_synthetic_parent",
        "score": 1.0,
        "source_id": row.source_id,
        "turn_id": row.turn_id,
    }


def _measure(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    tokens = sum(count_tokens(str(row["rendered_text"])) for row in rows)
    return tokens, tokens + 1


def _physical_id(row: Mapping[str, Any]) -> str:
    return str(row.get("backing_chunk_id", row["chunk_id"]))


def test_synthetic_shared_backing_anchors_one_source_safe_deduped_envelope() -> None:
    opener = _cached_row(
        "a-user",
        turn_id="a-turn-user",
        source_id="thread-a",
        role="user",
        ordinal=0,
        text="Please remember the launch plan.",
    )
    answer = _cached_row(
        "a-answer",
        turn_id="a-turn-answer",
        source_id="thread-a",
        role="assistant",
        ordinal=1,
        text="The assistant completion names cobalt and amber.",
    )
    other_opener = _cached_row(
        "b-user",
        turn_id="b-turn-user",
        source_id="thread-b",
        role="user",
        ordinal=2,
        text="An unrelated user request.",
    )
    other_answer = _cached_row(
        "b-answer",
        turn_id="b-turn-answer",
        source_id="thread-b",
        role="assistant",
        ordinal=3,
        text="An unrelated machine response.",
    )
    index = subject.build_user_led_envelope_shadow_index(
        _cache(opener, answer, other_opener, other_answer)
    )
    assert index.envelope_by_id is index.envelope_by_id
    with pytest.raises(TypeError):
        index.envelope_by_id["replacement"] = index.envelopes[0]  # type: ignore[index]
    parent = (
        _synthetic_parent(
            answer,
            logical_id="synthetic-cobalt",
            excerpt="cobalt",
        ),
        _synthetic_parent(
            answer,
            logical_id="synthetic-amber",
            excerpt="amber",
        ),
    )

    selection = subject.select_user_led_envelope_shadow(index, parent)

    assert len(selection.groups) == 1
    group = selection.groups[0]
    assert group.source_id == "thread-a"
    assert group.anchor_chunk_ids == (answer.chunk_id, answer.chunk_id)
    assert group.companion_chunk_ids == (opener.chunk_id,)
    assert other_opener.chunk_id not in group.ordered_chunk_ids
    assert other_answer.chunk_id not in group.ordered_chunk_ids
    assert set(group.companion_chunk_ids).isdisjoint(
        {_physical_id(row) for row in parent}
    )

    composition = subject.compose_user_led_envelope_shadow(
        index,
        selection,
        parent,
        measure_packet=_measure,
        max_context_tokens=1_000,
        max_prompt_tokens=1_000,
    )

    parent_hashes = Counter(identity_sha256(dict(row)) for row in parent)
    observed_parent_hashes = Counter(
        identity_sha256(dict(row))
        for row in composition.packed_evidence
        if row["route"] != subject.COMPANION_ROUTE
    )
    assert observed_parent_hashes == parent_hashes
    assert composition.admitted_companion_chunk_ids == (opener.chunk_id,)
    assert sum(
        _physical_id(row) == answer.chunk_id
        for row in composition.packed_evidence
    ) == 2


def test_synthetic_anchor_must_authenticate_source_and_exact_backing_text() -> None:
    opener = _cached_row(
        "auth-user",
        turn_id="auth-user-turn",
        source_id="auth-source",
        role="user",
        ordinal=0,
        text="Authenticated opener.",
    )
    answer = _cached_row(
        "auth-answer",
        turn_id="auth-answer-turn",
        source_id="auth-source",
        role="assistant",
        ordinal=1,
        text="The sealed answer contains indigo.",
    )
    index = subject.build_user_led_envelope_shadow_index(_cache(opener, answer))
    wrong_source = _synthetic_parent(
        answer,
        logical_id="wrong-source",
        excerpt="indigo",
    )
    wrong_source["source_id"] = "another-source"
    fabricated_text = _synthetic_parent(
        answer,
        logical_id="fabricated-text",
        excerpt="fabricated",
    )

    for raw in (wrong_source, fabricated_text):
        selection = subject.select_user_led_envelope_shadow(index, (raw,))
        assert selection.groups == ()
        assert selection.diagnostics[0]["reason"] == "parent_chunk_mismatch"


def test_envelope_cap_counts_selected_groups_and_backfills_skipped_anchors() -> None:
    empty = _cached_row(
        "empty-user",
        turn_id="empty-turn",
        source_id="thread-empty",
        role="user",
        ordinal=0,
        text="Already complete by itself.",
    )
    opener_a = _cached_row(
        "a-user",
        turn_id="a-user-turn",
        source_id="thread-a",
        role="user",
        ordinal=1,
        text="Question A.",
    )
    answer_a = _cached_row(
        "a-answer",
        turn_id="a-answer-turn",
        source_id="thread-a",
        role="assistant",
        ordinal=2,
        text="Answer A.",
    )
    opener_b = _cached_row(
        "b-user",
        turn_id="b-user-turn",
        source_id="thread-b",
        role="user",
        ordinal=3,
        text="Question B.",
    )
    answer_b = _cached_row(
        "b-answer",
        turn_id="b-answer-turn",
        source_id="thread-b",
        role="assistant",
        ordinal=4,
        text="Answer B.",
    )
    index = subject.build_user_led_envelope_shadow_index(
        _cache(empty, opener_a, answer_a, opener_b, answer_b)
    )
    parent = (_parent_row(empty), _parent_row(answer_a), _parent_row(answer_b))

    selection = subject.select_user_led_envelope_shadow(
        index,
        parent,
        budget=subject.UserLedEnvelopeShadowBudget(
            max_envelopes=1,
            max_turns_per_envelope=8,
            max_companion_chunks=16,
            max_companion_tokens=800,
        ),
    )

    assert len(selection.groups) == 1
    assert selection.groups[0].source_id == "thread-a"
    assert selection.groups[0].companion_chunk_ids == (opener_a.chunk_id,)
    assert any(row["reason"] == "no_novel_companion" for row in selection.diagnostics)
    cap_rows = [row for row in selection.diagnostics if row["reason"] == "max_envelopes"]
    assert len(cap_rows) == 1
    assert cap_rows[0]["envelope_id"] == index.envelope_by_chunk_id[answer_b.chunk_id]


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (
            subject.UserLedEnvelopeShadowBudget(
                max_envelopes=1,
                max_turns_per_envelope=1,
                max_companion_chunks=16,
                max_companion_tokens=800,
            ),
            "mandatory_turn_bound",
        ),
        (
            subject.UserLedEnvelopeShadowBudget(
                max_envelopes=1,
                max_turns_per_envelope=8,
                max_companion_chunks=0,
                max_companion_tokens=800,
            ),
            "mandatory_companion_bound",
        ),
        (
            subject.UserLedEnvelopeShadowBudget(
                max_envelopes=1,
                max_turns_per_envelope=8,
                max_companion_chunks=16,
                max_companion_tokens=0,
            ),
            "mandatory_companion_bound",
        ),
    ],
)
def test_turn_chunk_and_token_caps_fail_closed(
    budget: subject.UserLedEnvelopeShadowBudget,
    reason: str,
) -> None:
    opener = _cached_row(
        "cap-user",
        turn_id="cap-user-turn",
        source_id="cap-source",
        role="user",
        ordinal=0,
        text="A nonempty opener.",
    )
    answer = _cached_row(
        "cap-answer",
        turn_id="cap-answer-turn",
        source_id="cap-source",
        role="assistant",
        ordinal=1,
        text="A selected answer.",
    )
    index = subject.build_user_led_envelope_shadow_index(_cache(opener, answer))

    selection = subject.select_user_led_envelope_shadow(
        index,
        (_parent_row(answer),),
        budget=budget,
    )

    assert selection.groups == ()
    assert any(row["reason"] == reason for row in selection.diagnostics)


def test_final_packet_rejects_a_whole_group_without_partial_companions() -> None:
    opener = _cached_row(
        "atomic-user",
        turn_id="atomic-user-turn",
        source_id="atomic-source",
        role="user",
        ordinal=0,
        text="Atomic opener.",
    )
    system = _cached_row(
        "atomic-system",
        turn_id="atomic-system-turn",
        source_id="atomic-source",
        role="system",
        ordinal=1,
        text="Atomic system evidence.",
    )
    answer = _cached_row(
        "atomic-answer",
        turn_id="atomic-answer-turn",
        source_id="atomic-source",
        role="assistant",
        ordinal=2,
        text="Atomic selected answer.",
    )
    index = subject.build_user_led_envelope_shadow_index(
        _cache(opener, system, answer)
    )
    parent = (_parent_row(answer),)
    selection = subject.select_user_led_envelope_shadow(index, parent)
    assert selection.groups[0].companion_chunk_ids == (
        opener.chunk_id,
        system.chunk_id,
    )

    measure_calls = 0

    def reject_companions(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[int, int]:
        nonlocal measure_calls
        measure_calls += 1
        has_companion = any(
            row.get("route") == subject.COMPANION_ROUTE for row in rows
        )
        return (11, 11) if has_companion else (1, 1)

    composition = subject.compose_user_led_envelope_shadow(
        index,
        selection,
        parent,
        measure_packet=reject_companions,
        max_context_tokens=10,
        max_prompt_tokens=10,
    )

    assert composition.admitted_group_ids == ()
    assert composition.rejected_group_ids == (
        selection.groups[0].envelope_id,
    )
    assert composition.admitted_companion_chunk_ids == ()
    assert composition.packed_evidence == parent
    assert measure_calls == 2
