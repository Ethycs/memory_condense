from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Sequence

import pytest

from memory_condense.search.packing.ranked_prefix_prompt import PACKER_ID
from memory_condense.search.source_preserving_hybrid import (
    AUDIT_FORMAT,
    POLICY_ID,
    RESULT_FORMAT,
    ExactRawPromptFallback,
    HybridEvidence,
    HybridEvidencePlane,
    HybridPackMode,
    HybridPriorityPhase,
    SourceSeedCoverageError,
    pack_source_preserving_hybrid,
)


@dataclass(frozen=True, slots=True)
class Raw:
    chunk_id: str
    source_id: str
    tokens: int = 1


@dataclass(frozen=True, slots=True)
class Fact:
    chunk_id: str
    tokens: int = 1


def _prompt_sha(prompt: bytes) -> str:
    return hashlib.sha256(prompt).hexdigest()


def _context_tokens(rows: Sequence[HybridEvidence[Raw, Fact]]) -> int:
    return sum(row.value.tokens for row in rows)


def _render(rows: Sequence[HybridEvidence[Raw, Fact]]) -> bytes:
    body = b"|".join(
        f"{row.priority_phase.value}:{row.chunk_id}".encode("utf-8")
        for row in rows
    )
    return b"hybrid\n" + body


def _raw_fallback(
    raws: tuple[Raw, ...],
    *,
    prompt: bytes | None = None,
    reserve: int = 0,
) -> ExactRawPromptFallback[bytes]:
    rendered = (
        b"raw\n" + b"|".join(row.chunk_id.encode("utf-8") for row in raws)
        if prompt is None
        else prompt
    )
    prompt_tokens = len(rendered)
    return ExactRawPromptFallback(
        raw_chunk_ids=tuple(row.chunk_id for row in raws),
        rendered_prompt=rendered,
        context_token_count=sum(row.tokens for row in raws),
        prompt_token_count=prompt_tokens,
        output_token_reserve=reserve,
        prompt_workspace_token_count=prompt_tokens + reserve,
        prompt_sha256=_prompt_sha(rendered),
    )


def _pack(
    raws: tuple[Raw, ...],
    facts: tuple[Fact, ...],
    *,
    context_cap: int = 10_000,
    prompt_cap: int = 10_000,
    reserve: int = 0,
    fallback: ExactRawPromptFallback[bytes] | None = None,
    render: Any = _render,
):
    return pack_source_preserving_hybrid(
        raws,
        facts,
        raw_chunk_id=lambda row: row.chunk_id,
        raw_source_id=lambda row: row.source_id,
        projected_chunk_id=lambda row: row.chunk_id,
        count_context_tokens=_context_tokens,
        render_prompt=render,
        count_prompt_tokens=len,
        prompt_sha256=_prompt_sha,
        max_context_tokens=context_cap,
        max_prompt_tokens=prompt_cap,
        output_token_reserve=reserve,
        raw_fallback=fallback,
    )


def _identity(rows: Sequence[HybridEvidence[Raw, Fact]]) -> list[tuple[str, str]]:
    return [(row.priority_phase.value, row.chunk_id) for row in rows]


def test_three_band_order_and_post_selection_exact_dedup() -> None:
    raws = (
        Raw("a-1", "source-a"),
        Raw("a-2", "source-a"),
        Raw("b-1", "source-b"),
        Raw("b-2", "source-b"),
    )
    # Both selections reach the composer intact, including duplicate IDs.  The
    # source seed beats a projection duplicate, while projection beats a raw
    # remainder duplicate because dedup happens after three-band ordering.
    facts = (Fact("p-1"), Fact("a-2"), Fact("a-1"), Fact("p-1"))

    result = _pack(raws, facts)

    assert result.mode is HybridPackMode.HYBRID
    assert _identity(result.packed_items) == [
        ("raw_source_seed", "a-1"),
        ("raw_source_seed", "b-1"),
        ("projection", "p-1"),
        ("projection", "a-2"),
        ("raw_remainder", "b-2"),
    ]
    assert result.audit.raw_selected_chunk_ids == (
        "a-1",
        "a-2",
        "b-1",
        "b-2",
    )
    assert result.audit.projected_selected_chunk_ids == (
        "p-1",
        "a-2",
        "a-1",
        "p-1",
    )
    assert [row.chunk_id for row in result.audit.exact_duplicates] == [
        "a-1",
        "p-1",
        "a-2",
    ]
    assert result.audit.exact_duplicates[0].retained.priority_phase is (
        HybridPriorityPhase.RAW_SOURCE_SEED
    )
    assert result.audit.exact_duplicates[-1].retained.priority_phase is (
        HybridPriorityPhase.PROJECTION
    )


def test_all_source_seeds_are_a_proven_prefix_with_deterministic_receipts() -> None:
    raws = (
        Raw("one", "topic"),
        Raw("two", "TOPIC"),
        Raw("three", "topic "),
        Raw("four", "topic"),
    )
    facts = (Fact("fact"),)

    first = _pack(raws, facts)
    second = _pack(raws, facts)

    assert first == second
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.audit.receipt_sha256 == second.audit.receipt_sha256
    assert [row.chunk_id for row in first.audit.source_seeds] == [
        "one",
        "two",
        "three",
    ]
    assert len({row.source_id_sha256 for row in first.audit.source_seeds}) == 3
    assert first.audit.seed_gate.passed is True
    assert first.audit.seed_gate.missing_seed_chunk_ids == ()
    assert first.audit.seed_gate.action == "hybrid"
    assert first.audit.packing_audit.packer_id == PACKER_ID
    assert first.audit.projection()["audit_format"] == AUDIT_FORMAT
    assert first.audit.projection()["policy_id"] == POLICY_ID
    assert first.projection()["format"] == RESULT_FORMAT


def test_seed_loss_reuses_exact_raw_prompt_and_accounting() -> None:
    raws = (Raw("a", "source-a"), Raw("b", "source-b"))
    facts = (Fact("projected"),)
    fallback = _raw_fallback(raws, prompt=b"R", reserve=3)

    def expensive_hybrid(
        rows: Sequence[HybridEvidence[Raw, Fact]],
    ) -> bytes:
        return b"H" + (b"12345678" * len(rows))

    result = _pack(
        raws,
        facts,
        prompt_cap=4,
        reserve=3,
        fallback=fallback,
        render=expensive_hybrid,
    )

    assert result.mode is HybridPackMode.RAW_FALLBACK
    assert result.rendered_prompt is fallback.rendered_prompt
    assert result.effective_prompt_sha256 == fallback.prompt_sha256
    assert result.context_token_count == fallback.context_token_count
    assert result.prompt_token_count == fallback.prompt_token_count
    assert result.output_token_reserve == fallback.output_token_reserve
    assert result.prompt_workspace_token_count == (
        fallback.prompt_workspace_token_count
    )
    assert [row.value for row in result.packed_items] == list(raws)
    assert [row.value for row in result.dropped_items] == list(facts)
    assert result.audit.seed_gate.passed is False
    assert result.audit.seed_gate.action == "raw_fallback"
    assert result.audit.seed_gate.packed_seed_chunk_ids == ()
    assert result.audit.seed_gate.missing_seed_chunk_ids == ("a", "b")
    assert result.audit.raw_fallback_receipt_sha256 == fallback.receipt_sha256


def test_seed_loss_without_fallback_fails_closed_with_gate_receipt() -> None:
    raws = (Raw("a", "source-a"), Raw("b", "source-b"))

    with pytest.raises(SourceSeedCoverageError) as captured:
        _pack(
            raws,
            (),
            prompt_cap=5,
            render=lambda rows: b"H" + (b"12345678" * len(rows)),
        )

    gate = captured.value.gate
    assert gate.passed is False
    assert gate.action == "raise"
    assert gate.required_seed_chunk_ids == ("a", "b")
    assert gate.packed_seed_chunk_ids == ()
    assert gate.missing_seed_chunk_ids == ("a", "b")
    assert len(gate.receipt_sha256) == 64


def test_exact_seed_boundary_can_drop_projection_without_fallback() -> None:
    raws = (Raw("a", "source-a"), Raw("b", "source-b"))
    facts = (Fact("p"),)

    # The renderer emits two bytes per item plus one byte of fixed overhead.
    result = _pack(
        raws,
        facts,
        prompt_cap=5,
        render=lambda rows: b"H" + (b"xx" * len(rows)),
    )

    assert result.mode is HybridPackMode.HYBRID
    assert [row.chunk_id for row in result.packed_items] == ["a", "b"]
    assert [row.chunk_id for row in result.dropped_items] == ["p"]
    assert result.audit.seed_gate.passed is True


def test_passing_source_gate_can_drop_raw_remainder() -> None:
    raws = (Raw("seed", "source-a"), Raw("detail", "source-a"))

    # The source gate proves source representation, not byte preservation of
    # every raw row.  Once the one source seed fits, a later raw detail may be
    # dropped while the hybrid remains eligible.
    result = _pack(raws, (), context_cap=1)

    assert result.mode is HybridPackMode.HYBRID
    assert [row.chunk_id for row in result.packed_items] == ["seed"]
    assert [row.chunk_id for row in result.dropped_items] == ["detail"]
    assert result.audit.seed_gate.passed is True
    assert result.audit.seed_gate.required_seed_chunk_ids == ("seed",)


def test_infeasible_empty_hybrid_reuses_exact_valid_raw_fallback() -> None:
    raws = (Raw("seed", "source-a"),)
    fallback = _raw_fallback(raws, prompt=b"R")

    result = _pack(
        raws,
        (),
        context_cap=1,
        prompt_cap=1,
        fallback=fallback,
        render=lambda _rows: b"never-fits",
    )

    assert result.mode is HybridPackMode.RAW_FALLBACK
    assert result.rendered_prompt is fallback.rendered_prompt
    assert result.effective_prompt_sha256 == fallback.prompt_sha256
    assert result.audit.packing_status == "no_feasible_prefix"
    assert result.audit.packing_audit is None
    assert result.audit.hybrid_prompt_sha256 is None
    assert result.audit.seed_gate.action == "raw_fallback"
    assert result.audit.seed_gate.missing_seed_chunk_ids == ("seed",)


def test_infeasible_empty_hybrid_without_fallback_raises_audited_gate() -> None:
    raws = (Raw("seed", "source-a"),)

    with pytest.raises(SourceSeedCoverageError) as captured:
        _pack(
            raws,
            (),
            context_cap=1,
            prompt_cap=1,
            render=lambda _rows: b"never-fits",
        )

    assert captured.value.gate.action == "raise"
    assert captured.value.gate.missing_seed_chunk_ids == ("seed",)
    assert captured.value.audit is not None
    assert captured.value.audit.packing_status == "no_feasible_prefix"
    assert captured.value.audit.packing_audit is None
    assert captured.value.audit.hybrid_prompt_sha256 is None
    assert len(captured.value.audit.receipt_sha256) == 64


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("context_cap", "exceeds the context token cap"),
        ("workspace_cap", "exceeds the prompt workspace cap"),
        ("reserve", "output token reserve changed"),
    ),
)
def test_raw_fallback_must_match_all_live_budget_controls(
    failure: str,
    message: str,
) -> None:
    raws = (Raw("seed", "source-a", tokens=2),)
    fallback = _raw_fallback(raws, prompt=b"RR", reserve=1)
    controls = {"context_cap": 2, "prompt_cap": 3, "reserve": 1}
    if failure == "context_cap":
        controls["context_cap"] = 1
    elif failure == "workspace_cap":
        controls["prompt_cap"] = 2
    else:
        controls["reserve"] = 0

    with pytest.raises(ValueError, match=message):
        _pack(raws, (), fallback=fallback, **controls)


def test_exact_fallback_cap_boundaries_are_inclusive() -> None:
    raws = (Raw("a", "source-a"), Raw("b", "source-b"))
    fallback = _raw_fallback(raws, prompt=b"R", reserve=2)

    result = _pack(
        raws,
        (),
        context_cap=2,
        prompt_cap=3,
        reserve=2,
        fallback=fallback,
        render=lambda rows: b"H" + (b"12345678" * len(rows)),
    )

    assert result.mode is HybridPackMode.RAW_FALLBACK
    assert result.rendered_prompt is fallback.rendered_prompt
    assert result.context_token_count == 2
    assert result.prompt_workspace_token_count == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("ids", "evidence order changed"),
        ("context", "context token count changed"),
        ("prompt_count", "prompt token count changed"),
        ("prompt_sha", "prompt bytes changed"),
    ),
)
def test_raw_fallback_integrity_is_checked_before_use(
    mutation: str,
    message: str,
) -> None:
    raws = (Raw("a", "source-a"),)
    fallback = _raw_fallback(raws)
    if mutation == "ids":
        fallback = replace(fallback, raw_chunk_ids=("changed",), receipt_sha256="")
    elif mutation == "context":
        fallback = replace(
            fallback,
            context_token_count=fallback.context_token_count + 1,
            receipt_sha256="",
        )
    elif mutation == "prompt_count":
        fallback = replace(
            fallback,
            prompt_token_count=fallback.prompt_token_count + 1,
            prompt_workspace_token_count=(
                fallback.prompt_workspace_token_count + 1
            ),
            receipt_sha256="",
        )
    else:
        fallback = replace(fallback, prompt_sha256="0" * 64, receipt_sha256="")

    with pytest.raises(ValueError, match=message):
        _pack(raws, (), fallback=fallback)


def test_conflicting_raw_chunk_provenance_is_rejected() -> None:
    raws = (Raw("same", "source-a"), Raw("same", "source-b"))

    with pytest.raises(ValueError, match="conflicting exact sources"):
        _pack(raws, ())


def test_same_source_duplicate_raw_ids_remain_exact_in_fallback() -> None:
    raws = (Raw("same", "source-a"), Raw("same", "source-a"))
    fallback = _raw_fallback(raws, prompt=b"R")

    first = _pack(
        raws,
        (),
        prompt_cap=1,
        fallback=fallback,
        render=lambda rows: b"H" + (b"12345678" * len(rows)),
    )
    second = _pack(
        raws,
        (),
        prompt_cap=1,
        fallback=fallback,
        render=lambda rows: b"H" + (b"12345678" * len(rows)),
    )

    assert first == second
    assert first.rendered_prompt is fallback.rendered_prompt
    assert first.audit.raw_selected_chunk_ids == ("same", "same")
    assert first.audit.seed_gate.required_seed_chunk_ids == ("same",)
    assert [row.chunk_id for row in first.packed_items] == ["same", "same"]
    assert [row.chunk_id for row in first.audit.exact_duplicates] == ["same"]


def test_projection_only_input_has_a_vacuously_complete_seed_gate() -> None:
    result = _pack((), (Fact("p-1"), Fact("p-2")))

    assert result.mode is HybridPackMode.HYBRID
    assert result.audit.source_seeds == ()
    assert result.audit.seed_gate.required_seed_chunk_ids == ()
    assert result.audit.seed_gate.passed is True
    assert [row.plane for row in result.packed_items] == [
        HybridEvidencePlane.PROJECTION,
        HybridEvidencePlane.PROJECTION,
    ]


def test_audit_and_result_reject_inconsistent_reconstruction() -> None:
    result = _pack(
        (Raw("a", "source-a"), Raw("b", "source-a")),
        (Fact("p"),),
    )

    with pytest.raises(ValueError, match="packed items disagree"):
        replace(
            result,
            packed_items=result.packed_items[1:],
            receipt_sha256="",
        )
    with pytest.raises(TypeError, match="requires an exact packing audit"):
        replace(
            result.audit,
            packing_audit=None,
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="infeasible hybrid"):
        replace(
            result.audit,
            packing_status="no_feasible_prefix",
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="receipt changed"):
        replace(result, receipt_sha256="0" * 64)
