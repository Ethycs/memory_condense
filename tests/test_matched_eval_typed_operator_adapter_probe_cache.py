from __future__ import annotations

from collections import Counter
import hashlib

from tools.matched_eval import typed_operator_adapter as adapter
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = "[Question asked at 2026/09/01 12:00]\nWhat did I do?"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(index: int) -> EvidenceHandleBinding:
    citation = f"exact citation {index}"
    return EvidenceHandleBinding(
        handle_id=f"H{index:03d}",
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle=f"G{index:03d}",
        sealed_artifact_sha256=_sha("map-artifact"),
        parent_receipt_sha256=_sha("map-parent"),
        evidence_receipt_sha256=_sha(f"map-item-{index}"),
        payload_sha256=_sha(f"payload-{index}"),
        citation_sha256=_sha(citation),
        citation_char_count=len(citation),
        local_source_locator_sha256=_sha(f"source-{index}"),
    )


def _parsed(summaries: tuple[str, ...]):
    spec = compile_typed_operator_spec(QUESTION)
    bindings = tuple(_binding(index) for index in range(1, len(summaries) + 1))
    parsed = parse_typed_items(
        [
            {"handle_ids": [binding.handle_id], "summary": summary}
            for binding, summary in zip(bindings, summaries, strict=True)
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    return spec, bindings, parsed


def _build(spec, bindings, parsed):
    return build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        frontier_mode=FrontierMode.BOUNDED,
    )


def test_fit_probe_cache_reuses_identical_populations_and_rechecks_final(
    monkeypatch,
) -> None:
    spec, bindings, parsed = _parsed(
        ("I went walking.", "I cooked dinner.", "I read a book.")
    )
    original_frontier = adapter.build_frontier_receipt
    projected_populations: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def recording_frontier(spec, bindings, items, rejected, **kwargs):
        projected_populations.append(
            (
                tuple(row.receipt_sha256 for row in items),
                tuple(row.rejection_sha256 for row in rejected),
            )
        )
        return original_frontier(spec, bindings, items, rejected, **kwargs)

    original_count_tokens = adapter.count_tokens
    tokenized_payloads: list[str] = []

    def recording_count_tokens(text: str) -> int:
        tokenized_payloads.append(text)
        return original_count_tokens(text)

    monkeypatch.setattr(adapter, "build_frontier_receipt", recording_frontier)
    monkeypatch.setattr(adapter, "count_tokens", recording_count_tokens)

    packet = _build(spec, bindings, parsed)

    final_population = (
        tuple(row.receipt_sha256 for row in packet.items),
        tuple(row.rejection_sha256 for row in packet.rejected_items),
    )
    population_counts = Counter(projected_populations)
    assert population_counts[final_population] == 3
    assert all(
        count == 1
        for population, count in population_counts.items()
        if population != final_population
    )
    assert len(projected_populations) == len(parsed.accepted_items) + 2
    assert len(tokenized_payloads) == len(parsed.accepted_items) + 3
    assert tuple(row.receipt_sha256 for row in packet.items) == tuple(
        row.receipt_sha256 for row in parsed.accepted_items
    )
    assert not packet.rejected_items


def test_fit_probe_cache_preserves_overflow_bytes_order_and_rejection_receipts(
    monkeypatch,
) -> None:
    spec, bindings, parsed = _parsed(
        (
            "small first item",
            ("overflow " * 10_000).strip(),
            "small final item",
        )
    )
    expected = _build(spec, bindings, parsed)

    original_frontier = adapter.build_frontier_receipt
    projected_populations: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def recording_frontier(spec, bindings, items, rejected, **kwargs):
        projected_populations.append(
            (
                tuple(row.receipt_sha256 for row in items),
                tuple(row.rejection_sha256 for row in rejected),
            )
        )
        return original_frontier(spec, bindings, items, rejected, **kwargs)

    monkeypatch.setattr(adapter, "build_frontier_receipt", recording_frontier)
    actual = _build(spec, bindings, parsed)

    assert actual.projection() == expected.projection()
    assert actual.render_provider_payload() == expected.render_provider_payload()
    assert actual.provider_payload_token_proxy == expected.provider_payload_token_proxy
    assert tuple(row.receipt_sha256 for row in actual.items) == tuple(
        row.receipt_sha256 for row in expected.items
    )
    assert tuple(row.rejection_sha256 for row in actual.rejected_items) == tuple(
        row.rejection_sha256 for row in expected.rejected_items
    )
    assert tuple(row.reason for row in actual.rejected_items) == (
        "hard_8k_item_overflow",
    )

    final_population = (
        tuple(row.receipt_sha256 for row in actual.items),
        tuple(row.rejection_sha256 for row in actual.rejected_items),
    )
    population_counts = Counter(projected_populations)
    assert population_counts[final_population] == 3
    assert all(
        count == 1
        for population, count in population_counts.items()
        if population != final_population
    )
