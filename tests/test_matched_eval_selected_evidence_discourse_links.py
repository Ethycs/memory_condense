from __future__ import annotations

import hashlib

from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from tools.matched_eval.selected_evidence_discourse_links import (
    SelectedEvidenceLinkInput,
    link_selected_evidence,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _input(
    handle_id: str,
    ordinal: int,
    text: str,
    *,
    source: str = "thread-a",
    role: str = "user",
) -> SelectedEvidenceLinkInput:
    span = EvidenceSpan(
        chunk_id=f"chunk-{source}-{ordinal}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id=source,
        turn_id=f"turn-{source}-{ordinal}",
        role=role,
        created_at=f"2026-08-{ordinal:02d}T12:00:00+00:00",
    )
    return SelectedEvidenceLinkInput(
        handle_id=handle_id,
        span=span,
        quote=text,
        source_binding_receipt_sha256=_sha(f"binding-{handle_id}"),
        selected_evidence_receipt_sha256=_sha(f"selected-{handle_id}"),
    )


def test_selected_linker_preserves_sequence_revision_dependency_and_roles() -> None:
    compiled = link_selected_evidence(
        (
            _input("H001", 1, "We decided to use option A.", role="assistant"),
            _input(
                "H002",
                2,
                "We revised that decision; instead use option B.",
            ),
            _input("H003", 3, "The release must stay within the budget."),
            _input("H004", 4, "Deployment depends on that requirement."),
        )
    )

    by_relation = {}
    for link in compiled.links:
        by_relation.setdefault(link.relation, []).append(link)
    assert {"sequence", "revises", "depends_on"} <= set(by_relation)
    revision = by_relation["revises"][0]
    assert revision.handle_ids == ("H001", "H002")
    assert [member["role"] for member in revision.members] == [
        "predecessor",
        "successor",
    ]
    assert [member["evidence_role"] for member in revision.members] == [
        "assistant",
        "user",
    ]
    dependency = by_relation["depends_on"][0]
    assert dependency.handle_ids == ("H003", "H004")
    assert [member["role"] for member in dependency.members] == [
        "requirement",
        "dependent",
    ]


def test_selected_linker_does_not_link_across_sources() -> None:
    compiled = link_selected_evidence(
        (
            _input("H001", 1, "We decided to use option A.", source="alpha"),
            _input(
                "H002",
                2,
                "We revised that decision; instead use option B.",
                source="beta",
            ),
        )
    )

    assert compiled.links == ()


def test_selected_linker_keeps_exact_provenance_local_and_zero_token_state() -> None:
    first = _input("H001", 1, "The launch must stay within budget.")
    second = _input("H002", 2, "Deployment depends on that requirement.")
    compiled = link_selected_evidence((first, second))

    assert compiled.retained_transformer_token_state_bytes == 0
    assert compiled.projection()["provider_calls"] == 0
    provider_text = str([link.projection() for link in compiled.links])
    assert first.span.chunk_id not in provider_text
    assert first.span.source_id not in provider_text
    binding = compiled.local_bindings[0]
    assert binding["members"][0]["source_binding_receipt_sha256"] in {
        first.source_binding_receipt_sha256,
        second.source_binding_receipt_sha256,
    }
    assert binding["relation_identity_sha256"]
    assert binding["relation_id"]
