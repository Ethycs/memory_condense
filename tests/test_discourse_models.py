from __future__ import annotations

from dataclasses import fields

import pytest

from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    DiscourseArtifact,
    DiscourseSnapshot,
    Episode,
    EvidenceAtom,
    EvidenceObligation,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
    make_atom_id,
    make_episode_id,
    quote_sha256,
)


_DIGEST = "0" * 64


def _span(*, source: str = "source", ordinal: int = 1) -> EvidenceSpan:
    text = f"evidence-{ordinal}"
    return EvidenceSpan(
        chunk_id=f"chunk-{ordinal}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id=source,
    )


def test_artifact_identity_is_stable_and_binds_policy() -> None:
    first = DiscourseArtifact.create(
        kind="deterministic-source-boundaries",
        implementation_sha256=_DIGEST,
        policy={"max_episode_chunks": 8, "method": "source"},
        metadata={"retained_request_token_state_bytes": 0},
    )
    second = DiscourseArtifact.create(
        kind="deterministic-source-boundaries",
        implementation_sha256=_DIGEST,
        policy={"method": "source", "max_episode_chunks": 8},
        metadata={"retained_request_token_state_bytes": 0},
    )

    assert first == second
    assert len(first.policy_sha256) == 64


def test_evidence_atom_derives_and_cannot_forge_authoritative_role_or_time() -> None:
    text = "source fact"
    span = EvidenceSpan(
        chunk_id="c1",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=1,
        source_id="thread",
        turn_id="t1",
        role="user",
        created_at="2026-08-18",
    )
    atom = EvidenceAtom(make_atom_id(span), span, text, "fact")
    assert atom.role == "user"
    assert atom.created_at == "2026-08-18"
    with pytest.raises(ValueError, match="role contradicts"):
        EvidenceAtom(make_atom_id(span), span, text, "fact", role="assistant")
    with pytest.raises(ValueError, match="created_at contradicts"):
        EvidenceAtom(
            make_atom_id(span),
            span,
            text,
            "fact",
            created_at="2026-08-19",
        )


def test_nonempty_snapshot_cannot_use_placeholder_content_roots() -> None:
    with pytest.raises(ValueError, match="source snapshot requires"):
        DiscourseSnapshot(1, 1, 1, 11, ())
    with pytest.raises(ValueError, match="graph snapshot requires"):
        DiscourseSnapshot(
            0,
            0,
            1,
            11,
            ("artifact",),
            source_content_sha256="1" * 64,
        )


def test_episode_rejects_cross_source_evidence_and_tampered_receipt() -> None:
    first = _span(source="one", ordinal=1)
    second = _span(source="two", ordinal=2)
    with pytest.raises(ValueError, match="cannot cross source"):
        Episode(
            episode_id="episode-cross-source",
            artifact_id="artifact",
            source_id="one",
            sequence_no=0,
            first_ordinal=1,
            last_ordinal=2,
            evidence=(first, second),
            boundary_method="source",
        )

    episode_id = make_episode_id(
        artifact_id="artifact",
        source_id="one",
        sequence_no=0,
        evidence=(first,),
    )
    with pytest.raises(ValueError, match="receipt"):
        Episode(
            episode_id=episode_id,
            artifact_id="artifact",
            source_id="one",
            sequence_no=0,
            first_ordinal=1,
            last_ordinal=1,
            evidence=(first,),
            boundary_method="source",
            receipt_sha256="f" * 64,
        )


def test_query_program_rejects_unknown_obligation_dependency() -> None:
    with pytest.raises(ValueError, match="unknown obligation"):
        QueryProgram(
            query="What should change?",
            intent="recommend",
            subject_terms=("system",),
            obligations=(
                EvidenceObligation(
                    "goal",
                    "objective",
                    True,
                    1.0,
                    dependencies=("missing",),
                ),
            ),
        )


def test_closure_plan_cannot_claim_completion_with_a_missing_requirement() -> None:
    program = QueryProgram(
        query="What should change?",
        intent="recommend",
        subject_terms=("system",),
        obligations=(EvidenceObligation("goal", "objective", True, 1.0),),
    )
    with pytest.raises(ValueError, match="complete_claimed"):
        ClosurePlan(
            query_program=program,
            policy=ClosurePolicy(),
            snapshot=DiscourseSnapshot(
                1,
                1,
                0,
                10,
                (),
                source_content_sha256="1" * 64,
            ),
            seeds=(),
            atoms=(),
            bundles=(),
            obligation_results=(ObligationResult("goal", "not_found"),),
            visited_episode_ids=(),
            visited_unit_ids=(),
            visited_relation_ids=(),
            stopping_reason="complete",
            complete_claimed=True,
        )


def test_receipt_schema_cannot_retain_transformer_token_state() -> None:
    assert not {
        "token_ids",
        "kv_cache",
        "attention",
        "activations",
        "residual_stream",
    } & {field.name for field in fields(ClosureReceipt)}
    with pytest.raises(ValueError, match="zero retained"):
        ClosureReceipt(
            plan_sha256=_DIGEST,
            context_sha256=quote_sha256(""),
            selected_bundle_ids=(),
            selected_atom_ids=(),
            dropped_bundle_reasons={},
            context_token_proxy=0,
            max_context_token_proxy=0,
            tokenizer_identity="test",
            stopping_reason="not_found",
            complete_claimed=False,
            retained_request_token_state_bytes=1,
        )
