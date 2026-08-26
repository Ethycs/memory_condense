from __future__ import annotations

import inspect
import json
from dataclasses import fields, replace

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval import _fast_cav_link_synthesis_codec as codec
from memory_condense.eval.fast_cav_feature_session import (
    FAST_CAV_FEATURE_BACKEND_FORMAT,
    FAST_CAV_ORDERING_PROXY_ROLE,
    FastCAVFeatureSessionReceipt,
    FastCAVStageReceipt,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_GUIDE_PROJECTION_POLICY,
    FAST_CAV_LINK_SYNTHESIS_ARM_IDS,
    FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    FastCAVLinkSynthesisGuideGroup,
    build_fast_cav_link_synthesis_population,
    parse_fast_cav_link_synthesis,
)
from memory_condense.eval.fast_cav_links import (
    build_fast_cav_concepts,
    build_fast_cav_link_receipt,
)
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)
from memory_condense.search.fusion.steered_readout import MatchedSteeredReadout
from memory_condense.search.fusion.tensor_identity import canonical_float32_tensor


_QUESTION = "Which codes are current, in order?"
_DATED_QUESTION = "[Question asked at 2026/08/22 (Sat) 12:00]\n" + _QUESTION


def _digest(label: str) -> str:
    return quote_sha256(label)


def _order_sha256(ids: tuple[str, ...]) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-steered-readout-order-v1",
            "atom_ids": list(ids),
        }
    )


def _readout(ids: tuple[str, ...]) -> MatchedSteeredReadout:
    base_scores = tuple(0.9 - index * 0.1 for index in range(len(ids)))
    treatment_scores = tuple(reversed(base_scores))
    base_order = tuple(
        ids[index]
        for index in sorted(
            range(len(ids)), key=lambda index: (-base_scores[index], index)
        )
    )
    treatment_order = tuple(
        ids[index]
        for index in sorted(
            range(len(ids)), key=lambda index: (-treatment_scores[index], index)
        )
    )
    return MatchedSteeredReadout(
        original_atom_order=ids,
        base_scores=base_scores,
        treatment_scores=treatment_scores,
        base_order=base_order,
        treatment_order=treatment_order,
        atom_count=len(ids),
        hidden_dim=4,
        max_output_atoms=64,
        max_hidden_dim=4096,
        source_dtype="torch.float32",
        base_scores_sha256=canonical_float32_tensor(
            base_scores, label="fixture base scores"
        ).tensor_sha256,
        treatment_scores_sha256=canonical_float32_tensor(
            treatment_scores, label="fixture treatment scores"
        ).tensor_sha256,
        base_order_sha256=_order_sha256(base_order),
        treatment_order_sha256=_order_sha256(treatment_order),
    )


def _fixture(
    *, final_text: str = "Delta is the final current code.",
) -> tuple[FastRetrievalArtifact, FastCAVFeatureSessionReceipt]:
    evidence = (
        FastEvidence("e-alpha", "source-alpha", "Alpha was selected first."),
        FastEvidence("e-beta", "source-beta", "Beta replaced Alpha later."),
        FastEvidence("e-gamma", "source-gamma", "Gamma remained in position two."),
        FastEvidence("e-delta", "source-delta", final_text),
    )
    ladders = tuple(evidence[: index + 1] for index in range(len(STAGE_IDS)))
    stages: list[FastRetrievalStage] = []
    for stage_ordinal, (stage_id, rows) in enumerate(
        zip(STAGE_IDS, ladders, strict=True)
    ):
        context = "\n".join(row.text for row in rows)
        user = (
            "Retrieved excerpts from the conversation history:\n"
            f"{context}\n\nQuestion: {_DATED_QUESTION}\nShort answer:"
        )
        messages = (
            FastProviderMessage("system", "Use supplied evidence."),
            FastProviderMessage("user", user),
        )
        stages.append(
            FastRetrievalStage(
                stage_id=stage_id,
                stage_receipt_sha256=_digest(f"source-stage-{stage_ordinal}"),
                matched_controls_sha256=_digest("controls"),
                evidence_projection_sha256=identity_sha256(
                    [
                        {
                            "evidence_id": row.evidence_id,
                            "source_id": row.source_id,
                            "text_sha256": quote_sha256(row.text),
                        }
                        for row in rows
                    ]
                ),
                context_sha256=quote_sha256(context),
                prompt_messages_sha256=identity_sha256(
                    [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ]
                ),
                context_token_proxy=50,
                max_context_token_proxy=8_000,
                prompt_token_proxy=100,
                max_prompt_token_proxy=8_000,
                responder_output_token_reserve=256,
                admission_status="root" if stage_ordinal == 0 else "added",
                added_evidence_ids=(rows[-1].evidence_id,),
                context=context,
                evidence=rows,
                provider_messages=messages,
                feature_row_indices=tuple(range(len(rows))),
            )
        )
    final_user = stages[-1].provider_messages[-1]
    question = FastRetrievalQuestion(
        ordinal=0,
        question_id="question-0",
        question_sha256=quote_sha256(_QUESTION),
        dated_question_sha256=quote_sha256(_DATED_QUESTION),
        predecessor_receipt_sha256=_digest("predecessor"),
        retrieval_receipt_sha256=_digest("retrieval"),
        protected_chunk_ids=("chunk-alpha",),
        retained_request_token_state_bytes=0,
        question=_QUESTION,
        dated_question=_DATED_QUESTION,
        final_user_message=final_user,
        question_parse_receipt=FastQuestionParseReceipt(
            framing="memory-condense-qa-user-template-v1",
            source_stage_id=STAGE_IDS[-1],
            provider_message_index=1,
            provider_message_sha256=quote_sha256(final_user.content),
            question_marker_occurrences=1,
            matching_framing_candidates=1,
            dated_question_sha256=quote_sha256(_DATED_QUESTION),
            question_sha256=quote_sha256(_QUESTION),
            question_form="dated_header",
        ),
        feature_rows=tuple(
            FastFeatureRow(
                question=_QUESTION,
                evidence_text=row.text,
                row_sha256=identity_sha256(
                    {
                        "format": "memory-condense-fast-feature-row-v1",
                        "question": _QUESTION,
                        "evidence_text": row.text,
                    }
                ),
            )
            for row in evidence
        ),
        stages=tuple(stages),
    )
    artifact = FastRetrievalArtifact(
        source_path="fixture/retrieval.json",
        raw_sha256=_digest("artifact"),
        format=RETRIEVAL_FORMAT,
        campaign_format=CAMPAIGN_FORMAT,
        population_identity_sha256=_digest("population"),
        source_store_receipt_sha256=_digest("source-store"),
        combined_store_receipt_sha256=_digest("combined-store"),
        retrieval_implementation_sha256=_digest("retrieval-implementation"),
        retrieval_policy_sha256=_digest("retrieval-policy"),
        transcript_tokens=1_000_000,
        turn_count=5_000,
        retained_request_token_state_bytes=0,
        stage_ids=STAGE_IDS,
        questions=(question,),
    )

    bank_sha = _digest("router-bank")
    runtime_sha = _digest("router-runtime")
    concepts = build_fast_cav_concepts(
        bank_identity_sha256=bank_sha,
        artifact_file_sha256s=(_digest("concept-a"), _digest("concept-b")),
        tensor_keys=("private.concept_a.layer_2", "private.concept_b.layer_2"),
    )
    feature_stages: list[FastCAVStageReceipt] = []
    for stage_ordinal, stage in enumerate(stages):
        rows = stage.evidence
        width = len(rows)
        descending = tuple(
            float(width - index) / sum(range(1, width + 1))
            for index in range(width)
        )
        extraction = (descending, tuple(reversed(descending)))
        reinjection = tuple(
            (0.75, 0.25) if index % 2 == 0 else (0.25, 0.75)
            for index in range(width)
        )
        packet_sha = _digest(f"packet-{stage_ordinal}")
        links = build_fast_cav_link_receipt(
            packet_identity_sha256=packet_sha,
            router_runtime_identity_sha256=runtime_sha,
            router_bank_identity_sha256=bank_sha,
            concepts=concepts,
            evidence_ids=stage.evidence_ids,
            source_ids=stage.source_ids,
            evidence_text_sha256s=tuple(
                quote_sha256(row.text) for row in rows
            ),
            extraction_attention=extraction,
            reinjection_attention=reinjection,
        )
        feature_stages.append(
            FastCAVStageReceipt(
                artifact_sha256=artifact.raw_sha256,
                placement_ordinal=stage_ordinal,
                question_ordinal=0,
                question_id=question.question_id,
                question_sha256=question.question_sha256,
                dated_question_sha256=question.dated_question_sha256,
                stage_ordinal=stage_ordinal,
                stage_id=stage.stage_id,
                source_stage_receipt_sha256=stage.stage_receipt_sha256,
                evidence_projection_sha256=stage.evidence_projection_sha256,
                evidence_feature_row_indices=stage.feature_row_indices,
                evidence_ids=stage.evidence_ids,
                source_ids=stage.source_ids,
                evidence_text_sha256s=tuple(
                    quote_sha256(row.text) for row in rows
                ),
                packet_identity_sha256=packet_sha,
                feature_backend_identity_sha256=_digest("feature-backend"),
                feature_checkpoint_sha256=_digest("checkpoint"),
                feature_layer=2,
                feature_hidden_dim=4,
                feature_encoder_runtime_dtype="float32",
                feature_encoder_runtime_device="cpu",
                feature_encoder_prefix_layers=3,
                router_runtime_identity_sha256=runtime_sha,
                router_bank_identity_sha256=bank_sha,
                router_call_ordinal=stage_ordinal,
                reused_router_result=False,
                readout=_readout(stage.evidence_ids),
                links=links,
                readout_role=FAST_CAV_ORDERING_PROXY_ROLE,
            )
        )
    session = FastCAVFeatureSessionReceipt(
        artifact_sha256=artifact.raw_sha256,
        feature_backend_format=FAST_CAV_FEATURE_BACKEND_FORMAT,
        feature_backend_identity_sha256=_digest("feature-backend"),
        feature_checkpoint_sha256=_digest("checkpoint"),
        feature_layer=2,
        feature_hidden_dim=4,
        feature_source_dtype="torch.float32",
        feature_encoder_runtime_dtype="float32",
        feature_encoder_runtime_device="cpu",
        feature_encoder_prefix_layers=3,
        router_runtime_identity_sha256=runtime_sha,
        router_bank_identity_sha256=bank_sha,
        router_num_cavs=2,
        stage_ids=STAGE_IDS,
        question_count=1,
        stage_placement_count=4,
        logical_evidence_placement_count=sum(
            len(stage.evidence) for stage in stages
        ),
        per_question_unique_feature_row_count=4,
        global_unique_evidence_text_count=4,
        global_unique_question_text_count=1,
        global_unique_text_count=5,
        encoder_input_projection_sha256=_digest("encoder-input"),
        encoder_api_call_count=1,
        unique_router_call_count=4,
        batch_size=4,
        stage_receipts=tuple(feature_stages),
    )
    return artifact, session


def _guide_section(user: str) -> tuple[str, str, str]:
    before, separator, remainder = user.partition("\n\nLatent CAV link guide:\n")
    assert separator
    guide, separator, after = remainder.partition("\n\nTask:\n")
    assert separator
    return before, guide, after


def test_builds_matched_s3_arms_with_only_genuine_link_guide_intervention():
    artifact, session = _fixture()

    population = build_fast_cav_link_synthesis_population(artifact, session)

    assert population.stage_id == STAGE_IDS[-1]
    assert population.arm_ids == FAST_CAV_LINK_SYNTHESIS_ARM_IDS
    assert population.logical_prompt_count == population.unique_prompt_count == 2
    assert population.retained_token_id_count == 0
    assert population.retained_tensor_bytes == 0
    assert population.persisted_token_state_bytes == 0
    unlinked, linked = population.prompts
    assert unlinked.arm_id == "unlinked" and unlinked.link_exposed is False
    assert linked.arm_id == "linked" and linked.link_exposed is True
    assert unlinked.evidence_ids == linked.evidence_ids == (
        "e-alpha",
        "e-beta",
        "e-gamma",
        "e-delta",
    )
    assert unlinked.alias_order == linked.alias_order == (
        "E001",
        "E002",
        "E003",
        "E004",
    )
    assert unlinked.evidence_coordinates_sha256 == linked.evidence_coordinates_sha256
    assert unlinked.evidence_catalog_sha256 == linked.evidence_catalog_sha256
    assert unlinked.matched_scaffold_sha256 == linked.matched_scaffold_sha256
    assert unlinked.source_link_receipt_sha256 == linked.source_link_receipt_sha256
    assert unlinked.link_guide_projection_sha256 == linked.link_guide_projection_sha256
    assert unlinked.messages_sha256 != linked.messages_sha256

    messages = population.logical_message_population
    assert messages[0][0] == messages[1][0]
    left_before, left_guide, left_after = _guide_section(messages[0][1]["content"])
    right_before, right_guide, right_after = _guide_section(messages[1][1]["content"])
    assert left_before == right_before
    assert left_after == right_after
    assert left_guide == "unavailable; reason over the evidence independently."
    assert right_guide.splitlines() == [
        "C01 | extract-ranked: E001,E002,E003,E004 | reinject-rank1: E001,E003",
        "C02 | extract-ranked: E004,E003,E002,E001 | reinject-rank1: E002,E004",
    ]
    assert "private.concept" not in right_guide
    assert "->" not in right_guide and ">" not in right_guide
    for alias, evidence in zip(
        linked.alias_order, artifact.questions[0].stages[-1].evidence, strict=True
    ):
        assert left_before.index(f"[{alias}]") == right_before.index(f"[{alias}]")
        assert messages[0][1]["content"].count(evidence.text) == 1
        assert messages[1][1]["content"].count(evidence.text) == 1

    receipt = population.stage_receipts[0]
    assert receipt.source_link_receipt_sha256 == session.stage(
        "question-0", STAGE_IDS[-1]
    ).links.link_receipt_sha256
    assert receipt.evidence_pair_graph_constructed is False
    assert [group.concept_alias for group in receipt.link_guide_groups] == [
        "C01",
        "C02",
    ]
    assert [
        group.reinjection_evidence_aliases for group in receipt.link_guide_groups
    ] == [("E001", "E003"), ("E002", "E004")]
    assert population.completion_preflight.max_prompt_token_proxy == 8_000
    assert all(row.prompt_token_proxy <= 8_000 for row in population.prompts)
    rebuilt = preflight_fast_completion_prompts(
        population.logical_message_population,
        max_prompt_tokens=FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    )
    assert (
        rebuilt.prompt_population_sha256
        == population.completion_preflight.prompt_population_sha256
    )


def test_rejects_same_id_with_changed_exact_source_or_text_coordinate():
    artifact, session = _fixture()
    question = artifact.questions[0]
    s3 = question.stages[-1]
    changed = replace(
        s3.evidence[-1],
        source_id="substituted-source",
        text="Substituted text with the same evidence ID.",
    )
    changed_s3 = replace(s3, evidence=(*s3.evidence[:-1], changed))
    changed_question = replace(
        question, stages=(*question.stages[:-1], changed_s3)
    )
    changed_artifact = replace(artifact, questions=(changed_question,))

    with pytest.raises(ValueError, match="exact evidence ID/source/text"):
        build_fast_cav_link_synthesis_population(changed_artifact, session)


def test_rejects_tampered_nested_genuine_link_seal():
    artifact, session = _fixture()
    link = session.stage("question-0", STAGE_IDS[-1]).links.extraction_links[0]
    object.__setattr__(link, "weight", link.weight / 2.0)

    with pytest.raises(ValueError, match="seal does not match"):
        build_fast_cav_link_synthesis_population(artifact, session)


def test_preflights_complete_population_against_exact_8k_cap():
    artifact, session = _fixture(final_text="oversized " * 9_000)

    with pytest.raises(ValueError, match=r"exceeds the hard token cap.*> 8000"):
        build_fast_cav_link_synthesis_population(artifact, session)


def test_strict_parser_hydrates_exact_citation_and_binds_256_token_contract():
    artifact, session = _fixture()
    population = build_fast_cav_link_synthesis_population(artifact, session)
    completion = json.dumps(
        {
            "answer": "Delta, Gamma",
            "citations": [
                {
                    "evidence_alias": "E004",
                    "quote": "Delta is the final current code.",
                },
                {
                    "evidence_alias": "E003",
                    "quote": "Gamma remained in position two.",
                },
            ],
        },
        separators=(",", ":"),
    )

    parsed = parse_fast_cav_link_synthesis(
        completion,
        stage=artifact.questions[0].stages[-1],
        receipt=population.stage_receipts[0],
    )

    assert parsed.answer == "Delta, Gamma"
    assert parsed.completion_token_proxy <= FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS
    assert parsed.completion_sha256 == quote_sha256(completion)
    assert parsed.citations[0].evidence_id == "e-delta"
    assert parsed.citations[0].source_id == "source-delta"
    assert parsed.citations[0].evidence_text_sha256 == quote_sha256(
        "Delta is the final current code."
    )
    assert parsed.citations[0].quote_sha256 == quote_sha256(
        parsed.citations[0].quote
    )
    assert parsed.response_sha256 == identity_sha256(
        parsed.identity_payload(include_sha256=False)
    )


@pytest.mark.parametrize(
    ("completion", "match"),
    [
        (
            'prefix {"answer":"Delta","citations":[]} suffix',
            "strict synthesis JSON contract",
        ),
        (
            '{"answer":"Delta","citations":[],"extra":true}',
            "strict synthesis JSON contract",
        ),
        (
            '{"answer":"Delta","answer":"Gamma","citations":[]}',
            "strict synthesis JSON contract",
        ),
        (
            '{"answer":"Delta","citations":[]}',
            "strict synthesis JSON contract",
        ),
        (
            '{"answer":"I don\'t know","citations":'
            '[{"evidence_alias":"E004","quote":"Delta"}]}',
            "strict synthesis JSON contract",
        ),
        (
            '{"answer":"Delta","citations":'
            '[{"evidence_alias":"E999","quote":"Delta"}]}',
            "unknown S3 evidence alias",
        ),
        (
            '{"answer":"Delta","citations":'
            '[{"evidence_alias":"E004","quote":"not in evidence"}]}',
            "exact contiguous substring",
        ),
    ],
)
def test_parser_rejects_noncanonical_or_unsupported_json(
    completion: str,
    match: str,
):
    artifact, session = _fixture()
    population = build_fast_cav_link_synthesis_population(artifact, session)

    with pytest.raises(ValueError, match=match):
        parse_fast_cav_link_synthesis(
            completion,
            stage=artifact.questions[0].stages[-1],
            receipt=population.stage_receipts[0],
        )


def test_parser_rejects_completion_above_256_token_proxy(
    monkeypatch: pytest.MonkeyPatch,
):
    artifact, session = _fixture()
    population = build_fast_cav_link_synthesis_population(artifact, session)
    monkeypatch.setattr(
        codec,
        "count_tokens",
        lambda _text: FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS + 1,
    )

    with pytest.raises(ValueError, match="hard 256-token"):
        parse_fast_cav_link_synthesis(
            '{"answer":"I don\'t know","citations":[]}',
            stage=artifact.questions[0].stages[-1],
            receipt=population.stage_receipts[0],
        )


def test_contract_is_gold_free_rank_only_and_has_no_evidence_pair_schema():
    parameters = inspect.signature(
        build_fast_cav_link_synthesis_population
    ).parameters
    assert not any("gold" in name for name in parameters)
    assert FAST_CAV_LINK_GUIDE_PROJECTION_POLICY[
        "evidence_pair_graph_constructed"
    ] is False
    assert FAST_CAV_LINK_GUIDE_PROJECTION_POLICY[
        "evidence_pair_matrix_constructed"
    ] is False
    field_names = {field.name for field in fields(FastCAVLinkSynthesisGuideGroup)}
    assert not any("pair" in name or "graph" in name or "matrix" in name for name in field_names)
