from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memory_condense.eval.fast_em_fact_memory import (
    EMFactMemoryError,
    MAX_V2_CITED_PAYLOAD_ROWS,
    build_em_fact_answer_prompt,
    build_fact_compression_messages,
    episodic_neighborhood,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence


def _evidence(identity: str, text: str, source: str) -> FastEvidence:
    return FastEvidence(evidence_id=identity, source_id=source, text=text)


def _question(*, noisy: bool = False):
    root = _evidence("root", "The original preference is tea.", "source-root")
    distraction = _evidence(
        "noise",
        ("Unrelated catering discussion. " * 180 if noisy else "Unrelated note."),
        "source-noise",
    )
    answer = _evidence(
        "answer",
        "On Tuesday the preference changed from tea to coffee.",
        "source-answer",
    )
    root_copy = _evidence(
        "root-copy",
        root.text,
        root.source_id,
    )
    low_value = _evidence(
        "late-noise",
        "A later retrieval hop found another unrelated note.",
        "source-late-noise",
    )
    return SimpleNamespace(
        question_id="q-1",
        dated_question="What is the current drink preference?",
        stages=(
            SimpleNamespace(
                stage_id="causal_graph_coverage_predecessor",
                evidence=(root,),
            ),
            SimpleNamespace(
                stage_id="direct_episode_additions",
                evidence=(root, root_copy, distraction, answer),
            ),
            SimpleNamespace(
                stage_id="representative_episode_additions",
                evidence=(
                    root,
                    root_copy,
                    distraction,
                    answer,
                    low_value,
                ),
            ),
        ),
    )


def _compression(question):
    return parse_fact_compression(
        question,
        """{"facts":[{"text":"The current preference is coffee.","citations":[{"evidence_alias":"E002","quote":"preference changed from tea to coffee"}]}]}""",
    )


def test_extracts_only_final_stage_delta_and_builds_gold_free_request() -> None:
    question = _question()
    root, neighborhood = episodic_neighborhood(question)

    assert [row.evidence_id for row in root] == ["root"]
    assert [row.evidence_id for row in neighborhood] == ["noise", "answer"]
    messages = build_fact_compression_messages(question)
    assert "The original preference is tea" not in messages[-1]["content"]
    assert "[E001 | source=source-noise]" in messages[-1]["content"]
    assert "[E002 | source=source-answer]" in messages[-1]["content"]
    assert "Do not answer" in messages[0]["content"]


def test_v2_compression_prompt_preserves_atomic_question_relevant_structure() -> None:
    question = _question()
    original = build_fact_compression_messages(question)
    explicit_v1 = build_fact_compression_messages(question, policy="v1")
    v2 = build_fact_compression_messages(question, policy="v2")

    assert original == explicit_v1
    assert v2 != original
    instructions = v2[0]["content"]
    for requirement in (
        "directly relevant to the explicit question",
        "bridge or linking facts",
        "disambiguate similar entities",
        "temporal operands",
        "atomic facts",
        "dates and times as temporal metadata",
        "planned",
        "completed",
        "entity names, values, and units",
        "chronological ordering",
        "list member",
        "conflicts as separate attributed facts",
    ):
        assert requirement in instructions
    assert '"text":"one concise fact"' in instructions
    assert parse_fact_compression(question, '{"facts":[]}').facts == ()


def test_v2_compression_prompt_retains_two_hop_bridge_evidence() -> None:
    root = _evidence("root", "The project is Atlas.", "source-root")
    project_link = _evidence(
        "project-link",
        "Project Atlas uses access code cobalt.",
        "source-project-link",
    )
    owner_link = _evidence(
        "owner-link",
        "The owner of access code cobalt is Mira.",
        "source-owner-link",
    )
    question = SimpleNamespace(
        question_id="q-bridge",
        dated_question="Who owns Project Atlas?",
        stages=(
            SimpleNamespace(
                stage_id="causal_graph_coverage_predecessor",
                evidence=(root,),
            ),
            SimpleNamespace(
                stage_id="direct_episode_additions",
                evidence=(root, project_link, owner_link),
            ),
        ),
    )

    messages = build_fact_compression_messages(question, policy="v2")

    assert "bridge or linking facts" in messages[0]["content"]
    assert "[E001 | source=source-project-link]" in messages[1]["content"]
    assert "[E002 | source=source-owner-link]" in messages[1]["content"]


def test_parses_only_exact_source_grounded_fact_citations() -> None:
    question = _question()
    compression = _compression(question)

    fact = compression.facts[0]
    assert fact.text == "The current preference is coffee."
    assert fact.citations[0].evidence_id == "answer"
    assert fact.citations[0].source_id == "source-answer"
    assert compression.receipt_sha256
    assert compression.source_stage_id == "direct_episode_additions"

    with pytest.raises(EMFactMemoryError, match="source-exact"):
        parse_fact_compression(
            question,
            """{"facts":[{"text":"Coffee.","citations":[{"evidence_alias":"E002","quote":"changed to cocoa"}]}]}""",
        )
    with pytest.raises(EMFactMemoryError, match="unknown evidence"):
        parse_fact_compression(
            question,
            """{"facts":[{"text":"Coffee.","citations":[{"evidence_alias":"E999","quote":"coffee"}]}]}""",
        )

    two_quotes = parse_fact_compression(
        question,
        """{"facts":[{"text":"The Tuesday update selected coffee.","citations":[{"evidence_alias":"E002","quote":"On Tuesday"},{"evidence_alias":"E002","quote":"coffee"}]}]}""",
    )
    assert [row.quote for row in two_quotes.facts[0].citations] == [
        "On Tuesday",
        "coffee",
    ]


def test_rejects_duplicate_json_keys_and_ungrounded_facts() -> None:
    question = _question()
    with pytest.raises(EMFactMemoryError, match="repeats key"):
        parse_fact_compression(question, '{"facts":[],"facts":[]}')
    with pytest.raises(EMFactMemoryError, match="requires a citation"):
        parse_fact_compression(
            question,
            '{"facts":[{"text":"Coffee.","citations":[]}]}',
        )


def test_three_arms_keep_root_and_change_only_em_representation() -> None:
    question = _question()
    compression = _compression(question)
    prompts = {
        arm: build_em_fact_answer_prompt(question, compression, arm=arm)
        for arm in ("payload", "facts", "facts_payload")
    }

    for prompt in prompts.values():
        memory = prompt.messages[1].content
        assert "The original preference is tea." in memory
        assert prompt.prompt_token_proxy + prompt.responder_output_token_reserve <= 8000
    assert "Compact episodic facts" not in prompts["payload"].messages[1].content
    assert "Unrelated note." in prompts["payload"].messages[1].content
    assert "The current preference is coffee." in prompts["facts"].messages[1].content
    assert "Unrelated note." not in prompts["facts"].messages[1].content
    assert "The current preference is coffee." in prompts["facts_payload"].messages[1].content
    assert "Unrelated note." in prompts["facts_payload"].messages[1].content


def test_combined_arm_prioritizes_cited_payload_and_keeps_original_alias() -> None:
    question = _question(noisy=True)
    compression = _compression(question)
    facts_only = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts",
        max_prompt_tokens=900,
        responder_output_token_reserve=64,
    )
    combined = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
        max_prompt_tokens=900,
        responder_output_token_reserve=64,
    )

    assert facts_only.fact_ids == ("F1",)
    assert combined.selected_neighborhood_evidence_ids == ("answer",)
    assert combined.dropped_neighborhood_evidence_ids == ("noise",)
    assert "[E002 | source=source-answer]" in combined.messages[1].content
    assert "[E001 | source=source-answer]" not in combined.messages[1].content


def test_v2_combined_arm_includes_only_cited_rows_in_original_alias_order() -> None:
    question = _question()
    compression = parse_fact_compression(
        question,
        '{"facts":['
        '{"text":"A late retrieval hop found a note.","citations":['
        '{"evidence_alias":"E003","quote":"later retrieval hop"}]},'
        '{"text":"An unrelated note was retrieved.","citations":['
        '{"evidence_alias":"E001","quote":"Unrelated note."}]}'
        "]}",
        stage_id="representative_episode_additions",
    )
    prompt = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
        policy="v2",
    )

    assert prompt.selected_neighborhood_evidence_ids == ("noise", "late-noise")
    assert prompt.dropped_neighborhood_evidence_ids == ("answer",)
    memory = prompt.messages[1].content
    assert "[E002 | source=source-answer]" not in memory
    assert memory.index("[E001 | source=source-noise]") < memory.index(
        "[E003 | source=source-late-noise]"
    )


def test_v2_combined_arm_hard_caps_unique_cited_raw_rows() -> None:
    root = _evidence("root", "Root evidence.", "source-root")
    additions = tuple(
        _evidence(f"em-{index}", f"Linked fact {index}.", f"source-{index}")
        for index in range(1, MAX_V2_CITED_PAYLOAD_ROWS + 3)
    )
    question = SimpleNamespace(
        question_id="q-many-citations",
        dated_question="What linked facts were recorded?",
        stages=(
            SimpleNamespace(
                stage_id="causal_graph_coverage_predecessor",
                evidence=(root,),
            ),
            SimpleNamespace(
                stage_id="direct_episode_additions",
                evidence=(root, *additions),
            ),
        ),
    )
    response = json.dumps(
        {
            "facts": [
                {
                    "text": row.text,
                    "citations": [
                        {
                            "evidence_alias": f"E{index:03d}",
                            "quote": row.text,
                        }
                    ],
                }
                for index, row in enumerate(additions, start=1)
            ]
        },
        separators=(",", ":"),
    )
    compression = parse_fact_compression(question, response)

    prompt = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
        policy="v2",
    )

    assert prompt.selected_neighborhood_evidence_ids == tuple(
        row.evidence_id for row in additions[:MAX_V2_CITED_PAYLOAD_ROWS]
    )
    assert prompt.dropped_neighborhood_evidence_ids == tuple(
        row.evidence_id for row in additions[MAX_V2_CITED_PAYLOAD_ROWS:]
    )
    memory = prompt.messages[1].content
    assert f"[E{MAX_V2_CITED_PAYLOAD_ROWS:03d} |" in memory
    assert f"[E{MAX_V2_CITED_PAYLOAD_ROWS + 1:03d} |" not in memory


def test_v2_deterministically_packs_facts_under_a_near_cap_root() -> None:
    question = _question()
    first_only = _compression(question)
    first_prompt = build_em_fact_answer_prompt(
        question,
        first_only,
        arm="facts",
        policy="v2",
        responder_output_token_reserve=64,
    )
    response = json.dumps(
        {
            "facts": [
                {
                    "text": "The current preference is coffee.",
                    "citations": [
                        {
                            "evidence_alias": "E002",
                            "quote": "preference changed from tea to coffee",
                        }
                    ],
                },
                {
                    "text": "verbose " * 1_000,
                    "citations": [
                        {"evidence_alias": "E002", "quote": "coffee"}
                    ],
                },
            ]
        },
        separators=(",", ":"),
    )
    full = parse_fact_compression(question, response)

    packed = build_em_fact_answer_prompt(
        question,
        full,
        arm="facts",
        policy="v2",
        max_prompt_tokens=first_prompt.prompt_token_proxy + 64,
        responder_output_token_reserve=64,
    )

    assert packed.fact_ids == ("F1",)
    assert "verbose verbose" not in packed.messages[1].content


@pytest.mark.parametrize(
    ("dated_question", "expected"),
    (
        (
            "[Question asked at 2023/05/30]\nHow many pages remain?",
            "return only one scalar value",
        ),
        (
            "[Question asked at 2023/05/30]\nWhat is the order of the visits?",
            "comma-separated list in the requested order",
        ),
        (
            "[Question asked at 2023/05/30]\nWhere do I take classes?",
            "return only the single entity",
        ),
    ),
)
def test_v2_adds_gold_blind_answer_shape_guidance(
    dated_question: str,
    expected: str,
) -> None:
    question = _question()
    question.dated_question = dated_question
    prompt = build_em_fact_answer_prompt(
        question,
        _compression(question),
        arm="facts",
        policy="v2",
    )

    instruction = prompt.messages[-1].content
    assert expected in instruction
    assert "explanation" in instruction
    if "order" in dated_question:
        assert "Do not use arrows" in instruction
    if "How many" in dated_question:
        assert "bare number" in instruction


def test_empty_compression_falls_back_to_payload_without_inventing_facts() -> None:
    question = _question()
    compression = parse_fact_compression(question, '{"facts":[]}')
    prompt = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
    )

    assert prompt.fact_ids == ()
    assert "no useful episodic facts" in prompt.messages[1].content
    assert set(prompt.selected_neighborhood_evidence_ids) == {"noise", "answer"}


def test_v2_empty_compression_does_not_reinject_uncited_em() -> None:
    question = _question()
    compression = parse_fact_compression(question, '{"facts":[]}')
    prompt = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
        policy="v2",
    )

    assert prompt.selected_neighborhood_evidence_ids == ()
    assert prompt.dropped_neighborhood_evidence_ids == ("noise", "answer")
    assert "(no cited episodic rows)" in prompt.messages[1].content
    assert "Unrelated note." not in prompt.messages[1].content


def test_default_answer_builder_is_explicit_v1_and_rejects_unknown_policy() -> None:
    question = _question()
    compression = _compression(question)
    implicit = build_em_fact_answer_prompt(question, compression, arm="facts_payload")
    explicit = build_em_fact_answer_prompt(
        question,
        compression,
        arm="facts_payload",
        policy="v1",
    )

    assert implicit == explicit
    with pytest.raises(EMFactMemoryError, match="unknown EM fact-memory policy"):
        build_fact_compression_messages(question, policy="v3")  # type: ignore[arg-type]
    with pytest.raises(EMFactMemoryError, match="unknown EM fact-memory policy"):
        build_em_fact_answer_prompt(
            question,
            compression,
            arm="facts",
            policy="v3",  # type: ignore[arg-type]
        )
