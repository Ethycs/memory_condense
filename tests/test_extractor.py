import json

import pytest

from memory_condense.extractor import (
    MEMORY_OPS_SYSTEM_PROMPT,
    RULES,
    LLMExtractor,
    RuleBasedExtractor,
    parse_memory_ops,
)
from memory_condense.memory_store import MemoryStore
from memory_condense.schemas import Chunk, MemoryOps, MemoryType, Turn
from memory_condense.transcript_store import TranscriptStore
from memory_condense.validator import Validator


@pytest.fixture
def extractor():
    return RuleBasedExtractor()


def turn(text, role="user"):
    return Turn(role=role, text=text)


def types_for(text, extractor=None):
    ex = extractor or RuleBasedExtractor()
    ops = ex.extract([turn(text)])
    return [op.type for op in ops.create]


# ----------------------------------------------------------------------
# RuleBasedExtractor — cue coverage
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I prefer dark mode everywhere.", MemoryType.PREFERENCE),
        ("I like the tabs over spaces style.", MemoryType.PREFERENCE),
        ("I'd rather ship it on Friday.", MemoryType.PREFERENCE),
        ("We decided to use Postgres.", MemoryType.DECISION),
        ("Let's go with the hnswlib index.", MemoryType.DECISION),
        ("We'll use SQLite for the transcript.", MemoryType.DECISION),
        ("The API key must never be logged.", MemoryType.CONSTRAINT),
        ("Don't send anything to the network.", MemoryType.CONSTRAINT),
        ("Always run the tests before pushing.", MemoryType.CONSTRAINT),
        ("Actually the deadline is Thursday.", MemoryType.CORRECTION),
        ("I meant the staging cluster.", MemoryType.CORRECTION),
        ("Correction: the port is 8080.", MemoryType.CORRECTION),
        ("Heat is defined as the decayed energy tier.", MemoryType.DEFINITION),
        ("A chunk means a merged span of sentences.", MemoryType.DEFINITION),
        ("TODO: wire up the validator.", MemoryType.TASK),
        ("I need to renew the certificate.", MemoryType.TASK),
        ("The next step is the eval harness.", MemoryType.TASK),
    ],
)
def test_rule_cues_map_to_types(text, expected):
    assert types_for(text) == [expected]


def test_importance_is_high_for_consequential_types():
    ops = RuleBasedExtractor().extract(
        [
            turn("We decided to use Postgres."),
            turn("The token must never be logged."),
            turn("Actually it is Thursday."),
            turn("I prefer dark mode always on."),
        ]
    )
    by_type = {op.type: op.importance for op in ops.create}
    assert by_type[MemoryType.DECISION] == pytest.approx(0.8)
    assert by_type[MemoryType.CONSTRAINT] == pytest.approx(0.8)
    assert by_type[MemoryType.CORRECTION] == pytest.approx(0.8)


def test_preference_importance_is_base():
    ops = RuleBasedExtractor().extract([turn("I prefer dark mode.")])
    assert ops.create[0].importance == pytest.approx(0.5)


def test_no_cue_produces_nothing(extractor):
    ops = extractor.extract([turn("The weather is nice today.")])
    assert ops.is_empty()


def test_empty_input(extractor):
    assert extractor.extract([]).is_empty()
    assert extractor.extract([turn("   ")]).is_empty()


def test_short_fragments_are_skipped():
    assert RuleBasedExtractor(min_words=5).extract([turn("I prefer tabs.")]).is_empty()


def test_first_matching_rule_wins():
    """Correction is checked before constraint, so it takes precedence."""
    assert types_for("Actually you must never do that.") == [MemoryType.CORRECTION]


def test_multiple_sentences_produce_multiple_ops(extractor):
    ops = extractor.extract(
        [turn("I prefer dark mode. We decided to use Postgres. The sky is blue.")]
    )
    assert [op.type for op in ops.create] == [
        MemoryType.PREFERENCE,
        MemoryType.DECISION,
    ]


def test_newlines_split_sentences(extractor):
    ops = extractor.extract([turn("I prefer dark mode\nWe decided to use Postgres")])
    assert len(ops.create) == 2


def test_duplicate_sentences_in_one_turn_are_deduped(extractor):
    ops = extractor.extract([turn("I prefer dark mode. I prefer dark mode.")])
    assert len(ops.create) == 1


def test_role_filter():
    turns = [turn("I prefer dark mode.", "user"), turn("I prefer tabs.", "assistant")]
    assert len(RuleBasedExtractor().extract(turns).create) == 2
    assert len(RuleBasedExtractor(roles=["user"]).extract(turns).create) == 1


def test_is_deterministic(extractor):
    turns = [turn("We decided to use Postgres. I prefer dark mode.")]
    first = extractor.extract(turns)
    second = extractor.extract(turns)
    assert [c.content for c in first.create] == [c.content for c in second.create]


def test_long_content_is_truncated_but_quote_is_not():
    text = "I prefer " + ("x" * 400) + " for everything."
    op = RuleBasedExtractor(max_content_chars=50).extract([turn(text)]).create[0]
    assert len(op.content) == 50
    assert op.content.endswith("...")
    assert op.provenance[0].quote in text


def test_chunk_ids_are_attached_when_chunks_are_supplied():
    t = turn("I prefer dark mode.")
    chunk = Chunk(
        chunk_id="chunk-1",
        turn_id=t.turn_id,
        text=t.text,
        start_char=0,
        end_char=len(t.text),
        token_count=5,
    )
    op = RuleBasedExtractor().extract([t], [chunk]).create[0]
    assert op.provenance[0].chunk_id == "chunk-1"


def test_chunk_id_is_none_without_chunks(extractor):
    op = extractor.extract([turn("I prefer dark mode.")]).create[0]
    assert op.provenance[0].chunk_id is None


def test_rules_table_is_tunable():
    custom = [(RULES[0][0], MemoryType.ENTITY, 0.9)]
    ops = RuleBasedExtractor(rules=custom).extract([turn("Actually it is Thursday.")])
    assert ops.create[0].type is MemoryType.ENTITY
    assert ops.create[0].importance == pytest.approx(0.9)


# ----------------------------------------------------------------------
# RuleBasedExtractor — end-to-end through the Validator
# ----------------------------------------------------------------------


def test_quotes_are_exact_substrings_of_the_turn(extractor):
    text = "I prefer dark mode.  We decided to use Postgres!  TODO: write tests."
    t = turn(text)
    ops = extractor.extract([t])
    assert len(ops.create) == 3
    for op in ops.create:
        assert op.provenance[0].quote in text


def test_rule_output_passes_validation_end_to_end(db):
    transcripts = TranscriptStore(db)
    stored = [
        transcripts.append(
            "user",
            "I prefer dark mode. We decided to use Postgres.\n"
            "The API token must never be logged.",
        ),
        transcripts.append("user", "Actually, the deadline is Thursday."),
    ]

    ops = RuleBasedExtractor().extract(stored)
    assert ops.total_ops() == 4

    report = Validator(db).validate(ops)
    assert report.ok, [e.model_dump() for e in report.rejected]
    assert len(report.accepted.create) == 4

    summary = MemoryStore(db).apply(report)
    assert summary["created"] == 4
    assert MemoryStore(db).count() == 4


def test_rule_output_survives_indented_transcript_text(db):
    t = TranscriptStore(db).append(
        "user", "I prefer\n    dark    mode for everything."
    )
    ops = RuleBasedExtractor().extract([t])
    report = Validator(db).validate(ops)
    assert report.ok


# ----------------------------------------------------------------------
# LLMExtractor
# ----------------------------------------------------------------------


def _valid_payload(turn_id):
    return {
        "create": [
            {
                "type": "Decision",
                "content": "Use Postgres for storage",
                "importance": 0.8,
                "provenance": [
                    {"turn_id": turn_id, "quote": "We decided to use Postgres"}
                ],
            }
        ]
    }


def test_llm_extractor_parses_valid_json():
    t = turn("We decided to use Postgres.")
    calls = []

    def complete(system, user):
        calls.append((system, user))
        return json.dumps(_valid_payload(t.turn_id))

    ops = LLMExtractor(complete).extract([t])
    assert len(ops.create) == 1
    assert ops.create[0].type is MemoryType.DECISION
    assert ops.create[0].provenance[0].turn_id == t.turn_id
    assert calls[0][0] == MEMORY_OPS_SYSTEM_PROMPT
    assert t.turn_id in calls[0][1]
    assert "We decided to use Postgres." in calls[0][1]


def test_llm_extractor_strips_markdown_fences():
    t = turn("We decided to use Postgres.")
    payload = json.dumps(_valid_payload(t.turn_id))
    ops = LLMExtractor(lambda s, u: f"```json\n{payload}\n```").extract([t])
    assert len(ops.create) == 1


def test_llm_extractor_recovers_object_from_surrounding_prose():
    t = turn("We decided to use Postgres.")
    payload = json.dumps(_valid_payload(t.turn_id))
    raw = f"Sure! Here are the ops:\n{payload}\nHope that helps."
    assert len(LLMExtractor(lambda s, u: raw).extract([t]).create) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not json at all",
        "{ this is broken json ",
        "[]",
        "null",
        '{"create": "not a list"}',
        '{"create": [{"type": "NotAType", "content": "x", "provenance": []}]}',
        '{"create": [{"content": "missing type"}]}',
    ],
)
def test_llm_extractor_returns_empty_on_malformed_output(raw):
    ops = LLMExtractor(lambda s, u: raw).extract([turn("We decided to use Postgres.")])
    assert isinstance(ops, MemoryOps)
    assert ops.is_empty()


def test_llm_extractor_swallows_transport_errors():
    def boom(system, user):
        raise RuntimeError("provider exploded")

    ops = LLMExtractor(boom).extract([turn("We decided to use Postgres.")])
    assert ops.is_empty()


def test_llm_extractor_no_turns_means_no_call():
    def complete(system, user):  # pragma: no cover - must not run
        raise AssertionError("should not be called")

    assert LLMExtractor(complete).extract([]).is_empty()


def test_llm_extractor_prompt_window_is_capped():
    turns = [turn(f"We decided item {i}.") for i in range(5)]
    prompt = LLMExtractor(lambda s, u: "{}", max_turns=2).build_prompt(turns)
    assert "We decided item 4." in prompt
    assert "We decided item 0." not in prompt


def test_llm_extractor_prompt_lists_chunk_ids():
    t = turn("We decided to use Postgres.")
    chunk = Chunk(
        chunk_id="chunk-9",
        turn_id=t.turn_id,
        text=t.text,
        start_char=0,
        end_char=len(t.text),
        token_count=5,
    )
    prompt = LLMExtractor(lambda s, u: "{}").build_prompt([t], [chunk])
    assert "chunk-9" in prompt


def test_llm_output_still_goes_through_the_validator(db):
    """A fabricated quote from the model is rejected, not stored."""
    t = TranscriptStore(db).append("user", "We decided to use Postgres.")
    payload = {
        "create": [
            {
                "type": "Decision",
                "content": "Use MySQL",
                "provenance": [
                    {"turn_id": t.turn_id, "quote": "We decided to use MySQL"}
                ],
            }
        ]
    }
    ops = LLMExtractor(lambda s, u: json.dumps(payload)).extract([t])
    assert len(ops.create) == 1

    report = Validator(db).validate(ops)
    assert not report.ok
    assert report.rejected[0].reason == "quote_not_found"
    assert MemoryStore(db).apply(report)["created"] == 0


def test_parse_memory_ops_helper_is_total():
    assert parse_memory_ops("garbage").is_empty()
    assert parse_memory_ops('{"create": []}').is_empty()


def test_system_prompt_documents_the_provenance_rule():
    assert "provenance" in MEMORY_OPS_SYSTEM_PROMPT
    assert "verbatim" in MEMORY_OPS_SYSTEM_PROMPT or "EXACTLY" in MEMORY_OPS_SYSTEM_PROMPT
    assert "supersede" in MEMORY_OPS_SYSTEM_PROMPT
