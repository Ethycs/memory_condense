"""Provider binding tests. No network, no key, no SDK — everything injected."""

from __future__ import annotations

import json

import pytest

from memory_condense.ingest.extractor import LLMExtractor, RuleBasedExtractor
from memory_condense.application.llm_provider import (
    DEFAULT_MODEL,
    api_key_present,
    make_completer,
    resolve_extractor,
)
from memory_condense.domain.schemas import Turn


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """A test must not depend on whether this machine has a real key."""
    for var in (
        "MEMORY_CONDENSE_EXTRACTOR",
        "MEMORY_CONDENSE_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # resolve_extractor calls load_dotenv(), which would put the developer's
    # real key back into the environment and make these assertions machine-
    # dependent. Stub it out.
    monkeypatch.setattr("memory_condense.application.llm_provider._load_env", lambda: None)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": content})()})()
        ]


class TestResolveExtractor:
    def test_defaults_to_rules(self):
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, RuleBasedExtractor)
        assert "rule-based" in reason

    def test_auto_without_a_key_falls_back_to_rules(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "auto")
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, RuleBasedExtractor)
        assert "ANTHROPIC_API_KEY" in reason

    def test_llm_without_a_key_still_falls_back_rather_than_failing(self, monkeypatch):
        """Never raise: a missing key must not stop the MCP server starting."""
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "llm")
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, RuleBasedExtractor)
        assert "not set" in reason

    def test_auto_with_a_key_selects_the_llm(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "auto")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, LLMExtractor)
        assert DEFAULT_MODEL in reason

    def test_an_unknown_mode_falls_back_and_says_so(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "magic")
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, RuleBasedExtractor)
        assert "magic" in reason

    def test_model_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "auto")
        monkeypatch.setenv("MEMORY_CONDENSE_LLM_MODEL", "openai/gpt-4.1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        extractor, reason = resolve_extractor()
        assert isinstance(extractor, LLMExtractor)
        assert "openai/gpt-4.1" in reason

    def test_a_provider_with_no_known_key_variable_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMORY_CONDENSE_EXTRACTOR", "auto")
        monkeypatch.setenv("MEMORY_CONDENSE_LLM_MODEL", "somebody/their-model")
        extractor, _ = resolve_extractor()
        assert isinstance(extractor, RuleBasedExtractor)

    def test_an_injected_completion_fn_needs_no_key(self):
        extractor, reason = resolve_extractor(
            mode="llm", completion_fn=lambda **kw: _FakeResponse("{}")
        )
        assert isinstance(extractor, LLMExtractor)
        assert "LLM extraction" in reason


class TestApiKeyPresent:
    def test_false_without_a_key(self):
        assert not api_key_present()

    def test_true_with_one(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert api_key_present()

    def test_unqualified_model_assumes_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert api_key_present("claude-haiku-4-5")


class TestMakeCompleter:
    def test_passes_both_prompts_through_as_chat_messages(self):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("ok")

        result = make_completer(completion_fn=fake)("SYSTEM", "USER")

        assert result == "ok"
        assert seen["messages"] == [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "USER"},
        ]

    def test_sends_no_temperature(self):
        """Several current Claude models 400 on non-default sampling params."""
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("ok")

        make_completer(completion_fn=fake)("s", "u")
        assert "temperature" not in seen

    def test_none_content_becomes_empty_string(self):
        completer = make_completer(completion_fn=lambda **kw: _FakeResponse(None))
        assert completer("s", "u") == ""

    def test_a_malformed_response_does_not_raise(self):
        completer = make_completer(completion_fn=lambda **kw: object())
        assert completer("s", "u") == ""


class TestEndToEnd:
    def test_a_bound_extractor_still_produces_validatable_ops(self):
        """The binding must feed the normal CreateOp → Validator path."""
        turn = Turn(role="user", text="We decided to use Postgres for storage.")
        payload = {
            "create": [
                {
                    "type": "Decision",
                    "content": "Storage is Postgres.",
                    "importance": 0.8,
                    "provenance": [
                        {
                            "turn_id": turn.turn_id,
                            "quote": "We decided to use Postgres for storage.",
                            "chunk_id": None,
                        }
                    ],
                }
            ]
        }
        extractor, _ = resolve_extractor(
            mode="llm",
            completion_fn=lambda **kw: _FakeResponse(json.dumps(payload)),
        )

        ops = extractor.extract([turn])

        assert len(ops.create) == 1
        assert ops.create[0].provenance[0].turn_id == turn.turn_id

    def test_a_transport_failure_yields_no_memories_rather_than_raising(self):
        def boom(**kwargs):
            raise RuntimeError("network down")

        extractor, _ = resolve_extractor(mode="llm", completion_fn=boom)
        turn = Turn(role="user", text="We decided to use Postgres.")

        assert extractor.extract([turn]).is_empty()
