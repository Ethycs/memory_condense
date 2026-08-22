from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_qa_prompt,
)
from memory_condense.eval.mem0_adapter import (
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_SPACY_MODEL,
    MEM0AI_PIN,
)
from tools.mem0_eval.prompt_pack import (
    MEM0_CONFIGURED_RECENT_WINDOW,
    MEM0_EFFECTIVE_RECENT_WINDOW,
    MEM0_MAX_PROMPT_TOKEN_PROXY,
    MEM0_PROMPT_CAP_SEMANTICS,
    MEM0_RETRIEVAL_ROW_FORMAT,
    MEM0_RECENT_WINDOW_SEMANTICS,
    MEM0_SOURCE_JUDGE_MODEL,
    MEM0_SOURCE_RESPONDER_MODEL,
    Mem0PromptProtocolError,
    PromptMemory,
    pack_mem0_prompt as _pack_mem0_prompt,
    render_official_created_at_context,
    verify_provider_input_tokens,
)


def _sha256_json(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_identity(**updates):
    value = {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "certified": True,
        "local_owned_state": True,
        "on_disk": True,
        "stable_config_sha256": "a" * 64,
        "effective_config_sha256": "b" * 64,
        "stack": {
            "dependency_versions": {"mem0ai": MEM0AI_PIN},
            "bm25_model": MEM0_BM25_MODEL,
            "spacy_model": MEM0_SPACY_MODEL,
            "bm25_operational": True,
            "entity_extraction_operational": True,
        },
    }
    value.update(updates)
    return value


def _evaluation_identity(**updates):
    value = {
        "responder_model": MEM0_SOURCE_RESPONDER_MODEL,
        "judge_model": MEM0_SOURCE_JUDGE_MODEL,
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": MEM0_MAX_PROMPT_TOKEN_PROXY,
        "prompt_cap_semantics": MEM0_PROMPT_CAP_SEMANTICS,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": (
            BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }
    value.update(updates)
    return value


def pack_mem0_prompt(question, result, **kwargs):
    kwargs.setdefault("evaluation_identity", _evaluation_identity())
    return _pack_mem0_prompt(question, result, **kwargs)


def _candidate(
    rank: int,
    memory_id: str,
    text: str,
    created_at: str,
    *,
    score: float | None = 0.75,
    attribution_kind: str = MEM0_ATTRIBUTION_KIND,
):
    return SimpleNamespace(
        rank=rank,
        memory_id=memory_id,
        text=text,
        score=score,
        created_at=created_at,
        attribution_kind=attribution_kind,
    )


def _result(raw_pool=(), **updates):
    value = {
        "query": "Where?",
        "raw_pool": tuple(raw_pool),
        "official_longmemeval_protocol": True,
        "official_search_protocol": True,
        "rendering_mode": MEM0_CERTIFIED_RENDERING,
        "certified_rendering": True,
        "comparison_certified": True,
        "runtime_identity": _runtime_identity(),
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
    }
    value.update(updates)
    return SimpleNamespace(**value)


def _prompt_memory(
    rank: int,
    memory_id: str,
    text: str,
    created_at: str,
) -> PromptMemory:
    return PromptMemory(
        rank=rank,
        memory_id=memory_id,
        text=text,
        score=0.75,
        created_at=created_at,
        attribution_kind=MEM0_ATTRIBUTION_KIND,
    )


def _text_that_reaches_exact_8000_with(prefix: tuple[PromptMemory, ...]) -> str:
    """Find a normalized one-token-repeat suffix at the locked exact cap."""

    low = 1
    high = MEM0_MAX_PROMPT_TOKEN_PROXY * 2
    while low <= high:
        midpoint = (low + high) // 2
        text = "beta" + " beta" * (midpoint - 1)
        candidate = _prompt_memory(
            len(prefix) + 1,
            "boundary",
            text,
            "2024-01-01T01:00:00Z",
        )
        proxy = count_chat_prompt_token_proxy(
            build_qa_prompt(
                "Where?",
                [render_official_created_at_context((*prefix, candidate))],
            )
        )
        if proxy == MEM0_MAX_PROMPT_TOKEN_PROXY:
            return text
        if proxy < MEM0_MAX_PROMPT_TOKEN_PROXY:
            low = midpoint + 1
        else:
            high = midpoint - 1
    raise AssertionError("could not construct an exact 8000-token QA prompt")


def test_rank_admission_renders_created_at_chronology_and_exact_qa_messages():
    result = _result(
        [
            _candidate(
                1,
                "new",
                "newer fact",
                "2024-02-02T18:00:00-08:00",
            ),
            _candidate(
                2,
                "old",
                "older fact",
                "2024-01-01T09:00:00+00:00",
            ),
        ]
    )

    packed = pack_mem0_prompt("Where?", result)

    assert [item.memory_id for item in packed.packed_pool] == ["new", "old"]
    assert packed.context == (
        "--- Monday, January 01, 2024 ---\n- older fact\n\n"
        "--- Saturday, February 03, 2024 ---\n- newer fact"
    )
    expected = build_qa_prompt("Where?", [packed.context])
    assert packed.provider_messages() == expected
    assert list(packed.messages) == expected
    assert len(packed.messages) == 2
    assert [message["role"] for message in packed.messages] == ["system", "user"]
    assert packed.prompt_token_proxy == count_chat_prompt_token_proxy(expected)
    assert packed.prompt_token_proxy <= MEM0_MAX_PROMPT_TOKEN_PROXY
    assert packed.configured_recent_window == MEM0_CONFIGURED_RECENT_WINDOW
    assert packed.effective_recent_window == MEM0_EFFECTIVE_RECENT_WINDOW == 0
    assert packed.recent_window_semantics == MEM0_RECENT_WINDOW_SEMANTICS
    # The configured replay default is metadata only for completed-haystack
    # LongMemEval QA; the exact provider input contains retrieved memory text
    # and no independently appended raw conversation tail.
    assert "recent turn" not in json.dumps(packed.provider_messages()).casefold()


def test_full_recount_admits_exact_cap_and_rejects_next_rank():
    first = _prompt_memory(1, "one", "alpha", "2024-01-01T00:00:00Z")
    boundary_text = _text_that_reaches_exact_8000_with((first,))
    memories = (
        first,
        _prompt_memory(
            2,
            "two",
            boundary_text,
            "2024-01-01T01:00:00Z",
        ),
    )
    exact_context = render_official_created_at_context(memories)
    exact_cap = count_chat_prompt_token_proxy(
        build_qa_prompt("Where?", [exact_context])
    )
    assert exact_cap == MEM0_MAX_PROMPT_TOKEN_PROXY
    result = _result(
        [
            _candidate(1, "one", memories[0].text, memories[0].created_at),
            _candidate(2, "two", memories[1].text, memories[1].created_at),
            _candidate(
                3,
                "three",
                "gamma " * 60,
                "2024-01-01T02:00:00Z",
            ),
        ]
    )

    packed = pack_mem0_prompt("Where?", result)

    assert [item.memory_id for item in packed.packed_pool] == ["one", "two"]
    assert packed.prompt_token_proxy == exact_cap
    assert packed.residual_prompt_token_proxy == 0
    assert [item.reason for item in packed.diagnostics] == [
        "selected",
        "selected",
        "prompt_token_budget",
    ]
    assert packed.diagnostics[-1].proposed_prompt_token_proxy > exact_cap
    # Repeated date headings in singleton estimates are not additive.  The
    # implementation must recount the fully assembled context and prompt.
    singleton_total = sum(
        item.rendered_tokens for item in packed.diagnostics[:2]
    )
    assert singleton_total != packed.context_tokens
    assert packed.context_tokens == count_tokens(packed.context)


def test_oversized_early_rank_is_skipped_and_later_short_rank_can_fit():
    small = _prompt_memory(2, "small", "small fact", "2024-01-01T00:00:00Z")
    small_proxy = count_chat_prompt_token_proxy(
        build_qa_prompt("Where?", [render_official_created_at_context((small,))])
    )
    result = _result(
        [
            _candidate(
                1,
                "huge",
                "oversized " * 10_000,
                "2024-02-01T00:00:00Z",
            ),
            _candidate(2, "small", small.text, small.created_at),
        ]
    )

    packed = pack_mem0_prompt("Where?", result)

    assert [item.memory_id for item in packed.packed_pool] == ["small"]
    assert [item.reason for item in packed.diagnostics] == [
        "prompt_token_budget",
        "selected",
    ]
    assert packed.prompt_token_proxy == small_proxy


def test_empty_raw_pool_produces_exact_no_context_two_message_input():
    packed = pack_mem0_prompt("Where?", _result())
    expected = build_qa_prompt("Where?", [])

    assert packed.context == ""
    assert packed.context_tokens == 0
    assert packed.raw_memory_count == 0
    assert packed.packed_memory_count == 0
    assert packed.provider_messages() == expected
    assert packed.prompt_token_proxy == count_chat_prompt_token_proxy(expected)
    assert packed.context_sha256 == hashlib.sha256(b"").hexdigest()


def test_empty_memory_is_audited_but_never_enters_context():
    result = _result(
        [_candidate(1, "empty", "", "2024-01-01T00:00:00Z", score=None)]
    )

    packed = pack_mem0_prompt("Where?", result)

    assert packed.packed_pool == ()
    assert packed.diagnostics[0].reason == "empty_memory"
    assert packed.diagnostics[0].selected is False
    assert packed.provider_messages() == build_qa_prompt("Where?", [])


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"official_search_protocol": False}, "official_search_protocol"),
        ({"certified_rendering": False}, "certified_rendering"),
        (
            {"rendering_mode": "enriched-attribution-noncertifying"},
            "rendering_mode",
        ),
        (
            {"supports_exact_source_provenance": True},
            "supports_exact_source_provenance",
        ),
        (
            {"runtime_identity": _runtime_identity(on_disk=False)},
            "on_disk",
        ),
    ],
)
def test_comparison_certified_bit_is_never_sufficient(updates, match):
    result = _result(comparison_certified=True, **updates)

    with pytest.raises(Mem0PromptProtocolError, match=match):
        pack_mem0_prompt("Where?", result)


def test_raw_pool_rank_date_identity_and_attribution_are_fail_closed():
    valid = _candidate(1, "one", "fact", "2024-01-01T00:00:00Z")
    cases = [
        (_candidate(2, "one", "fact", valid.created_at), "retrieval position"),
        (_candidate(1, "one", "fact", "not-a-date"), "created_at is invalid"),
        (
            _candidate(
                1,
                "one",
                "fact",
                valid.created_at,
                attribution_kind="exact_evidence",
            ),
            "attribution_kind",
        ),
    ]
    for candidate, match in cases:
        with pytest.raises(Mem0PromptProtocolError, match=match):
            pack_mem0_prompt("Where?", _result([candidate]))

    with pytest.raises(Mem0PromptProtocolError, match="repeats a memory_id"):
        pack_mem0_prompt(
            "Where?",
            _result(
                [
                    valid,
                    _candidate(2, "one", "other", "2024-01-02T00:00:00Z"),
                ]
            ),
        )


def test_output_reserve_is_separate_and_provider_usage_is_postchecked():
    packed = pack_mem0_prompt("Where?", _result())

    assert (
        packed.responder_output_token_reserve
        == BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    )
    assert packed.request_token_proxy == (
        packed.prompt_token_proxy + BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    )
    assert packed.prompt_token_proxy <= packed.max_prompt_token_proxy
    assert verify_provider_input_tokens(packed, 0) is None
    assert verify_provider_input_tokens(packed, packed.max_prompt_token_proxy) is True
    with pytest.raises(Mem0PromptProtocolError, match="provider input usage exceeds"):
        verify_provider_input_tokens(packed, packed.max_prompt_token_proxy + 1)


def test_call_cap_must_equal_source_identity_and_locked_8000():
    empty_proxy = count_chat_prompt_token_proxy(build_qa_prompt("Where?", []))
    with pytest.raises(Mem0PromptProtocolError, match="disagrees"):
        pack_mem0_prompt(
            "Where?",
            _result(),
            max_prompt_tokens=empty_proxy,
        )
    with pytest.raises(Mem0PromptProtocolError, match="frozen validation policy"):
        _pack_mem0_prompt(
            "Where?",
            _result(),
            evaluation_identity=_evaluation_identity(max_prompt_tokens=empty_proxy),
        )


def test_source_evaluation_identity_is_required_and_fully_bound():
    with pytest.raises(TypeError, match="evaluation_identity"):
        _pack_mem0_prompt("Where?", _result())
    with pytest.raises(Mem0PromptProtocolError, match="responder_model"):
        _pack_mem0_prompt(
            "Where?",
            _result(),
            evaluation_identity=_evaluation_identity(responder_model="wrong"),
        )
    with pytest.raises(Mem0PromptProtocolError, match="prompt_token_proxy_identity"):
        _pack_mem0_prompt(
            "Where?",
            _result(),
            evaluation_identity=_evaluation_identity(
                prompt_token_proxy_identity={"schema": "tampered"}
            ),
        )


def test_json_pools_and_all_hashes_bind_exact_provider_artifact():
    result = _result(
        [
            _candidate(1, "one", "first", "2024-02-01T00:00:00Z"),
            _candidate(2, "two", "second", "2024-01-01T00:00:00Z"),
        ]
    )
    packed = pack_mem0_prompt("Where?", result)
    row = packed.to_retrieval_row(question_id="q-1", search_latency_s=0.25)

    assert row["format"] == MEM0_RETRIEVAL_ROW_FORMAT
    assert row["question_id"] == "q-1"
    assert row["messages"] == build_qa_prompt("Where?", [packed.context])
    assert row["raw_memory_count"] == len(row["raw_pool"])
    assert row["packed_memory_count"] == len(row["packed_pool"])
    assert row["raw_pool_sha256"] == _sha256_json(row["raw_pool"])
    assert row["packed_pool_sha256"] == _sha256_json(row["packed_pool"])
    assert row["context_sha256"] == hashlib.sha256(
        row["context"].encode("utf-8")
    ).hexdigest()
    assert row["messages_sha256"] == _sha256_json(row["messages"])
    receipt_hash = row.pop("retrieval_row_sha256")
    assert receipt_hash == _sha256_json(row)
    assert row["prompt_token_proxy_identity"] == tokenizer_proxy_identity()
    assert row["source_evaluation_identity"] == _evaluation_identity()
    assert row["source_evaluation_identity_sha256"] == _sha256_json(
        row["source_evaluation_identity"]
    )
    assert row["prompt_pack_protocol"] == packed.protocol
    assert row["configured_recent_window"] == MEM0_CONFIGURED_RECENT_WINDOW
    assert row["effective_recent_window"] == MEM0_EFFECTIVE_RECENT_WINDOW
    assert row["recent_window_semantics"] == MEM0_RECENT_WINDOW_SEMANTICS
    assert row["provenance"] == {
        "kind": MEM0_ATTRIBUTION_KIND,
        "supports_exact_source_provenance": False,
    }
    raw_positions = {
        item["memory_id"]: index for index, item in enumerate(row["raw_pool"])
    }
    packed_positions = [raw_positions[item["memory_id"]] for item in row["packed_pool"]]
    assert packed_positions == sorted(packed_positions)


def test_mapping_artifact_input_is_supported_without_adapter_prompt_fields():
    candidate = {
        "rank": 1,
        "memory_id": "one",
        "text": "fact",
        "score": None,
        "created_at": "2024-01-01T00:00:00Z",
        "attribution_kind": MEM0_ATTRIBUTION_KIND,
    }
    result = vars(_result([candidate])).copy()
    # The independent packer neither needs nor trusts these adapter outputs.
    result.update(
        {
            "packed": "tampered",
            "context": "tampered",
            "prompt": "tampered",
            "prompt_token_proxy": -1,
        }
    )

    packed = pack_mem0_prompt("Where?", result)

    assert packed.context.endswith("- fact")
    assert "tampered" not in packed.context
    assert packed.prompt_token_proxy > 0
