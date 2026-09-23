from __future__ import annotations

import copy
import hashlib
from collections import Counter

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._retrieval_qa_prompt import build_qa_prompt
from tools.matched_eval.contracts import (
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
)
from tools.matched_eval.hot_v5_user_spine_prompt import (
    HARD_WORKSPACE_TOKEN_CAP,
    OPERATION_AWARE_SYSTEM_PROMPT,
    OUTPUT_TOKEN_RESERVE,
    USER_SPINE_USER_TEMPLATE,
    UserSpinePromptError,
    render_user_spine_prompt,
    replace_system_prompt_only,
)


def _row(
    evidence_id: str,
    source_id: str,
    role: str,
    text: str,
    *,
    turn_id: str | None = None,
) -> dict[str, object]:
    rendered = f"[2026-09-06T00:00:00+00:00 | {role}] {text}"
    return {
        "chunk_id": evidence_id,
        "created_at": "2026-09-06T00:00:00+00:00",
        "evidence_id": evidence_id,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
        "role": role,
        "route": "fixture",
        "score": 1.0,
        "source_id": source_id,
        "turn_id": turn_id or f"turn-{evidence_id}",
    }


def test_operation_a_replaces_only_system_content() -> None:
    original = build_qa_prompt(
        "[Question asked at 2026-09-07] What happened?",
        ["[2026-09-06 | user] The launch completed."],
    )
    frozen = copy.deepcopy(original)

    rendered = replace_system_prompt_only(original)

    assert original == frozen
    assert rendered[0] == {
        "role": "system",
        "content": OPERATION_AWARE_SYSTEM_PROMPT,
    }
    assert rendered[1] == original[1]
    assert rendered[1]["content"].encode("utf-8") == original[1][
        "content"
    ].encode("utf-8")


@pytest.mark.parametrize(
    "messages",
    (
        [],
        [{"role": "system", "content": "x"}],
        [
            {"role": "system", "content": "x"},
            {"role": "assistant", "content": "y"},
        ],
        [
            {"role": "system", "content": "x", "extra": "z"},
            {"role": "user", "content": "y"},
        ],
    ),
)
def test_operation_a_rejects_noncanonical_message_envelopes(
    messages: list[dict[str, str]],
) -> None:
    with pytest.raises(UserSpinePromptError):
        replace_system_prompt_only(messages)


def test_universal_policy_names_every_required_answer_operation() -> None:
    policy = OPERATION_AWARE_SYSTEM_PROMPT.casefold()
    for phrase in (
        "scan all excerpts",
        "entity, relation, scope, time, and output fields",
        "exact entities",
        "latest state",
        "stated event time",
        "elapsed time",
        "count physical items as written",
        "completed event",
        "exactly which field is unknown",
        "synthesize preferences",
        "already completed",
        "what the assistant said",
        "title plus url",
        "contradiction",
    ):
        assert phrase in policy


def test_user_spine_order_is_deterministic_and_source_local() -> None:
    selected = [
        _row("b-assistant", "source-b", "assistant", "B recommendation."),
        _row("a-user", "source-a", "user", "A request.", turn_id="shared"),
        _row("b-user", "source-b", "user", "B request.", turn_id="shared"),
        _row("a-assistant", "source-a", "assistant", "A recommendation."),
    ]

    first = render_user_spine_prompt("[Dated] What was recommended?", selected)
    second = render_user_spine_prompt("[Dated] What was recommended?", selected)

    assert first == second
    assert first["receipt_sha256"] == identity_sha256(
        {key: value for key, value in first.items() if key != "receipt_sha256"}
    )
    assert [row["source_id"] for row in first["source_blocks"]] == [
        "source-b",
        "source-a",
    ]
    assert [row["block_label"] for row in first["source_blocks"]] == [
        "S1",
        "S2",
    ]
    assert first["rendered_evidence_ids"] == [
        "b-user",
        "b-assistant",
        "a-user",
        "a-assistant",
    ]
    assert first["rendered_parent_ranks"] == [3, 1, 2, 4]
    context = first["provider_messages"][1]["content"]
    assert context.index("B request.") < context.index("B recommendation.")
    assert context.index("B recommendation.") < context.index("A request.")
    assert context.count("shared") == 0  # turn IDs never merge source blocks.


def test_no_loss_multiset_retains_same_text_under_distinct_ids_then_exact_id_dedups() -> None:
    one = _row("one", "source", "user", "Same exact evidence text.")
    two = _row("two", "source", "user", "Same exact evidence text.")
    selected = [one, two, copy.deepcopy(one)]
    frozen = copy.deepcopy(selected)

    rendered = render_user_spine_prompt("[Dated] What happened?", selected)

    assert selected == frozen
    assert rendered["selected_evidence_count"] == 3
    assert rendered["retained_evidence_count"] == 2
    assert rendered["dedup_excluded_evidence_count"] == 1
    assert rendered["retained_evidence_ids"] == ["one", "two"]
    assert Counter(rendered["selected_row_sha256s"]) == Counter(
        (identity_sha256(one), identity_sha256(two), identity_sha256(one))
    )
    duplicate = rendered["dedup_excluded_exact_id_bindings"][0]
    assert duplicate == {
        "evidence_id": "one",
        "excluded_parent_rank": 3,
        "retained_parent_rank": 1,
        "row_sha256": identity_sha256(one),
    }
    user_message = rendered["provider_messages"][1]["content"]
    assert user_message.count("Same exact evidence text.") == 2
    assert rendered["unique_selected_rows_omitted"] == 0


def test_assistant_lookup_keeps_both_roles_and_labels_them() -> None:
    selected = [
        _row("request", "session-1", "user", "Recommend a blue bicycle."),
        _row("reply", "session-1", "assistant", "I recommend the Swift Blue."),
    ]
    rendered = render_user_spine_prompt(
        "[Dated] Which bicycle did the assistant recommend?", selected
    )
    system, user = rendered["provider_messages"]

    assert "what the assistant said" in system["content"].casefold()
    assert "<S1>" in user["content"]
    assert "<U>" in user["content"]
    assert "<A>" in user["content"]
    assert "Recommend a blue bicycle." in user["content"]
    assert "I recommend the Swift Blue." in user["content"]


def test_renderer_is_gold_blind_and_uses_exact_token_arithmetic() -> None:
    question = "[Question asked at 2026-09-07] What happened?"
    rendered = render_user_spine_prompt(
        question,
        [_row("event", "session-1", "user", "The event completed.")],
    )

    assert_gold_blind(rendered)
    forbidden = {"gold", "category", "reference", "ordinal"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested for child in value.values() for nested in keys(child)
            }
        if isinstance(value, list):
            return {nested for child in value for nested in keys(child)}
        return set()

    assert not (forbidden & {key.casefold() for key in keys(rendered)})
    assert len(rendered["provider_messages"]) == 2
    assert rendered["provider_calls"] == 0
    assert rendered["prompt_workspace_token_proxy"] == (
        rendered["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
    )
    assert rendered["prompt_workspace_token_proxy"] <= HARD_WORKSPACE_TOKEN_CAP
    messages = rendered["provider_messages"]
    assert rendered["messages_sha256"] == identity_sha256(messages)
    assert rendered["prompt_token_proxy"] == count_chat_prompt_token_proxy(messages)
    before, after = USER_SPINE_USER_TEMPLATE.split("{context}")
    suffix = after.format(question=question)
    user_content = messages[1]["content"]
    context = user_content[len(before) : -len(suffix)]
    assert user_content.startswith(before) and user_content.endswith(suffix)
    assert rendered["context_sha256"] == quote_sha256(context)
    assert rendered["context_token_proxy"] == count_tokens(context)
    payload = canonical_json_bytes({"messages": messages})
    assert rendered["provider_payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert rendered["provider_payload_utf8_bytes"] == len(payload)


def test_same_id_with_different_bytes_fails_closed() -> None:
    first = _row("collision", "source", "user", "First bytes.")
    second = _row("collision", "source", "user", "Different bytes.")
    with pytest.raises(UserSpinePromptError, match="different row bytes"):
        render_user_spine_prompt("[Dated] What happened?", [first, second])


def test_budget_rejection_never_truncates_or_drops_evidence() -> None:
    huge = _row("huge", "source", "user", " ".join(["token"] * 9_000))
    with pytest.raises(UserSpinePromptError, match="exceeds 8000"):
        render_user_spine_prompt("[Dated] What happened?", [huge])
