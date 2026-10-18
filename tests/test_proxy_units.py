import json

import pytest

from memory_condense.interfaces import proxy_units as U


# -- fixtures: conversations -----------------------------------------------------------

def _tool_turn(i: int, big: int = 0) -> list[dict]:
    return [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm", "signature": f"sig{i}"},
            {"type": "tool_use", "id": f"tu{i}", "name": "Read", "input": {"path": f"f{i}.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"tu{i}", "content": "x" * big},
        ]},
    ]


def anthropic_messages(n_tools: int = 6, big: int = 2000) -> list[dict]:
    msgs = [{"role": "user", "content": "Fix the login bug"}]
    for i in range(n_tools):
        msgs += _tool_turn(i, big)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"step {i} done"}]})
        msgs.append({"role": "user", "content": f"continue {i}"})
    msgs.append({"role": "user", "content": "now what?"})
    return msgs


def responses_input(n_tools: int = 6, big: int = 2000) -> list[dict]:
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the login bug"}]},
    ]
    for i in range(n_tools):
        items += [
            {"type": "reasoning", "id": f"rs{i}", "summary": [], "encrypted_content": "..."},
            {"type": "function_call", "call_id": f"c{i}", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "y" * big},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"step {i}"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"continue {i}"}]},
        ]
    return items


def chat_messages(n_tools: int = 3, big: int = 200) -> list[dict]:
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the login bug"},
    ]
    for i in range(n_tools):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call{i}a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": f"call{i}b", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": f"call{i}a", "content": "x" * big},
            {"role": "tool", "tool_call_id": f"call{i}b", "content": "y" * big},
            {"role": "assistant", "content": f"step {i} done"},
            {"role": "user", "content": [{"type": "text", "text": f"continue {i}"}]},
        ]
    return msgs


# -- grouping: anthropic / responses (ported from test_context_proxy) ----------------------

def test_anthropic_grouping_keeps_tool_pairs_and_thinking_together():
    units = U.group_anthropic(anthropic_messages(2))
    kinds = [u.kind for u in units]
    assert kinds == ["user", "exchange", "assistant", "user", "exchange", "assistant", "user", "user"]
    ex = units[1]
    assert ex.items[0]["role"] == "assistant" and ex.items[1]["role"] == "user"
    assert ex.tool_names == ["Read"]
    assert units[0].user_text == "Fix the login bug"


def test_responses_grouping_attaches_reasoning_and_outputs():
    units = U.group_responses(responses_input(2))
    assert [u.kind for u in units][:5] == ["user", "user", "exchange", "assistant", "user"]
    ex = units[2]
    assert [i["type"] for i in ex.items] == ["reasoning", "function_call", "function_call_output"]
    assert U.head_length(units) == 2


# -- grouping: chat completions ----------------------------------------------------------

def test_chat_grouping_joins_tool_calls_with_their_tool_messages():
    units = U.group_chat(chat_messages(2))
    kinds = [u.kind for u in units]
    assert kinds == ["context", "user", "exchange", "assistant", "user", "exchange", "assistant", "user"]
    ex = units[2]
    assert [m["role"] for m in ex.items] == ["assistant", "tool", "tool"]
    assert ex.tool_names == ["read_file", "grep"]
    assert units[1].user_text == "Fix the login bug"
    assert units[4].user_text == "continue 0"  # list-of-parts content
    assert U.head_length(units) == 2
    assert U.tool_histogram(units) == {"read_file": 2, "grep": 2}


def test_chat_orphan_tool_message_does_not_join_the_previous_unit():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "tool_call_id": "ghost", "content": "z"},
    ]
    units = U.group_chat(msgs)
    assert [u.kind for u in units] == ["user", "assistant", "other"]


# -- check: pairing ------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,conv", [
    ("anthropic", anthropic_messages(3)),
    ("chat", chat_messages(3)),
    ("responses", responses_input(3)),
])
def test_valid_conversations_pass_check(fmt, conv):
    assert U.check(fmt, conv) == []


@pytest.mark.parametrize("fmt,conv", [
    ("anthropic", anthropic_messages(3)),
    ("chat", chat_messages(3)),
    ("responses", responses_input(3)),
])
def test_dropping_whole_units_preserves_pairing_but_splitting_one_does_not(fmt, conv):
    units = U.group(fmt, conv)
    head = U.head_length(units)
    # Drop the middle run of units wholesale, keep head and tail: still valid.
    kept = units[:head] + units[-2:]
    out = [item for u in kept for item in u.items]
    assert len(out) < len(conv)
    assert U.check(fmt, out) == []
    # Cut inside an exchange unit (keep only its trailing result, lose its call): invalid.
    ex = next(u for u in units if u.kind == "exchange")
    split = [item for u in units for item in (u.items[-1:] if u is ex else u.items)]
    assert U.check(fmt, split) != []


def test_chat_check_rejects_orphan_tool_message():
    msgs = chat_messages(1)
    del msgs[2]  # the assistant tool_calls message
    problems = U.check_chat(msgs)
    assert len(problems) == 2
    assert all("has no preceding tool_calls entry" in p for p in problems)


def test_chat_check_rejects_tool_message_after_a_user_turn():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "user", "content": "actually wait"},
        {"role": "tool", "tool_call_id": "c1", "content": "late"},
    ]
    assert U.check_chat(msgs) == ["message 3: tool message c1 has no preceding tool_calls entry"]


def test_chat_check_rejects_bad_first_message():
    assert U.check_chat([{"role": "assistant", "content": "hello"}]) == [
        "first message is 'assistant', not system/developer or user"
    ]
    assert U.check_chat([{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]) == []
    assert U.check_chat([]) == ["no messages"]


def test_format_aliases_and_unknown_formats():
    msgs = chat_messages(1)
    assert U.check("openai", msgs) == [] and U.group("openai", msgs) == U.group("chat", msgs)
    with pytest.raises(ValueError):
        U.check("sse", msgs)
    with pytest.raises(ValueError):
        U.group(None, msgs)


# -- guard_served_body -------------------------------------------------------------------

def test_guard_served_body_passes_valid_bodies_in_every_format():
    assert U.guard_served_body("anthropic", {"model": "m", "messages": anthropic_messages(2)}) == []
    assert U.guard_served_body("chat", {"model": "m", "messages": chat_messages(2)}) == []
    assert U.guard_served_body("openai", {"model": "m", "messages": chat_messages(2)}) == []
    assert U.guard_served_body("responses", {"model": "m", "input": responses_input(2)}) == []


def test_guard_served_body_catches_tool_result_that_lost_its_tool_use():
    msgs = anthropic_messages(2)
    # Simulate a window that opened on the tool_result and dropped the assistant tool_use.
    served = [msgs[0]] + msgs[2:]
    problems = U.guard_served_body("anthropic", {"model": "m", "messages": served})
    assert problems == ["message 1: tool_result tu0 has no preceding tool_use"]


def test_guard_served_body_catches_chat_and_responses_orphans():
    chat = chat_messages(1)
    del chat[2]
    assert U.guard_served_body("chat", {"messages": chat}) != []
    items = responses_input(1)
    items = [i for i in items if i.get("type") != "function_call"]
    assert U.guard_served_body("responses", {"input": items}) == [
        "item 3: function_call_output c0 has no preceding call"
    ]


def test_guard_served_body_never_blocks_what_it_does_not_understand():
    broken = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "nope"}]}]
    assert U.guard_served_body("sse", {"messages": broken}) == []
    assert U.guard_served_body(None, {"messages": broken}) == []
    # Right format, wrong key or non-list conversation: nothing to check.
    assert U.guard_served_body("anthropic", {"input": broken}) == []
    assert U.guard_served_body("responses", {"messages": broken}) == []
    assert U.guard_served_body("anthropic", {"messages": "not a list"}) == []
    assert U.guard_served_body("anthropic", json.dumps({"messages": broken})) == []
    assert U.guard_served_body("anthropic", None) == []


# -- misc ----------------------------------------------------------------------------------

def test_estimate_tokens_and_unit_tokens_are_positive_and_monotone():
    small = U.Unit(kind="user", items=[{"role": "user", "content": "hi"}])
    big = U.Unit(kind="user", items=[{"role": "user", "content": "x" * 4000}])
    assert 1 <= small.tokens < big.tokens
    assert U.estimate_tokens("") == 1
