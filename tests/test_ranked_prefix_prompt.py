from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from memory_condense.search.packing.ranked_prefix_prompt import (
    AUDIT_FORMAT,
    PACKER_ID,
    NoFeasiblePrefixError,
    PrefixCostMonotonicityError,
    pack_ranked_prefix_prompt,
)


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    context_cost: int


@dataclass(slots=True)
class Harness:
    context_calls: int = 0
    render_calls: int = 0
    prompt_calls: int = 0

    def count_context(self, items: Sequence[Item]) -> int:
        self.context_calls += 1
        return sum(item.context_cost for item in items)

    def render(self, items: Sequence[Item]) -> bytes:
        self.render_calls += 1
        body = b"|".join(item.item_id.encode("utf-8") for item in items)
        return b"question\n" + body

    def count_prompt(self, prompt: bytes) -> int:
        self.prompt_calls += 1
        return len(prompt)


def _linear_reference(
    items: Sequence[Item],
    *,
    max_context_tokens: int,
    max_prompt_tokens: int,
    output_token_reserve: int,
) -> tuple[tuple[Item, ...], bytes, int, int]:
    """Obvious linear oracle: retain the last fitting ranked prefix."""

    accepted: tuple[Item, ...] | None = None
    accepted_prompt: bytes | None = None
    accepted_context = 0
    accepted_prompt_tokens = 0
    for end in range(len(items) + 1):
        prefix = tuple(items[:end])
        context_tokens = sum(item.context_cost for item in prefix)
        prompt = b"question\n" + b"|".join(
            item.item_id.encode("utf-8") for item in prefix
        )
        prompt_tokens = len(prompt)
        if (
            context_tokens > max_context_tokens
            or prompt_tokens + output_token_reserve > max_prompt_tokens
        ):
            break
        accepted = prefix
        accepted_prompt = prompt
        accepted_context = context_tokens
        accepted_prompt_tokens = prompt_tokens
    if accepted is None or accepted_prompt is None:
        raise NoFeasiblePrefixError
    return (
        accepted,
        accepted_prompt,
        accepted_context,
        accepted_prompt_tokens,
    )


def _pack(
    items: Sequence[Item],
    *,
    context_cap: int,
    prompt_cap: int,
    reserve: int = 0,
):
    harness = Harness()
    result = pack_ranked_prefix_prompt(
        items,
        count_context_tokens=harness.count_context,
        render_prompt=harness.render,
        count_prompt_tokens=harness.count_prompt,
        max_context_tokens=context_cap,
        max_prompt_tokens=prompt_cap,
        output_token_reserve=reserve,
    )
    reference = _linear_reference(
        items,
        max_context_tokens=context_cap,
        max_prompt_tokens=prompt_cap,
        output_token_reserve=reserve,
    )
    assert result.packed_items == reference[0]
    assert result.rendered_prompt == reference[1]
    assert result.context_token_count == reference[2]
    assert result.prompt_token_count == reference[3]
    assert result.prompt_workspace_token_count == reference[3] + reserve
    assert result.packed_items + result.dropped_items == tuple(items)
    assert result.audit.packed_count == len(reference[0])
    assert result.audit.maximal_prefix_boundary_validated is True
    assert result.audit.sampled_monotonicity_validated is True
    assert harness.context_calls == result.audit.context_count_call_count
    assert harness.render_calls == result.audit.prompt_render_call_count
    assert harness.prompt_calls == result.audit.prompt_count_call_count
    return result


def test_all_fit_uses_one_render_and_is_byte_equivalent_to_linear() -> None:
    items = tuple(Item(chr(ord("a") + index), 2) for index in range(8))

    result = _pack(items, context_cap=100, prompt_cap=100)

    assert result.packed_items == items
    assert result.dropped_items == ()
    assert result.audit.complete_prefix_fast_path is True
    assert result.audit.context_count_call_count == 1
    assert result.audit.prompt_render_call_count == 1
    assert result.audit.prompt_count_call_count == 1
    assert result.audit.binary_search_iteration_count == 0
    assert [row.prefix_count for row in result.audit.probes] == [8]
    projection = result.audit.projection()
    assert projection["audit_format"] == AUDIT_FORMAT
    assert projection["packer_id"] == PACKER_ID
    assert projection["probed_prefix_counts"] == [8]


def test_empty_input_renders_the_empty_prefix_once() -> None:
    result = _pack((), context_cap=0, prompt_cap=len(b"question\n"))

    assert result.packed_items == ()
    assert result.dropped_items == ()
    assert result.rendered_prompt == b"question\n"
    assert result.context_token_count == 0
    assert result.audit.complete_prefix_fast_path is True
    assert result.audit.context_count_call_count == 1
    assert result.audit.prompt_render_call_count == 1
    assert result.audit.prompt_count_call_count == 1
    assert [row.prefix_count for row in result.audit.probes] == [0]


def test_context_bound_short_circuits_rejected_prompt_renders() -> None:
    items = tuple(Item(str(index), 2) for index in range(8))

    result = _pack(items, context_cap=7, prompt_cap=1_000)

    assert tuple(item.item_id for item in result.packed_items) == ("0", "1", "2")
    assert tuple(item.item_id for item in result.dropped_items) == (
        "3",
        "4",
        "5",
        "6",
        "7",
    )
    assert result.audit.complete_prefix_fast_path is False
    assert result.audit.context_count_call_count == 5
    assert result.audit.prompt_render_call_count == 3
    assert result.audit.prompt_count_call_count == 3
    assert result.audit.binary_search_iteration_count == 3
    assert [row.prefix_count for row in result.audit.probes] == [8, 0, 4, 2, 3]
    assert all(
        row.prompt_token_count is None
        for row in result.audit.probes
        if not row.context_fits
    )


def test_prompt_plus_reserve_bound_uses_longest_fitting_prefix() -> None:
    items = tuple(Item(str(index), 1) for index in range(8))
    # Four one-byte IDs render to 16 bytes; the three-token response reserve
    # makes that exactly fit 19.  A fifth ID renders to 18 + 3 and is rejected.
    result = _pack(items, context_cap=1_000, prompt_cap=19, reserve=3)

    assert tuple(item.item_id for item in result.packed_items) == (
        "0",
        "1",
        "2",
        "3",
    )
    assert result.prompt_token_count == 16
    assert result.prompt_workspace_token_count == 19
    assert result.audit.context_count_call_count == 5
    assert result.audit.prompt_render_call_count == 5
    assert result.audit.prompt_count_call_count == 5
    assert [row.prefix_count for row in result.audit.probes] == [8, 0, 4, 6, 5]


def test_exact_context_and_prompt_bounds_are_inclusive() -> None:
    items = (Item("alpha", 4), Item("beta", 5))
    exact_prompt_tokens = len(b"question\nalpha|beta")

    result = _pack(
        items,
        context_cap=9,
        prompt_cap=exact_prompt_tokens + 7,
        reserve=7,
    )

    assert result.packed_items == items
    assert result.context_token_count == 9
    assert result.prompt_workspace_token_count == exact_prompt_tokens + 7
    assert result.audit.complete_prefix_fast_path is True
    assert result.audit.prompt_render_call_count == 1


def test_observed_nonmonotone_context_cost_fails_closed() -> None:
    items = tuple(Item(str(index), 0) for index in range(4))
    costs = {0: 0, 1: 9, 2: 8, 4: 10}

    with pytest.raises(
        PrefixCostMonotonicityError,
        match="context token count decreased from prefix 1 to prefix 2",
    ):
        pack_ranked_prefix_prompt(
            items,
            count_context_tokens=lambda prefix: costs[len(prefix)],
            render_prompt=lambda prefix: tuple(prefix),
            count_prompt_tokens=lambda _prompt: 0,
            max_context_tokens=5,
            max_prompt_tokens=5,
        )


def test_no_feasible_empty_prompt_fails_closed() -> None:
    with pytest.raises(NoFeasiblePrefixError, match="empty ranked prefix"):
        pack_ranked_prefix_prompt(
            (Item("a", 1),),
            count_context_tokens=lambda prefix: len(prefix),
            render_prompt=lambda prefix: tuple(prefix),
            count_prompt_tokens=lambda _prompt: 4,
            max_context_tokens=10,
            max_prompt_tokens=3,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("max_context_tokens", -1, ValueError),
        ("max_context_tokens", True, TypeError),
        ("max_prompt_tokens", -1, ValueError),
        ("max_prompt_tokens", False, TypeError),
        ("output_token_reserve", -1, ValueError),
        ("output_token_reserve", 1.5, TypeError),
    ],
)
def test_rejects_invalid_caps(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    controls: dict[str, object] = {
        "max_context_tokens": 10,
        "max_prompt_tokens": 10,
        "output_token_reserve": 0,
    }
    controls[keyword] = value

    with pytest.raises(error):
        pack_ranked_prefix_prompt(
            (),
            count_context_tokens=lambda _prefix: 0,
            render_prompt=lambda prefix: tuple(prefix),
            count_prompt_tokens=lambda _prompt: 0,
            **controls,  # type: ignore[arg-type]
        )
