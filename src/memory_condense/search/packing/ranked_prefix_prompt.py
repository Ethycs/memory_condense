"""Deterministic binary search for a budgeted ranked prompt prefix.

The packer is intentionally renderer-agnostic.  Callers retain ownership of
the exact provider prompt format and token counters while this module owns the
selection invariant: the result is always the longest fitting prefix of the
input ranking.  Under the required non-decreasing prefix-cost contract, that
is byte-for-byte equivalent to rendering prefixes linearly and stopping at the
first rejection.

The complete prefix is tried first because it is the common case.  When it
does not fit, a binary search reduces prompt renders and token counts from
linear in the candidate count to logarithmic.  Context rejection short-circuits
prompt rendering.  Every actual callback invocation is reported in a stable
audit so a caller can measure that saving without timing-dependent metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


ItemT = TypeVar("ItemT")
PromptT = TypeVar("PromptT")

PACKER_ID = "deterministic-binary-ranked-prefix-prompt-v1"
AUDIT_FORMAT = "memory-condense-ranked-prefix-prompt-pack-audit-v1"


class PrefixCostMonotonicityError(ValueError):
    """Raised when observed prefix costs contradict binary-search safety."""


class NoFeasiblePrefixError(ValueError):
    """Raised when even the rendered empty prefix exceeds a configured cap."""


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class PrefixProbeAudit:
    """One newly evaluated prefix, in callback invocation order."""

    prefix_count: int
    context_token_count: int
    prompt_token_count: int | None
    prompt_workspace_token_count: int | None
    context_fits: bool
    prompt_workspace_fits: bool | None
    fits: bool

    def projection(self) -> dict[str, int | bool | None]:
        """Return a JSON-compatible deterministic projection."""

        return {
            "prefix_count": self.prefix_count,
            "context_token_count": self.context_token_count,
            "prompt_token_count": self.prompt_token_count,
            "prompt_workspace_token_count": self.prompt_workspace_token_count,
            "context_fits": self.context_fits,
            "prompt_workspace_fits": self.prompt_workspace_fits,
            "fits": self.fits,
        }


@dataclass(frozen=True, slots=True)
class RankedPrefixPackingAudit:
    """Stable callback accounting and boundary proof for one pack operation."""

    audit_format: str
    packer_id: str
    candidate_count: int
    packed_count: int
    dropped_count: int
    max_context_tokens: int
    max_prompt_tokens: int
    output_token_reserve: int
    context_count_call_count: int
    prompt_render_call_count: int
    prompt_count_call_count: int
    binary_search_iteration_count: int
    complete_prefix_fast_path: bool
    sampled_monotonicity_validated: bool
    maximal_prefix_boundary_validated: bool
    probes: tuple[PrefixProbeAudit, ...]

    def projection(self) -> dict[str, Any]:
        """Return a JSON-compatible deterministic projection."""

        return {
            "audit_format": self.audit_format,
            "packer_id": self.packer_id,
            "candidate_count": self.candidate_count,
            "packed_count": self.packed_count,
            "dropped_count": self.dropped_count,
            "max_context_tokens": self.max_context_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "output_token_reserve": self.output_token_reserve,
            "context_count_call_count": self.context_count_call_count,
            "prompt_render_call_count": self.prompt_render_call_count,
            "prompt_count_call_count": self.prompt_count_call_count,
            "binary_search_iteration_count": self.binary_search_iteration_count,
            "complete_prefix_fast_path": self.complete_prefix_fast_path,
            "sampled_monotonicity_validated": (
                self.sampled_monotonicity_validated
            ),
            "maximal_prefix_boundary_validated": (
                self.maximal_prefix_boundary_validated
            ),
            "probed_prefix_counts": [row.prefix_count for row in self.probes],
            "probes": [row.projection() for row in self.probes],
        }


@dataclass(frozen=True, slots=True)
class RankedPrefixPromptPack(Generic[ItemT, PromptT]):
    """The exact accepted prefix, rendered prompt, costs, and call audit."""

    packed_items: tuple[ItemT, ...]
    dropped_items: tuple[ItemT, ...]
    rendered_prompt: PromptT
    context_token_count: int
    prompt_token_count: int
    prompt_workspace_token_count: int
    audit: RankedPrefixPackingAudit

    @property
    def packed_count(self) -> int:
        return len(self.packed_items)


@dataclass(frozen=True, slots=True)
class _PrefixEvaluation(Generic[PromptT]):
    probe: PrefixProbeAudit
    rendered_prompt: PromptT | None


def _raise_nonmonotonic(
    *,
    smaller: PrefixProbeAudit,
    larger: PrefixProbeAudit,
    metric: str,
) -> None:
    raise PrefixCostMonotonicityError(
        f"{metric} decreased from prefix {smaller.prefix_count} to "
        f"prefix {larger.prefix_count}"
    )


def _validate_observed_monotonicity(
    probe: PrefixProbeAudit,
    observed: Sequence[PrefixProbeAudit],
) -> None:
    """Reject every monotonicity contradiction visible in sampled prefixes.

    Binary search cannot prove an arbitrary opaque callback monotone without
    evaluating every prefix.  Its explicit contract is therefore that context
    and prompt token costs are non-decreasing as ranked items are appended.
    This check validates every pair the logarithmic search actually observes;
    the terminal accepted/rejected neighbors provide the maximality boundary.
    """

    for other in observed:
        if other.prefix_count == probe.prefix_count:
            raise RuntimeError("a prefix was evaluated more than once")
        smaller, larger = (
            (other, probe)
            if other.prefix_count < probe.prefix_count
            else (probe, other)
        )
        if smaller.context_token_count > larger.context_token_count:
            _raise_nonmonotonic(
                smaller=smaller,
                larger=larger,
                metric="context token count",
            )
        if (
            smaller.prompt_token_count is not None
            and larger.prompt_token_count is not None
            and smaller.prompt_token_count > larger.prompt_token_count
        ):
            _raise_nonmonotonic(
                smaller=smaller,
                larger=larger,
                metric="prompt token count",
            )
        if not smaller.fits and larger.fits:
            _raise_nonmonotonic(
                smaller=smaller,
                larger=larger,
                metric="prefix fit predicate",
            )


def pack_ranked_prefix_prompt(
    ranked_items: Sequence[ItemT],
    *,
    count_context_tokens: Callable[[Sequence[ItemT]], int],
    render_prompt: Callable[[Sequence[ItemT]], PromptT],
    count_prompt_tokens: Callable[[PromptT], int],
    max_context_tokens: int,
    max_prompt_tokens: int,
    output_token_reserve: int = 0,
) -> RankedPrefixPromptPack[ItemT, PromptT]:
    """Render the longest ranked prefix satisfying both independent caps.

    ``count_context_tokens`` and ``render_prompt`` receive the same immutable
    tuple prefix.  ``max_prompt_tokens`` applies to the rendered prompt plus
    ``output_token_reserve``; the context cap is checked separately.  Callback
    costs must be deterministic, non-negative integers and non-decreasing as
    the prefix grows.  Observed violations fail closed instead of returning a
    potentially non-maximal prefix.

    The final ``rendered_prompt`` is the exact object produced while testing
    the winning prefix.  It is not reconstructed, normalized, or serialized by
    this module, which preserves the caller's linear packer's byte semantics.
    """

    context_cap = _nonnegative_int(max_context_tokens, "max_context_tokens")
    prompt_cap = _nonnegative_int(max_prompt_tokens, "max_prompt_tokens")
    reserve = _nonnegative_int(output_token_reserve, "output_token_reserve")
    for callback, label in (
        (count_context_tokens, "count_context_tokens"),
        (render_prompt, "render_prompt"),
        (count_prompt_tokens, "count_prompt_tokens"),
    ):
        if not callable(callback):
            raise TypeError(f"{label} must be callable")

    candidates = tuple(ranked_items)
    evaluations: dict[int, _PrefixEvaluation[PromptT]] = {}
    probes: list[PrefixProbeAudit] = []
    context_count_calls = 0
    prompt_render_calls = 0
    prompt_count_calls = 0

    def evaluate(prefix_count: int) -> _PrefixEvaluation[PromptT]:
        nonlocal context_count_calls, prompt_render_calls, prompt_count_calls
        cached = evaluations.get(prefix_count)
        if cached is not None:
            return cached
        if prefix_count < 0 or prefix_count > len(candidates):
            raise RuntimeError("internal prefix count is outside candidate bounds")

        prefix = candidates[:prefix_count]
        context_tokens = _nonnegative_int(
            count_context_tokens(prefix),
            "count_context_tokens result",
        )
        context_count_calls += 1
        context_fits = context_tokens <= context_cap
        rendered: PromptT | None = None
        prompt_tokens: int | None = None
        workspace_tokens: int | None = None
        prompt_fits: bool | None = None
        if context_fits:
            rendered = render_prompt(prefix)
            prompt_render_calls += 1
            prompt_tokens = _nonnegative_int(
                count_prompt_tokens(rendered),
                "count_prompt_tokens result",
            )
            prompt_count_calls += 1
            workspace_tokens = prompt_tokens + reserve
            prompt_fits = workspace_tokens <= prompt_cap

        probe = PrefixProbeAudit(
            prefix_count=prefix_count,
            context_token_count=context_tokens,
            prompt_token_count=prompt_tokens,
            prompt_workspace_token_count=workspace_tokens,
            context_fits=context_fits,
            prompt_workspace_fits=prompt_fits,
            fits=context_fits and prompt_fits is True,
        )
        _validate_observed_monotonicity(probe, probes)
        evaluation = _PrefixEvaluation(probe=probe, rendered_prompt=rendered)
        evaluations[prefix_count] = evaluation
        probes.append(probe)
        return evaluation

    candidate_count = len(candidates)
    complete = evaluate(candidate_count)
    iterations = 0
    complete_fast_path = complete.probe.fits
    if complete_fast_path:
        accepted_count = candidate_count
        accepted = complete
    else:
        empty = evaluate(0)
        if not empty.probe.fits:
            raise NoFeasiblePrefixError(
                "the empty ranked prefix exceeds a configured prompt cap"
            )
        lower = 0
        upper = candidate_count
        while upper - lower > 1:
            iterations += 1
            middle = (lower + upper) // 2
            proposal = evaluate(middle)
            if proposal.probe.fits:
                lower = middle
            else:
                upper = middle
        accepted_count = lower
        accepted = evaluations[lower]
        rejected_neighbor = evaluations[upper]
        if accepted.probe.fits is not True or rejected_neighbor.probe.fits:
            raise RuntimeError("binary search did not establish a fit boundary")

    if accepted.rendered_prompt is None:
        raise RuntimeError("accepted prefix has no rendered prompt")
    if accepted.probe.prompt_token_count is None:
        raise RuntimeError("accepted prefix has no prompt token count")
    if accepted.probe.prompt_workspace_token_count is None:
        raise RuntimeError("accepted prefix has no prompt workspace token count")

    audit = RankedPrefixPackingAudit(
        audit_format=AUDIT_FORMAT,
        packer_id=PACKER_ID,
        candidate_count=candidate_count,
        packed_count=accepted_count,
        dropped_count=candidate_count - accepted_count,
        max_context_tokens=context_cap,
        max_prompt_tokens=prompt_cap,
        output_token_reserve=reserve,
        context_count_call_count=context_count_calls,
        prompt_render_call_count=prompt_render_calls,
        prompt_count_call_count=prompt_count_calls,
        binary_search_iteration_count=iterations,
        complete_prefix_fast_path=complete_fast_path,
        sampled_monotonicity_validated=True,
        maximal_prefix_boundary_validated=(
            accepted_count == candidate_count
            or (
                evaluations[accepted_count + 1].probe.fits is False
                and evaluations[accepted_count].probe.fits is True
            )
        ),
        probes=tuple(probes),
    )
    return RankedPrefixPromptPack(
        packed_items=candidates[:accepted_count],
        dropped_items=candidates[accepted_count:],
        rendered_prompt=accepted.rendered_prompt,
        context_token_count=accepted.probe.context_token_count,
        prompt_token_count=accepted.probe.prompt_token_count,
        prompt_workspace_token_count=(
            accepted.probe.prompt_workspace_token_count
        ),
        audit=audit,
    )


__all__ = [
    "AUDIT_FORMAT",
    "PACKER_ID",
    "NoFeasiblePrefixError",
    "PrefixCostMonotonicityError",
    "PrefixProbeAudit",
    "RankedPrefixPackingAudit",
    "RankedPrefixPromptPack",
    "pack_ranked_prefix_prompt",
]
