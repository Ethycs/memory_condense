"""Causal, compact learning for stepwise attention over external memory.

The transformer may propose a small set of memory transitions at turn ``t``.
Only after turn ``t + 1`` arrives does this module score those proposals.  It
retains decayed scalar utility statistics by role/head and role/edge; token
keys, values, residual sequences, text, and pending CAV vectors are never part
of the serializable policy state.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Sequence

from memory_condense.domain.decay import decay_factor


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _role_key(from_role: str, to_role: str) -> str:
    left = from_role.strip().lower()
    right = to_role.strip().lower()
    if not left or not right:
        raise ValueError("transition roles must be non-empty")
    return f"{left}->{right}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("predicted and observed CAV deltas need equal dimensions")
    if not left:
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


@dataclass(frozen=True, slots=True)
class TransitionCandidate:
    """One external-memory destination proposed by transient QK/OV work.

    ``head_attention`` is the QK addressing mass for each inspected head.
    ``head_cav_deltas`` is optional projected OV, one fixed-width CAV delta per
    head.  These vectors live only in a one-turn :class:`TransitionDecision`.
    """

    destination_id: str
    base_score: float
    head_attention: tuple[float, ...]
    head_cav_deltas: tuple[tuple[float, ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.destination_id:
            raise ValueError("destination_id must be non-empty")
        _finite(self.base_score, "base_score")
        if not self.head_attention:
            raise ValueError("head_attention must be non-empty")
        for mass in self.head_attention:
            if _finite(mass, "head attention") < 0.0:
                raise ValueError("head attention must be non-negative")
        if self.head_cav_deltas and len(self.head_cav_deltas) != len(
            self.head_attention
        ):
            raise ValueError("head_cav_deltas must align with head_attention")
        for delta in self.head_cav_deltas:
            for value in delta:
                _finite(value, "projected OV CAV delta")


@dataclass(frozen=True, slots=True)
class ScoredTransition:
    candidate: TransitionCandidate
    score: float
    head_gates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    """Ephemeral prediction made before the following turn is visible."""

    source_id: str
    from_role: str
    expected_next_role: str
    source_cav: tuple[float, ...]
    turn: int
    selected: tuple[ScoredTransition, ...]


@dataclass(frozen=True, slots=True)
class TransitionFeedback:
    actual_destination_id: str | None
    target_was_selected: bool


@dataclass(slots=True)
class _DecayedStatistic:
    reward_sum: float = 0.0
    mass: float = 0.0
    observations: int = 0
    last_turn: int = 0

    def _factor(self, turn: int, half_life: float) -> float:
        if turn < self.last_turn:
            raise ValueError("turn must be monotonic for each statistic")
        return decay_factor(self.last_turn, turn, half_life)

    def value(self, turn: int, half_life: float, prior_mass: float) -> float:
        factor = self._factor(turn, half_life)
        return self.reward_sum * factor / (prior_mass + self.mass * factor)

    def effective_mass(self, turn: int, half_life: float) -> float:
        return self.mass * self._factor(turn, half_life)

    def update(self, reward: float, turn: int, half_life: float) -> None:
        factor = self._factor(turn, half_life)
        self.reward_sum = self.reward_sum * factor + reward
        self.mass = self.mass * factor + 1.0
        self.observations += 1
        self.last_turn = turn

    def as_dict(self) -> dict[str, float | int]:
        return {spec.name: getattr(self, spec.name) for spec in fields(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _DecayedStatistic:
        return cls(
            **{
                spec.name: _STATISTIC_PARSERS[spec.name](
                    value.get(spec.name, spec.default), spec.name
                )
                for spec in fields(cls)
            }
        )


#: One parser per statistic field. Both directions of the round trip are
#: driven by ``fields()``, so a new field that forgets its parser fails loudly
#: here rather than silently dropping out of persisted snapshots.
_STATISTIC_PARSERS: dict[str, Callable[[Any, str], float | int]] = {
    "reward_sum": _finite,
    "mass": lambda value, name: max(0.0, _finite(value, name)),
    "observations": lambda value, _name: max(0, int(value)),
    "last_turn": lambda value, _name: max(0, int(value)),
}


class CausalTransitionPolicy:
    """Learn role-specific head gates and sparse edge utility one turn late.

    ``propose`` never updates the policy. ``observe`` consumes that frozen
    decision after the next turn exists. This API makes future-turn leakage
    harder during offline replay and matches the live system's event order.
    """

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        *,
        half_life_turns: float = 128.0,
        prior_mass: float = 1.0,
        transition_weight: float = 0.25,
        velocity_weight: float = 0.0,
        gate_temperature: float = 1.0,
        max_edge_statistics: int = 10_000,
    ) -> None:
        if half_life_turns <= 0.0:
            raise ValueError("half_life_turns must be positive")
        if prior_mass <= 0.0:
            raise ValueError("prior_mass must be positive")
        if transition_weight < 0.0 or velocity_weight < 0.0 or gate_temperature < 0.0:
            raise ValueError("transition and gate weights must be non-negative")
        if max_edge_statistics < 1:
            raise ValueError("max_edge_statistics must be positive")
        self.half_life_turns = float(half_life_turns)
        self.prior_mass = float(prior_mass)
        self.transition_weight = float(transition_weight)
        self.velocity_weight = float(velocity_weight)
        self.gate_temperature = float(gate_temperature)
        self.max_edge_statistics = int(max_edge_statistics)
        self._heads: dict[tuple[str, int], _DecayedStatistic] = {}
        self._edges: dict[tuple[str, str, str], _DecayedStatistic] = {}

    def _head_value(self, role: str, head: int, turn: int) -> float:
        statistic = self._heads.get((role, head))
        if statistic is None:
            return 0.0
        return statistic.value(turn, self.half_life_turns, self.prior_mass)

    def _edge_value(
        self, role: str, source_id: str, destination_id: str, turn: int
    ) -> float:
        statistic = self._edges.get((role, source_id, destination_id))
        if statistic is None:
            return 0.0
        return statistic.value(turn, self.half_life_turns, self.prior_mass)

    def head_gates(
        self, from_role: str, to_role: str, *, head_count: int, turn: int
    ) -> tuple[float, ...]:
        """Return positive multiplicative gates for the next QK distribution."""
        if head_count < 1:
            raise ValueError("head_count must be positive")
        role = _role_key(from_role, to_role)
        return tuple(
            math.exp(
                self.gate_temperature * self._head_value(role, head, int(turn))
            )
            for head in range(head_count)
        )

    def propose(
        self,
        *,
        source_id: str,
        from_role: str,
        expected_next_role: str,
        source_cav: Sequence[float],
        cav_velocity: Sequence[float] | None = None,
        candidates: Sequence[TransitionCandidate],
        turn: int,
        top_k: int = 1,
    ) -> TransitionDecision:
        """Rank with history through ``turn`` and return an ephemeral decision."""
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if turn < 0:
            raise ValueError("turn must be non-negative")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        bounded = tuple(candidates)
        if not bounded:
            raise ValueError("at least one transition candidate is required")
        head_count = len(bounded[0].head_attention)
        if any(len(candidate.head_attention) != head_count for candidate in bounded):
            raise ValueError("all candidates must expose the same heads")
        cav = tuple(_finite(value, "source CAV") for value in source_cav)
        velocity = (
            tuple(_finite(value, "CAV velocity") for value in cav_velocity)
            if cav_velocity is not None
            else ()
        )
        if velocity and len(velocity) != len(cav):
            raise ValueError("CAV velocity and source CAV need equal dimensions")
        role = _role_key(from_role, expected_next_role)
        gates = self.head_gates(
            from_role, expected_next_role, head_count=head_count, turn=turn
        )

        scored: list[ScoredTransition] = []
        for candidate in bounded:
            attention_total = sum(candidate.head_attention)
            head_utility = (
                sum(
                    mass * self._head_value(role, head, turn)
                    for head, mass in enumerate(candidate.head_attention)
                )
                / attention_total
                if attention_total > 0.0
                else 0.0
            )
            edge_utility = self._edge_value(
                role, source_id, candidate.destination_id, turn
            )
            velocity_utility = 0.0
            if velocity and candidate.head_cav_deltas and attention_total > 0.0:
                velocity_utility = sum(
                    mass * _cosine(candidate.head_cav_deltas[head], velocity)
                    for head, mass in enumerate(candidate.head_attention)
                ) / attention_total
            scored.append(
                ScoredTransition(
                    candidate=candidate,
                    score=(
                        candidate.base_score
                        + self.transition_weight * (head_utility + edge_utility)
                        + self.velocity_weight * velocity_utility
                    ),
                    head_gates=gates,
                )
            )
        scored.sort(
            key=lambda item: (item.score, item.candidate.destination_id), reverse=True
        )
        return TransitionDecision(
            source_id=source_id,
            from_role=from_role,
            expected_next_role=expected_next_role,
            source_cav=cav,
            turn=int(turn),
            selected=tuple(scored[: min(top_k, len(scored))]),
        )

    def observe(
        self,
        decision: TransitionDecision,
        *,
        actual_destination_id: str | None,
        actual_next_role: str,
        next_cav: Sequence[float],
        usefulness: float = 1.0,
        turn: int | None = None,
    ) -> TransitionFeedback:
        """Reveal ``t + 1`` and update only scalar head/edge statistics."""
        if actual_destination_id is not None and not actual_destination_id:
            raise ValueError("actual_destination_id must be non-empty when supplied")
        observation_turn = decision.turn + 1 if turn is None else int(turn)
        if observation_turn <= decision.turn:
            raise ValueError("feedback must arrive after the decision turn")
        utility = _finite(usefulness, "usefulness")
        next_values = tuple(_finite(value, "next CAV") for value in next_cav)
        if len(next_values) != len(decision.source_cav):
            raise ValueError("source and next CAVs need equal dimensions")
        observed_delta = tuple(
            right - left
            for left, right in zip(decision.source_cav, next_values, strict=True)
        )
        role = _role_key(decision.from_role, actual_next_role)
        target_was_selected = False

        for scored in decision.selected:
            candidate = scored.candidate
            target = (
                actual_destination_id is not None
                and candidate.destination_id == actual_destination_id
            )
            target_was_selected = target_was_selected or target
            head_rewards: list[float] = []
            for head, attention in enumerate(candidate.head_attention):
                if candidate.head_cav_deltas:
                    signed_alignment = _cosine(
                        candidate.head_cav_deltas[head], observed_delta
                    )
                else:
                    if actual_destination_id is None:
                        raise ValueError(
                            "CAV-only feedback requires projected OV CAV deltas"
                        )
                    signed_alignment = 1.0
                if actual_destination_id is None:
                    # Self-supervised CAV-direction learning remains useful
                    # even when the next turn addresses an exact chunk that
                    # was absent from t's bounded candidate set.
                    reward = utility * float(attention) * signed_alignment
                else:
                    # Map cosine to [0, 1]. A wrong but anti-aligned proposal
                    # is ignored, not accidentally rewarded by two negatives.
                    alignment = (1.0 + signed_alignment) / 2.0
                    reward = utility * float(attention) * alignment
                    if not target:
                        reward = -reward
                statistic = self._heads.setdefault(
                    (role, head), _DecayedStatistic(last_turn=observation_turn)
                )
                statistic.update(reward, observation_turn, self.half_life_turns)
                head_rewards.append(reward)

            edge = self._edges.setdefault(
                (role, decision.source_id, candidate.destination_id),
                _DecayedStatistic(last_turn=observation_turn),
            )
            edge.update(
                sum(head_rewards) / len(head_rewards),
                observation_turn,
                self.half_life_turns,
            )

        self._prune_edges(observation_turn)
        return TransitionFeedback(
            actual_destination_id=actual_destination_id,
            target_was_selected=target_was_selected,
        )

    def _prune_edges(self, turn: int) -> None:
        overflow = len(self._edges) - self.max_edge_statistics
        if overflow <= 0:
            return
        weakest = sorted(
            self._edges,
            key=lambda key: (
                self._edges[key].effective_mass(turn, self.half_life_turns),
                abs(
                    self._edges[key].value(
                        turn, self.half_life_turns, self.prior_mass
                    )
                ),
                key,
            ),
        )
        for key in weakest[:overflow]:
            del self._edges[key]

    @classmethod
    def _config_names(cls) -> tuple[str, ...]:
        """Every constructor knob, in declaration order.

        Each is stored under its own name, so the snapshot's ``config`` block
        and the ``cls(**config)`` restore stay in step with the signature
        instead of with two hand-maintained lists.
        """

        return tuple(
            name
            for name, parameter in inspect.signature(cls).parameters.items()
            if parameter.kind is parameter.KEYWORD_ONLY
        )

    def snapshot(self) -> dict[str, Any]:
        """Serialize compact learned statistics, never pending turn state."""
        return {
            "version": self.SNAPSHOT_VERSION,
            "config": {
                name: getattr(self, name) for name in self._config_names()
            },
            "heads": [
                {"role": role, "head": head, **statistic.as_dict()}
                for (role, head), statistic in sorted(self._heads.items())
            ],
            "edges": [
                {
                    "role": role,
                    "source_id": source_id,
                    "destination_id": destination_id,
                    **statistic.as_dict(),
                }
                for (role, source_id, destination_id), statistic in sorted(
                    self._edges.items()
                )
            ],
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> CausalTransitionPolicy:
        if int(payload.get("version", 0)) != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported transition-policy snapshot version")
        # Retired knobs may still sit in snapshots written by an older build;
        # replaying one must continue to work at this snapshot version.
        config = dict(payload.get("config", {}))
        policy = cls(
            **{
                name: config[name]
                for name in cls._config_names()
                if name in config
            }
        )
        for raw in payload.get("heads", []):
            role = str(raw["role"])
            head = int(raw["head"])
            policy._heads[(role, head)] = _DecayedStatistic.from_dict(raw)
        for raw in payload.get("edges", []):
            key = (
                str(raw["role"]),
                str(raw["source_id"]),
                str(raw["destination_id"]),
            )
            policy._edges[key] = _DecayedStatistic.from_dict(raw)
        if len(policy._edges) > policy.max_edge_statistics:
            latest_turn = max(
                (statistic.last_turn for statistic in policy._edges.values()),
                default=0,
            )
            policy._prune_edges(latest_turn)
        return policy
