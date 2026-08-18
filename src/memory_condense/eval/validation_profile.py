"""Named claim profiles for benchmark certification."""

from __future__ import annotations

from collections.abc import Mapping


LONGMEMEVAL_1M_95_PROFILE = "longmemeval-s-1m-100q-95-v1"
LONGMEMEVAL_ACCURACY_TARGET = 0.95
LONGMEMEVAL_MIN_QUESTIONS = 100
LONGMEMEVAL_STRESS_TOKENS = 1_000_000
LONGMEMEVAL_QUESTIONS_PER_SHARD = 10
LONGMEMEVAL_RECENT_WINDOW = 4


class ValidationClaimProfileError(ValueError):
    """A policy claims a named certification profile but does not satisfy it."""


def claimed_validation_profile(policy: Mapping[str, object]) -> str:
    """Return the declared profile, allowing an absent legacy diagnostic policy."""

    raw = policy.get("claim_profile")
    if raw is None:
        return ""
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationClaimProfileError("validation claim_profile must be text")
    profile = raw.strip()
    if profile != LONGMEMEVAL_1M_95_PROFILE:
        raise ValidationClaimProfileError(
            f"unsupported validation claim_profile: {profile!r}"
        )
    return profile


def validate_longmemeval_claim_profile(
    policy: Mapping[str, object],
    evaluation: Mapping[str, object],
    *,
    population_size: int | None = None,
) -> str:
    """Fail closed unless a policy exactly represents the advertised claim.

    The locked-population reconstruction remains authoritative.  This helper
    adds the semantic meaning that raw positive thresholds cannot express:
    95% accuracy over at least 100 exact validation questions, each retrieved
    from a one-million-token stress memory in deterministic ten-question shards.
    """

    profile = claimed_validation_profile(policy)
    if profile != LONGMEMEVAL_1M_95_PROFILE:
        raise ValidationClaimProfileError(
            "validation policy does not declare the LongMemEval 1M/95 claim profile"
        )

    checks = (
        ("accuracy_target", LONGMEMEVAL_ACCURACY_TARGET),
        ("stress_context_tokens", LONGMEMEVAL_STRESS_TOKENS),
        ("stress_questions", LONGMEMEVAL_QUESTIONS_PER_SHARD),
        ("stress_question_offset", 0),
        ("max_samples", 1),
        ("recent_window", LONGMEMEVAL_RECENT_WINDOW),
    )
    for field, expected in checks:
        value = evaluation.get(field)
        if isinstance(value, bool) or value != expected:
            raise ValidationClaimProfileError(
                f"{profile} requires evaluation.{field}={expected!r}"
            )

    minimum = evaluation.get("min_target_questions")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise ValidationClaimProfileError(
            f"{profile} requires integer evaluation.min_target_questions"
        )
    if minimum < LONGMEMEVAL_MIN_QUESTIONS:
        raise ValidationClaimProfileError(
            f"{profile} requires at least {LONGMEMEVAL_MIN_QUESTIONS} questions"
        )
    if minimum % LONGMEMEVAL_QUESTIONS_PER_SHARD:
        raise ValidationClaimProfileError(
            f"{profile} requires a population divisible into ten-question shards"
        )

    offsets = evaluation.get("sample_offsets")
    expected_offsets = list(
        range(0, minimum, LONGMEMEVAL_QUESTIONS_PER_SHARD)
    )
    if offsets != expected_offsets:
        raise ValidationClaimProfileError(
            f"{profile} requires sample_offsets={expected_offsets!r}"
        )

    if population_size is not None:
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size < LONGMEMEVAL_MIN_QUESTIONS
        ):
            raise ValidationClaimProfileError(
                f"{profile} requires an exact locked population of at least "
                f"{LONGMEMEVAL_MIN_QUESTIONS} questions"
            )
        if minimum != population_size:
            raise ValidationClaimProfileError(
                f"{profile} min_target_questions does not equal the exact locked "
                "population"
            )
    return profile
