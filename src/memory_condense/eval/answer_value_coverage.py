"""Pure lexical answer-reachability and multi-value coverage transforms."""

from __future__ import annotations

import re

from memory_condense.eval.benchmark import f1_score, normalize_answer
from memory_condense.eval.recall_models import AnswerValueCoverage

_NUMBERED_ANSWER_COMPONENT_RE = re.compile(
    r"(?<!\w)(?P<number>[1-9]\d*)[.)]\s+"
)


_ANSWER_VALUE_TOKEN_RECALL_NUMERATOR = 4


_ANSWER_VALUE_TOKEN_RECALL_DENOMINATOR = 5


_ANSWER_VALUE_MIN_FALLBACK_TOKENS = 4


def contains_answer(texts: list[str], gold: str) -> bool:
    """Whether ``gold`` appears in any of ``texts`` under SQuAD normalization.

    Containment, not equality: the context is a passage and the answer is a
    span inside it. Normalization (lowercase, strip articles and punctuation)
    is the same one the benchmark grades with, so this measures the same notion
    of "the answer is there" that F1 does.
    """
    needle = normalize_answer(gold)
    if not needle:
        return False
    return any(needle in normalize_answer(t) for t in texts)


def best_f1(texts: list[str], gold: str) -> float:
    """Highest token-F1 between the gold answer and any single context piece.

    A softer signal than containment: it still scores when the answer is
    present but reworded, which containment misses entirely.
    """
    return max((f1_score(t, gold) for t in texts), default=0.0)


def _parse_answer_value_components(
    gold: str,
    expected_count: int,
) -> tuple[list[str], str] | None:
    """Parse only list shapes whose cardinality is independently known.

    LongMemEval stores aggregate multi-answer golds as either numbered lists
    (whose items may themselves contain commas) or plain comma-separated
    lists. The benchmark's evidence-source count supplies an independent
    cardinality check. Derived numeric answers and ambiguous prose return
    ``None`` instead of being counted as misses.
    """

    if expected_count < 2 or not gold.strip():
        return None

    markers = list(_NUMBERED_ANSWER_COMPONENT_RE.finditer(gold))
    if markers:
        numbers = [int(marker.group("number")) for marker in markers]
        if len(markers) != expected_count or numbers != list(
            range(1, expected_count + 1)
        ):
            return None
        components = [
            gold[
                marker.end() : (
                    markers[index + 1].start()
                    if index + 1 < len(markers)
                    else len(gold)
                )
            ].strip(" \t\r\n,;")
            for index, marker in enumerate(markers)
        ]
        parse_kind = "numbered_list"
    else:
        components = [part.strip() for part in gold.split(",")]
        if len(components) != expected_count:
            return None
        parse_kind = "comma_list"

    normalized = [normalize_answer(component) for component in components]
    if (
        any(not component for component in normalized)
        or len(set(normalized)) != expected_count
        or any(not any(character.isalpha() for character in component) for component in components)
    ):
        return None
    return components, parse_kind


def _answer_value_component_in_excerpt(component: str, excerpt: str) -> bool:
    """Match one value within one excerpt using transparent lexical rules."""

    normalized_component = normalize_answer(component)
    normalized_excerpt = normalize_answer(excerpt)
    if not normalized_component or not normalized_excerpt:
        return False
    if normalized_component in normalized_excerpt:
        return True

    component_tokens = normalized_component.split()
    if len(component_tokens) < _ANSWER_VALUE_MIN_FALLBACK_TOKENS:
        return False
    # Preserve token order for the paraphrase fallback. Bag overlap falsely
    # treated "contemporary art ... museum of modern art" as evidence for the
    # distinct venue "Museum of Contemporary Art". An LCS still accepts mild
    # paraphrases such as "Queen live with Adam Lambert ..." while requiring
    # the identifying words to occur in a compatible sequence in one excerpt.
    excerpt_tokens = normalized_excerpt.split()
    previous = [0] * (len(excerpt_tokens) + 1)
    for component_token in component_tokens:
        current = [0]
        for index, excerpt_token in enumerate(excerpt_tokens, start=1):
            if component_token == excerpt_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    overlap = previous[-1]
    return (
        overlap * _ANSWER_VALUE_TOKEN_RECALL_DENOMINATOR
        >= len(component_tokens) * _ANSWER_VALUE_TOKEN_RECALL_NUMERATOR
    )


def answer_value_component_coverage(
    gold: str,
    evidence_source_count: int,
    packed_raw_excerpts: list[str],
) -> AnswerValueCoverage | None:
    """Measure explicit answer values across final packed raw excerpts.

    This intentionally ignores source and chunk identity: equivalent raw
    evidence can operationally provide the same answer value. Each component
    must occur within one excerpt; tokens are never assembled across chunks.
    The caller must supply the post-budget body with metadata rows removed.
    """

    parsed = _parse_answer_value_components(gold, evidence_source_count)
    if parsed is None:
        return None
    components, parse_kind = parsed
    hit_mask = tuple(
        any(
            _answer_value_component_in_excerpt(component, excerpt)
            for excerpt in packed_raw_excerpts
        )
        for component in components
    )
    found = sum(hit_mask)
    return AnswerValueCoverage(
        expected=len(components),
        found=found,
        recall=found / len(components),
        all_components=found == len(components),
        hit_mask=hit_mask,
        metric_kind=(
            f"{parse_kind}:normalized_literal_or_80pct_ordered_token_recall_same_excerpt"
        ),
    )
