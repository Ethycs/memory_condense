"""Gold-blind, operator-aware numeric mention classification.

The memory adapters previously treated any isolated digit as an answer
operand.  That promoted calendar days, durations, and ranks into unrelated
counts or prices.  This module is the single lexical contract shared by the
typed scanner and adapters.  It consumes only evidence text plus the dated
question/operator specification and retains no model state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .typed_operator_spec import AnswerShape, TypedOperatorSpec


class NumericDimension(str, Enum):
    COUNT = "count"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DURATION = "duration"
    MEASURE = "measure"
    GENERIC = "generic"


class NumericQualifier(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"


@dataclass(frozen=True, slots=True)
class NumericMention:
    value: float
    dimension: NumericDimension
    qualifier: NumericQualifier
    unit: str | None
    start: int
    end: int
    surface: str

    def __post_init__(self) -> None:
        if type(self.value) not in {int, float}:
            raise TypeError("numeric mention value must be exact")
        if type(self.dimension) is not NumericDimension:
            raise TypeError("numeric mention dimension must be canonical")
        if type(self.qualifier) is not NumericQualifier:
            raise TypeError("numeric mention qualifier must be canonical")
        if self.unit is not None and (
            type(self.unit) is not str or not self.unit
        ):
            raise TypeError("numeric mention unit must be exact text")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise TypeError("numeric mention span is invalid")
        if type(self.surface) is not str or not self.surface:
            raise TypeError("numeric mention surface must be exact text")


_MONTH = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December"
)
_DATE_SPAN_RE = re.compile(
    rf"\b(?:19|20)\d{{2}}[-/]\d{{1,2}}(?:[-/]\d{{1,2}})?\b|"
    rf"\b(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?"
    rf"(?:,?\s+(?:19|20)\d{{2}})?\b|"
    rf"\b(?:{_MONTH})\s+(?:19|20)\d{{2}}\b",
    re.I,
)
_DIGIT_RE = re.compile(
    r"(?<![\w.])(?P<sign>[+-]?)(?P<number>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<decimal>\.\d+)?(?P<ordinal>st|nd|rd|th)?(?!\w)",
    re.I,
)
_NUMBER_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand)\b",
    re.I,
)
_NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}
_DURATION_UNIT_RE = re.compile(
    r"^\s*-?\s*(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.I,
)
_MEASURE_UNIT_RE = re.compile(
    r"^\s*-?\s*(?P<unit>lb|lbs|pounds?|oz|ounces?|kg|kgs|kilograms?|"
    r"g|grams?|miles?|kilometers?|km|meters?|metres?|feet|foot|inches?)\b",
    re.I,
)
_PERCENT_UNIT_RE = re.compile(r"^\s*(?:%|percent(?:age)?\b)", re.I)
_CURRENCY_AFTER_RE = re.compile(
    r"^\s*(?:USD\b|US\s+dollars?\b|dollars?\b)", re.I
)
_APPROX_PREFIX_RE = re.compile(
    r"(?:\babout|\baround|\bapproximately|\bapprox\.?|\broughly|"
    r"\bnearly|\balmost)\s*(?:\$|USD\s*)?$",
    re.I,
)
_LOWER_PREFIX_RE = re.compile(
    r"(?:\bover|\babove|\bmore\s+than|\bat\s+least|\bminimum(?:\s+of)?)"
    r"\s*(?:\$|USD\s*)?$",
    re.I,
)
_UPPER_PREFIX_RE = re.compile(
    r"(?:\bunder|\bbelow|\bless\s+than|\bat\s+most|\bup\s+to|"
    r"\bmaximum(?:\s+of)?)\s*(?:\$|USD\s*)?$",
    re.I,
)
_CURRENCY_QUESTION_RE = re.compile(
    r"\b(?:spend|spent|pay|paid|cost|price|accommodations?|lodg(?:e|ing)|"
    r"per\s+night|dollars?|usd)\b|\$",
    re.I,
)
_CURRENCY_EVIDENCE_RE = re.compile(
    r"\b(?:spend|spent|pay|paid|cost|costs|costing|price|priced|"
    r"per\s+night|nightly)\b",
    re.I,
)
_PERCENT_QUESTION_RE = re.compile(
    r"\b(?:percent(?:age)?|discount|rate)\b|%", re.I
)
_MEASURE_QUESTION_RE = re.compile(
    r"\b(?:lb|lbs|pounds?|oz|ounces?|kg|kgs|kilograms?|grams?|miles?|"
    r"kilometers?|km|meters?|metres?|feet|foot|inches?)\b",
    re.I,
)
_DURATION_QUESTION_RE = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:seconds?|minutes?|hours?|days?|"
    r"weeks?|months?|years?)\b",
    re.I,
)
_COUNT_QUESTION_RE = re.compile(r"\bhow\s+many\b", re.I)
_RANK_PREFIX_RE = re.compile(
    r"(?:\btop|\brank(?:ed)?(?:\s+(?:number|no\.?))?|#)\s*$", re.I
)


def expected_numeric_dimension(
    *,
    operator_spec: TypedOperatorSpec | None = None,
    question: str | None = None,
) -> NumericDimension:
    """Derive the answer dimension from question text, then the sealed spec."""

    if operator_spec is not None and type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("numeric semantics operator spec must be exact")
    if question is not None and type(question) is not str:
        raise TypeError("numeric semantics question must be exact text")
    body = question or ""
    if body and _PERCENT_QUESTION_RE.search(body):
        return NumericDimension.PERCENTAGE
    if body and _CURRENCY_QUESTION_RE.search(body) and re.search(
        r"\bhow\s+much\b|\b(?:higher|lower|more|less)\b", body, re.I
    ):
        return NumericDimension.CURRENCY
    if body and _MEASURE_QUESTION_RE.search(body) and re.search(
        r"\bhow\s+(?:many|much)\b", body, re.I
    ):
        return NumericDimension.MEASURE
    if (
        operator_spec is not None
        and operator_spec.answer_shape is AnswerShape.DURATION
    ) or (body and _DURATION_QUESTION_RE.search(body)):
        return NumericDimension.DURATION
    if body and _COUNT_QUESTION_RE.search(body):
        return NumericDimension.COUNT
    return NumericDimension.GENERIC


def _overlaps(span: tuple[int, int], protected: tuple[tuple[int, int], ...]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in protected)


def _qualifier(prefix: str) -> NumericQualifier:
    if _APPROX_PREFIX_RE.search(prefix):
        return NumericQualifier.APPROXIMATE
    if _LOWER_PREFIX_RE.search(prefix):
        return NumericQualifier.LOWER_BOUND
    if _UPPER_PREFIX_RE.search(prefix):
        return NumericQualifier.UPPER_BOUND
    return NumericQualifier.EXACT


def _measure_unit(raw: str) -> str:
    folded = raw.casefold()
    if folded in {"lb", "lbs", "pound", "pounds"}:
        return "lb"
    if folded in {"oz", "ounce", "ounces"}:
        return "oz"
    if folded in {"kg", "kgs", "kilogram", "kilograms"}:
        return "kg"
    if folded in {"g", "gram", "grams"}:
        return "g"
    if folded in {"km", "kilometer", "kilometers"}:
        return "km"
    return folded.rstrip("s")


def _compatible(
    actual: NumericDimension, expected: NumericDimension
) -> bool:
    if expected is NumericDimension.GENERIC:
        return actual is not NumericDimension.DURATION
    if expected is NumericDimension.COUNT:
        return actual is NumericDimension.COUNT
    return actual is expected


def numeric_mentions(
    text: str,
    *,
    operator_spec: TypedOperatorSpec | None = None,
    question: str | None = None,
    expected_dimension: NumericDimension | None = None,
) -> tuple[NumericMention, ...]:
    """Return compatible answer-value mentions in deterministic text order.

    Calendar values and ranks are never answer operands.  Duration values are
    admitted only for duration questions; other dimensions must match the
    question-derived expectation.  Supplying neither a question nor a spec is
    still conservative: it admits scalar/currency/percent/measure mentions but
    not dates, ranks, or durations.
    """

    if type(text) is not str:
        raise TypeError("numeric mention text must be exact")
    if expected_dimension is not None and type(expected_dimension) is not NumericDimension:
        raise TypeError("expected numeric dimension must be canonical")
    expected = expected_dimension or expected_numeric_dimension(
        operator_spec=operator_spec,
        question=question,
    )
    protected = tuple(match.span() for match in _DATE_SPAN_RE.finditer(text))
    raw_matches: list[tuple[int, int, float, str, bool]] = []
    for match in _DIGIT_RE.finditer(text):
        raw = f"{match.group('sign')}{match.group('number')}{match.group('decimal') or ''}"
        raw_matches.append(
            (
                match.start(),
                match.end(),
                float(raw.replace(",", "")),
                match.group(0),
                bool(match.group("ordinal")),
            )
        )
    for match in _NUMBER_WORD_RE.finditer(text):
        raw_matches.append(
            (
                match.start(),
                match.end(),
                float(_NUMBER_WORD_VALUES[match.group(0).casefold()]),
                match.group(0),
                False,
            )
        )

    output: list[NumericMention] = []
    for start, end, value, surface, ordinal in sorted(raw_matches):
        if _overlaps((start, end), protected):
            continue
        prefix = text[max(0, start - 40) : start]
        suffix = text[end : min(len(text), end + 40)]
        if ordinal or _RANK_PREFIX_RE.search(prefix):
            continue

        duration = _DURATION_UNIT_RE.match(suffix)
        measure = _MEASURE_UNIT_RE.match(suffix)
        percentage = _PERCENT_UNIT_RE.match(suffix)
        currency_after = _CURRENCY_AFTER_RE.match(suffix)
        currency_before = re.search(r"(?:\$|USD\s*)\s*$", prefix, re.I)
        if percentage is not None:
            dimension = NumericDimension.PERCENTAGE
            unit = "%"
        elif currency_before is not None or currency_after is not None:
            dimension = NumericDimension.CURRENCY
            unit = "$"
        elif duration is not None:
            dimension = NumericDimension.DURATION
            unit = duration.group("unit").casefold().rstrip("s")
        elif measure is not None:
            dimension = NumericDimension.MEASURE
            unit = _measure_unit(measure.group("unit"))
        elif (
            expected is NumericDimension.CURRENCY
            and _CURRENCY_EVIDENCE_RE.search(f"{prefix[-24:]} {suffix[:24]}")
        ):
            dimension = NumericDimension.CURRENCY
            unit = "$"
        elif expected is NumericDimension.COUNT:
            dimension = NumericDimension.COUNT
            unit = None
        else:
            dimension = NumericDimension.GENERIC
            unit = None
        if not _compatible(dimension, expected):
            continue
        output.append(
            NumericMention(
                value=value,
                dimension=dimension,
                qualifier=_qualifier(prefix),
                unit=unit,
                start=start,
                end=end,
                surface=surface,
            )
        )
    return tuple(output)


def single_numeric_mention(
    text: str,
    *,
    operator_spec: TypedOperatorSpec | None = None,
    question: str | None = None,
    expected_dimension: NumericDimension | None = None,
) -> NumericMention | None:
    mentions = numeric_mentions(
        text,
        operator_spec=operator_spec,
        question=question,
        expected_dimension=expected_dimension,
    )
    return mentions[0] if len(mentions) == 1 else None


__all__ = [
    "NumericDimension",
    "NumericMention",
    "NumericQualifier",
    "expected_numeric_dimension",
    "numeric_mentions",
    "single_numeric_mention",
]
