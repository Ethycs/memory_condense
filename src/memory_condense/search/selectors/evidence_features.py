"""Transient, source-grounded evidence features for coverage selection."""

from __future__ import annotations

import math
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.packing.source_provenance import (
    provenance_timestamp_key,
)

def _source_id(result: RetrievalResult) -> str:
    return result.durable_source_id


def _normalized_event_key(value: str | None) -> str | None:
    if value is None:
        return None
    key = re.sub(r"\s+", " ", value).strip().casefold()
    return key or None

def _timestamp_key(value: str | None) -> float | None:
    return provenance_timestamp_key(value, allow_bare_year=True)

def _normalized_transport(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _normalized_scalars(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low <= 1e-12:
        # An invariant component contains no ranking information. Treat it as
        # neutral rather than as uniformly strong evidence.
        return [0.5 for _value in values]
    return [(float(value) - low) / (high - low) for value in values]


def _energy_softmax(energies: Sequence[float]) -> list[float]:
    """Stable normalization for posterior-shaped, explicitly uncalibrated scores."""

    if not energies:
        return []
    peak = max(energies)
    weights = [math.exp(max(-60.0, min(60.0, value - peak))) for value in energies]
    total = sum(weights)
    return [value / total for value in weights]


def _surface_value_evidence(text: str, timestamp: str | None) -> float:
    """Cheap category-free evidence that a row states a recoverable value.

    The score is deliberately only a surface prior: proper-name-shaped spans,
    numbers, and a complete-enough clause raise it; bare anaphora lowers it.
    It does not claim entity recognition and never inspects answer labels.
    """

    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return 0.0
    name_spans = re.findall(
        r"\b(?:[A-Z][\w&.'-]+|[A-Z]{2,})"
        r"(?:\s+(?:(?:of|the|and|at|in|on)\s+)?"
        r"(?:[A-Z][\w&.'-]+|[A-Z]{2,}))+\b",
        text,
    )
    standalone_names = re.findall(
        r"\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][\w'-]*)\b",
        text,
    )
    named_tokens = sum(len(span.split()) for span in name_spans) + len(
        standalone_names
    )
    named = min(1.0, named_tokens / 4.0)
    numeric = min(1.0, len(re.findall(r"\b\d[\d,.:/-]*\b", text)) / 2.0)
    completion = min(1.0, math.log1p(len(words)) / math.log(33.0))
    value = 0.50 * named + 0.15 * numeric + 0.25 * completion
    if timestamp:
        value += 0.10
    if named == 0.0 and numeric == 0.0 and re.search(
        r"\b(?:it|this|that|there|them|the place|the event|the show)\b",
        text,
        re.IGNORECASE,
    ):
        value *= 0.65
    return max(0.0, min(1.0, value))


_VENUE_QUERY_RE = re.compile(r"\b(?:museum|museums|gallery|galleries)\b", re.I)
_PROPER_VENUE_TOKEN = r"(?:[A-Z][\w&.'’:-]*|[A-Z]{2,})"
_PROPER_VENUE_RE = re.compile(
    rf"\b(?:"
    rf"(?:{_PROPER_VENUE_TOKEN}\s+){{1,5}}(?:Museum|Gallery)"
    rf"(?:\s+of\s+{_PROPER_VENUE_TOKEN}(?:\s+{_PROPER_VENUE_TOKEN}){{0,4}})?"
    rf"|(?:Museum|Gallery)\s+of\s+{_PROPER_VENUE_TOKEN}"
    rf"(?:\s+{_PROPER_VENUE_TOKEN}){{0,4}}"
    rf")\b"
)


def _canonical_answer_object_key(query: str, text: str) -> str | None:
    """Return one transient, query-anchored museum/gallery identity.

    This deliberately narrow parser is a conservative identity control, not
    a general NER system.  It activates only for a matching query head and
    only when the row contains exactly one unambiguous proper-name venue.
    The normalized key is consumed inside one ``select`` call and is never
    written to a trace or durable store.
    """

    if not _VENUE_QUERY_RE.search(query):
        return None
    keys: set[str] = set()
    for match in _PROPER_VENUE_RE.finditer(text):
        value = re.sub(r"['’]s\b", "", match.group(0), flags=re.I)
        value = re.sub(r"[^\w&]+", " ", value, flags=re.UNICODE)
        value = re.sub(r"\s+", " ", value).strip().casefold()
        value = re.sub(r"^(?:the|a|an|my|our)\s+", "", value)
        if value not in {"museum", "gallery"}:
            keys.add(value)
    return next(iter(keys)) if len(keys) == 1 else None


def _optional_probability(value: Any, *names: str) -> float | None:
    """Read a transient scorer row without coupling to its concrete class."""

    if value is None:
        return None
    inspected = (
        value.get("inspected")
        if isinstance(value, Mapping)
        else getattr(value, "inspected", None)
    )
    if inspected is False:
        return None
    candidate: Any = value
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
            break
        attribute = getattr(value, name, None)
        if attribute is not None:
            candidate = attribute
            break
    try:
        number = float(candidate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number < 0.0 or number > 1.0:
        number = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, number))))
    return max(0.0, min(1.0, number))


def resolve_surface_value_evidence():
    """Honor the legacy facade's monkeypatch seam without coupling imports."""

    facade = sys.modules.get(
        "memory_condense.search.selectors.coverage_selector"
    )
    return getattr(facade, "_surface_value_evidence", _surface_value_evidence)
