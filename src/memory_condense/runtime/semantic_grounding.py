"""BGE-filtered semantic location followed by exact source hydration."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
import threading

import numpy as np


_TOKEN = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
_CRITICAL = re.compile(
    r"https?://\S+|[\w.+-]+@[\w.-]+|\b\d[\w./:%+\-]*\b|\b[\w-]*\d[\w-]*\b",
    flags=re.UNICODE,
)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9_-]{1,}\b")
_CAPITALIZED_STOP = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "In",
    "It", "Its", "My", "On", "Our", "She", "That", "The", "Their", "They", "This", "To",
    "We", "You", "Your",
}
_NEGATION = {"no", "not", "never", "neither", "nor", "without"}
_COMPLETION = {
    "complete", "completed", "deploy", "deployed", "done", "finish", "finished", "launch",
    "launched", "release", "released", "ship", "shipped",
}
_PENDING = {
    "intend", "intended", "pending", "plan", "planned", "propose", "proposed", "schedule",
    "scheduled",
}
_POLICY_PATH = Path(__file__).parent / "data" / "bge-summary-validator-v1.json"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.casefold())


def _critical_values(text: str) -> set[str]:
    values = {match.group(0).casefold().rstrip(".,;:!?") for match in _CRITICAL.finditer(text)}
    values.update(token.casefold() for token in _CAPITALIZED.findall(text)
                  if token not in _CAPITALIZED_STOP)
    return values


def _status(text: str) -> set[str]:
    tokens = _tokens(text)
    result: set[str] = set()
    for index, token in enumerate(tokens):
        negated = any(value in _NEGATION for value in tokens[max(0, index-3):index])
        if token in _COMPLETION:
            result.add("pending" if negated else "completed")
        elif token in _PENDING:
            result.add("pending")
    return result


def _polarity_conflict(proposal: str, candidate: str) -> bool:
    proposal_tokens, candidate_tokens = _tokens(proposal), _tokens(candidate)
    shared = set(proposal_tokens) & set(candidate_tokens) & _COMPLETION
    for term in shared:
        def polarities(tokens):
            values = set()
            for index, token in enumerate(tokens):
                if token == term:
                    values.add(any(value in _NEGATION for value in tokens[max(0, index-3):index]))
            return values
        if polarities(proposal_tokens).isdisjoint(polarities(candidate_tokens)):
            return True
    proposal_status, candidate_status = _status(proposal), _status(candidate)
    return bool(proposal_status and candidate_status and proposal_status.isdisjoint(candidate_status))


def exact_candidate_spans(source: str, proposal: str) -> tuple[str, ...]:
    """Return bounded literal source spans for semantic ranking.

    Sentence candidates preserve ordinary prose. Overlapping token windows
    cover transcripts split by timestamp lines and long punctuation-free text.
    Every returned value is a literal substring of ``source``.
    """
    words = list(_TOKEN.finditer(source))
    query_words = list(_TOKEN.finditer(proposal))
    if not words or not query_words:
        return ()
    target = min(64, max(8, len(query_words) + 8))
    values: list[str] = []
    seen: set[str] = set()

    def add(left: int, right: int) -> None:
        value = source[left:right].strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)

    for match in re.finditer(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)", source):
        sentence_words = list(_TOKEN.finditer(source, match.start(), match.end()))
        if not sentence_words:
            continue
        if len(sentence_words) <= 64:
            add(sentence_words[0].start(), sentence_words[-1].end())
            continue
        window = min(64, max(target, 24))
        stride = max(1, window // 3)
        starts = list(range(0, len(sentence_words) - window + 1, stride))
        if starts[-1] != len(sentence_words) - window:
            starts.append(len(sentence_words) - window)
        for start in starts:
            add(sentence_words[start].start(), sentence_words[start + window - 1].end())

    window = min(len(words), target)
    stride = max(1, window // 3)
    starts = list(range(0, len(words) - window + 1, stride))
    if starts and starts[-1] != len(words) - window:
        starts.append(len(words) - window)
    for start in starts:
        add(words[start].start(), words[start + window - 1].end())
    return tuple(values)


class BGESourceGrounder:
    """Use the resident BGE encoder only to locate an exact source span."""

    def __init__(self, embedding, *, policy=None, cache_size=128):
        self.embedding = embedding
        self.policy = policy or json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
        if (self.policy.get("format") != "memory-condense-bge-summary-validator-calibration-v1"
                or self.policy.get("model_id") != getattr(embedding, "model_name", None)
                or self.policy.get("model_revision") != getattr(embedding, "model_revision", None)
                or self.policy.get("candidate_algorithm") != "exact-sentence-and-overlapping-token-windows-v1"):
            raise ValueError("BGE summary-validator calibration does not match the resident encoder")
        self.pre_review_cutoff = float(self.policy["pre_review_cutoff"])
        self.reviewed_cutoff = float(self.policy["reviewed_cutoff"])
        if not 0.0 <= self.reviewed_cutoff <= self.pre_review_cutoff <= 1.0:
            raise ValueError("Invalid BGE summary-validator cutoffs")
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("BGE source grounding cache must be positive")
        self._cache: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _candidate_vectors(self, candidates: tuple[str, ...]) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(candidates)
            if cached is not None:
                self._cache.move_to_end(candidates)
                return cached
        generated = np.asarray(self.embedding.embed_queries(candidates), dtype=np.float32)
        if generated.shape != (len(candidates), 1024) or not np.isfinite(generated).all():
            raise ValueError("BGE source grounding returned invalid candidate vectors")
        with self._lock:
            self._cache[candidates] = generated
            self._cache.move_to_end(candidates)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return generated

    def __call__(self, source: str, proposal: str, *, reviewed=False) -> str | None:
        candidates = exact_candidate_spans(source, proposal)
        if not candidates:
            return None
        query = np.asarray(self.embedding.embed_queries([proposal]), dtype=np.float32)
        if query.shape != (1, 1024) or not np.isfinite(query).all():
            raise ValueError("BGE source grounding returned an invalid proposal vector")
        scores = self._candidate_vectors(candidates) @ query[0]
        eligible = [index for index, candidate in enumerate(candidates)
                    if _critical_values(proposal).issubset(_critical_values(candidate))
                    and not _polarity_conflict(proposal, candidate)]
        if not eligible:
            return None
        index = max(eligible, key=lambda value: float(scores[value]))
        cutoff = self.reviewed_cutoff if reviewed else self.pre_review_cutoff
        return candidates[index] if float(scores[index]) >= cutoff else None
