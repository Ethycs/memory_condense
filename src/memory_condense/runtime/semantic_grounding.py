"""BGE-filtered semantic location followed by exact source hydration."""
from __future__ import annotations

from collections import Counter, OrderedDict
from difflib import SequenceMatcher
import json
import math
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
_CANDIDATE_ALGORITHM = "lexical-prefilter-top4-exact-source-v2"


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


def semantic_candidate_spans(source: str, proposal: str, *, limit: int = 4) -> tuple[str, ...]:
    """Keep the few literal spans most likely to need semantic disambiguation.

    BGE remains the admission score. This cheap lexical rank only prevents a
    validator miss in one long fragment from embedding hundreds of irrelevant
    source windows. The sealed failure calibration checks that top four retains
    every known positive source span.
    """
    candidates = exact_candidate_spans(source, proposal)
    if len(candidates) <= limit:
        return candidates
    query = _tokens(proposal)
    if not query:
        return candidates[:limit]
    tokenized = [_tokens(candidate) for candidate in candidates]
    document_frequency = Counter(term for words in tokenized for term in set(words))
    population = len(candidates)
    idf = {term: math.log((population + 1) / (frequency + 1)) + 1
           for term, frequency in document_frequency.items()}
    query_terms = set(query)
    query_weight = sum(idf.get(term, math.log(population + 1) + 1)
                       for term in query_terms)

    def rank(index: int) -> tuple[float, float, float, int]:
        words = tokenized[index]
        terms = set(words)
        shared_weight = sum(idf.get(term, 1) for term in query_terms & terms)
        coverage = shared_weight / query_weight if query_weight else 0.0
        candidate_weight = sum(idf.get(term, 1) for term in terms)
        precision = shared_weight / candidate_weight if candidate_weight else 0.0
        ordered = SequenceMatcher(None, query, words, autojunk=False).find_longest_match().size
        order_coverage = ordered / len(query)
        length_fit = min(len(query) / len(words), len(words) / len(query))
        score = .55 * coverage + .25 * order_coverage + .10 * precision + .10 * length_fit
        return score, coverage, order_coverage, -index

    selected = sorted(range(len(candidates)), key=rank, reverse=True)[:limit]
    return tuple(candidates[index] for index in selected)


class BGESourceGrounder:
    """Use the resident BGE encoder only to locate an exact source span."""

    def __init__(self, embedding, *, policy=None, cache_size=128):
        self.embedding = embedding
        self.policy = policy or json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
        if (self.policy.get("format") != "memory-condense-bge-summary-validator-calibration-v1"
                or self.policy.get("model_id") != getattr(embedding, "model_name", None)
                or self.policy.get("model_revision") != getattr(embedding, "model_revision", None)
                or self.policy.get("candidate_algorithm") != _CANDIDATE_ALGORITHM):
            raise ValueError("BGE summary-validator calibration does not match the resident encoder")
        self.pre_review_cutoff = float(self.policy["pre_review_cutoff"])
        self.reviewed_cutoff = float(self.policy["reviewed_cutoff"])
        if not 0.0 <= self.reviewed_cutoff <= self.pre_review_cutoff <= 1.0:
            raise ValueError("Invalid BGE summary-validator cutoffs")
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("BGE source grounding cache must be positive")
        # Cache individual texts rather than whole candidate tuples. A source
        # span can recur across retries with a different proposal, and the
        # proposal itself is reused when a sealed response is revalidated.
        # Per-text entries also let one admitted FastEmbed batch contain both
        # the proposal and its eligible source spans.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _vectors(self, texts: tuple[str, ...]) -> np.ndarray:
        values: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._lock:
            for text in texts:
                cached = self._cache.get(text)
                if cached is not None:
                    values[text] = cached
                    self._cache.move_to_end(text)
                elif text not in values and text not in missing:
                    missing.append(text)
        if missing:
            generated = np.asarray(self.embedding.embed_queries(missing), dtype=np.float32)
            if generated.shape != (len(missing), 1024) or not np.isfinite(generated).all():
                raise ValueError("BGE source grounding returned invalid vectors")
            with self._lock:
                for text, vector in zip(missing, generated, strict=True):
                    values[text] = vector
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
        with self._lock:
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return np.stack([values[text] for text in texts])

    def __call__(self, source: str, proposal: str, *, reviewed=False) -> str | None:
        candidates = semantic_candidate_spans(source, proposal)
        if not candidates:
            return None
        critical = _critical_values(proposal)
        eligible = tuple(candidate for candidate in candidates
                         if critical.issubset(_critical_values(candidate))
                         and not _polarity_conflict(proposal, candidate))
        if not eligible:
            return None
        vectors = self._vectors((proposal, *eligible))
        scores = vectors[1:] @ vectors[0]
        index = int(np.argmax(scores))
        cutoff = self.reviewed_cutoff if reviewed else self.pre_review_cutoff
        return eligible[index] if float(scores[index]) >= cutoff else None
