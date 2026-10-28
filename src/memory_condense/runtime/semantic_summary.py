"""Proxy-only semantic summary validation with exact evidence receipts.

The frozen benchmark keeps the stricter extractive validator.  Live proxy
summaries may paraphrase, but only when the pinned BGE encoder maps them to a
short exact span from their own fragment and deterministic critical-fact gates
preserve identifiers, quantities, named entities, negation, and completion
status.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.local import extract_rows
from memory_condense.runtime.summary_fallback import LocalFirstSummaries


SEMANTIC_VALIDATION_POLICY = {
    "format": "semantic-summary-grounding-v1",
    "verifier": "pinned-bge-m3-cosine-plus-critical-facts-v1",
    "embedding_model": "BAAI/bge-m3-pinned-upstream",
    "minimum_support_cosine": 0.60,
    "maximum_summary_tokens": 128,
    "maximum_summary_words": 96,
    "maximum_support_tokens": 32,
    "support_quotes": "one exact contiguous span from the atom's own fragment",
    "critical_facts": ["identifiers", "quantities", "named-entity tokens", "negation", "status"],
}
SEMANTIC_VALIDATION_SHA256 = identity_sha256(SEMANTIC_VALIDATION_POLICY)

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
_COMPLETION_TERMS = {
    "complete", "completed", "deploy", "deployed", "done", "finish", "finished", "launch",
    "launched", "release", "released", "ship", "shipped",
}
_PENDING_TERMS = {"intend", "intended", "pending", "plan", "planned", "propose", "proposed", "schedule", "scheduled"}


def policy_root_name() -> str:
    return f"semantic-{SEMANTIC_VALIDATION_SHA256[:16]}"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.casefold())


def _critical_values(text: str) -> set[str]:
    values = {match.group(0).casefold().rstrip(".,;:!?") for match in _CRITICAL.finditer(text)}
    values.update(token.casefold() for token in _CAPITALIZED.findall(text) if token not in _CAPITALIZED_STOP)
    return values


def _status(text: str) -> set[str]:
    tokens = _tokens(text)
    result: set[str] = set()
    for index, token in enumerate(tokens):
        negated = any(value in _NEGATION for value in tokens[max(0, index-3):index])
        if token in _COMPLETION_TERMS:
            result.add("pending" if negated else "completed")
        elif token in _PENDING_TERMS:
            result.add("pending")
    return result


def _polarity_conflict(summary: str, support: str) -> bool:
    summary_tokens, support_tokens = _tokens(summary), _tokens(support)
    shared = set(summary_tokens) & set(support_tokens) & _COMPLETION_TERMS
    for term in shared:
        def polarities(tokens):
            values = set()
            for index, token in enumerate(tokens):
                if token == term:
                    values.add(any(value in _NEGATION for value in tokens[max(0, index-3):index]))
            return values
        if polarities(summary_tokens).isdisjoint(polarities(support_tokens)):
            return True
    summary_status, support_status = _status(summary), _status(support)
    return bool(summary_status and support_status and summary_status.isdisjoint(support_status))


def _support_candidates(source: str, *, maximum_tokens: int = 32) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", source):
        segment = match.group(0).strip()
        if not segment:
            continue
        if count_tokens(segment) <= maximum_tokens:
            pieces = [segment]
        else:
            words = list(re.finditer(r"\S+", segment))
            pieces = []
            start = 0
            while start < len(words):
                end = min(len(words), start + maximum_tokens)
                while end > start + 1:
                    piece = segment[words[start].start():words[end-1].end()]
                    if count_tokens(piece) <= maximum_tokens:
                        break
                    end -= 1
                piece = segment[words[start].start():words[end-1].end()]
                pieces.append(piece)
                if end == len(words):
                    break
                start += max(1, (end-start)//2)
        for piece in pieces:
            if piece and piece not in seen and piece in source:
                seen.add(piece)
                candidates.append(piece)
    return candidates


def _cosines(encoder, summary: str, candidates: list[str]) -> np.ndarray:
    vectors = np.asarray(encoder.embed_queries([summary, *candidates]), dtype=np.float32)
    if vectors.shape[0] != len(candidates) + 1 or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Semantic validator received invalid BGE vectors")
    query, matrix = vectors[0], vectors[1:]
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Semantic validator received zero-norm BGE vectors")
    return (matrix @ query) / denominator


class SemanticLocalFirstSummaries(LocalFirstSummaries):
    """Local-first summaries whose cache identity includes semantic validation."""

    def __init__(self, local, root, api, *, require_provider=False, handoff='failure',
                 saturated=None):
        super().__init__(local, root, self._bind_policy(api), require_provider=require_provider,
                         handoff=handoff, saturated=saturated)

    @staticmethod
    def _bind_policy(api):
        if api is None:
            return None
        identity = dict(api.identity)
        bound = identity.get("summary_validation")
        expected = {
            "policy": SEMANTIC_VALIDATION_POLICY,
            "sha256": SEMANTIC_VALIDATION_SHA256,
        }
        if bound is not None and bound != expected:
            raise ValueError("Summary provider is already bound to another validation policy")
        api.identity = dict(identity, summary_validation=expected)
        return api

    def configure(self, api):
        super().configure(self._bind_policy(api))

    def _validate(self, kind, messages, response, typed_request, *, ground_near_exact=False,
                  semantic_grounder=None, semantic_reviewed=False):
        try:
            return super()._validate(kind, messages, response, typed_request,
                ground_near_exact=ground_near_exact, semantic_grounder=semantic_grounder,
                semantic_reviewed=semantic_reviewed)
        except ValueError:
            if kind != "raw":
                raise
        finish = response.get("generated_finish_reason", response.get("finish_reason"))
        if finish != "stop":
            raise ValueError("Summary generation did not complete")
        if response.get("local_extractive_fallbacks", 0) and "generated_content" not in response:
            raise ValueError("Semantic validation requires original generated content")
        content = response.get("generated_content", response.get("content", ""))
        fragments = json.loads(messages[1]["content"])["fragments"]
        rows = extract_rows(content, "atoms")
        if not isinstance(rows, list) or len(rows) != len(fragments):
            raise ValueError("Semantic summary coverage differs from its fragments")
        encoder = self.local.embedding()
        atoms = []
        for fragment, item in zip(fragments, rows, strict=True):
            label, source = fragment["label"], fragment["fragment"]
            summary = item.get("summary") if isinstance(item, dict) else None
            if (not isinstance(item, dict) or item.get("label") != label
                    or not isinstance(summary, str) or not summary.strip()
                    or count_tokens(summary) > SEMANTIC_VALIDATION_POLICY["maximum_summary_tokens"]
                    or len(summary.split()) > SEMANTIC_VALIDATION_POLICY["maximum_summary_words"]):
                raise ValueError("Semantic summary shape is invalid")
            candidates = _support_candidates(source)
            if not candidates:
                raise ValueError("Semantic summary has no exact support candidate")
            scores = _cosines(encoder, summary, candidates)
            best_index = int(np.argmax(scores))
            support, score = candidates[best_index], float(scores[best_index])
            if (not math.isfinite(score)
                    or score < SEMANTIC_VALIDATION_POLICY["minimum_support_cosine"]
                    or not _critical_values(summary).issubset(_critical_values(source))
                    or _polarity_conflict(summary, support)):
                raise ValueError("Semantic summary is not grounded in its own fragment")
            atoms.append({"label": label, "summary": summary.strip(), "support": [support]})
        return {"atoms": atoms}


__all__ = [
    "SEMANTIC_VALIDATION_POLICY",
    "SEMANTIC_VALIDATION_SHA256",
    "SemanticLocalFirstSummaries",
    "policy_root_name",
]
