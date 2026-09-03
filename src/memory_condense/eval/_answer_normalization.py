"""Dependency-free SQuAD/LongMemEval lexical answer normalization."""

from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace."""

    if not text:
        return ""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 after the frozen lexical normalization."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


__all__ = ["f1_score", "normalize_answer"]
