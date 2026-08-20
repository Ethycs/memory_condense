"""Stable facade for episodic surprise objects and transformations."""

from memory_condense.domain._tokenizer import tokenizer_proxy_identity

from .qwen_episode_signal import (
    QwenAttentionHeadSurpriseScorer,
    _attention_head_implementation_sha256,
    _lossless_proxy_prefix,
    _normalized_transport_signature,
    _owned_qwen_runtime_binding,
    _owned_qwen_receipt_matches,
)
from .surprise_controls import (
    LexicalEmbeddingChangeScorer,
    dense_cosine,
    lexical_cosine,
    score_surprise_sequence,
)
from .surprise_models import (
    ATTENTION_HEAD_SIMILARITY_ALGORITHM,
    ATTENTION_HEAD_SURPRISE_ALGORITHM,
    ATTENTION_HEAD_SURPRISE_FORMAT,
    ATTENTION_HEAD_SURPRISE_SCORE_FORMULA,
    EPISODIC_SURPRISE_PROBE,
    AttentionHeadSurpriseReceipt,
    ScoredSurpriseSequence,
    SurpriseScorer,
    SurpriseSequenceScorer,
)


__all__ = [
    "ATTENTION_HEAD_SIMILARITY_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_FORMAT",
    "ATTENTION_HEAD_SURPRISE_SCORE_FORMULA",
    "EPISODIC_SURPRISE_PROBE",
    "AttentionHeadSurpriseReceipt",
    "LexicalEmbeddingChangeScorer",
    "QwenAttentionHeadSurpriseScorer",
    "ScoredSurpriseSequence",
    "SurpriseScorer",
    "SurpriseSequenceScorer",
    "dense_cosine",
    "lexical_cosine",
    "score_surprise_sequence",
]
