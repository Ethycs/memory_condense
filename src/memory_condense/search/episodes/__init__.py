"""Provider-free episodic retrieval with optional transient head signals."""

from .boundaries import (
    AdaptiveBoundaryDetector,
    BoundaryDetector,
    BoundaryProposal,
    BoundaryRefinement,
    CohesionBoundaryRefiner,
    FixedIntervalBoundaryDetector,
)
from .builder import EpisodeBuildResult, EpisodeBuilder
from .representatives import select_episode_representatives
from .retrieval import (
    DirectChunkSeed,
    EpisodeLookup,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
    expand_episode_seeds,
)
from .surprise import (
    ATTENTION_HEAD_SIMILARITY_ALGORITHM,
    ATTENTION_HEAD_SURPRISE_ALGORITHM,
    ATTENTION_HEAD_SURPRISE_FORMAT,
    ATTENTION_HEAD_SURPRISE_SCORE_FORMULA,
    EPISODIC_SURPRISE_PROBE,
    AttentionHeadSurpriseReceipt,
    LexicalEmbeddingChangeScorer,
    QwenAttentionHeadSurpriseScorer,
    ScoredSurpriseSequence,
    SurpriseScorer,
    SurpriseSequenceScorer,
    dense_cosine,
    lexical_cosine,
    score_surprise_sequence,
)


__all__ = [
    "ATTENTION_HEAD_SIMILARITY_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_FORMAT",
    "ATTENTION_HEAD_SURPRISE_SCORE_FORMULA",
    "AdaptiveBoundaryDetector",
    "AttentionHeadSurpriseReceipt",
    "BoundaryDetector",
    "BoundaryProposal",
    "BoundaryRefinement",
    "CohesionBoundaryRefiner",
    "DirectChunkSeed",
    "EPISODIC_SURPRISE_PROBE",
    "EpisodeBuildResult",
    "EpisodeBuilder",
    "EpisodeLookup",
    "EpisodeRetrievalPlan",
    "EpisodeRetrievalPolicy",
    "FixedIntervalBoundaryDetector",
    "LexicalEmbeddingChangeScorer",
    "QwenAttentionHeadSurpriseScorer",
    "ScoredSurpriseSequence",
    "SurpriseScorer",
    "SurpriseSequenceScorer",
    "dense_cosine",
    "expand_episode_seeds",
    "lexical_cosine",
    "score_episode_surprises",
    "score_surprise_sequence",
    "select_episode_representatives",
]


# Readable alias for callers operating at the episode abstraction.
score_episode_surprises = score_surprise_sequence
