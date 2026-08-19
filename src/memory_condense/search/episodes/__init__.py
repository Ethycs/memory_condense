"""Provider-free episodic retrieval; EM-style surprise is an optional input."""

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
    LexicalEmbeddingChangeScorer,
    SurpriseScorer,
    dense_cosine,
    lexical_cosine,
    score_surprise_sequence,
)


__all__ = [
    "AdaptiveBoundaryDetector",
    "BoundaryDetector",
    "BoundaryProposal",
    "BoundaryRefinement",
    "CohesionBoundaryRefiner",
    "DirectChunkSeed",
    "EpisodeBuildResult",
    "EpisodeBuilder",
    "EpisodeLookup",
    "EpisodeRetrievalPlan",
    "EpisodeRetrievalPolicy",
    "FixedIntervalBoundaryDetector",
    "LexicalEmbeddingChangeScorer",
    "SurpriseScorer",
    "dense_cosine",
    "expand_episode_seeds",
    "lexical_cosine",
    "score_episode_surprises",
    "score_surprise_sequence",
    "select_episode_representatives",
]


# Readable alias for callers operating at the episode abstraction.
score_episode_surprises = score_surprise_sequence
