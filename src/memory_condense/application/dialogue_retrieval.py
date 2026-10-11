"""Prepared dialogue-wide retrieval; compilation happens before recall.

The policy is independent of the reader and provider. It selects original
addresses using stored summaries; native hydration and learning own evidence.
Each prepared instance belongs to exactly one published snapshot.
"""
from dataclasses import dataclass

from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.dialogue_neighborhood import DialogueNeighborhoodRouter
from memory_condense.search.section_routing import SectionRoutePlan
from memory_condense.search.section_summary import bound_int


@dataclass(frozen=True, slots=True)
class DialogueRetrievalPolicy(SealedIdentity):
    """Versioned tuning profile. Defaults reproduce the closing-response assay."""

    version: int = 1
    episode_limit: int = 12
    chronology_bands: int = 4
    closing_responses: bool = True
    neighbor_turns: int = 1
    max_context_tokens: int = 6144
    max_raw_spans: int = 128
    receipt_sha256: str = ""

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('Unsupported dialogue retrieval policy version')
        for name, minimum, maximum in (
            ('episode_limit', 1, 256), ('chronology_bands', 1, 256),
            ('neighbor_turns', 0, 4), ('max_context_tokens', 1, 65536),
            ('max_raw_spans', 1, 4096),
        ):
            value = bound_int(getattr(self, name), name, minimum)
            if value > maximum:
                raise ValueError(f'{name} must be <= {maximum}')
        if self.chronology_bands > self.episode_limit:
            raise ValueError('chronology_bands must not exceed episode_limit')
        if type(self.closing_responses) is not bool:
            raise ValueError('closing_responses must be a boolean')
        self._seal()

    @classmethod
    def from_mapping(cls, value):
        if type(value) is not dict:
            raise ValueError('Dialogue retrieval policy must be a JSON object')
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f'Invalid dialogue retrieval policy: {exc}') from exc


@dataclass(frozen=True, slots=True)
class DialogueNeighborhoodRoute(SealedIdentity):
    baseline: SectionRoutePlan
    expanded: SectionRoutePlan
    hierarchy_sha256: str
    base_route_sha256: str
    policy: DialogueRetrievalPolicy
    selected_episode_ids: tuple[str, ...]
    closing_atomic_ids: tuple[str, ...]
    eligible_episodes: int
    selected_chronology_bands: int
    raw_reads_during_routing: int = 0
    query_qwen_passes: int = 0
    exhaustive: bool = False
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ('selected_episode_ids', 'closing_atomic_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        b, e = self.baseline, self.expanded
        delivered = {r.section.section_id: r.section for r in e.routes}
        if (b.query_sha256 != e.query_sha256 or b.eligible_source_ids != e.eligible_source_ids
                or any(delivered.get(r.section.section_id) != r.section for r in b.routes)
                or any(r.section.source_id not in b.eligible_source_ids for r in e.routes)
                or not set(self.closing_atomic_ids) <= delivered.keys()
                or len(self.selected_episode_ids) > self.policy.episode_limit
                or self.raw_reads_during_routing != 0 or self.query_qwen_passes != 0
                or self.exhaustive is not False):
            raise ValueError('Dialogue route changed source scope or evidence identity')
        self._seal()


class PreparedDialogueRetrieval:
    """No raw loader, encoder, Qwen, or reader callback enters this module."""

    def __init__(self, native_router, episode_matrix, *, embedding_identity, policy, turn_positions=None):
        if type(policy) is not DialogueRetrievalPolicy:
            raise TypeError('Expected a validated DialogueRetrievalPolicy')
        self.policy = policy
        self.router = DialogueNeighborhoodRouter(native_router, episode_matrix,
                                                  embedding_identity=embedding_identity)
        from memory_condense.application.hierarchy_context import PreparedHierarchyContext
        self.hierarchy_context = PreparedHierarchyContext(native_router, turn_positions=turn_positions)

    def route(self, base, vector, *, embedding_identity):
        p = self.policy
        plan, coverage = self.router.route(base, vector, embedding_identity=embedding_identity,
            episode_limit=p.episode_limit, chronology_bands=p.chronology_bands,
            closing_responses=p.closing_responses, neighbor_turns=p.neighbor_turns)
        return DialogueNeighborhoodRoute(base.baseline, plan, base.hierarchy_sha256,
            base.receipt_sha256, p, tuple(coverage['selected_episode_ids']),
            tuple(coverage['closing_atomic_ids']), coverage['eligible_episodes'],
            coverage['selected_chronology_bands'])
