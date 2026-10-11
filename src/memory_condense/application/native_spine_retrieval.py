"""One live query embedding followed by exact native evidence hydration."""
from dataclasses import asdict, dataclass

from memory_condense.application.dialogue_retrieval import DialogueNeighborhoodRoute, PreparedDialogueRetrieval
from memory_condense.application.section_retrieval import SectionRetrievalResult, hydrate_section_plan
from memory_condense.search.native_spine_routing import NativeSpineRoute, NativeSpineRouter
from memory_condense.search.summary_semantic_index import summary_embedding_identity, compatible_summary_embedding
from memory_condense.search.summary_time_prior_v2 import question_day


@dataclass(frozen=True)
class NativeSpineRetrieval:
    routing: NativeSpineRoute | DialogueNeighborhoodRoute
    hydration: SectionRetrievalResult
    live_query_embedding: bool = True
    tactic: dict | None = None
    hierarchy_context: object | None = None
    experiments: object | None = None


class ResidentNativeSpineMemory:
    """The caller owns the resident encoder and the cold-admitted namespace."""

    def __init__(self, atomic_semantic, hierarchy, *, encoder, load_turn):
        self.router = NativeSpineRouter(atomic_semantic, hierarchy)
        if not compatible_summary_embedding(encoder, atomic_semantic.embedding_identity):
            raise ValueError("native query encoder differs from the stored summary vectors")
        self.encoder, self.load_turn = encoder, load_turn
        self._query_embedding_identity = summary_embedding_identity(encoder)
        self.dialogue = None
        self.tactic_classifier = None
        self.include_hierarchy = False
        self.forced_tactic = None
        self.experiments = None

    def configure_tactics(self, classifier):
        if self.dialogue is None:
            raise ValueError('Automatic tactics require a prepared dialogue index')
        if classifier.encoder is not self.encoder:
            raise ValueError('Tactic classifier must share the resident query encoder')
        self.tactic_classifier = classifier
        self.include_hierarchy = True

    def configure_dialogue(self, policy, episode_matrix, *, embedding_identity, turn_positions=None):
        """Prepare off the recall path, after the final native router is installed."""
        if embedding_identity != self._query_embedding_identity:
            raise ValueError('Dialogue summary vectors differ from the live query encoder')
        self.dialogue = PreparedDialogueRetrieval(self.router, episode_matrix,
            embedding_identity=embedding_identity, policy=policy, turn_positions=turn_positions)

    def retrieve(self, query, dated_question, *, augment=True, max_context_tokens=None,
                 max_raw_spans=None, recent=(), **routing_limits):
        question_day(query, dated_question)
        identity = summary_embedding_identity(self.encoder)
        if identity != self._query_embedding_identity:
            raise ValueError("native query encoder changed")
        if self.dialogue is not None and self.dialogue.router.native is not self.router:
            raise ValueError('Dialogue index belongs to another native snapshot')
        vector = self.encoder.embed_query(query)
        if summary_embedding_identity(self.encoder) != identity:
            raise ValueError("native query encoder changed during live encoding")
        tactic = (dict(selected_tactic=self.forced_tactic, reason='forced')
                  if augment and self.forced_tactic is not None else None)
        use_dialogue = self.dialogue is not None and augment
        if self.tactic_classifier is not None and augment:
            from memory_condense.application.tactic_router import render_input
            from memory_condense.domain._discourse_identity import quote_sha256
            from memory_condense.domain._tokenizer import count_tokens
            if count_tokens(query.strip()) > 512:
                tactic = dict(preferred_tactic=None, abstained=True, reason='query_budget',
                              selected_tactic='cap8')
            else:
                text,_ = render_input(query)
                bound = dict(vector=vector, text_sha256=quote_sha256(query), embedding_identity=identity)
                decision = self.tactic_classifier.prioritize(query, input_vector=bound if text==query else None)
                initial = asdict(decision)
                # Clear requests reuse the retrieval vector. Only ambiguity
                # pays for an additional embedding of recent conversation.
                if decision.abstained and recent:
                    decision = self.tactic_classifier.prioritize(query, recent=recent)
                tactic = dict(asdict(decision), initial_decision=initial,
                              selected_tactic=decision.preferred_tactic or 'cap8')
            use_dialogue = tactic['selected_tactic'] in ('hierarchy_neighborhoods', 'decision_history')
        policy = self.dialogue.policy if use_dialogue else None
        if max_context_tokens is None:
            from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
            focused_budget = 2048 if isinstance(self.router, NativeSpineUserCompletionRouter) else 3072
            max_context_tokens = policy.max_context_tokens if policy else focused_budget
        if max_raw_spans is None:
            max_raw_spans = policy.max_raw_spans if policy else 128
        # Admission above authenticates the opt-in device transition. The
        # router's namespace retains the identity of its stored vectors.
        routing = self.router.route_vector(query, dated_question, vector,
                                            embedding_identity=self.router.semantic.embedding_identity,
                                            **routing_limits)
        if policy is not None:
            routing = self.dialogue.route(routing, vector, embedding_identity=identity)
        experiment = (self.experiments.select(query,vector,routing)
                      if self.experiments is not None and augment else None)
        hydration = hydrate_section_plan(routing.expanded if augment else routing.baseline,
            load_turn=self.load_turn, max_context_tokens=max_context_tokens, max_raw_spans=max_raw_spans)
        hierarchy_context = (self.dialogue.hierarchy_context.render(
            routing.baseline.eligible_source_ids, routing.selected_episode_ids,
            max_tokens=24576) if policy and self.include_hierarchy else None)
        additions = (self.experiments.hydrate(experiment,hydration,load_turn=self.load_turn)
                     if experiment is not None else None)
        return NativeSpineRetrieval(routing, hydration, tactic=tactic,
                                   hierarchy_context=hierarchy_context,experiments=additions)
