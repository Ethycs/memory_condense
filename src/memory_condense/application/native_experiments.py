"""Opt-in experimental recall over authenticated summaries and frozen scalar links.

Experiments cannot replace baseline evidence, read transcripts during routing,
or give their own additions independent Hebbian learning credit.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from types import MappingProxyType

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.application.section_retrieval import hydrate_section_plan


MODULES = ('learned_associations', 'heat_diffusion', 'qwen_summary_routing',
           'qwen_associations', 'cav_associations', 'cav_reinjection',
           'spectral_reinjection', 'live_head_memory', 'qwen_consolidation')
GPU_MODULES = frozenset(('qwen_summary_routing','qwen_associations','cav_associations',
                         'live_head_memory','qwen_consolidation'))
SHORTLIST_MODULES = frozenset((*(GPU_MODULES-{'qwen_consolidation'}),'cav_reinjection'))


@dataclass(frozen=True)
class NativeExperimentPolicy(SealedIdentity):
    version: int = 1
    modules: tuple[str, ...] = ()
    candidate_limit: int = 16
    additions: int = 4
    extra_context_tokens: int = 1024
    extra_raw_spans: int = 8
    graph_edge_limit: int = 8192
    min_observations: int = 2
    graph_hops: int = 2
    neighbors_per_node: int = 4
    qwen_layer: int = 4
    workspace_tokens: int = 1024
    max_qwen_passes: int = 12
    cav_bundle: str | None = None
    cav_extraction_temperature: float = .05
    cav_reinjection_temperature: float = .05
    receipt_sha256: str = ''

    def __post_init__(self):
        if not isinstance(self.modules, (list, tuple)):
            raise ValueError('Experiment modules must be a list')
        object.__setattr__(self, 'modules', tuple(self.modules))
        if len(set(self.modules)) != len(self.modules) or any(m not in MODULES for m in self.modules):
            raise ValueError('Unknown or duplicate native experiment')
        for name, low, high in (('version',1,1), ('candidate_limit',2,16), ('additions',1,16),
                ('extra_context_tokens',0,8192), ('extra_raw_spans',0,32),
                ('graph_edge_limit',1,65536), ('min_observations',1,100), ('graph_hops',1,3),
                ('neighbors_per_node',1,16), ('qwen_layer',0,4), ('workspace_tokens',128,2048),
                ('max_qwen_passes',1,32)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid experimental {name}')
        if self.additions > self.candidate_limit:
            raise ValueError('Experiment additions exceed candidate limit')
        if self.cav_bundle is not None and (type(self.cav_bundle) is not str or not self.cav_bundle):
            raise ValueError('CAV bundle must be a filename')
        if {'cav_associations','cav_reinjection'} & set(self.modules) and not self.cav_bundle:
            raise ValueError('CAV experiments require a checkpoint-bound CAV bundle')
        for name in ('cav_extraction_temperature','cav_reinjection_temperature'):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'Invalid experimental {name}')
        self._seal()

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Experiment configuration must be an object')
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError('Unknown experiment configuration field') from exc


@dataclass(frozen=True)
class SummaryGraph:
    """A publication-local graph. No SQLite connection or raw text is retained."""
    adjacency: object
    receipt: object

    @classmethod
    def empty(cls):
        return cls(MappingProxyType({}), MappingProxyType(dict(edges=0, nodes=0)))

    @classmethod
    def capture(cls, app, router, policy):
        from memory_condense.associations.coaccess_graph import coaccess_edge_score
        # Cap scalar state independently of corpus size. Recency truncation is
        # disclosed; unobserved/missing chunks never acquire fabricated edges.
        edges = app._db.execute('SELECT node_low,node_high,coactivation_mass,'
            'coactivation_count,causal_count,last_reinforced_turn FROM consolidation_edges '
            'WHERE coactivation_count>=? OR causal_count>=1 '
            'ORDER BY last_reinforced_turn DESC,node_low,node_high LIMIT ?',
            (policy.min_observations, policy.graph_edge_limit+1)).fetchall()
        truncated = len(edges) > policy.graph_edge_limit
        edges = edges[:policy.graph_edge_limit]
        keys = {key for row in edges for key in row[:2]}
        nodes = {}
        # Keep every query below SQLite's portable variable limit.
        ordered = sorted(keys)
        for start in range(0,len(ordered),400):
            part = ordered[start:start+400]
            placeholders = ','.join('?' for _ in part)
            rows = app._db.execute('SELECT n.node_key,n.access_mass,n.last_access_turn,'
                'c.turn_id,c.start_char,c.end_char FROM consolidation_nodes n '
                'JOIN chunks c ON c.chunk_id=n.chunk_id '
                f'WHERE n.node_key IN ({placeholders})', tuple(part)).fetchall()
            nodes.update((r[0],r) for r in rows)
        by_turn = defaultdict(list)
        for atom in router.atoms.values():
            span = atom.spans[0]
            by_turn[span.turn_id].append(atom)
        atoms = {key: tuple(a.section_id for a in by_turn.get(row[3],())
                    if a.spans[0].start_char < row[5] and a.spans[0].end_char > row[4])
                 for key,row in nodes.items()}
        now = app._db.current_turn()
        pairs = {}
        for low,high,mass,count,causal,turn in edges:
            if low not in nodes or high not in nodes:
                continue
            left,right = nodes[low],nodes[high]
            score = coaccess_edge_score(edge_mass=mass,last_reinforced_turn=turn,
                left_mass=left[1],left_turn=left[2],right_mass=right[1],right_turn=right[2],
                now_turn=now,half_life_turns=200.)
            if score < .05:
                continue
            for a in atoms[low]:
                for b in atoms[high]:
                    if a != b:
                        pair = tuple(sorted((a,b)))
                        pairs[pair] = max(pairs.get(pair,0.),score)
        adjacency = defaultdict(list)
        for (a,b),score in pairs.items():
            adjacency[a].append((b,score)); adjacency[b].append((a,score))
        frozen = {k:tuple(sorted(v,key=lambda x:(-x[1],x[0]))) for k,v in adjacency.items()}
        return cls(MappingProxyType(frozen),MappingProxyType(dict(
            edges=len(pairs),nodes=len(frozen),stored_edges=len(edges),truncated=truncated,
            now_turn=now,sha256=identity_sha256(frozen))))


@dataclass(frozen=True)
class ExperimentSelection:
    plan: SectionRoutePlan
    receipt: dict


@dataclass(frozen=True)
class ExperimentHydration:
    hydration: object
    receipt: dict


class PreparedNativeExperiments:
    """Per-publication immutable inputs; the shared GPU adapter owns serialization."""
    def __init__(self, router, policy, *, graph=None, attention=None, reinjection=None,
                 spectral=None):
        if type(policy) is not NativeExperimentPolicy:
            raise TypeError('Expected NativeExperimentPolicy')
        self.router, self.policy = router, policy
        self.graph = graph or SummaryGraph.empty()
        self.attention = attention
        self.reinjection = reinjection
        self.spectral = spectral

    def select(self, query, vector, routing):
        p = self.policy
        scope = routing.baseline.eligible_source_ids
        allowed = {key for key,a in self.router.atoms.items() if scope is None or a.source_id in scope}
        seeds = [(r.section.section_id,r.score) for r in routing.baseline.routes
                 if r.section.section_id in allowed][:p.candidate_limit]
        # Reuse this turn's BGE query vector and the existing resident address
        # index. The separate shortlist is not allowed to change baseline ranks.
        existing = {r.section.section_id for r in routing.expanded.routes}
        summaries = ()
        summary_scores = {}
        if set(p.modules) & (SHORTLIST_MODULES|{'spectral_reinjection'}):
            shortlist = self.router.semantic.route_vector(query,vector,
                embedding_identity=self.router.semantic.embedding_identity,
                max_sections=p.candidate_limit+len(existing),lexical_reserve=0,eligible_source_ids=scope)
            summary_routes = tuple(r for r in shortlist.routes
                                   if r.section.section_id not in existing)[:p.candidate_limit]
            summaries = tuple(r.section for r in summary_routes)
            summary_scores = {r.section.section_id:r.score for r in summary_routes}
        adjacency = {key:tuple((other,score) for other,score in self.graph.adjacency.get(key,())
                         if other in allowed)[:p.neighbors_per_node] for key in allowed
                         if key in self.graph.adjacency}
        arms, queues, passes = [], [], 0
        for module in p.modules:
            row = dict(module=module,raw_reads=0,model_passes=0)
            scores = []
            if module == 'qwen_consolidation':
                row['status'] = 'learning_only'
            elif module == 'learned_associations':
                weighted = defaultdict(float)
                for rank,(seed,_) in enumerate(seeds,1):
                    for key,score in adjacency.get(seed,()):
                        weighted[key] += score/math.sqrt(rank)
                scores = sorted(weighted.items(),key=lambda x:(-x[1],x[0]))
                row['status'] = 'selected' if scores else 'no_learned_links'
            elif module == 'heat_diffusion':
                from memory_condense.associations.heat_diffusion import diffuse_scalar_heat
                heat = diffuse_scalar_heat(dict(seeds),adjacency,hops=p.graph_hops,
                    neighbors_per_node=p.neighbors_per_node,max_nodes=p.candidate_limit*2)
                scores = [(n.chunk_id,n.heat) for n in heat.nodes if n.hop>0]
                row.update(status='selected' if scores else 'no_learned_links',total_heat=heat.total_heat)
            elif module == 'cav_reinjection':
                if self.reinjection is None:
                    row['status'] = 'unavailable_precomputed_state'
                else:
                    scores,details = self.reinjection.select(seeds,allowed=allowed,
                        candidates=tuple(summary_scores.items()),limit=p.candidate_limit)
                    row.update(details)
            elif module == 'spectral_reinjection':
                if self.spectral is None:
                    row['status'] = 'unavailable_precomputed_state'
                else:
                    scores,details = self.spectral.select(seeds,allowed=allowed,
                        candidates=tuple(summary_scores.items()),
                        limit=p.candidate_limit+len(existing))
                    row.update(details)
            elif not callable(getattr(self.attention,'experiment',None)):
                row['status'] = 'unavailable_attention'
            else:
                scores, details = self.attention.experiment(module,query,summaries,
                    policy=p,remaining_passes=p.max_qwen_passes-passes)
                row.update(details)
                if (type(row['model_passes']) is not int or row['model_passes']<0
                        or row['raw_reads']!=0
                        or row.get('retained_transformer_token_state_bytes',0)!=0):
                    raise ValueError('Experimental model violated summary-only execution bounds')
                passes += row['model_passes']
                if passes > p.max_qwen_passes:
                    raise ValueError('Experiment exceeded the Qwen pass budget')
            seen = set()
            accepted = []
            for key,score in scores:
                if key not in allowed or not math.isfinite(score) or score <= 0:
                    raise ValueError('Experimental proposal escaped its namespace or score bounds')
                if module in SHORTLIST_MODULES and key not in {s.section_id for s in summaries}:
                    raise ValueError('Experimental model selected an uninspected summary')
                if key not in seen:
                    accepted.append((key,float(score))); seen.add(key)
            row['proposed_section_ids'] = [key for key,_ in accepted]
            arms.append(row); queues.append(accepted)
        # Round-robin gives combined arms a bounded, reproducible share. Already
        # proposed baseline atoms are excluded, even when baseline hydration
        # rejected them, so experiment budgets cannot backfill the control.
        seen = {r.section.section_id for r in routing.expanded.routes}
        routes, provenance = [], {}
        while any(queues) and len(routes)<p.additions:
            for row,queue in zip(arms,queues,strict=True):
                while queue and queue[0][0] in seen:
                    queue.pop(0)
                if queue and len(routes)<p.additions:
                    key,score = queue.pop(0)
                    routes.append(SectionRoute(self.router.atoms[key],score,()))
                    provenance[key] = row['module']; seen.add(key)
        plan = SectionRoutePlan(self.router.semantic.hierarchy.receipt_sha256,
            routing.baseline.query_sha256,tuple(routes),len(routes),scope,p.additions,
            routing_backend='summary_hybrid')
        return ExperimentSelection(plan,dict(policy=p.identity_payload(),arms=arms,
            graph=dict(self.graph.receipt),query_qwen_passes=passes,raw_reads_during_routing=0,
            provenance=provenance,plan_sha256=plan.receipt_sha256,
            baseline_preserved=True,learning_credit=False))

    def hydrate(self, selection, baseline, *, load_turn):
        # Whole atoms overlapping delivered baseline spans are skipped. Hydration
        # remains the same exact-source validator, under an independent budget.
        spans = [e.span for s in baseline.sections for e in s.evidence]
        routes = tuple(r for r in selection.plan.routes if not any(
            a.turn_id==b.turn_id and a.start_char<b.end_char and a.end_char>b.start_char
            for a in r.section.spans for b in spans))
        from dataclasses import replace
        plan = replace(selection.plan,routes=routes,receipt_sha256='')
        hydration = hydrate_section_plan(plan,load_turn=load_turn,
            max_context_tokens=self.policy.extra_context_tokens,max_raw_spans=self.policy.extra_raw_spans)
        receipt = dict(selection.receipt,hydration=hydration.identity_payload(),
            delivered_section_ids=[s.section.section_id for s in hydration.sections])
        return ExperimentHydration(hydration,receipt)
