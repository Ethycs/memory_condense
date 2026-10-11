"""Experimental dialogue survey over stored summary neighborhoods.

No raw loader or model callback enters routing. Semantic episode ranking,
chronological coverage and role-balanced neighbors produce original addresses;
the usual exact hydrator still enforces the evidence budget. This is an explicit
experimental route, not a replacement for the production cap-eight policy.
"""
from collections import deque
import hashlib

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.section_summary import bound_int


class DialogueNeighborhoodRouter:
    def __init__(self, native_router, episode_matrix, *, embedding_identity):
        self.native = native_router
        self.episodes = tuple(s for s in native_router.hierarchy.sections if not s.child_section_ids)
        matrix = np.asarray(episode_matrix, dtype=np.float32)
        self.dense = ExactDenseAddressIndex([s.section_id for s in self.episodes], matrix)
        if matrix.shape[1] != native_router.semantic._dense._matrix.shape[1]:
            raise ValueError('Dialogue summary dimensions differ from the atomic index')
        self.embedding_identity = embedding_identity
        self.positions = {s.section_id: i for i, s in enumerate(native_router.semantic.sections)}
        self.episode_positions = {s.section_id: min(self.positions[a.section_id]
            for a in native_router.leaf_atoms[s.section_id]) for s in self.episodes}
        self.identity = identity_sha256(dict(format='dialogue-neighborhood-survey-v1',
            hierarchy=native_router.hierarchy.receipt_sha256,
            atomic=native_router.semantic.receipt_sha256,
            episodes=[s.section_id for s in self.episodes],
            matrix=hashlib.sha256(matrix.tobytes()).hexdigest(),embedding_identity=embedding_identity))

    def route(self, base, vector, *, embedding_identity, episode_limit=12, chronology_bands=4,
              closing_responses=False, neighbor_turns=1):
        bound_int(episode_limit, 'episode_limit', 1)
        bound_int(chronology_bands, 'chronology_bands', 1)
        bound_int(neighbor_turns, 'neighbor_turns')
        if neighbor_turns > 4 or type(closing_responses) is not bool:
            raise ValueError('Dialogue survey requires bounded neighbors and boolean closing_responses')
        if embedding_identity != self.embedding_identity:
            raise ValueError('Dialogue survey query encoder changed')
        if base.hierarchy_sha256 != self.native.hierarchy.receipt_sha256:
            raise ValueError('Dialogue survey baseline belongs to another hierarchy')
        if base.baseline.index_sha256 != self.native.semantic.hierarchy.receipt_sha256:
            raise ValueError('Dialogue survey baseline belongs to another atomic index')
        allowed = set(base.baseline.eligible_source_ids)
        episode_scores = self.dense.score_all(vector)
        atomic_scores = self.native.semantic._dense.score_all(vector)
        available = [i for i,s in enumerate(self.episodes) if s.source_id in allowed]
        available.sort(key=lambda i:self.episode_positions[self.episodes[i].section_id])
        ranked = sorted(available, key=lambda i:(-float(episode_scores[i]),i))
        bands = min(chronology_bands,episode_limit,len(available))
        selected = []
        # Cover different parts of the archived dialogue, not just one cluster
        # of near-duplicate tool traces. Coordinates never become relevance text.
        for b in range(bands):
            group=available[b*len(available)//bands:(b+1)*len(available)//bands]
            selected.append(min(group,key=lambda i:(-float(episode_scores[i]),i)))
        selected.extend(i for i in ranked if i not in selected)
        selected=selected[:episode_limit]
        queues=[]
        closure_ids=[]
        for i in selected:
            atoms=self.native.leaf_atoms[self.episodes[i].section_id]
            ranked_atoms=sorted(atoms,key=lambda a:(-float(atomic_scores[self.positions[a.section_id]]),
                                                   self.positions[a.section_id]))
            best=[]
            roles=[]
            for atom in ranked_atoms:
                role=atom.spans[0].role
                if role not in roles:
                    roles.append(role)
                    best.append(atom)
            neighbors=[]
            # Bring the observation immediately before/after a selected action
            # into contention. Preserve the original speaker and source spans.
            turn_order=list(dict.fromkeys(a.spans[0].turn_id for a in atoms))
            for seed in best:
                pos=turn_order.index(seed.spans[0].turn_id)
                nearby=turn_order[max(0,pos-neighbor_turns):pos+neighbor_turns+1]
                for tid in nearby:
                    candidates=[a for a in ranked_atoms if a.spans[0].turn_id==tid]
                    if candidates:
                        neighbors.append(candidates[0])
            queue=[]
            seen=set()
            closing=[]
            if closing_responses:
                # A completed exchange's last assistant turn carries the result
                # more often than its intermediate actions. Only metadata is
                # consulted: do not mistake an open tool cycle for a final reply.
                final=atoms[-1].spans[0]
                if final.role=='assistant':
                    closing=[a for a in atoms if a.spans[0].turn_id==final.turn_id]
                    closure_ids.extend(a.section_id for a in closing)
            for atom in [*closing,*best,*neighbors,*ranked_atoms]:
                if atom.section_id not in seen:
                    seen.add(atom.section_id)
                    queue.append(atom)
            queues.append(deque(queue))
        ordered=[]
        seen=set()
        while any(queues):
            for queue in queues:
                if queue:
                    atom=queue.popleft()
                    if atom.section_id not in seen:
                        seen.add(atom.section_id)
                        ordered.append(atom)
        # Retain baseline candidates even when their neighborhood was not among
        # the survey's selected episodes. Budgeting still happens at hydration.
        for route in base.expanded.routes:
            if route.section.section_id not in seen:
                seen.add(route.section.section_id)
                ordered.append(route.section)
        plan=SectionRoutePlan(identity_sha256(dict(index=self.identity,episode_limit=episode_limit,
            chronology_bands=chronology_bands,closing_responses=closing_responses,
            neighbor_turns=neighbor_turns)),base.baseline.query_sha256,
            tuple(SectionRoute(a,1/(i+1),()) for i,a in enumerate(ordered)),
            max(len(ordered),base.baseline.matched_section_count),base.baseline.eligible_source_ids,
            max(1,len(ordered)),routing_backend='summary_hybrid')
        return plan,dict(policy='dialogue-neighborhood-survey-v1',
            eligible_episodes=len(available),inspected_episode_summaries=len(available),
            selected_episode_ids=[self.episodes[i].section_id for i in selected],
            closing_responses=closing_responses,closing_atomic_ids=closure_ids,neighbor_turns=neighbor_turns,
            selected_chronology_bands=bands,raw_reads_during_routing=0,query_qwen_passes=0,
            exhaustive=False,coverage_note='Survey coverage is not proof of complete factual evidence.')
