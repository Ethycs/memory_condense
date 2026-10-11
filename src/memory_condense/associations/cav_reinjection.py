"""Query-free low-rank CAV reinjection over precomputed summary features.

The expensive Qwen residuals are produced once when a summary is published.
For N summaries and K fixed concepts, publication retains only the rectangular
extraction and reinjection matrices::

    E = softmax(C0_hat @ X_hat.T / tau_e)       # [K, N]
    C1 = E @ X                                   # [K, D]
    R = softmax(X_hat @ C1_hat.T / tau_r)       # [N, K]

At recall, the existing BGE results seed evidence nodes.  Relevance crosses the
two retained matrices as ``R @ (E @ seed)``.  The recall path does not retain X,
run Qwen, or construct an N by N evidence graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from types import MappingProxyType
from typing import Iterable, Sequence

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256


ALGORITHM = 'memory-condense-precomputed-cav-reinjection-v1'


def _matrix_sha256(value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype='<f4', order='C')
    return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    result = np.exp(shifted)
    result /= np.sum(result, axis=axis, keepdims=True)
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return result


@dataclass(frozen=True)
class CAVReinjectionIndex:
    """Compact publication-local CAV state with a CPU-only recall readout."""

    section_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    extraction: np.ndarray
    reinjection: np.ndarray
    receipt: object

    def __post_init__(self):
        section_ids = tuple(self.section_ids)
        source_ids = tuple(self.source_ids)
        if (not section_ids or len(section_ids) != len(set(section_ids))
                or len(source_ids) != len(section_ids)):
            raise ValueError('CAV reinjection addresses must be nonempty and unique')
        extraction = np.asarray(self.extraction, dtype=np.float32, order='C')
        reinjection = np.asarray(self.reinjection, dtype=np.float32, order='C')
        concepts, nodes = extraction.shape
        if concepts < 1 or reinjection.shape != (nodes, concepts) or nodes != len(section_ids):
            raise ValueError('CAV extraction/reinjection shapes disagree with addresses')
        if (not np.isfinite(extraction).all() or not np.isfinite(reinjection).all()
                or not np.allclose(extraction.sum(axis=1), 1., atol=2e-5)
                or not np.allclose(reinjection.sum(axis=1), 1., atol=2e-5)):
            raise ValueError('CAV reinjection matrices are invalid')
        extraction.setflags(write=False)
        reinjection.setflags(write=False)
        object.__setattr__(self, 'section_ids', section_ids)
        object.__setattr__(self, 'source_ids', source_ids)
        object.__setattr__(self, 'extraction', extraction)
        object.__setattr__(self, 'reinjection', reinjection)
        object.__setattr__(self, 'receipt', MappingProxyType(dict(self.receipt)))

    @classmethod
    def build(cls, section_ids: Sequence[str], source_ids: Sequence[str],
              node_features: np.ndarray, cav_vectors: np.ndarray, *,
              extraction_temperature: float, reinjection_temperature: float,
              provenance: dict) -> 'CAVReinjectionIndex':
        """Compile Qwen features into two compact matrices and release X/C1."""
        tau_e = _positive_float(extraction_temperature, 'extraction_temperature')
        tau_r = _positive_float(reinjection_temperature, 'reinjection_temperature')
        x = np.asarray(node_features, dtype=np.float32)
        cav = np.asarray(cav_vectors, dtype=np.float32)
        if (x.ndim != 2 or cav.ndim != 2 or not len(x)
                or x.shape[1] != cav.shape[1] or not len(cav)):
            raise ValueError('CAV features must have compatible [N,D] and [K,D] shapes')
        if not np.isfinite(x).all() or not np.isfinite(cav).all():
            raise ValueError('CAV features contain nonfinite values')
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        cav_norm = np.linalg.norm(cav, axis=1, keepdims=True)
        if np.any(x_norm <= 0) or np.any(cav_norm <= 0):
            raise ValueError('CAV features must have nonzero rows')
        x_keys = x / x_norm
        cav_keys = cav / cav_norm
        extraction = _softmax((cav_keys @ x_keys.T) / tau_e, axis=1)
        concepts = extraction @ x
        concept_norm = np.linalg.norm(concepts, axis=1, keepdims=True)
        if np.any(concept_norm <= 0):
            raise ValueError('CAV extraction produced an empty concept')
        reinjection = _softmax((x_keys @ (concepts / concept_norm).T) / tau_r, axis=1)
        extraction = np.asarray(extraction, dtype=np.float32, order='C')
        reinjection = np.asarray(reinjection, dtype=np.float32, order='C')
        receipt = dict(provenance, algorithm=ALGORITHM, node_count=int(x.shape[0]),
            concept_count=int(cav.shape[0]), hidden_dim=int(x.shape[1]),
            extraction_temperature=tau_e, reinjection_temperature=tau_r,
            extraction_sha256=_matrix_sha256(extraction),
            reinjection_sha256=_matrix_sha256(reinjection),
            retained_state_bytes=int(extraction.nbytes + reinjection.nbytes),
            retained_transformer_token_state_bytes=0,
            all_pairs_attention_cells=0,
            low_rank_state_cells=int(extraction.size + reinjection.size),
            address_sha256=identity_sha256(dict(section_ids=list(section_ids),source_ids=list(source_ids))))
        receipt['state_sha256'] = identity_sha256(receipt)
        return cls(tuple(section_ids),tuple(source_ids),extraction,reinjection,receipt)

    def select(self, seeds: Sequence[tuple[str, float]], *, allowed: Iterable[str],
               candidates: Sequence[tuple[str,float]], limit: int):
        """Propagate BGE seed relevance through E then R without a model call."""
        started = time.perf_counter()
        if type(limit) is not int or limit < 1:
            raise ValueError('CAV reinjection selection limit must be positive')
        positions = {key:i for i,key in enumerate(self.section_ids)}
        allowed = set(allowed)
        # C1 was compiled from this exact publication. A historical as-of query
        # must not observe later summaries indirectly through the concept pool.
        # Prefix-specific states can be added later; until then fail closed.
        if allowed != set(self.section_ids):
            return [],dict(status='temporal_scope_mismatch',model_passes=0,raw_reads=0,
                retained_transformer_token_state_bytes=0,query_qwen_passes=0,
                state_sha256=self.receipt['state_sha256'],
                retained_state_bytes=self.receipt['retained_state_bytes'],
                admitted_nodes=len(allowed),state_nodes=len(self.section_ids),
                query_compute_s=time.perf_counter()-started)
        admitted = [(positions[key],float(score)) for key,score in seeds
                    if key in positions and math.isfinite(float(score))]
        candidate_rows = [(positions[key],key,float(score)) for key,score in candidates
                          if key in positions and key in allowed and math.isfinite(float(score))]
        base = dict(status='no_seed',model_passes=0,raw_reads=0,
            retained_transformer_token_state_bytes=0,query_qwen_passes=0,
            state_sha256=self.receipt['state_sha256'],
            retained_state_bytes=self.receipt['retained_state_bytes'])
        if not admitted:
            return [],dict(base,query_compute_s=time.perf_counter()-started)
        if not candidate_rows:
            return [],dict(base,status='no_candidates',query_compute_s=time.perf_counter()-started)
        logits = np.asarray([score for _,score in admitted],dtype=np.float64)
        weights = _softmax(logits,axis=0).astype(np.float32)
        seed_positions = np.asarray([position for position,_ in admitted],dtype=np.int64)
        concept = self.extraction[:,seed_positions] @ weights
        total = float(concept.sum())
        if not math.isfinite(total) or total <= 0:
            return [],dict(base,status='empty_concept',query_compute_s=time.perf_counter()-started)
        concept /= total
        candidate_positions = np.asarray([position for position,_,_ in candidate_rows],dtype=np.int64)
        cav_values = self.reinjection[candidate_positions] @ concept
        cav_values /= max(float(cav_values.sum()),1e-12)
        semantic_logits = np.asarray([score for _,_,score in candidate_rows],dtype=np.float64)
        semantic_values = _softmax(semantic_logits,axis=0).astype(np.float32)
        values = np.sqrt(cav_values*semantic_values)
        ranked = sorted(((key,float(values[i])) for i,(_,key,_) in enumerate(candidate_rows)
                         if math.isfinite(float(values[i])) and values[i] > 0),
                        key=lambda item:(-item[1],item[0]))[:limit]
        cells = self.extraction.shape[0]*(len(admitted)+len(candidate_rows))
        return ranked,dict(base,status='selected' if ranked else 'no_candidates',
            seed_count=len(admitted),candidate_count=len(candidate_rows),selected_count=len(ranked),
            semantic_fusion='geometric_mean',query_compute_cells=int(cells),
            dense_attention_cells=int(len(self.section_ids)**2),
            compute_cell_ratio=cells/max(1,len(self.section_ids)**2),
            query_compute_s=time.perf_counter()-started)


__all__ = ['ALGORITHM','CAVReinjectionIndex']
