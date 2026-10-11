"""Sparse restart-preserving graph filtering over resident summary addresses.

This is the polynomial-filter form of a graph spectral operation: recall never
forms eigenvectors or an N by N attention matrix. Publication compiles a small
directed k-nearest-neighbor graph from normalized summary embeddings. A query
starts from baseline semantic seeds, retains a restart component at each hop,
and mixes the propagated signal with the direct BGE shortlist.
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


ALGORITHM='memory-condense-sparse-spectral-reinjection-v1'


def _array_sha256(value):
    return hashlib.sha256(np.asarray(value,order='C').tobytes(order='C')).hexdigest()


def _softmax(value):
    values=np.asarray(value,dtype=np.float64)
    values=np.exp(values-values.max())
    values/=values.sum()
    return values.astype(np.float32)


@dataclass(frozen=True)
class SpectralReinjectionIndex:
    section_ids: tuple[str,...]
    source_ids: tuple[str,...]
    neighbors: np.ndarray
    weights: np.ndarray
    restart: float
    hops: int
    direct_mix: float
    receipt: object

    def __post_init__(self):
        ids=tuple(self.section_ids); sources=tuple(self.source_ids)
        neighbors=np.asarray(self.neighbors,dtype=np.int32,order='C')
        weights=np.asarray(self.weights,dtype=np.float32,order='C')
        if not ids or len(ids)!=len(set(ids)) or len(sources)!=len(ids):
            raise ValueError('Spectral reinjection addresses must be nonempty and unique')
        if (neighbors.ndim!=2 or neighbors.shape!=weights.shape or neighbors.shape[0]!=len(ids)
                or neighbors.shape[1]<1 or np.any(neighbors<0) or np.any(neighbors>=len(ids))
                or not np.isfinite(weights).all() or np.any(weights<0)
                or not np.allclose(weights.sum(axis=1),1.,atol=2e-5)):
            raise ValueError('Spectral reinjection graph is invalid')
        if not 0<float(self.restart)<=1 or not 0<=float(self.direct_mix)<=1:
            raise ValueError('Spectral mixing coefficients are invalid')
        if type(self.hops) is not int or not 1<=self.hops<=8:
            raise ValueError('Spectral hop count is invalid')
        neighbors.setflags(write=False); weights.setflags(write=False)
        object.__setattr__(self,'section_ids',ids); object.__setattr__(self,'source_ids',sources)
        object.__setattr__(self,'neighbors',neighbors); object.__setattr__(self,'weights',weights)
        object.__setattr__(self,'receipt',MappingProxyType(dict(self.receipt)))

    @classmethod
    def build(cls,section_ids:Sequence[str],source_ids:Sequence[str],embeddings:np.ndarray,*,
              degree:int=8,restart:float=.7,hops:int=2,direct_mix:float=.7,
              block_rows:int=256,provenance:dict|None=None):
        started=time.perf_counter(); ids=tuple(section_ids)
        x=np.asarray(embeddings,dtype=np.float32)
        if (x.ndim!=2 or x.shape[0]!=len(ids) or len(ids)<1 or not np.isfinite(x).all()
                or not np.allclose(np.linalg.norm(x,axis=1),1.,rtol=2e-5,atol=2e-5)):
            raise ValueError('Spectral graph requires aligned normalized embeddings')
        if (type(degree) is not int or degree<1
                or (len(ids)>1 and degree>=len(ids)) or (len(ids)==1 and degree!=1)):
            raise ValueError('Spectral graph degree is invalid')
        if type(block_rows) is not int or block_rows<1:
            raise ValueError('Spectral graph block size is invalid')
        neighbors=np.empty((len(ids),degree),dtype=np.int32)
        weights=np.empty((len(ids),degree),dtype=np.float32)
        for start in range(0,len(ids),block_rows):
            end=min(start+block_rows,len(ids))
            scores=np.asarray(x[start:end]@x.T,dtype=np.float32)
            scores[np.arange(end-start),np.arange(start,end)]=-np.inf
            # Stable sorting uses publication order as the exact tie break.
            order=np.argsort(-scores,axis=1,kind='stable')[:,:degree]
            values=np.maximum(np.take_along_axis(scores,order,axis=1),0.)
            totals=values.sum(axis=1,keepdims=True)
            empty=totals[:,0]<=0
            if np.any(empty):
                values[empty]=1.; totals[empty]=degree
            neighbors[start:end]=order.astype(np.int32)
            weights[start:end]=(values/totals).astype(np.float32)
        state_bytes=int(neighbors.nbytes+weights.nbytes)
        receipt=dict(provenance or {},algorithm=ALGORITHM,node_count=len(ids),degree=degree,
            restart=float(restart),hops=hops,direct_mix=float(direct_mix),
            retained_state_bytes=state_bytes,retained_transformer_token_state_bytes=0,
            graph_edges=int(neighbors.size),all_pairs_state_cells=0,
            publication_similarity_cells=int(len(ids)**2),
            neighbors_sha256=_array_sha256(neighbors),weights_sha256=_array_sha256(weights),
            address_sha256=identity_sha256(dict(section_ids=list(ids),source_ids=list(source_ids))),
            build_s=time.perf_counter()-started)
        receipt['state_sha256']=identity_sha256(receipt)
        return cls(ids,tuple(source_ids),neighbors,weights,float(restart),hops,float(direct_mix),receipt)

    def select(self,seeds:Sequence[tuple[str,float]],*,allowed:Iterable[str],
               candidates:Sequence[tuple[str,float]],limit:int):
        started=time.perf_counter()
        if type(limit) is not int or limit<1:
            raise ValueError('Spectral selection limit must be positive')
        positions={key:i for i,key in enumerate(self.section_ids)}; allowed=set(allowed)
        base=dict(model_passes=0,query_qwen_passes=0,raw_reads=0,
            retained_transformer_token_state_bytes=0,state_sha256=self.receipt['state_sha256'],
            retained_state_bytes=self.receipt['retained_state_bytes'])
        if allowed!=set(self.section_ids):
            return [],dict(base,status='temporal_scope_mismatch',admitted_nodes=len(allowed),
                state_nodes=len(self.section_ids),query_compute_s=time.perf_counter()-started)
        admitted=[(positions[key],float(score)) for key,score in seeds
                  if key in positions and math.isfinite(float(score))]
        if not admitted:
            return [],dict(base,status='no_seed',query_compute_s=time.perf_counter()-started)
        seed=np.zeros(len(self.section_ids),dtype=np.float32)
        seed_weights=_softmax([score for _,score in admitted])
        for (row,_),value in zip(admitted,seed_weights,strict=True): seed[row]+=value
        signal=seed.copy()
        for _ in range(self.hops):
            propagated=np.zeros_like(signal)
            np.add.at(propagated,self.neighbors.reshape(-1),
                (signal[:,None]*self.weights).reshape(-1))
            signal=self.restart*seed+(1.-self.restart)*propagated
        values=(1.-self.direct_mix)*signal
        direct=[(positions[key],float(score)) for key,score in candidates
                if key in positions and math.isfinite(float(score))]
        if direct:
            for (row,_),value in zip(direct,_softmax([score for _,score in direct]),strict=True):
                values[row]+=self.direct_mix*value
        ranked=sorted(((key,float(values[row])) for row,key in enumerate(self.section_ids)
                       if key in allowed and math.isfinite(float(values[row])) and values[row]>0),
                      key=lambda item:(-item[1],item[0]))[:limit]
        cells=self.hops*self.neighbors.size+len(direct)
        return ranked,dict(base,status='selected' if ranked else 'no_candidates',
            seed_count=len(admitted),direct_candidate_count=len(direct),selected_count=len(ranked),
            graph_hops=self.hops,restart=self.restart,direct_mix=self.direct_mix,
            query_compute_cells=int(cells),dense_attention_cells=0,
            query_compute_s=time.perf_counter()-started)


__all__=['ALGORITHM','SpectralReinjectionIndex']
