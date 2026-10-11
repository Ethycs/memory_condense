import numpy as np
import pytest

from memory_condense.associations.spectral_reinjection import SpectralReinjectionIndex


def test_sparse_spectral_reinjection_preserves_restart_and_has_no_query_model():
    matrix=np.asarray([[1.,0.],[.99,.1],[0.,1.],[.1,.99]],dtype=np.float32)
    matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
    index=SpectralReinjectionIndex.build(('a','b','c','d'),('s','s','s','s'),matrix,
        degree=1,restart=.7,hops=2,direct_mix=0.,provenance={'test':True})
    assert index.neighbors.shape==(4,1)
    assert index.receipt['retained_state_bytes']==32
    assert index.receipt['all_pairs_state_cells']==0
    scores,receipt=index.select((('a',1.),),allowed={'a','b','c','d'},candidates=(),limit=4)
    assert [key for key,_ in scores][:2]==['a','b']
    assert receipt['query_qwen_passes']==receipt['model_passes']==0
    assert receipt['dense_attention_cells']==0
    with pytest.raises(ValueError): index.weights[0,0]=0


def test_sparse_spectral_reinjection_fails_closed_for_historical_scope():
    matrix=np.eye(3,dtype=np.float32)
    index=SpectralReinjectionIndex.build(('a','b','future'),('old','old','future'),matrix,
        degree=1)
    scores,receipt=index.select((('a',1.),),allowed={'a','b'},candidates=(),limit=3)
    assert scores==[] and receipt['status']=='temporal_scope_mismatch'


def test_sparse_spectral_reinjection_admits_single_summary_publication():
    index=SpectralReinjectionIndex.build(('only',),('source',),np.asarray([[1.,0.]],dtype=np.float32),
        degree=1)
    scores,receipt=index.select((('only',1.),),allowed={'only'},candidates=(),limit=1)
    assert scores[0][0]=='only' and receipt['status']=='selected'
