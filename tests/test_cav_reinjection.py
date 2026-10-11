import numpy as np
import pytest

from memory_condense.associations.cav_reinjection import CAVReinjectionIndex


def test_precomputed_reinjection_is_rectangular_and_query_free():
    features=np.asarray([[1.,0.,0.],[.9,.1,0.],[0.,1.,0.],[0.,.9,.1]],dtype=np.float32)
    cavs=np.asarray([[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    index=CAVReinjectionIndex.build(('a','b','c','d'),('s','s','s','s'),features,cavs,
        extraction_temperature=.2,reinjection_temperature=.2,
        provenance=dict(bundle_sha256='b'*64,checkpoint_sha256='c'*64,layer=4,
                        concept_names=['first','second'],build_qwen_batches=1))
    assert index.extraction.shape==(2,4)
    assert index.reinjection.shape==(4,2)
    assert index.receipt['low_rank_state_cells']==16
    assert index.receipt['all_pairs_attention_cells']==0
    assert index.receipt['retained_transformer_token_state_bytes']==0
    assert index.receipt['retained_state_bytes']==64
    assert np.allclose(index.extraction.sum(axis=1),1.)
    assert np.allclose(index.reinjection.sum(axis=1),1.)
    with pytest.raises(ValueError):
        index.extraction[0,0]=0
    scores,receipt=index.select((('a',.8),),allowed={'a','b','c','d'},
                                candidates=(('a',.9),('b',.8),('c',.1),('d',0.)),limit=4)
    assert [key for key,_ in scores][:2]==['a','b']
    assert receipt['model_passes']==receipt['query_qwen_passes']==0
    assert receipt['raw_reads']==0
    assert receipt['query_compute_cells']==10
    assert receipt['dense_attention_cells']==16
    assert receipt['compute_cell_ratio']==pytest.approx(.625)


def test_reinjection_fails_closed_when_as_of_scope_differs_from_compiled_state():
    features=np.eye(3,dtype=np.float32)
    cavs=np.asarray([[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    index=CAVReinjectionIndex.build(('a','b','future'),('old','old','future'),features,cavs,
        extraction_temperature=1.,reinjection_temperature=1.,provenance={})
    first,one=index.select((('a',1.),),allowed={'a','b'},candidates=(('a',1.),('b',.5)),limit=3)
    second,two=index.select((('a',1.),),allowed={'a','b'},candidates=(('a',1.),('b',.5)),limit=3)
    assert first==second
    assert {k:v for k,v in one.items() if k!='query_compute_s'}=={
        k:v for k,v in two.items() if k!='query_compute_s'}
    assert first==[]
    assert one['status']=='temporal_scope_mismatch'
    assert one['admitted_nodes']==2 and one['state_nodes']==3
