"""The admitted FastEmbed scheduler preserves the batch-one vector contract."""
from types import SimpleNamespace

import numpy as np

from memory_condense.runtime.embedding import FastEmbedBGE


class _Tokenizer:
    def encode(self,text):
        return SimpleNamespace(ids=range(int(text)))


class _Model:
    def __init__(self):
        self.model=SimpleNamespace(tokenizer=_Tokenizer())
        self.calls=[]

    def embed(self,texts,*,batch_size):
        self.calls.append((list(texts),batch_size))
        return (np.full(1024,float(text),dtype=np.float32) for text in texts)


def test_fastembed_batches_by_token_length_and_restores_caller_order():
    encoder=FastEmbedBGE.__new__(FastEmbedBGE)
    encoder.model=_Model()
    encoder.preferred_background_batch_size=4
    incoming=['8','1','6','3','7','2']
    vectors=encoder.embed_queries(incoming)
    assert encoder.model.calls==[(['1','2','3','6','7','8'],4)]
    assert vectors.shape==(6,1024)
    assert vectors[:,0].tolist()==[8.,1.,6.,3.,7.,2.]
