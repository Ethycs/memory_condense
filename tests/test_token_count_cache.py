"""Exact bounded count reuse keeps untrusted text and tokenizer semantics."""
from memory_condense.domain import _tokenizer as tokens


def test_counts_are_exact_for_literal_special_tokens_unicode_and_encoding():
    tokens._count_short_text.cache_clear()
    for encoding in ('cl100k_base',):
        for text in ('','café 日本語 🦉','<|endoftext|>','word '*3000):
            expected=len(tokens._get_encoder(encoding).encode(text,disallowed_special=()))
            assert tokens.count_tokens(text,encoding)==expected
            assert tokens.count_tokens(text,encoding)==expected
    info=tokens._count_short_text.cache_info()
    assert info.hits==3 and info.currsize==3 and info.maxsize==8192


def test_short_count_cache_is_bounded_and_large_inputs_are_not_retained(monkeypatch):
    calls=[]
    class Encoder:
        def __init__(self,name): self.name=name
        def encode(self,text,**kwargs):
            assert kwargs=={'disallowed_special':()}
            calls.append(text)
            return text.split() if self.name=='cl100k_base' else list(text)
    monkeypatch.setattr(tokens,'_get_encoder',Encoder)
    tokens._count_short_text.cache_clear()
    try:
        assert tokens.count_tokens('one two')==2
        assert tokens.count_tokens('one two')==2
        assert calls==['one two']
        assert tokens.count_tokens('one two','other-encoding')==7
        assert tokens.count_tokens('one changed three')==3
        large='x '*3000
        assert tokens.count_tokens(large)==tokens.count_tokens(large)==3000
        assert calls.count(large)==2
        for i in range(8300): tokens.count_tokens(str(i))
        assert tokens._count_short_text.cache_info().currsize==8192
    finally:
        tokens._count_short_text.cache_clear()
