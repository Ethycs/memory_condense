"""Explicit packaged assets reuse BGE without changing standalone MCP defaults."""
import pytest

pytest.importorskip('mcp')

from memory_condense.interfaces import mcp_server
from memory_condense.runtime import embedding


@pytest.mark.parametrize('configured',[False,True])
def test_mcp_assets_are_explicit_and_do_not_download_another_checkpoint(tmp_path,monkeypatch,configured):
    calls=[]
    sentinel=object()
    if configured:
        monkeypatch.setenv('MEMORY_CONDENSE_ASSETS',str(tmp_path/'assets'))
    else:
        monkeypatch.delenv('MEMORY_CONDENSE_ASSETS',raising=False)
    monkeypatch.setenv('MEMORY_CONDENSE_DATA_DIR',str(tmp_path/'memory'))
    monkeypatch.setattr(mcp_server,'_condenser',None)
    monkeypatch.setattr(mcp_server,'resolve_extractor',lambda:(None,'test'))
    monkeypatch.setattr(embedding,'FastEmbedBGE',lambda **kwargs: sentinel)
    monkeypatch.setattr(mcp_server,'MemoryCondenser',lambda **kwargs:calls.append(kwargs) or object())
    result=mcp_server._condense()
    assert mcp_server._condense() is result
    assert len(calls)==1
    assert calls[0]['data_dir']==tmp_path/'memory'
    assert calls[0]['embedder'] is (sentinel if configured else None)
