"""Package boundaries, asset setup, and installed-command behavior."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import httpx

import pytest

from memory_condense.runtime.config import RuntimeAssets, config_path
from memory_condense.runtime import setup as provision


def test_runtime_imports_without_research_tools_or_working_directory(tmp_path):
    source = Path(__file__).resolve().parents[1]/'src'
    script = '''
import sys, importlib.abc
sys.path.insert(0, sys.argv[1])
class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tools','tests'):
            raise AssertionError('Shipping runtime imports '+fullname)
sys.meta_path.insert(0, NoResearch())
from memory_condense.runtime.sessions import NativeProxySessions
from memory_condense.runtime.embedding import FastEmbedBGE
from memory_condense.interfaces.proxy_server import build_app
from memory_condense.runtime.artifacts import read
from pathlib import Path
import memory_condense.runtime.embedding as embedding
assert read(Path(embedding.__file__).parent/'data/bge-admission/admission.json')['files']
print('runtime-import-ok')
'''
    result = subprocess.run([sys.executable,'-I','-c',script,str(source)],cwd=tmp_path,
                            capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert 'runtime-import-ok' in result.stdout


def test_installed_command_enables_memory_and_allows_explicit_observe(monkeypatch):
    from memory_condense.cli import main
    from memory_condense.interfaces import proxy_server
    calls=[]
    monkeypatch.setattr(proxy_server,'main',lambda args:calls.append(args) or 0)
    assert main(['proxy','--mode','observe','--port','8899'])==0
    assert calls[0][:2]==['--mode','augment']
    assert calls[0][-4:]==['--mode','observe','--port','8899']


def test_asset_paths_are_independent_of_cwd_and_explicit_override_wins(tmp_path,monkeypatch):
    config=tmp_path/'settings.json'
    root=tmp_path/'existing models'
    config.write_text(json.dumps({'assets_dir':str(root)}),encoding='utf-8')
    monkeypatch.setenv('MEMORY_CONDENSE_CONFIG',str(config))
    monkeypatch.delenv('MEMORY_CONDENSE_ASSETS',raising=False)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert RuntimeAssets.resolve().qwen==root/'models/Qwen3-8B'
    assert RuntimeAssets.resolve(tmp_path/'override').root==tmp_path/'override'


def test_setup_reuse_verifies_before_saving_and_never_downloads(tmp_path,monkeypatch):
    root=tmp_path/'assets';root.mkdir()
    sentinel=root/'model';sentinel.write_bytes(b'pinned-model')
    model_source=provision.read(provision.DATA/'bge-admission/plan.json')['model']
    bge=root/'models/bge-m3-onnx-fp32';bge.mkdir(parents=True)
    (bge/'source.json').write_text(json.dumps(model_source),encoding='utf-8')
    monkeypatch.setenv('MEMORY_CONDENSE_CONFIG',str(tmp_path/'config.json'))
    monkeypatch.setattr(provision.platform,'system',lambda:'Windows')
    monkeypatch.setattr(provision.platform,'machine',lambda:'AMD64')
    monkeypatch.setattr(RuntimeAssets,'require_files',lambda self:None)
    monkeypatch.setattr(provision,'manifests',lambda assets:[(root,{'model':hashlib.sha256(b'pinned-model').hexdigest()})])
    monkeypatch.setattr(provision,'download_assets',lambda _:pytest.fail('reuse must not download'))
    assert provision.setup(reuse=root)==0
    before=config_path().read_bytes()
    sentinel.write_bytes(b'changed-model')
    with pytest.raises(ValueError,match='differs'):
        provision.setup(reuse=root)
    assert config_path().read_bytes()==before


@pytest.mark.parametrize('name',['../escape.dll','/escape.dll','C:/escape.dll','..\\escape.dll'])
def test_runtime_archive_rejects_traversal_before_extracting(tmp_path,name):
    archive=tmp_path/'runtime.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('ordinary.dll',b'valid')
        z.writestr(name,b'invalid')
    with pytest.raises(ValueError,match='Unsafe'):
        provision.extract_archive(archive,tmp_path/'destination')
    assert not (tmp_path/'destination/ordinary.dll').exists()


def test_required_assets_error_explains_setup(tmp_path):
    with pytest.raises(FileNotFoundError,match='memory-condense setup'):
        RuntimeAssets.resolve(tmp_path).require_files()


def test_cold_asset_download_uses_pins_verifies_archives_and_reuses_valid_files(tmp_path,monkeypatch):
    import huggingface_hub
    import truststore
    assets=RuntimeAssets.resolve(tmp_path/'assets')
    metadata=tmp_path/'metadata';metadata.mkdir()
    data=b'pinned'
    digest=hashlib.sha256(data).hexdigest()
    groups=[(assets.qwen,{'config.json':digest}),(assets.bge,{'onnx/model.onnx':digest}),
            (assets.llama_model.parent,{assets.llama_model.name:digest})]
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as bundle: bundle.writestr('llama-server.exe',data)
    archive=blob.getvalue()
    manifest=[dict(name='runtime.zip',url='https://example.test/runtime.zip',
                   sha256=hashlib.sha256(archive).hexdigest(),files={'llama-server.exe':digest})]
    (metadata/'llama-runtime.json').write_text(json.dumps(manifest),encoding='utf-8')
    monkeypatch.setattr(provision,'DATA',metadata)
    monkeypatch.setattr(provision,'manifests',lambda _:groups)
    monkeypatch.setattr(provision,'read',lambda _:dict(model=dict(repo='test/bge',revision='pinned-bge')))
    monkeypatch.setattr(truststore,'inject_into_ssl',lambda:None)
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs['revision'] and kwargs['allow_patterns']
        for name in kwargs['allow_patterns']:
            path=Path(kwargs['local_dir'])/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    original_client=httpx.Client
    remote=[]
    def get(request):
        remote.append(str(request.url));return httpx.Response(200,content=archive)
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original_client(transport=httpx.MockTransport(get)))
    provision.download_assets(assets)
    assert len(calls)==3 and len(remote)==1
    assert (assets.root/'runtimes/runtime/llama-server.exe').read_bytes()==data
    provision.download_assets(assets)
    assert len(calls)==3 and len(remote)==1
