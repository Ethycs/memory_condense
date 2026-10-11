"""Package boundaries, asset setup, and installed-command behavior."""
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import subprocess
import os
import sys
import tarfile
import zipfile
import httpx

import pytest

from memory_condense.runtime.config import LLAMA_RUNTIME, RuntimeAssets, config_path
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
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy, PreparedDialogueRetrieval
assert DialogueRetrievalPolicy().max_context_tokens == 6144
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


def test_proxy_cli_wires_dialogue_profile_to_each_native_session(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import uvicorn
    from memory_condense.cli import main
    from memory_condense.interfaces import proxy_server
    from memory_condense.runtime import sessions
    from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
    policy = DialogueRetrievalPolicy(episode_limit=8, max_context_tokens=4096)
    path = tmp_path/'dialogue.json'
    path.write_text(json.dumps(policy.identity_payload()), encoding='utf-8')
    monkeypatch.setattr(RuntimeAssets, 'require_files', lambda *a, **k: None)
    monkeypatch.setattr(sessions, 'LocalRuntime', lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sessions, 'ChatSession', lambda path, name, backend, **k: SimpleNamespace(backend=backend))
    observed = []
    def serve(config, memory):
        assert memory.status()['retrieval'] == dict(mode='dialogue', policy=policy.identity_payload())
        for name in ('a', 'b'):
            session = memory.factory(name, tmp_path/name)
            assert session.backend.dialogue_policy == policy
        observed.append(memory.factory)
        return object()
    monkeypatch.setattr(proxy_server, 'build_app', serve)
    monkeypatch.setattr(uvicorn, 'run', lambda *a, **k: None)
    assert main(['proxy', '--data-dir', str(tmp_path/'service'), '--memory-retrieval-policy', 'dialogue',
                 '--memory-dialogue-config', str(path)]) == 0
    assert len(observed) == 1


@pytest.mark.parametrize('arguments', [
    ['--memory-dialogue-config', 'unused.json'],
    ['--mode', 'observe', '--memory-retrieval-policy', 'dialogue'],
])
def test_invalid_dialogue_cli_never_loads_models(monkeypatch, arguments):
    from memory_condense.cli import main
    monkeypatch.setattr(RuntimeAssets, 'require_files', lambda *a, **k: pytest.fail('Models must not start'))
    with pytest.raises(SystemExit) as error:
        main(['proxy', *arguments])
    assert error.value.code == 2


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


def test_windows_setup_uncompresses_only_present_compressed_mmap_models(tmp_path,monkeypatch):
    assets=RuntimeAssets(tmp_path/'assets',None,'win-amd64')
    from memory_condense.modeling.qwen_prefix import FIRST_SHARD
    targets=(assets.qwen/FIRST_SHARD,assets.bge/'onnx/model.onnx',
             assets.bge/'onnx/model.onnx_data',assets.llama_model)
    for path in targets:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'model')
    compressed={str(path):True for path in targets}
    calls=[]
    monkeypatch.setattr(provision,'windows_file_is_compressed',
                        lambda path:compressed[str(path)])
    def run(command,check):
        assert command[:4]==['compact.exe','/u','/i','/q'] and check is True
        calls.append(command[-1]);compressed[command[-1]]=False
    monkeypatch.setattr(provision.subprocess,'run',run)
    assert provision.prepare_memory_mapped_models(assets,windows=True)==4
    assert calls==[str(path) for path in targets]
    assert provision.prepare_memory_mapped_models(assets,windows=True)==0


def _tarball(path, entries):
    """entries: (name, mode, payload) with payload None meaning a directory."""
    with tarfile.open(path, 'w:gz') as bundle:
        for name, mode, payload in entries:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if payload is None:
                info.type = tarfile.DIRTYPE
                bundle.addfile(info)
            else:
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))


def test_runtime_tarball_strips_one_top_level_directory_and_restores_the_exec_bit(tmp_path):
    archive = tmp_path/'llama-b11272-bin-ubuntu-cuda-12.8-x64.tar.gz'
    _tarball(archive, [
        ('llama-b11272/', 0o755, None),
        ('llama-b11272/llama-server', 0o755, b'server'),
        ('llama-b11272/libggml-cuda.so', 0o755, b'cuda'),
        ('llama-b11272/LICENSE', 0o644, b'license'),
        ('llama-b11272/sub/extra.so', 0o755, b'extra'),
    ])
    destination = tmp_path/'runtimes'/provision.archive_stem(archive.name)
    provision.extract_archive(archive, destination)
    # The upstream top-level component is unwrapped; nesting below it survives.
    assert (destination/'llama-server').read_bytes() == b'server'
    assert (destination/'LICENSE').read_bytes() == b'license'
    assert (destination/'sub/extra.so').read_bytes() == b'extra'
    assert not (destination/'llama-b11272').exists()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits')
def test_runtime_tarball_extraction_marks_only_executables_executable(tmp_path):
    archive = tmp_path/'runtime.tar.gz'
    _tarball(archive, [('top/', 0o755, None), ('top/llama-server', 0o755, b'server'),
                       ('top/LICENSE', 0o644, b'license')])
    destination = tmp_path/'out'
    provision.extract_archive(archive, destination)
    assert os.access(destination/'llama-server', os.X_OK)
    assert not os.access(destination/'LICENSE', os.X_OK)


@pytest.mark.parametrize('member', [
    ('/escape.so', 0o644, b'bad'),
    ('../escape.so', 0o644, b'bad'),
    ('top/../../escape.so', 0o644, b'bad'),
    ('C:/escape.so', 0o644, b'bad'),
])
def test_runtime_tarball_rejects_unsafe_paths_before_extracting(tmp_path, member):
    archive = tmp_path/'runtime.tar.gz'
    _tarball(archive, [('top/ordinary.so', 0o644, b'valid'), member])
    with pytest.raises(ValueError, match='Unsafe'):
        provision.extract_archive(archive, tmp_path/'destination')
    assert not (tmp_path/'destination/ordinary.so').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE,
                                  tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_runtime_tarball_rejects_non_regular_members_before_extracting(tmp_path, kind):
    """Hardlinks, devices and FIFOs are threats zip never had; a hardlink target
    is resolved by the kernel and would bypass the path check entirely."""
    archive = tmp_path/'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        ordinary = tarfile.TarInfo('top/ordinary.so')
        ordinary.size = 5
        bundle.addfile(ordinary, io.BytesIO(b'valid'))
        hostile = tarfile.TarInfo('top/hostile')
        hostile.type = kind
        hostile.linkname = '/etc/passwd'
        bundle.addfile(hostile)
    with pytest.raises(ValueError, match='Unsafe'):
        provision.extract_archive(archive, tmp_path/'destination')
    assert not (tmp_path/'destination/ordinary.so').exists()


@pytest.mark.parametrize('names', [['a.dll', 'b.dll'], ['only.dll']])
def test_flat_zip_archives_are_extracted_without_stripping(tmp_path, names):
    """Regression lock for the Windows half: nothing may be unwrapped there."""
    archive = tmp_path/'runtime.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name in names:
            bundle.writestr(name, b'payload')
    destination = tmp_path/'out'
    provision.extract_archive(archive, destination)
    for name in names:
        assert (destination/name).read_bytes() == b'payload'


def test_archive_names_cover_both_pinned_suffixes_and_fail_closed():
    assert provision.archive_stem('a.zip') == 'a'
    assert provision.archive_stem('a.tar.gz') == 'a'
    assert provision.archive_suffix('a.tar.gz') == '.tar.gz'
    for rejected in ('a.tar.xz', 'a.7z', 'a'):
        with pytest.raises(ValueError, match='Unsupported pinned runtime archive'):
            provision.archive_stem(rejected)


def test_pinned_runtime_manifest_matches_the_asset_paths_for_every_platform(tmp_path):
    """Any drift between config.LLAMA_RUNTIME and the shipped manifest is caught
    here, on either operating system, before it can reach a release."""
    entries = json.loads((provision.DATA/'llama-runtime.json').read_text(encoding='utf-8'))
    assert isinstance(entries, list) and entries
    for entry in entries:
        assert set(entry) == {'name', 'url', 'sha256', 'platforms', 'files'}, entry.get('name')
        assert entry['platforms'] and set(entry['platforms']) <= set(LLAMA_RUNTIME)
        assert len(entry['sha256']) == 64
        assert entry['url'].endswith(entry['name'])
        provision.archive_suffix(entry['name'])
        assert entry['files']
        for member in entry['files']:
            assert PureWindowsPath(member).name == member, member
            assert len(entry['files'][member]) == 64

    for key in LLAMA_RUNTIME:
        selected = provision.platform_archives(key)
        if not selected:
            continue
        assets = RuntimeAssets(tmp_path, None, key)
        stems = {provision.archive_stem(entry['name']) for entry in selected}
        assert stems == {assets.llama_runtime.name, assets.cuda_runtime.name}
        by_stem = {provision.archive_stem(e['name']): e for e in selected}
        assert assets.llama_server.name in by_stem[assets.llama_runtime.name]['files']
        assert assets.cuda_runtime_sentinel.name in by_stem[assets.cuda_runtime.name]['files']


def test_runtime_tarball_preserves_soname_chains(tmp_path):
    """The pinned Linux runtime ships libggml.so -> .so.0 -> .so.0.25.3 and
    llama-server loads by SONAME, so the chain must survive extraction."""
    archive = tmp_path/'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        real = tarfile.TarInfo('top/libggml.so.0.25.3')
        real.mode, real.size = 0o755, 7
        bundle.addfile(real, io.BytesIO(b'payload'))
        for name, linkname in (('top/libggml.so.0', 'libggml.so.0.25.3'),
                               ('top/libggml.so', 'libggml.so.0')):
            link = tarfile.TarInfo(name)
            link.type, link.linkname, link.mode = tarfile.SYMTYPE, linkname, 0o777
            bundle.addfile(link)
    destination = tmp_path/'out'
    provision.extract_archive(archive, destination)
    # Readable through either hop, so a digest taken on the link matches the
    # regular file it resolves to, which is what verify_assets relies on.
    for name in ('libggml.so.0.25.3', 'libggml.so.0', 'libggml.so'):
        assert (destination/name).read_bytes() == b'payload', name
