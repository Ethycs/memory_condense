"""Provision pinned public assets or register an existing verified cache."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import ssl
import sys
import zipfile

from memory_condense.runtime.artifacts import read
from memory_condense.runtime.config import RuntimeAssets, config_path
from memory_condense.runtime.llama import MODEL_SHA


DATA = Path(__file__).parent/'data'
LLAMA_REPO = 'bartowski/Llama-3.2-3B-Instruct-GGUF'
LLAMA_REVISION = '5ab33fa94d1d04e903623ae72c95d1696f09f9e8'


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def manifests(assets):
    from memory_condense.modeling.qwen_prefix import QWEN3_8B_FILE_SHA256, FIRST_SHARD
    qwen = {n:h for n,h in QWEN3_8B_FILE_SHA256.items()
            if not n.endswith('.safetensors') or n == FIRST_SHARD}
    yield assets.qwen, qwen
    yield assets.bge, read(DATA/'bge-admission'/'admission.json')['files']
    yield assets.llama_model.parent, {assets.llama_model.name: MODEL_SHA}
    for archive in json.loads((DATA/'llama-runtime.json').read_text(encoding='utf-8')):
        yield assets.root/'runtimes'/archive['name'][:-4], archive['files']


def verify_assets(assets):
    assets.require_files()
    checked = 0
    for root, files in manifests(assets):
        print('Verifying '+str(root), file=sys.stderr, flush=True)
        for name, expected in files.items():
            path = root/name
            if not path.is_file() or file_sha(path) != expected:
                raise ValueError('Asset is missing or differs from the pinned runtime: '+str(path))
            checked += 1
    source = json.loads((assets.bge/'source.json').read_text(encoding='utf-8'))
    if source != read(DATA/'bge-admission'/'plan.json')['model']:
        raise ValueError('BGE source identity differs from the pinned export')
    return checked


def extract_archive(archive, destination):
    """Validate every member before writing anything from an archive."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = entry.filename.replace('\\', '/')
            target = (destination/name).resolve()
            if (not target.is_relative_to(destination) or ':' in name
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe runtime archive member: '+name)
        bundle.extractall(destination)


def download_assets(assets):
    import httpx
    import truststore
    from huggingface_hub import snapshot_download
    from memory_condense.modeling.qwen_prefix import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
    truststore.inject_into_ssl()
    groups = list(manifests(assets))
    bge_source = read(DATA/'bge-admission'/'plan.json')['model']
    for repo, revision, directory, files in (
        (DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, assets.qwen, groups[0][1]),
        (bge_source['repo'], bge_source['revision'], assets.bge, groups[1][1]),
        (LLAMA_REPO, LLAMA_REVISION, assets.llama_model.parent, groups[2][1]),
    ):
        print('Preparing '+repo+' at '+revision, flush=True)
        if not all((directory/name).is_file() and file_sha(directory/name)==sha for name,sha in files.items()):
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(directory), allow_patterns=list(files))
    (assets.bge/'source.json').write_text(json.dumps(bge_source)+'\n', encoding='utf-8')
    archives = json.loads((DATA/'llama-runtime.json').read_text(encoding='utf-8'))
    for item in archives:
        directory = assets.root/'runtimes'/item['name'][:-4]
        if all((directory/n).is_file() and file_sha(directory/n)==h for n,h in item['files'].items()):
            continue
        archive = assets.root/'downloads'/item['name']
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or file_sha(archive)!=item['sha256']:
            partial = archive.with_suffix('.zip.partial')
            print('Downloading '+item['name'], flush=True)
            with httpx.Client(follow_redirects=True, timeout=120,
                             verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
                with client.stream('GET', item['url']) as response, partial.open('wb') as output:
                    response.raise_for_status()
                    for data in response.iter_bytes(): output.write(data)
            if file_sha(partial)!=item['sha256']:
                raise ValueError('Runtime download checksum mismatch: '+str(partial))
            partial.replace(archive)
        extract_archive(archive, directory)


def setup(*, assets_dir=None, reuse=None):
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64','x86_64'):
        raise RuntimeError('The packaged native runtime currently supports Windows x64 with NVIDIA CUDA; observe mode is independent.')
    assets = RuntimeAssets.resolve(reuse or assets_dir)
    if not reuse:
        download_assets(assets)
    count = verify_assets(assets)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'version':1,'assets_dir':str(assets.root)}, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({'configured':True,'assets_dir':str(assets.root),'verified_files':count,'config':str(path)}))
    return 0


def doctor(*, assets_dir=None, verify=False):
    assets = RuntimeAssets.resolve(assets_dir)
    checks = {}
    def check(name, operation):
        try:
            checks[name] = {'ok':True,'detail':operation()}
        except Exception as exc:
            checks[name] = {'ok':False,'detail':str(exc)}
    def platform_check():
        if platform.system()!='Windows' or platform.machine().lower() not in ('amd64','x86_64'):
            raise RuntimeError('Native proxy runtime requires Windows x64')
        if not (3,11)<=sys.version_info[:2]<(3,13):
            raise RuntimeError('Use Python 3.11 or 3.12')
        return platform.platform()
    check('platform', platform_check)
    check('assets', lambda: verify_assets(assets) if verify else assets.require_files() or 'Required files present; use --verify-models to hash them')
    for path in (assets.fastembed_package, assets.nvcomp_package):
        if path is not None: sys.path.insert(0,str(path))
    for name, wanted in (('fastembed','0.8.1'),('onnxruntime','1.30.0'),('nvidia-nvcomp-cu12','5.3.0.16')):
        def version(name=name, wanted=wanted):
            actual = metadata.version(name)
            if actual != wanted: raise RuntimeError('Install '+name+'=='+wanted+'; found '+actual)
            return actual
        check(name, version)
    def gpu_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA-enabled PyTorch and an NVIDIA GPU are required; see installation instructions')
        props = torch.cuda.get_device_properties(0)
        return {'name':props.name,'total_vram_gib':round(props.total_memory/2**30,2),'torch':torch.__version__}
    check('gpu', gpu_check)
    def imports():
        from fastembed import TextEmbedding
        from nvidia import nvcomp
        import onnxruntime
        return 'FastEmbed, ONNX Runtime, and nvCOMP imported'
    check('runtime_imports', imports)
    result = {'ok':all(c['ok'] for c in checks.values()),'assets_dir':str(assets.root),'checks':checks}
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1
