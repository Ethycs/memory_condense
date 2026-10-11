"""Provision pinned public assets or register an existing verified cache."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import zipfile

from memory_condense.runtime.artifacts import read
from memory_condense.runtime.config import RuntimeAssets, config_path, detect_platform_key
from memory_condense.runtime.llama import MODEL_SHA


DATA = Path(__file__).parent/'data'
LLAMA_REPO = 'bartowski/Llama-3.2-3B-Instruct-GGUF'
LLAMA_REVISION = '5ab33fa94d1d04e903623ae72c95d1696f09f9e8'


#: Pinned archive kinds. The manifest name carries the kind explicitly, and an
#: unknown suffix fails closed so no one can add a .tar.xz or .7z pin without
#: also teaching the extractor to validate it.
ARCHIVE_SUFFIXES = ('.zip', '.tar.gz')
WINDOWS_FILE_ATTRIBUTE_COMPRESSED = 0x800


def windows_file_is_compressed(path):
    attributes = int(getattr(Path(path).stat(), 'st_file_attributes', 0))
    return bool(attributes & WINDOWS_FILE_ATTRIBUTE_COMPRESSED)


def archive_suffix(name):
    for suffix in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    raise ValueError('Unsupported pinned runtime archive: '+name)


def archive_stem(name):
    """Directory this archive is extracted into, named by us rather than by the
    archive: the Windows zips are flat while the Linux tarballs nest everything
    under one top-level component that extract_archive strips."""
    return name[:-len(archive_suffix(name))]


def platform_archives(key=None):
    """Pinned runtime archives for one asset flavor.

    An entry with no explicit platform list applies everywhere, which keeps a
    synthetic single-platform manifest (tests, private rebuilds) valid against
    this reader. The shipped file always lists platforms explicitly, and
    tests/test_proxy_package.py asserts that.
    """
    key = key or detect_platform_key()
    return [a for a in json.loads((DATA/'llama-runtime.json').read_text(encoding='utf-8'))
            if key in a.get('platforms', (key,))]


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
    for archive in platform_archives(assets.platform_key):
        yield assets.root/'runtimes'/archive_stem(archive['name']), archive['files']


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


def prepare_memory_mapped_models(assets, *, windows=None):
    """Remove transparent NTFS compression from large mmap model files.

    The model payloads are already compressed or dense binary formats, so NTFS
    compression saves effectively no space.  It can make llama.cpp and the
    safetensors/ONNX loaders fault compressed filesystem blocks for minutes on
    a cold start.  Setup performs this conversion once, outside the interactive
    path; model bytes and their pinned hashes do not change.
    """

    active = os.name == 'nt' if windows is None else bool(windows)
    if not active:
        return 0
    from memory_condense.modeling.qwen_prefix import FIRST_SHARD
    paths = (
        assets.qwen/FIRST_SHARD,
        assets.bge/'onnx'/'model.onnx',
        assets.bge/'onnx'/'model.onnx_data',
        assets.llama_model,
    )
    converted = 0
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if not windows_file_is_compressed(path):
            continue
        print('Removing NTFS compression from '+str(path), flush=True)
        subprocess.run(
            ['compact.exe', '/u', '/i', '/q', str(path)],
            check=True,
        )
        if windows_file_is_compressed(path):
            raise RuntimeError('NTFS compression remains enabled for '+str(path))
        converted += 1
    return converted


def archive_members(bundle):
    """Normalize zip and tar entries to (name, kind, mode, linkname).

    kind is 'dir', 'file', 'link' or 'other'; only the first three are ever
    extracted, and 'link' only where the format allows it. The zip mapping
    reproduces the predicate that shipped before Linux support, which rejected
    symlinks and admitted everything else, so Windows behaviour cannot drift.
    """
    if isinstance(bundle, zipfile.ZipFile):
        for entry in bundle.infolist():
            attributes = entry.external_attr >> 16
            kind = ('link' if attributes & 0o170000 == 0o120000
                    else 'dir' if entry.is_dir() else 'file')
            yield (entry.filename.replace('\\', '/'), kind, attributes & 0o777, '')
    else:
        for entry in bundle.getmembers():
            kind = ('dir' if entry.isdir() else 'file' if entry.isfile()
                    else 'link' if entry.issym() else 'other')
            yield (entry.name.replace('\\', '/'), kind, entry.mode & 0o777, entry.linkname)


def strip_prefix(names):
    """One shared leading component to drop, or '' when the archive is flat.

    The llama.cpp Linux tarballs nest every file under a single top-level
    directory while the Windows zips are flat, and both must land directly in
    the destination this module names itself. A wrong inference can only
    misplace files inside the destination, never outside it, because
    validate_members re-checks containment after stripping.
    """
    cleaned = [name.rstrip('/') for name in names if name.rstrip('/')]
    if not cleaned:
        return ''
    first = cleaned[0].split('/')[0]
    if first in ('', '.', '..') or ':' in first:
        return ''
    if any(name != first and not name.startswith(first+'/') for name in cleaned):
        return ''
    if all(name == first for name in cleaned):
        return ''                      # a lone root entry is a file, not a tree
    return first+'/'


def safe_link(linkname):
    """True for a sibling link: one plain component that cannot leave its directory.

    The pinned Linux runtime ships SONAME chains, libggml.so -> libggml.so.0 ->
    libggml.so.0.25.3, and llama-server loads by SONAME, so the chain has to
    survive extraction. Every such link names a sibling, so restricting links to
    a single component keeps them while still refusing any path or traversal.
    """
    return (bool(linkname) and '/' not in linkname and ':' not in linkname
            and linkname not in ('.', '..') and not linkname.startswith('~'))


def validate_members(destination, members, prefix, allow_links):
    for name, kind, _mode, linkname in members:
        relative = name[len(prefix):].rstrip('/')
        target = (destination/relative).resolve() if relative else destination
        admissible = (kind in ('dir', 'file')
                      or (kind == 'link' and allow_links and safe_link(linkname)))
        if not admissible or ':' in name or not target.is_relative_to(destination):
            raise ValueError('Unsafe runtime archive member: '+name)


def resolve_link(directory, linkname, links, depth=8):
    """Follow a sibling link chain to the regular file it ends at."""
    while depth and linkname in links:
        linkname = links[linkname]
        depth -= 1
    return directory/linkname


def extract_archive(archive, destination):
    """Validate every member before writing anything from an archive."""
    destination = Path(destination).resolve()
    archive = Path(archive)
    if archive_suffix(archive.name) == '.zip':
        with zipfile.ZipFile(archive) as bundle:
            members = list(archive_members(bundle))
            if strip_prefix([name for name, *_ in members]):
                raise ValueError('Pinned zip runtime archives are flat: '+archive.name)
            validate_members(destination, members, '', allow_links=False)
            bundle.extractall(destination)
        return
    # 'r:gz' rather than 'r:*' so a renamed or disguised compression cannot load.
    with tarfile.open(archive, 'r:gz') as bundle:
        members = list(archive_members(bundle))
        prefix = strip_prefix([name for name, *_ in members])
        validate_members(destination, members, prefix, allow_links=True)
        pending = []
        for name, kind, mode, linkname in members:
            relative = name[len(prefix):].rstrip('/')
            if not relative:
                continue               # the stripped top-level entry itself
            target = destination/relative
            if kind == 'dir':
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind == 'link':
                pending.append((target, relative, linkname))
                continue
            with bundle.extractfile(name) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
            # extractall() drops the POSIX exec bit, and llama-server plus every
            # libggml-*.so ship 0755 and are unusable without it. Clamp to two
            # outcomes so setuid, setgid, sticky and group-write never survive.
            if os.name != 'nt':
                target.chmod(0o755 if mode & 0o111 else 0o644)
        # Links come last: a chain may name a member extracted later, and the
        # Windows fallback copy needs the file it resolves to already on disk.
        links = {relative: linkname for _target, relative, linkname in pending}
        for target, _relative, linkname in pending:
            if os.name != 'nt':
                target.symlink_to(linkname)
                continue
            # Windows symlink creation needs privilege the pinning tool may not
            # hold; copy instead, so the recorded digest still matches what a
            # Linux host reads through the link.
            shutil.copyfile(resolve_link(target.parent, linkname, links), target)


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
    for item in platform_archives(assets.platform_key):
        directory = assets.root/'runtimes'/archive_stem(item['name'])
        if all((directory/n).is_file() and file_sha(directory/n)==h for n,h in item['files'].items()):
            continue
        archive = assets.root/'downloads'/item['name']
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or file_sha(archive)!=item['sha256']:
            partial = archive.with_name(archive.name+'.partial')
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
    prepare_memory_mapped_models(assets)


def require_supported_platform():
    """Reject an unpinned host before touching the disk or the network."""
    key = detect_platform_key()
    if key is None:
        raise RuntimeError('The packaged native runtime currently supports Windows x64 and '
            'Linux x86_64 with NVIDIA CUDA; observe mode is independent.')
    return key


def setup(*, assets_dir=None, reuse=None):
    require_supported_platform()
    assets = RuntimeAssets.resolve(reuse or assets_dir)
    if not reuse:
        download_assets(assets)
    count = verify_assets(assets)
    uncompressed = prepare_memory_mapped_models(assets)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'version':1,'assets_dir':str(assets.root)}, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({'configured':True,'assets_dir':str(assets.root),'verified_files':count,
                      'uncompressed_model_files':uncompressed,'config':str(path)}))
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
        require_supported_platform()
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
        # The PyPI nvcomp wheel's libnvcomp.so links libcudart.so.12 with no
        # RPATH into a conda prefix, so it resolves only once torch has already
        # loaded that SONAME. Import torch first deliberately; a no-op on Windows.
        import torch                                                  # noqa: F401
        from fastembed import TextEmbedding
        from nvidia import nvcomp
        import onnxruntime
        return 'FastEmbed, ONNX Runtime, and nvCOMP imported'
    check('runtime_imports', imports)
    def driver_check():
        from memory_condense.runtime.llama import gpu
        return gpu(strict=True)
    check('nvidia_smi', driver_check)
    result = {'ok':all(c['ok'] for c in checks.values()),'platform':detect_platform_key(),
              'assets_dir':str(assets.root),'checks':checks}
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1
