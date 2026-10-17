"""Bundle a locally built Pixi artifact with its installer and instructions."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tomllib
import zipfile


#: One bundle per platform. The conda package itself is noarch, so both bundles
#: carry the same package bytes and only the installer and container differ.
#: The manifest name is suffixed too, not just the loose file: verify checks the
#: archive's member set exactly, and a user extracting both bundles into one
#: directory must not collide on a shared SHA256SUMS.json.
PLATFORMS = {
    'win64': dict(installer='install-proxy.ps1', suffix='.zip', format='zip'),
    'linux64': dict(installer='install-proxy.sh', suffix='.tar.gz', format='tar'),
}


def manifest_name(platform):
    return f'SHA256SUMS-{platform}.json'


def bundle_name(version, platform):
    return f'memory-condense-{version}-{platform}{PLATFORMS[platform]["suffix"]}'


def members(repo, output, version, platform):
    spec = PLATFORMS[platform]
    packages = list(output.glob(f'memory_condense-{version}-*.conda'))
    if len(packages) != 1:
        raise ValueError(f'Run pixi build --output-dir dist first; expected one {version} package')
    package = packages[0]
    installer = output/spec['installer']
    shutil.copyfile(repo/'packaging'/spec['installer'], installer)
    if spec['format'] == 'tar':
        installer.chmod(0o755)          # CI executes the loose copy directly
    guide = repo/'docs/02 - Implementation/06 - Provider Proxy and Transcript Import.md'
    files = {
        package.name: package,
        installer.name: installer,
        'README.md': repo/'README.md',
        'packaging/MODEL_ASSETS.md': repo/'packaging/MODEL_ASSETS.md',
        'RELEASE_NOTES.md': repo/'packaging/RELEASE_NOTES.md',
        'docs/02 - Implementation/06 - Provider Proxy and Transcript Import.md': guide,
        'docs/02 - Implementation/32 - Codex Client and Windows Startup.md':
            repo/'docs/02 - Implementation/32 - Codex Client and Windows Startup.md',
        'docs/verification/codex-windows-startup.md': repo/'docs/verification/codex-windows-startup.md',
    }
    if platform == 'win64':
        # The installer copies this helper into the independent application.
        startup = output/'manage-startup.ps1'
        shutil.copyfile(repo/'packaging'/startup.name, startup)
        files[startup.name] = startup
    return package, files


def write_bundle(archive, files, manifest_path, platform):
    """The checksum manifest is always the last member, for both formats."""
    spec = PLATFORMS[platform]
    if spec['format'] == 'zip':
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, path in files.items():
                bundle.write(path, name)
            bundle.write(manifest_path, manifest_path.name)
        return
    executable = {spec['installer']}

    def reproducible(info):
        # Fixed ownership and mtime so the same inputs give the same bytes, and
        # a forced mode so the installer is executable regardless of the
        # builder's umask or git's record of the file mode.
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        info.mtime = 0
        info.mode = 0o755 if info.name in executable else 0o644
        return info

    with gzip.GzipFile(archive, 'wb', compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as bundle:
            for name, path in files.items():
                bundle.add(path, arcname=name, filter=reproducible)
            bundle.add(manifest_path, arcname=manifest_path.name, filter=reproducible)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('dist'))
    parser.add_argument('--platform', choices=sorted(PLATFORMS), default='win64')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    version = tomllib.loads((repo/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    pixi_version = tomllib.loads((repo/'pixi.toml').read_text(encoding='utf-8'))['package']['version']
    if version != pixi_version:
        raise ValueError('Python and Pixi package versions disagree')
    package, files = members(repo, output, version, args.platform)
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in files.items()}
    manifest_path = output/manifest_name(args.platform)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    archive = output/bundle_name(version, args.platform)
    write_bundle(archive, files, manifest_path, args.platform)
    print(json.dumps(dict(platform=args.platform, package=str(package), bundle=str(archive),
                          manifest=manifest_path.name, bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
