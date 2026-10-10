"""Verify the complete release bundle and its version before publication."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile

from build_proxy_release import PLATFORMS, bundle_name, manifest_name


def tar_entries(bundle):
    """Names and a reader, refusing anything that is not a plain file.

    Tar has no zip analogue for this: a symlink member would make extractfile
    follow it inside the archive, so a hostile bundle could pass a digest check
    against content other than the member it claims to be.
    """
    names = []
    for member in bundle.getmembers():
        if not member.isfile():
            raise ValueError('Unexpected non-file release member: '+member.name)
        if member.name.startswith('/') or '..' in Path(member.name).parts:
            raise ValueError('Unsafe release member path: '+member.name)
        names.append(member.name)
    return names, lambda name: bundle.extractfile(name).read()


def verify(output, tag='', platform='win64'):
    repo = Path(__file__).resolve().parents[1]
    version = tomllib.loads((repo/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if tag.startswith('v') and tag not in (f'v{version}',) and not tag.startswith(f'v{version}-'):
        raise ValueError('Release tag does not match package version')
    archive = output/bundle_name(version, platform)
    checksums = manifest_name(platform)
    opener = (zipfile.ZipFile(archive) if PLATFORMS[platform]['format'] == 'zip'
              else tarfile.open(archive, 'r:gz'))
    with opener as bundle:
        if isinstance(bundle, zipfile.ZipFile):
            names, read = bundle.namelist(), bundle.read
        else:
            names, read = tar_entries(bundle)
        manifest = json.loads(read(checksums))
        if len(set(names)) != len(names) or set(names) != set(manifest)|{checksums}:
            raise ValueError('Unexpected or duplicate release members')
        if json.loads((output/checksums).read_text(encoding='utf-8')) != manifest:
            raise ValueError('External and bundled checksums disagree')
        for name, sha in manifest.items():
            if hashlib.sha256(read(name)).hexdigest() != sha:
                raise ValueError('Corrupt release member: '+name)
            local = output/name
            # Catches a loose upload drifting from the bundled copy, which is
            # how a package built on another runner would be detected.
            if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() != sha:
                raise ValueError('Upload differs from bundled member: '+name)
    return dict(version=version, platform=platform, bundle=archive.name,
                manifest=checksums, verified_members=len(manifest),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('dist'))
    parser.add_argument('--platform', choices=sorted(PLATFORMS), default='win64')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.tag, args.platform)))
