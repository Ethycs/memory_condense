"""Verify the complete release bundle and its version before publication."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile


def verify(output, tag=''):
    repo=Path(__file__).resolve().parents[1]
    version=tomllib.loads((repo/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if tag.startswith('v') and tag not in (f'v{version}',) and not tag.startswith(f'v{version}-'):
        raise ValueError('Release tag does not match package version')
    archive=output/f'memory-condense-{version}-win64.zip'
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('SHA256SUMS.json'))
        names=bundle.namelist()
        if len(set(names))!=len(names) or set(names)!=set(manifest)|{'SHA256SUMS.json'}:
            raise ValueError('Unexpected or duplicate release members')
        if json.loads((output/'SHA256SUMS.json').read_text(encoding='utf-8'))!=manifest:
            raise ValueError('External and bundled checksums disagree')
        for name,sha in manifest.items():
            if hashlib.sha256(bundle.read(name)).hexdigest()!=sha:
                raise ValueError('Corrupt release member: '+name)
            local=output/name
            if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest()!=sha:
                raise ValueError('Upload differs from bundled member: '+name)
    return dict(version=version,bundle=archive.name,verified_members=len(manifest),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('dist'))
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    print(json.dumps(verify(args.output_dir,args.tag)))
