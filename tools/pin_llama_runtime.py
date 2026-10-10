"""Pin an upstream llama.cpp runtime archive into llama-runtime.json.

Research tool, deliberately outside the shipping package: the installed runtime
must never import from tools/, which tests/test_proxy_package.py asserts.

The archive names are derived from LLAMA_RUNTIME rather than passed in, so the
invariant that every flavor's directory equals archive_stem(entry['name'])
holds by construction. Extraction runs through the production extract_archive
for the same reason: it is the only way the recorded 'files' basenames are
guaranteed to match what setup will actually produce after prefix stripping.

Usage:
    pixi run -e dev python -m tools.pin_llama_runtime --platform linux-x86_64
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import sys
import tempfile

from memory_condense.runtime.config import LLAMA_RUNTIME
from memory_condense.runtime.setup import archive_stem, extract_archive, file_sha

RELEASE = 'https://github.com/ggml-org/llama.cpp/releases/download/'
#: Upstream ships Windows as zip and Linux as gzipped tar.
SUFFIX = {'win-amd64': '.zip', 'linux-x86_64': '.tar.gz'}


def archive_names(key):
    flavor = LLAMA_RUNTIME[key]
    suffix = SUFFIX[key]
    return [flavor['runtime']+suffix, flavor['cudart']+suffix]


def download(url, destination):
    import httpx
    import truststore
    truststore.inject_into_ssl()
    partial = destination.with_name(destination.name+'.partial')
    digest = hashlib.sha256()
    with httpx.Client(follow_redirects=True, timeout=120,
                      verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
        with client.stream('GET', url) as response, partial.open('wb') as output:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            seen = 0
            for block in response.iter_bytes():
                digest.update(block)
                output.write(block)
                seen += len(block)
                if total:
                    print(f'\r  {destination.name}: {seen/2**20:,.0f}/{total/2**20:,.0f} MiB',
                          end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    partial.replace(destination)
    return digest.hexdigest()


def pin(key, tag, downloads=None):
    entries = []
    with tempfile.TemporaryDirectory() as scratch:
        for name in archive_names(key):
            store = Path(downloads or scratch)
            store.mkdir(parents=True, exist_ok=True)
            archive = store/name
            url = RELEASE+tag+'/'+name
            if archive.is_file():
                print('Reusing '+str(archive), file=sys.stderr)
                sha = file_sha(archive)
            else:
                print('Downloading '+url, file=sys.stderr)
                sha = download(url, archive)
            extracted = Path(scratch)/'extracted'/archive_stem(name)
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True)
            extract_archive(archive, extracted)
            files = {}
            for path in sorted(extracted.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(extracted).as_posix()
                if '/' in relative:
                    raise ValueError('Extracted member is nested, which the flat '
                                     "'files' map cannot express: "+relative)
                files[relative] = file_sha(path)
            if not files:
                raise ValueError('Archive produced no files: '+name)
            entries.append(dict(name=name, url=url, sha256=sha,
                                platforms=[key], files=files))
            print(f'  {name}: {len(files)} members', file=sys.stderr)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform', required=True, choices=sorted(LLAMA_RUNTIME))
    parser.add_argument('--tag', default='b11272', help='Upstream llama.cpp release tag')
    parser.add_argument('--downloads', help='Keep archives here instead of a temporary directory')
    parser.add_argument('--output', help='Write the entries here instead of stdout')
    parsed = parser.parse_args(argv)
    entries = pin(parsed.platform, parsed.tag, parsed.downloads)
    text = json.dumps(entries, indent=2)+'\n'
    if parsed.output:
        Path(parsed.output).write_text(text, encoding='utf-8', newline='\n')
        print('Wrote '+parsed.output, file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
