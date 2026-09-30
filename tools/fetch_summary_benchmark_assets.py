"""Fetch pinned official CPU runtime and quantized summarizers into .cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ASSETS = [
    ('https://github.com/ggml-org/llama.cpp/releases/download/b11272/llama-b11272-bin-win-cpu-x64.zip',
     '.cache/runtimes/llama-b11272-bin-win-cpu-x64.zip', 'f6d6c2d547b8f87ac1cecf1614207ee300aae1e0cf9b1966bd01129ef58855f7'),
    ('https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript-GGUF/resolve/586791c82a1699507624755ffe4fcdef0ff8ed95/LFM2-2.6B-Transcript-Q4_K_M.gguf',
     '.cache/models/LFM2-2.6B-Transcript-GGUF/LFM2-2.6B-Transcript-Q4_K_M.gguf', '74832bbf09a5321bfd119e13c6c1fd6361517d3f085545cf00b79c7dc8cceac6'),
    ('https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_K_M.gguf',
     '.cache/models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf', 'b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5'),
]


def fetch(asset):
    url, name, expected = asset
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
            while block := response.read(4*1024*1024): out.write(block)
        with partial.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest()==expected
        partial.replace(path)
    with path.open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest()==expected
    if path.suffix=='.zip':
        destination = path.with_suffix('')
        with zipfile.ZipFile(path) as archive:
            assert all((destination/m.filename).resolve().is_relative_to(destination.resolve()) for m in archive.infolist())
            archive.extractall(destination)
    print(json.dumps(dict(asset=str(path), bytes=path.stat().st_size, sha256=expected)), flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(fetch, ASSETS))
