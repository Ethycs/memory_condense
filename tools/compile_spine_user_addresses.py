"""Compile user-only summary addresses without evaluation questions or raw text."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex, user_spine_text
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def compile_addresses(index_root, root):
    manifest, index = load_index(index_root)
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": manifest.sha256, "document_inputs": "stored user-spine summaries only",
        "raw_inputs": False, "gold_inputs": False, "query_inputs_during_compilation": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/compile_spine_user_addresses.py", "src/memory_condense/search/user_spine_addresses.py",
            "src/memory_condense/modeling/embedding.py", "src/memory_condense/search/summary_semantic_index.py")}})
    path = root / "spine-vectors.npy"
    texts = [user_spine_text(s) for s in index.sections]
    encoder = EmbeddingService(device="cuda", batch_size=8)
    try:
        encoder.embed_queries(texts[:1])
        identity = summary_embedding_identity(encoder)
        if identity != index.embedding_identity:
            raise ValueError("user and combined summary encoders differ")
        if path.exists():
            prior = read_sealed_json(root / "addresses.json")
            if (prior.payload["preflight_sha256"] != preflight.sha256 or
                    prior.payload["matrix_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError("compiled user addresses changed")
            matrix = np.load(path, allow_pickle=False)
        else:
            matrix = np.array(encoder.embed_queries(texts), dtype=np.float32, copy=True)
            if matrix.ndim != 2 or len(matrix) != len(texts) or not np.isfinite(matrix).all():
                raise ValueError("invalid user-summary matrix")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("zero user-summary vector")
            matrix /= norms
            with path.open("xb") as handle:
                np.save(handle, matrix, allow_pickle=False)
        if summary_embedding_identity(encoder) != identity:
            raise ValueError("summary encoder changed during compilation")
        addresses = UserSpineAddressIndex(index.hierarchy, np.load(index_root / "summary-vectors.npy", allow_pickle=False),
                                         matrix, embedding_identity=identity)
        artifact, _ = publish_sealed_json(root / "addresses.json", {
            "preflight_sha256": preflight.sha256, "base_index_sha256": manifest.sha256,
            "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "address_index_sha256": addresses.receipt_sha256, "embedding_identity": identity,
            "section_count": len(index.sections), "new_provider_calls": 0})
        print({"addresses_sha256": artifact.sha256, "sections": len(index.sections), "new_provider_calls": 0}, flush=True)
    finally:
        encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    compile_addresses(args.index_root, args.output_root)
