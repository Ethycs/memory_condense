"""Compile additional addresses from stored user summaries, without questions."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex, summary_facets
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


IMPLEMENTATION = ("tools/compile_spine_facet_addresses.py", "src/memory_condense/search/spine_summary_facets.py",
    "src/memory_condense/search/user_spine_addresses.py", "src/memory_condense/modeling/embedding.py",
    "src/memory_condense/search/summary_semantic_index.py")


def compile_addresses(index_root, root):
    manifest, index = load_index(index_root)
    facets = tuple(f for s in index.sections for f in summary_facets(s))
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": manifest.sha256, "facets": [f.identity_payload() for f in facets],
        "document_inputs": "exact passages within stored user-spine summaries only",
        "raw_inputs": False, "gold_inputs": False, "query_inputs_during_compilation": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    path = root / "facet-vectors.npy"
    encoder = EmbeddingService(device="cuda", batch_size=8)
    try:
        encoder.embed_queries([facets[0].text])
        identity = summary_embedding_identity(encoder)
        if identity != index.embedding_identity:
            raise ValueError("facet and whole-summary encoders differ")
        if path.exists():
            prior = read_sealed_json(root / "addresses.json")
            if (prior.payload["preflight_sha256"] != preflight.sha256 or
                    prior.payload["matrix_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError("compiled facet addresses changed")
            matrix = np.load(path, allow_pickle=False)
        else:
            matrix = np.array(encoder.embed_queries([f.text for f in facets]), dtype=np.float32, copy=True)
            if matrix.ndim != 2 or len(matrix) != len(facets) or not np.isfinite(matrix).all():
                raise ValueError("invalid facet matrix")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("zero facet vector")
            matrix /= norms
            with path.open("xb") as handle:
                np.save(handle, matrix, allow_pickle=False)
        if summary_embedding_identity(encoder) != identity:
            raise ValueError("facet encoder changed during compilation")
        addresses = SpineFacetAddressIndex(index.hierarchy, matrix, embedding_identity=identity)
        artifact, _ = publish_sealed_json(root / "addresses.json", {
            "preflight_sha256": preflight.sha256, "base_index_sha256": manifest.sha256,
            "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "address_index_sha256": addresses.receipt_sha256, "embedding_identity": identity,
            "section_count": len(index.sections), "facet_count": len(facets), "new_provider_calls": 0})
        print({"facet_addresses_sha256": artifact.sha256, "sections": len(index.sections),
            "facets": len(facets), "new_provider_calls": 0}, flush=True)
    finally:
        encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    compile_addresses(args.index_root, args.output_root)
