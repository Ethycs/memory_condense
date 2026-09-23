"""Persist complete-namespace summary vectors and authenticated raw hydration data."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from tools.execute_spine_corpus import fragments_from_request
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def restore_turns(fragments):
    """Reassemble every exact fragment; reject gaps, aliases and changed bytes."""
    groups = {}
    for fragment in fragments:
        groups.setdefault(fragment.span.turn_id, []).append(fragment)
    turns = []
    for rows in groups.values():
        first = rows[0].span
        text = "".join(row.text for row in rows)
        turn = Turn(turn_id=first.turn_id, source_id=first.source_id, role=first.role,
                    created_at=datetime.fromisoformat(first.created_at), text=text)
        cursor = 0
        for row in rows:
            if row.span.start_char != cursor or row.span != RawSectionSpan.from_turn(
                turn, start_char=cursor, end_char=cursor + len(row.text)):
                raise ValueError("raw hydration fragments changed, overlap or contain a gap")
            cursor += len(row.text)
        turns.append(turn)
    return tuple(turns)


def persist_vectors(root, index, matrix, embedding_identity):
    semantic = SemanticSectionIndex(index, matrix, embedding_identity=embedding_identity)
    path = root / "summary-vectors.npy"
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = np.load(path, allow_pickle=False)
        if prior.dtype != np.float32 or not np.array_equal(prior, matrix):
            raise ValueError("compiled summary vectors changed")
    else:
        with path.open("xb") as handle:
            np.save(handle, matrix, allow_pickle=False)
    return semantic, hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(root):
    manifest = read_sealed_json(root / "index.json")
    payload = manifest.payload
    path = root / "summary-vectors.npy"
    if hashlib.sha256(path.read_bytes()).hexdigest() != payload["matrix_file_sha256"]:
        raise ValueError("compiled semantic matrix file changed")
    index = SectionSummaryIndex.from_json(payload["index_json"])
    semantic = SemanticSectionIndex(index, np.load(path, allow_pickle=False), embedding_identity=payload["embedding_identity"])
    if semantic.receipt_sha256 != payload["semantic_index_sha256"]:
        raise ValueError("semantic index metadata changed")
    return manifest, semantic


def compilation_policy_sha256(payload):
    """Compare the method across namespaces without mixing in cache provenance."""
    excluded = {"atoms_sha256", "raw_span_population_sha256", "complete_namespace"}
    declared = payload.get("method_policy_sha256")
    if declared is not None:
        excluded.update({"method_policy_sha256", "parent_caches"})
    digest = identity_sha256({k: v for k, v in payload.items() if k not in excluded})
    if declared is not None and declared != digest:
        raise ValueError("hierarchy method policy changed")
    return digest


def compile_index(corpus_root, hierarchy_root, offset, root):
    corpus = read_sealed_json(corpus_root / "preflight.json")
    namespace = next(row for row in corpus.payload["namespaces"] if row["shard_offset"] == offset)
    hierarchy = read_sealed_json(hierarchy_root / "hierarchy.json")
    compilation = read_sealed_json(hierarchy_root / "preflight.json")
    if compilation.sha256 != hierarchy.payload["preflight_sha256"]:
        raise ValueError("hierarchy compilation binding changed")
    compilation_policy = compilation_policy_sha256(compilation.payload)
    atoms = read_sealed_json(corpus_root / f"offset-{offset:03d}" / f"source-bound-atoms-prefix-{namespace['request_count']:04d}.json")
    if (not hierarchy.payload["complete_namespace"] or not atoms.payload["complete_namespace"] or
        hierarchy.payload["atoms_sha256"] != atoms.sha256 or atoms.payload["corpus_preflight_sha256"] != corpus.sha256 or
        hierarchy.payload["raw_span_population_sha256"] != namespace["span_population_sha256"]):
        raise ValueError("semantic evaluation requires the complete bound namespace")
    index = SectionSummaryIndex.from_json(hierarchy.payload["index_json"])
    fragments = []
    for binding in namespace["requests"]:
        request = read_sealed_json(corpus_root / binding["path"])
        if request.sha256 != binding["sha256"]:
            raise ValueError("raw request changed")
        fragments.extend(fragments_from_request(request.payload))
    if identity_sha256([row.span.receipt_sha256 for row in fragments]) != namespace["span_population_sha256"]:
        raise ValueError("raw namespace population changed")
    expected = sorted(row.span.receipt_sha256 for row in fragments)
    observed = sorted(span.receipt_sha256 for section in index.sections if not section.child_section_ids
                      for span in section.spans)
    if observed != expected or len(set(expected)) != len(expected):
        raise ValueError("hierarchy leaves do not partition the complete raw namespace exactly once")
    turns = restore_turns(fragments)
    if len(turns) != namespace["turn_count"]:
        raise ValueError("raw hydration turn population changed")
    if (len({turn.source_id for turn in turns}) != namespace["source_count"] or
        sum(count_tokens(turn.text) for turn in turns) != namespace["raw_token_proxy"]):
        raise ValueError("raw hydration source or token population changed")
    raw, _ = publish_sealed_json(root / "raw-turns.json", {"corpus_preflight_sha256": corpus.sha256,
        "shard_offset": offset, "raw_span_population_sha256": namespace["span_population_sha256"],
        "turns": [{"turn_id": t.turn_id, "source_id": t.source_id, "role": t.role,
                   "created_at": t.created_at.isoformat(), "text": t.text, "text_sha256": quote_sha256(t.text)} for t in turns]})
    encoder = EmbeddingService(device="cuda", batch_size=8)
    try:
        before = summary_embedding_identity(encoder)
        leaves = [s for s in index.sections if not s.child_section_ids]
        matrix = np.array(encoder.embed_queries([s.summary for s in leaves]), dtype=np.float32, copy=True)
        if summary_embedding_identity(encoder) != before or not np.isfinite(matrix).all():
            raise ValueError("embedding identity or vector values changed")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero summary vector")
        matrix /= norms
        semantic, file_sha = persist_vectors(root, index, matrix, before)
    finally:
        encoder.close()
    manifest, _ = publish_sealed_json(root / "index.json", {
        "format": "memory-condense-persisted-semantic-spine-index-v1", "shard_offset": offset,
        "corpus_preflight_sha256": corpus.sha256, "hierarchy_sha256": hierarchy.sha256,
        "hierarchy_compilation_policy_sha256": compilation_policy,
        "raw_turns_sha256": raw.sha256, "raw_span_population_sha256": namespace["span_population_sha256"],
        "complete_namespace": True, "raw_token_proxy": namespace["raw_token_proxy"],
        "turn_count": len(turns), "source_count": namespace["source_count"],
        "leaf_count": len(leaves), "matrix_file_sha256": file_sha, "embedding_identity": before,
        "semantic_index_sha256": semantic.receipt_sha256, "semantic_metadata": semantic.metadata_json,
        "index_json": index.to_json(), "raw_embedding_inputs": False, "question_inputs": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                           ("tools/compile_spine_semantic_index.py",
                            "src/memory_condense/search/summary_semantic_index.py", "src/memory_condense/modeling/embedding.py")}})
    print({"index_sha256": manifest.sha256, "leaf_count": len(leaves), "raw_tokens": namespace["raw_token_proxy"],
           "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--hierarchy-root", type=Path, required=True)
    parser.add_argument("--shard-offset", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    compile_index(args.corpus_root, args.hierarchy_root, args.shard_offset, args.output_root)
