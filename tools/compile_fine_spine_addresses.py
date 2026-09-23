"""Index existing user-fragment summaries across the ten complete memories."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.source_spine_hydration import SourceSpineHydrationIndex
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.compile_spine_semantic_index import load_index, persist_vectors
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


IMPLEMENTATION = ('tools/compile_fine_spine_addresses.py', 'tools/compile_spine_semantic_index.py',
    'src/memory_condense/search/summary_semantic_index.py', 'src/memory_condense/modeling/embedding.py',
    'src/memory_condense/search/source_spine_hydration.py')


def compile_all(source_root, root):
    implementation = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}
    inputs = []
    for offset in range(0, 100, 10):
        source = read_sealed_json(source_root / 'namespaces' / f'offset-{offset:03}' / 'preflight.json')
        p = source.payload
        inputs.append({'offset': offset, 'index_root': p['index_root'], 'index_sha256': p['index_manifest_sha256'],
                       'atoms_path': p['atoms_path'], 'atoms_sha256': p['atoms_sha256']})
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-fine-spine-compilation-v1', 'inputs': inputs, 'implementation': implementation,
        'embedding_inputs': 'existing user-fragment summaries only', 'raw_embedding_inputs': False,
        'question_embedding_inputs': False, 'new_summary_generation': False, 'new_qwen_calls': 0})
    encoder = EmbeddingService(device='cuda', batch_size=8)
    cache, outputs, cache_identity = {}, [], None
    try:
        for row in inputs:
            manifest, original = load_index(Path(row['index_root']))
            atoms = read_sealed_json(Path(row['atoms_path']))
            if (manifest.sha256 != row['index_sha256'] or atoms.sha256 != row['atoms_sha256'] or
                    not manifest.payload['complete_namespace'] or not atoms.payload['complete_namespace']):
                raise ValueError('complete source bindings changed')
            all_atoms = tuple(SectionSummary.from_dict(a) for a in atoms.payload['atoms'])
            source_spine = SourceSpineHydrationIndex(original.hierarchy, all_atoms)
            index = SectionSummaryIndex(tuple(a for a in all_atoms if a.spans[0].role == 'user'))
            texts = list(dict.fromkeys(s.summary for s in index.sections))
            missing = [text for text in texts if quote_sha256(text) not in cache]
            encoder.embed_queries(texts[:1])
            identity = summary_embedding_identity(encoder)
            if identity != original.embedding_identity:
                raise ValueError('fine and coarse summary encoders differ')
            if cache_identity is not None and cache_identity != identity:
                raise ValueError('shared summary cache encoder changed')
            cache_identity = identity
            for start in range(0, len(missing), 128):
                chunk = missing[start:start + 128]
                matrix = np.array(encoder.embed_queries(chunk), dtype=np.float32, copy=True)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if matrix.ndim != 2 or len(matrix) != len(chunk) or not np.isfinite(matrix).all() or np.any(norms == 0):
                    raise ValueError('invalid fine summary vectors')
                matrix /= norms
                for text, vector in zip(chunk, matrix, strict=True):
                    cache[quote_sha256(text)] = vector
                print({'offset': row['offset'], 'embedded': min(start + 128, len(missing)),
                       'new_summary_count': len(missing)}, flush=True)
            if summary_embedding_identity(encoder) != identity:
                raise ValueError('embedding identity changed during compilation')
            matrix = np.array([cache[quote_sha256(s.summary)] for s in index.sections], dtype=np.float32)
            namespace = root / f"offset-{row['offset']:03}"
            fine, file_sha = persist_vectors(namespace, index, matrix, identity)
            artifact, _ = publish_sealed_json(namespace / 'index.json', {
                'preflight_sha256': preflight.sha256, **row,
                'source_partition_sha256': source_spine.index.receipt_sha256,
                'matrix_file_sha256': file_sha, 'embedding_identity': identity,
                'semantic_index_sha256': fine.receipt_sha256, 'index_json': index.to_json(),
                'raw_embedding_inputs': False, 'user_fragment_count': len(index.sections),
                'whole_user_turn_count': sum(len(rows) for rows in source_spine.by_source.values())})
            outputs.append({'root': str(namespace.resolve()), 'sha256': artifact.sha256})
            print({'offset': row['offset'], 'index_sha256': artifact.sha256,
                   'user_fragments': len(index.sections)}, flush=True)
    finally:
        encoder.close()
    if implementation != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}:
        raise ValueError('compiler changed during compilation')
    publish_sealed_json(root / 'complete.json', {'preflight_sha256': preflight.sha256, 'indexes': outputs,
                                              'new_provider_calls': 0, 'new_qwen_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    compile_all(args.source_root, args.output_root)
