"""Compile only stored root user summaries beside the already ingested memory."""
import argparse
from contextlib import closing
from pathlib import Path
import time

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.persistence import native_spine_parent_store as store
from memory_condense.search import native_spine_parent_users as projection
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound
from tools.run_spine_reader_after_timeout import require_idle


APPLICATION = Path('eval_results/native-spine-application-lifecycle-20260915-r1')


def run(root):
    if root.exists():
        raise ValueError('parent compilation requires a fresh output root')
    require_idle()
    verification = read_sealed_json(APPLICATION / 'reopen-verification.json')
    ingested = bound(verification.payload['ingest_complete'])
    app_path = APPLICATION / 'application'
    for name, sha in ingested.payload['application_files'].items():
        if digest(app_path / name) != sha:
            raise ValueError('base application files changed before parent compilation')
    target = app_path / store.FILENAME
    if target.exists():
        raise ValueError('preserve existing parent cache rather than silently recompiling it')
    started = time.perf_counter()
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with MemoryCondenser(app_path, embedder=encoder, auto_extract=False, read_only=True) as app:
            receipt = app.native_spine_receipt()
            if receipt != verification.payload['snapshot']:
                raise ValueError('native snapshot differs from verified application memory')
            snapshot = app._load_native_spine()[1]
            parents = projection.project_parent_users(snapshot.hierarchy)
            texts = [s.summary for s in parents.sections]
            identity = summary_embedding_identity(encoder)
            model = encoder._load_model()
            lengths = model.tokenizer(texts, add_special_tokens=True, truncation=False,
                                      padding=False, return_length=True)['length']
            if not texts or max(lengths) > model.max_seq_length:
                raise ValueError('parent user summaries would be truncated by the encoder')
            plan, _ = publish_sealed_json(root / 'preflight.json', {
                'application_verification': binding(verification), 'native_snapshot': receipt,
                'projection_sha256': parents.receipt_sha256, 'parent_count': len(texts),
                'embedding_identity': identity, 'max_input_tokens': max(lengths),
                'encoder_token_limit': model.max_seq_length, 'document_batch_token_cap': 8192,
                'implementation': {__file__: digest(__file__), projection.__file__: digest(projection.__file__),
                                   store.__file__: digest(store.__file__)},
                'embedding_inputs': 'stored root user_spine strings only',
                'question_inputs': False, 'gold_inputs': False, 'raw_inputs_to_encoder': False,
                'new_qwen_calls': 0, 'new_answer_calls': 0, 'new_ingestions': 0})
            print({'parent_count': len(texts), 'max_input_tokens': max(lengths),
                   'preflight_sha256': plan.sha256}, flush=True)
            vectors = []
            cursor = 0
            while cursor < len(texts):
                end = min(cursor + 8, len(texts))
                while end > cursor + 1 and max(lengths[cursor:end]) * (end - cursor) > 8192:
                    end -= 1
                vectors.extend(encoder.embed_queries(texts[cursor:end]))
                cursor = end
                if cursor % 64 == 0 or cursor == len(texts):
                    print({'parent_summaries_embedded': cursor, 'required': len(texts)}, flush=True)
            if summary_embedding_identity(encoder) != identity:
                raise ValueError('summary encoder identity changed during compilation')
            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(matrix).all() or np.any(norms == 0):
                raise ValueError('invalid parent summary vectors')
            matrix /= norms
            semantic = SemanticSectionIndex(parents, matrix, embedding_identity=identity)
            persisted = store.publish(target, hierarchy=snapshot.hierarchy, matrix=matrix, native_receipt=receipt)
            reopened, again = store.load(target, hierarchy=snapshot.hierarchy, native_receipt=receipt)
            if again != persisted or reopened.receipt_sha256 != semantic.receipt_sha256:
                raise ValueError('persisted parent vectors changed on reopen')
    for name, sha in ingested.payload['application_files'].items():
        if digest(app_path / name) != sha:
            raise ValueError('parent compilation modified the original application files')
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(plan), 'parent_snapshot': persisted,
        'parent_file': {'path': str(target.resolve()), 'sha256': digest(target)},
        'parent_count': len(texts), 'elapsed_s': time.perf_counter() - started,
        'original_application_files_unchanged': True,
        'new_ingestions': 0, 'new_qwen_calls': 0, 'new_answer_calls': 0})
    print({'report_sha256': report.sha256, 'parent_count': len(texts),
           'elapsed_s': report.payload['elapsed_s']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
