"""FP32 FastEmbed adapter with an explicit, sealed BGE-M3 transition assay."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.modeling.embedding import EmbeddingService, DEFAULT_MODEL_NAME
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.runtime.artifacts import read
from memory_condense.runtime.config import RuntimeAssets


class FastEmbedBGE:
    allow_fp32_device_compatibility = True
    model_name = DEFAULT_MODEL_NAME
    dim = 1024

    def __init__(self, *, assets=None, admission=None):
        assets = assets or RuntimeAssets.resolve()
        admission = admission or Path(__file__).parent/'data'/'bge-admission'
        model_dir = assets.bge
        plan, result = read(admission/'plan.json'), read(admission/'report.json')
        batching = read(admission/'batching.json')
        sealed=read(admission/'admission.json')
        if sealed['report_sha256']!=hashlib.sha256((admission/'report.json').read_bytes()).hexdigest():
            raise ValueError('FastEmbed admission report changed')
        source = json.loads((model_dir/'source.json').read_text())
        if (source != plan['model'] or not result['faithful'] or plan['precision'] != 'FP32'
                or plan['pooling'] != 'CLS' or plan['normalization'] is not True
                or {k:r['fidelity']['vectors'] for k,r in result['results'].items()}
                   != {'query':24,'summary':17,'chunk':16}):
            raise ValueError('FastEmbed export lacks the required pinned transition assay')
        policy = batching.get('policy', {})
        fidelity = batching.get('fidelity', {})
        if (batching.get('format') != 'memory-condense-fastembed-batching-admission-v1'
                or policy != {'batch_size':4, 'ordering':'bge-token-length-ascending-v1'}
                or fidelity.get('max_absolute_vector_delta') != 0.0
                or fidelity.get('top10_exact_query_fraction') != 1.0):
            raise ValueError('FastEmbed batching policy lacks an exact-vector admission')
        # Setup/doctor already hash every pinned export file. Re-hashing the
        # 2.27 GB ONNX data file here made every process cold start pay for a
        # second full disk scan. Runtime admission uses the sealed manifest
        # identity and fails fast for missing files; explicit verification
        # remains available through ``memory-condense doctor --verify-models``.
        missing = [name for name in sealed['files'] if not (model_dir/name).is_file()]
        if missing:
            raise FileNotFoundError('FastEmbed export is incomplete: '+', '.join(missing))
        self.model_revision = source['revision']
        self.checkpoint_sha256 = identity_sha256(dict(source=source,files=sealed['files']))
        self.validated_source_embedding_identity = summary_embedding_identity(EmbeddingService(device='cpu',batch_size=8))
        self.execution_identity = dict(backend='fastembed-onnx-fp32-v1',device='cpu',batch_size=1,
            output_dtype='float32',pooling='cls',normalize_embeddings=True,
            compatibility_policy='sealed-bge-m3-fp32-device-export-assay-v1',
            admission_sha256=hashlib.sha256((admission/'report.json').read_bytes()).hexdigest(),
            source_embedding_identity=self.validated_source_embedding_identity)
        self.model = load(result['selected_threads'], model_dir, assets.fastembed_package)
        # Batch size is deliberately outside ``execution_identity``. The
        # production-text admission proves bit-identical vectors against the
        # batch-one reference, so changing this scheduler must not invalidate
        # existing indexes or trained BGE classifiers.
        self.preferred_background_batch_size = policy['batch_size']
        self.embedding_schedule = dict(policy, admission_sha256=hashlib.sha256(
            (admission/'batching.json').read_bytes()).hexdigest())
        import fastembed, onnxruntime
        if fastembed.__version__!=plan['fastembed'] or onnxruntime.__version__!=plan['onnxruntime']:
            raise ValueError('FastEmbed runtime differs from the admitted assay')

    def _load_model(self):
        return self.model

    def embed_query(self, query):
        return np.asarray(next(iter(self.model.embed([query],batch_size=1))),dtype=np.float32)

    def background_batch_key(self, text):
        """Return the admitted padding key from BGE's own tokenizer."""
        return len(self.model.model.tokenizer.encode(text).ids)

    def embed_queries(self, queries):
        if not queries:
            return np.zeros((0,self.dim),dtype=np.float32)
        population = list(queries)
        order = list(range(len(population)))
        if len(population) > self.preferred_background_batch_size:
            order.sort(key=lambda index:(self.background_batch_key(population[index]),index))
        ordered = [population[index] for index in order]
        generated = np.asarray(list(self.model.embed(
            ordered,batch_size=min(self.preferred_background_batch_size,len(ordered)))),dtype=np.float32)
        if generated.shape != (len(population),self.dim):
            raise ValueError('FastEmbed query embedding cardinality or dimension changed')
        restored = np.empty_like(generated)
        restored[order] = generated
        return restored

    def embed_chunks(self, chunks):
        vectors=self.embed_queries([c.text for c in chunks])
        return [c.model_copy(update={'embedding':v.tolist()}) for c,v in zip(chunks,vectors,strict=True)]

    def park(self):
        pass

    def close(self):
        self.model=None


NAME='local/bge-m3-fp32-pinned'
def load(threads, model_dir, package=None):
    if package is not None:
        sys.path.insert(0,str(package))
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource
    if not any(m['model']==NAME for m in TextEmbedding.list_supported_models()):
        TextEmbedding.add_custom_model(model=NAME,pooling=PoolingType.CLS,normalization=True,
            sources=ModelSource(hf='onnx-community/bge-m3-ONNX'),dim=1024,
            model_file='onnx/model.onnx',additional_files=['onnx/model.onnx_data'])
    return TextEmbedding(NAME,threads=threads,providers=['CPUExecutionProvider'],
        specific_model_path=str(model_dir.resolve()),local_files_only=True)
