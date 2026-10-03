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
        sealed=read(admission/'admission.json')
        if sealed['report_sha256']!=hashlib.sha256((admission/'report.json').read_bytes()).hexdigest():
            raise ValueError('FastEmbed admission report changed')
        source = json.loads((model_dir/'source.json').read_text())
        if (source != plan['model'] or not result['faithful'] or plan['precision'] != 'FP32'
                or plan['pooling'] != 'CLS' or plan['normalization'] is not True
                or {k:r['fidelity']['vectors'] for k,r in result['results'].items()}
                   != {'query':24,'summary':17,'chunk':16}):
            raise ValueError('FastEmbed export lacks the required pinned transition assay')
        for name, expected in sealed['files'].items():
            with (model_dir/name).open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
                    raise ValueError('FastEmbed export changed')
        self.model_revision = source['revision']
        self.checkpoint_sha256 = identity_sha256(dict(source=source,files=sealed['files']))
        self.validated_source_embedding_identity = summary_embedding_identity(EmbeddingService(device='cpu',batch_size=8))
        self.execution_identity = dict(backend='fastembed-onnx-fp32-v1',device='cpu',batch_size=1,
            output_dtype='float32',pooling='cls',normalize_embeddings=True,
            compatibility_policy='sealed-bge-m3-fp32-device-export-assay-v1',
            admission_sha256=hashlib.sha256((admission/'report.json').read_bytes()).hexdigest(),
            source_embedding_identity=self.validated_source_embedding_identity)
        self.model = load(result['selected_threads'], model_dir, assets.fastembed_package)
        import fastembed, onnxruntime
        if fastembed.__version__!=plan['fastembed'] or onnxruntime.__version__!=plan['onnxruntime']:
            raise ValueError('FastEmbed runtime differs from the admitted assay')

    def _load_model(self):
        return self.model

    def embed_query(self, query):
        return np.asarray(next(iter(self.model.embed([query],batch_size=1))),dtype=np.float32)

    def embed_queries(self, queries):
        if not queries:
            return np.zeros((0,self.dim),dtype=np.float32)
        return np.asarray(list(self.model.embed(queries,batch_size=1)),dtype=np.float32)

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
