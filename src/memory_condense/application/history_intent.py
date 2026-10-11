"""A small decision/fix-history head on the existing BGE query vector."""
import json
from pathlib import Path

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256

FORMAT = 'memory-condense-bge-history-intent-v1'
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'runtime'/'data'/'history-intent-v1.json'


class HistoryIntent:
    def __init__(self, artifact, *, embedding_identity, dimension):
        value = json.loads(Path(artifact).read_text(encoding='utf-8')) if isinstance(artifact, (str, Path)) else dict(artifact)
        sha = value.pop('receipt_sha256', None)
        if sha != identity_sha256(value):
            raise ValueError('History classifier identity changed')
        if (value.get('format') != FORMAT or value.get('embedding_identity') != embedding_identity
                or value.get('dimension') != dimension or value.get('input_format') != 'bounded-recent-v1'):
            raise ValueError('History classifier does not match the query encoder/input')
        weights = np.array(value['weights'], dtype=np.float64, copy=True)
        intercept, threshold = value['intercept'], value['threshold']
        if (weights.shape != (dimension,) or not np.isfinite(weights).all()
                or type(intercept) not in (int, float) or not np.isfinite(intercept)
                or type(threshold) not in (int, float) or not 0 < threshold < 1):
            raise ValueError('Invalid history classifier parameters')
        weights.flags.writeable = False
        self.weights, self.intercept, self.threshold = weights, intercept, threshold
        self.receipt_sha256 = sha

    def classify(self, normalized_vector):
        logit = float(self.weights @ normalized_vector + self.intercept)
        score = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
        return dict(selected=score >= self.threshold, score=score, threshold=self.threshold,
                    classifier_sha256=self.receipt_sha256, calibrated=False)
