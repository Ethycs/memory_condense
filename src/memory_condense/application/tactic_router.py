"""Local tactic prioritization using the existing BGE encoder and a linear head.

This module proposes a tactic; it never reads memories, changes source scope,
hydrates evidence, calls a provider, or activates an unimplemented retrieval path.
The classifier head is data, not another transformer or generative model.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.summary_semantic_index import summary_embedding_identity

FORMAT = 'memory-condense-bge-tactic-router-v1'
LABELS = ('focused','overview','uncertain')
TACTICS = {'focused':'cap8','overview':'hierarchy_neighborhoods'}
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'runtime'/'data'/'tactic-router-v1.json'


def render_input(query, recent=(), *, max_query_tokens=512, max_context_tokens=256):
    """Bound recent context explicitly; never silently truncate the question."""
    if not isinstance(query,str) or not query.strip():
        raise ValueError('Tactic routing requires a nonempty question')
    if isinstance(recent,(str,bytes)) or any(not isinstance(t,str) for t in recent):
        raise TypeError('Recent context must be a sequence of text messages or summaries')
    query = query.strip()
    if count_tokens(query) > max_query_tokens:
        raise ValueError('Question exceeds tactic classifier input budget')
    recent = tuple(t.strip() for t in recent if t.strip())[-6:]
    context = '\n'.join(recent)
    # Keep the most recent suffix rather than interpreting an old topic as current.
    low, high = 0, len(context)
    while low < high:
        mid = (low+high)//2
        if count_tokens(context[mid:]) <= max_context_tokens:
            high = mid
        else:
            low = mid+1
    context = context[low:].strip()
    text = f'Recent conversation:\n{context}\n\nCurrent request:\n{query}' if context else query
    return text, bool(context)


@dataclass(frozen=True, slots=True)
class TacticDecision:
    preferred_tactic: str | None
    priorities: tuple[str, ...]
    predicted_scope: str
    scores: tuple[tuple[str, float], ...]
    margin: float
    abstained: bool
    reason: str
    contextualized: bool
    classifier_sha256: str
    input_sha256: str
    embedding_calls: int
    elapsed_s: float
    history_intent: dict | None = None


class BGETacticRouter:
    """Immutable head with an externally owned, already-resident encoder."""

    def __init__(self, encoder, artifact=None, *, history_artifact=None):
        packaged = artifact is None
        if artifact is None:
            artifact = json.loads(DEFAULT_MODEL.read_text(encoding='utf-8'))
        elif isinstance(artifact,(str,Path)):
            artifact = json.loads(Path(artifact).read_text(encoding='utf-8'))
        if not isinstance(artifact,dict):
            raise ValueError('Expected a tactic classifier object')
        value = dict(artifact)
        sha = value.pop('receipt_sha256',None)
        if sha != identity_sha256(value):
            raise ValueError('Tactic classifier identity changed')
        if value.get('format') != FORMAT or tuple(value.get('labels',())) != LABELS:
            raise ValueError('Unsupported tactic classifier')
        identity = summary_embedding_identity(encoder)
        if identity != value.get('embedding_identity'):
            raise ValueError('Tactic classifier encoder identity differs from training')
        weights = np.array(value['weights'],dtype=np.float64,copy=True)
        intercept = np.array(value['intercept'],dtype=np.float64,copy=True)
        if (weights.ndim != 2 or weights.shape != (3,value['dimension'])
                or intercept.shape != (3,) or not np.isfinite(weights).all()
                or not np.isfinite(intercept).all()):
            raise ValueError('Invalid tactic classifier weights')
        for name in ('min_score','min_margin'):
            if type(value[name]) not in (int,float) or not 0 <= value[name] <= 1:
                raise ValueError('Invalid tactic abstention threshold')
        if value.get('input_format') != 'bounded-recent-v1':
            raise ValueError('Unsupported tactic input format')
        if value.get('max_query_tokens') != 512 or value.get('max_context_tokens') != 256:
            raise ValueError('Unsupported tactic input budget')
        weights.flags.writeable = intercept.flags.writeable = False
        self.encoder, self.embedding_identity = encoder, identity
        self.weights, self.intercept = weights, intercept
        self.min_score, self.min_margin = value['min_score'],value['min_margin']
        self.receipt_sha256 = sha
        from memory_condense.application.history_intent import HistoryIntent, DEFAULT_MODEL as HISTORY_MODEL
        self.history = (HistoryIntent(history_artifact or HISTORY_MODEL,
                            embedding_identity=identity, dimension=weights.shape[1])
                        if packaged or history_artifact is not None else None)

    def prioritize(self, query, *, recent=(), input_vector=None):
        """Reuse a vector only when it embeds exactly render_input(query,recent).

        A raw query vector is suitable when recent is empty. With context, callers
        must encode the rendered pair; a query-only vector has different semantics.
        Scores are softmax classifier scores, not calibrated reliability estimates.
        """
        started = time.perf_counter()
        text, contextualized = render_input(query,recent)
        if summary_embedding_identity(self.encoder) != self.embedding_identity:
            raise ValueError('Tactic encoder identity changed')
        calls = 0
        if input_vector is None:
            vector = self.encoder.embed_query(text)
            calls = 1
        else:
            # Explicit text binding prevents accidentally reusing the retrieval
            # question embedding for a different contextual classifier input.
            if (not isinstance(input_vector,dict) or input_vector.get('text_sha256') != quote_sha256(text)
                    or input_vector.get('embedding_identity') != self.embedding_identity):
                raise ValueError('Tactic vector does not match the complete classifier input')
            vector = input_vector['vector']
        if summary_embedding_identity(self.encoder) != self.embedding_identity:
            raise ValueError('Tactic encoder identity changed during inference')
        vector = np.asarray(vector,dtype=np.float64)
        if vector.shape != (self.weights.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('Invalid tactic query vector')
        norm = np.linalg.norm(vector)
        if not norm or not np.isfinite(norm):
            raise ValueError('Tactic query vector must be nonzero and finite')
        normalized = vector/norm
        history = self.history.classify(normalized) if self.history is not None else None
        logits = self.weights @ normalized+self.intercept
        exp = np.exp(logits-logits.max())
        scores = exp/exp.sum()
        order = sorted(range(3),key=lambda i:(-float(scores[i]),i))
        top = order[0]
        margin = float(scores[top]-scores[order[1]])
        label = LABELS[top]
        reason = ('underspecified' if label == 'uncertain' else
                  'low_score' if scores[top] < self.min_score else
                  'low_margin' if margin < self.min_margin else 'classified')
        abstained = reason != 'classified'
        preferred = None if abstained else TACTICS[label]
        priorities = tuple(TACTICS[LABELS[i]] for i in order if LABELS[i] in TACTICS)
        if history is not None and history['selected']:
            preferred, abstained, reason = 'decision_history', False, 'history_intent'
            priorities = ('decision_history', *priorities)
        return TacticDecision(preferred, priorities,label,
            tuple((name,float(score)) for name,score in zip(LABELS,scores,strict=True)),
            margin,abstained,reason,contextualized,self.receipt_sha256,quote_sha256(text),
            calls,time.perf_counter()-started,history)
