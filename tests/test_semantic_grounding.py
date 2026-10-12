import math

import numpy as np
import pytest

from memory_condense.runtime.semantic_grounding import (
    BGESourceGrounder,
    exact_candidate_spans,
    semantic_candidate_spans,
)


class Embedding:
    model_name = 'BAAI/bge-m3'
    model_revision = 'test-revision'

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def embed_queries(self, texts):
        self.calls.append(tuple(texts))
        result = []
        for text in texts:
            score = self.scores.get(text, 0.0)
            vector = np.zeros(1024, dtype=np.float32)
            vector[0] = score
            vector[1] = math.sqrt(max(0.0, 1.0-score*score))
            result.append(vector)
        return np.stack(result)


def policy(pre=0.8, reviewed=0.6):
    return {
        'format': 'memory-condense-bge-summary-validator-calibration-v1',
        'model_id': 'BAAI/bge-m3',
        'model_revision': 'test-revision',
        'candidate_algorithm': 'lexical-prefilter-top4-exact-source-v2',
        'pre_review_cutoff': pre,
        'reviewed_cutoff': reviewed,
    }


def test_candidate_spans_are_bounded_exact_source_substrings():
    source = ('First sentence establishes context. For example, creating short-form videos '
              'could appeal to younger viewers. Final sentence closes the topic.')
    values = exact_candidate_spans(source, 'create short-form videos')
    assert values
    assert all(value in source for value in values)
    assert any('creating short-form videos' in value for value in values)


def test_semantic_candidate_prefilter_caps_embedding_population_and_keeps_match():
    source = ' '.join(f'Unrelated sentence {index}.' for index in range(20))
    source += ' For example, creating short-form videos could appeal to younger viewers.'
    values = semantic_candidate_spans(source, 'create short-form videos')
    assert 1 <= len(values) <= 4
    assert any('creating short-form videos' in value for value in values)


def test_grounder_uses_strict_then_reviewed_cutoff_and_caches_source_vectors():
    source = 'For example, creating short-form videos could appeal to younger viewers.'
    proposal = 'create short-form videos'
    candidates = semantic_candidate_spans(source, proposal)
    target = next(value for value in candidates if 'creating short-form videos' in value)
    scores = {proposal: 1.0, target: 0.7}
    embedding = Embedding(scores)
    grounder = BGESourceGrounder(embedding, policy=policy())
    assert grounder(source, proposal) is None
    assert grounder(source, proposal, reviewed=True) == target
    candidate_batches = [call for call in embedding.calls if call != (proposal,)]
    assert len(candidate_batches) == 1


def test_grounder_rejects_calibration_for_another_encoder():
    mismatched = policy()
    mismatched['model_revision'] = 'other'
    with pytest.raises(ValueError, match='does not match'):
        BGESourceGrounder(Embedding({}), policy=mismatched)


@pytest.mark.parametrize('proposal', ['Release r19 is planned.', 'Release r18 deployed.'])
def test_grounder_does_not_use_similarity_to_overwrite_critical_facts(proposal):
    source = 'Release r18 is planned, not deployed.'
    candidates = semantic_candidate_spans(source, proposal)
    scores = {proposal: 1.0, **{candidate: 0.99 for candidate in candidates}}
    grounder = BGESourceGrounder(Embedding(scores), policy=policy(pre=0.5, reviewed=0.4))
    assert grounder(source, proposal, reviewed=True) is None
