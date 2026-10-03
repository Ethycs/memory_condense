"""A multi-history union must preserve the semantic meaning of every row."""
from dataclasses import replace

import numpy as np
import pytest

from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex
from tests.test_semantic_section_index import hierarchy
from tools.run_memory_scale_stress import align_atomic_vectors


def test_interleaved_source_histories_keep_their_dense_routes():
    _, source = hierarchy()
    leaves = [s for s in source.sections if not s.child_section_ids]
    # Source A contains 0,2; source B contains 1. The union sorts them to 0,1,2.
    sections = [leaves[0], leaves[2], leaves[1]]
    vectors = np.eye(3, dtype=np.float32)
    combined = SectionSummaryIndex(sections)
    aligned = align_atomic_vectors(combined, sections, vectors)
    index = SemanticSectionIndex(combined, aligned, embedding_identity='frozen-test')
    for section, query in zip(sections, vectors, strict=True):
        result = index.route_vector('unrelated', query, embedding_identity='frozen-test', max_sections=1)
        assert result.routes[0].section.section_id == section.section_id
    np.testing.assert_array_equal(aligned, vectors[[0, 2, 1]])


def test_changed_or_duplicate_source_bindings_are_rejected():
    _, source = hierarchy()
    leaves = [s for s in source.sections if not s.child_section_ids]
    combined = SectionSummaryIndex(leaves)
    with pytest.raises(ValueError, match='Duplicate'):
        align_atomic_vectors(combined, [leaves[0]]*3, np.eye(3, dtype=np.float32))
    changed = [replace(leaves[0], summary='different', receipt_sha256=''), *leaves[1:]]
    with pytest.raises(ValueError, match='source section changed'):
        align_atomic_vectors(combined, changed, np.eye(3, dtype=np.float32))
