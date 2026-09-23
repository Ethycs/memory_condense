from pathlib import Path

import numpy as np
import pytest

from memory_condense.search.hot_retrieval import (
    ExactDenseAddressIndex,
    ExactSourceDescriptorIndex,
)


def test_exact_dense_search_normalizes_query_and_breaks_ties_by_chunk_id() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index = ExactDenseAddressIndex(("a", "b", "c"), matrix)

    hits = index.search(np.asarray([4.0, 0.0], dtype=np.float32), limit=3)

    assert [hit.chunk_id for hit in hits] == ["a", "b", "c"]
    assert [hit.score for hit in hits] == [1.0, 1.0, 0.0]
    assert all(hit.route == "exact_dense" for hit in hits)


def test_exact_dense_index_opens_a_read_only_npy_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "dense.npy"
    with path.open("wb") as handle:
        np.save(handle, np.eye(2, dtype=np.float32), allow_pickle=False)

    index = ExactDenseAddressIndex.open(("a", "b"), path)

    assert index.count == 2
    assert index.dimension == 2
    assert index.nbytes == 16
    assert index.search(np.asarray([0.0, 2.0], dtype=np.float32), limit=1)[0].chunk_id == "b"


def test_exact_dense_index_owns_a_read_only_copy_of_caller_matrix() -> None:
    matrix = np.eye(2, dtype=np.float32)
    index = ExactDenseAddressIndex(("a", "b"), matrix)

    matrix[:] = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    assert index.score_all(np.asarray([1.0, 0.0], dtype=np.float32)).tolist() == [
        1.0,
        0.0,
    ]
    assert not index._matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        index._matrix[0, 0] = 0.0


def test_exact_dense_index_rejects_bad_rows_queries_and_id_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ExactDenseAddressIndex(("b", "a"), np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="L2-normalized"):
        ExactDenseAddressIndex(("a",), np.asarray([[2.0, 0.0]], dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        ExactDenseAddressIndex(("a",), np.asarray([[1.0]], dtype=np.float64))

    index = ExactDenseAddressIndex(("a",), np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="wrong shape"):
        index.search(np.asarray([1.0], dtype=np.float32), limit=1)
    with pytest.raises(ValueError, match="non-zero"):
        index.search(np.zeros(2, dtype=np.float32), limit=1)
    assert index.search(np.ones(2, dtype=np.float32), limit=0) == ()


def test_dense_score_all_is_read_only_and_matches_ranked_search() -> None:
    index = ExactDenseAddressIndex(
        ("a", "b"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    scores = index.score_all(np.asarray([3.0, 4.0], dtype=np.float32))

    assert scores.tolist() == pytest.approx([0.6, 0.8])
    assert not scores.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        scores[0] = 0.0
    assert index.search(np.asarray([3.0, 4.0]), limit=2)[0].chunk_id == "b"


def test_source_descriptors_rank_centroids_and_reuse_chunk_scores() -> None:
    chunk_ids = ("a", "b", "c", "d")
    matrix = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    chunk_index = ExactDenseAddressIndex(chunk_ids, matrix)
    source_index = ExactSourceDescriptorIndex(
        chunk_ids,
        matrix,
        ("mixed", "horizontal", "mixed", "vertical"),
    )
    query = np.asarray([0.0, 5.0], dtype=np.float32)

    hits = source_index.search(
        query,
        chunk_scores=chunk_index.score_all(query),
        limit=3,
    )

    assert [hit.source_id for hit in hits] == ["vertical", "mixed", "horizontal"]
    assert [hit.representative_chunk_id for hit in hits] == ["d", "c", "b"]
    assert [hit.member_count for hit in hits] == [1, 2, 1]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5)
    assert hits[1].representative_score == pytest.approx(1.0)
    assert all(hit.route == "exact_source_centroid" for hit in hits)
    assert source_index.members("mixed") == ("a", "c")


def test_source_descriptors_break_source_and_member_ties_by_identity() -> None:
    matrix = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    index = ExactSourceDescriptorIndex(
        ("a", "b", "c", "d"),
        matrix,
        ("z-source", "z-source", "a-source", "a-source"),
    )

    hits = index.search(
        np.asarray([1.0, 1.0], dtype=np.float32),
        chunk_scores=np.full(4, 2**-0.5, dtype=np.float32),
        limit=2,
    )

    assert [hit.source_id for hit in hits] == ["a-source", "z-source"]
    assert [hit.representative_chunk_id for hit in hits] == ["c", "a"]


def test_source_descriptor_metadata_and_numeric_arrays_are_immutable() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = ExactSourceDescriptorIndex(("a", "b"), matrix, ("one", "two"))
    matrix[:] = -1.0

    assert index.chunk_ids == ("a", "b")
    assert index.chunk_source_ids == ("one", "two")
    assert index.source_ids == ("one", "two")
    assert index.source_count == 2
    assert index.chunk_count == 2
    assert index.dimension == 2
    assert index.nbytes == 48
    assert np.array_equal(index.centroid_matrix, np.eye(2, dtype=np.float32))
    assert np.linalg.norm(index.centroid_matrix, axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert not index.centroid_matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        index.centroid_matrix[0, 0] = 0.0
    with pytest.raises(AttributeError, match="immutable"):
        index._source_ids = ("changed",)  # type: ignore[misc]
    with pytest.raises(KeyError):
        index.members("missing")


@pytest.mark.parametrize(
    ("chunk_ids", "matrix", "source_ids", "error", "message"),
    [
        (("b", "a"), np.eye(2, dtype=np.float32), ("x", "y"), ValueError, "strictly increasing"),
        (("a", "b"), np.eye(2, dtype=np.float32), ("x",), ValueError, "align"),
        (("a",), np.ones((1, 1), dtype=np.float32), ("",), ValueError, "non-empty"),
        (("a",), np.ones((1, 1), dtype=np.float64), ("x",), TypeError, "float32"),
        (("a",), np.asarray([[2.0]], dtype=np.float32), ("x",), ValueError, "L2-normalized"),
        (
            ("a", "b"),
            np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
            ("x", "x"),
            ValueError,
            "zero centroid",
        ),
    ],
)
def test_source_descriptor_compile_rejects_invalid_artifacts(
    chunk_ids: tuple[str, ...],
    matrix: np.ndarray,
    source_ids: tuple[str, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ExactSourceDescriptorIndex(chunk_ids, matrix, source_ids)


def test_source_descriptor_search_rejects_misaligned_chunk_scores() -> None:
    index = ExactSourceDescriptorIndex(
        ("a", "b"),
        np.eye(2, dtype=np.float32),
        ("one", "two"),
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="align"):
        index.search(query, chunk_scores=np.ones((2, 1), dtype=np.float32), limit=1)
    with pytest.raises(TypeError, match="floating"):
        index.search(query, chunk_scores=np.ones(2, dtype=np.int64), limit=1)
    with pytest.raises(ValueError, match="non-finite"):
        index.search(query, chunk_scores=np.asarray([np.nan, 0.0]), limit=1)
    with pytest.raises(ValueError, match="cosine"):
        index.search(query, chunk_scores=np.asarray([2.0, 0.0]), limit=1)
    with pytest.raises(ValueError, match="route"):
        index.search(query, chunk_scores=np.ones(2), limit=1, route="")
    assert index.search(query, chunk_scores=np.ones(2), limit=0) == ()
