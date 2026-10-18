from __future__ import annotations

import numpy as np
import pytest

from memory_condense.associations.qwen_int8_landmarks import (
    Int8KVLandmarks,
    balanced_token_buckets,
    compile_int8_kv_landmarks,
    grouped_query_head_indices,
    score_int8_landmarks,
    score_landmark_keys,
    score_symmetric_int8_kv_landmark_batch,
    score_symmetric_int8_kv_landmarks,
)


def test_eight_int8_landmarks_have_the_expected_payload_size() -> None:
    rng = np.random.default_rng(11)
    keys = rng.normal(size=(8, 128, 128)).astype(np.float32)
    values = rng.normal(size=(8, 128, 128)).astype(np.float32)

    code, pooled_keys, pooled_values = compile_int8_kv_landmarks(
        keys,
        values,
        landmarks=8,
    )

    assert isinstance(code, Int8KVLandmarks)
    assert code.keys.shape == (8, 8, 128)
    assert code.values.shape == (8, 8, 128)
    assert code.retained_bytes == 16_432
    assert code.source_tokens == 128
    assert code.token_counts.tolist() == [16] * 8
    assert pooled_keys.shape == code.keys.shape
    assert pooled_values.shape == code.values.shape


def test_balanced_landmarks_cover_short_inputs_without_empty_buckets() -> None:
    assert balanced_token_buckets(3, 8) == (
        slice(0, 1),
        slice(1, 2),
        slice(2, 3),
    )
    keys = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    code, pooled_keys, _ = compile_int8_kv_landmarks(keys, keys, landmarks=8)

    assert code.token_counts.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(pooled_keys[:, :3, :], keys)


def test_int8_quantization_preserves_fp32_landmark_ranking() -> None:
    rng = np.random.default_rng(19)
    query = rng.normal(size=(8, 3, 16)).astype(np.float32)
    groups = grouped_query_head_indices(8, 2)
    fp32_scores: list[float] = []
    int8_scores: list[float] = []
    for bias in (-0.8, -0.1, 0.4, 1.1):
        keys = rng.normal(scale=0.25, size=(2, 32, 16)).astype(np.float32)
        keys += bias * query.reshape(8, -1).mean(axis=0)[:16][None, None, :]
        values = rng.normal(size=(2, 32, 16)).astype(np.float32)
        code, pooled_keys, _ = compile_int8_kv_landmarks(
            keys,
            values,
            landmarks=8,
        )
        fp32_score, _ = score_landmark_keys(
            query,
            pooled_keys,
            code.token_counts,
            key_group_indices=groups,
            scaling=0.25,
            head_vote_k=2,
        )
        int8_score, _ = score_int8_landmarks(
            query,
            code,
            key_group_indices=groups,
            scaling=0.25,
            head_vote_k=2,
        )
        fp32_scores.append(fp32_score)
        int8_scores.append(int8_score)

    assert np.argsort(fp32_scores).tolist() == np.argsort(int8_scores).tolist()
    np.testing.assert_allclose(int8_scores, fp32_scores, rtol=0.01, atol=0.01)


def test_landmark_code_rejects_counts_that_do_not_cover_source() -> None:
    with pytest.raises(ValueError, match="cover every source token"):
        Int8KVLandmarks(
            keys=np.zeros((2, 2, 4), dtype=np.int8),
            values=np.zeros((2, 2, 4), dtype=np.int8),
            key_scales=np.ones(2, dtype=np.float16),
            value_scales=np.ones(2, dtype=np.float16),
            token_counts=np.asarray([1, 0], dtype=np.uint16),
            source_tokens=2,
        )


def test_grouped_query_heads_requires_an_exact_multiple() -> None:
    np.testing.assert_array_equal(
        grouped_query_head_indices(8, 2),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="multiple"):
        grouped_query_head_indices(7, 2)


def test_symmetric_kv_score_prefers_a_matching_landmark_code() -> None:
    rng = np.random.default_rng(23)
    query_keys = rng.normal(size=(2, 3, 8)).astype(np.float32)
    query_values = rng.normal(size=(2, 3, 8)).astype(np.float32)
    matching_keys = np.repeat(query_keys.mean(axis=1)[:, None, :], 16, axis=1)
    matching_values = np.repeat(query_values.mean(axis=1)[:, None, :], 16, axis=1)
    opposite_keys = -matching_keys
    opposite_values = -matching_values
    matching, _, _ = compile_int8_kv_landmarks(
        matching_keys, matching_values, landmarks=8
    )
    opposite, _, _ = compile_int8_kv_landmarks(
        opposite_keys, opposite_values, landmarks=8
    )

    matching_score, matching_heads = score_symmetric_int8_kv_landmarks(
        query_keys, query_values, matching, head_vote_k=1
    )
    opposite_score, _ = score_symmetric_int8_kv_landmarks(
        query_keys, query_values, opposite, head_vote_k=1
    )

    assert matching_score > 0.99
    assert matching_heads.shape == (2,)
    assert opposite_score < -0.99


def test_vectorized_symmetric_score_matches_scalar_candidates() -> None:
    rng = np.random.default_rng(29)
    query_keys = rng.normal(size=(2, 5, 8)).astype(np.float32)
    query_values = rng.normal(size=(2, 5, 8)).astype(np.float32)
    codes = []
    for _ in range(7):
        code, _, _ = compile_int8_kv_landmarks(
            rng.normal(size=(2, 19, 8)).astype(np.float32),
            rng.normal(size=(2, 19, 8)).astype(np.float32),
            landmarks=8,
        )
        codes.append(code)

    batch_scores, batch_heads = score_symmetric_int8_kv_landmark_batch(
        query_keys,
        query_values,
        keys=np.ascontiguousarray(np.stack([code.keys for code in codes])),
        values=np.ascontiguousarray(np.stack([code.values for code in codes])),
        key_scales=np.ascontiguousarray(np.stack([code.key_scales for code in codes])),
        value_scales=np.ascontiguousarray(np.stack([code.value_scales for code in codes])),
        token_counts=np.ascontiguousarray(np.stack([code.token_counts for code in codes])),
        head_vote_k=1,
    )
    def reference(code):
        counts = code.token_counts.astype(np.float32)
        valid = counts > 0
        weights = counts[valid] / counts[valid].sum()

        def head_cosines(query, candidates):
            query_pooled = query.mean(axis=1)
            query_pooled /= np.maximum(
                np.linalg.norm(query_pooled, axis=-1, keepdims=True), 1e-12
            )
            candidate_valid = candidates[:, valid, :]
            candidate_valid /= np.maximum(
                np.linalg.norm(candidate_valid, axis=-1, keepdims=True), 1e-12
            )
            candidate_pooled = np.einsum(
                "hld,l->hd", candidate_valid, weights, optimize=True
            )
            candidate_pooled /= np.maximum(
                np.linalg.norm(candidate_pooled, axis=-1, keepdims=True), 1e-12
            )
            return np.einsum(
                "hd,hd->h", query_pooled, candidate_pooled, optimize=True
            )

        heads = (
            head_cosines(query_keys, code.dequantized_keys())
            + head_cosines(query_values, code.dequantized_values())
        ) * np.float32(0.5)
        return float(np.max(heads)), heads

    scalar = [reference(code) for code in codes]

    np.testing.assert_allclose(batch_scores, [row[0] for row in scalar], atol=1e-7)
    np.testing.assert_allclose(batch_heads, [row[1] for row in scalar], atol=1e-7)
    assert batch_scores.flags.c_contiguous
    assert batch_heads.flags.c_contiguous
