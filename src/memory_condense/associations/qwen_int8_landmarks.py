"""Compact query-independent K/V landmarks for experimental Qwen routing.

The production linker deliberately sheds token activations.  This module is a
bounded experiment for the alternative design: compile a small number of
contiguous token-bucket centroids, quantize them symmetrically per KV head, and
score later query states without re-encoding the candidate text.

Landmarks are built from *unrotated* selected-layer K/V projections.  Query
states must therefore also be unrotated.  This makes the artifact a semantic
cross-document index rather than a continuation KV cache; it does not claim to
reproduce a joint causal forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


LANDMARK_FORMAT = "memory-condense-qwen-int8-kv-landmarks-v1"


@dataclass(frozen=True, slots=True)
class Int8KVLandmarks:
    """One candidate's fixed-width, per-head symmetric INT8 K/V code."""

    keys: np.ndarray
    values: np.ndarray
    key_scales: np.ndarray
    value_scales: np.ndarray
    token_counts: np.ndarray
    source_tokens: int

    def __post_init__(self) -> None:
        keys = np.asarray(self.keys)
        values = np.asarray(self.values)
        key_scales = np.asarray(self.key_scales)
        value_scales = np.asarray(self.value_scales)
        token_counts = np.asarray(self.token_counts)
        if keys.dtype != np.int8 or values.dtype != np.int8:
            raise TypeError("landmark K/V arrays must be int8")
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("landmark K/V arrays must share [heads, landmarks, width]")
        if key_scales.shape != (keys.shape[0],) or value_scales.shape != (
            keys.shape[0],
        ):
            raise ValueError("landmark scales must contain one value per KV head")
        if key_scales.dtype != np.float16 or value_scales.dtype != np.float16:
            raise TypeError("landmark scales must be float16")
        if token_counts.shape != (keys.shape[1],) or token_counts.dtype != np.uint16:
            raise TypeError("landmark token counts must be a uint16 vector")
        if int(self.source_tokens) < 1:
            raise ValueError("source_tokens must be positive")
        if int(token_counts.sum(dtype=np.uint64)) != int(self.source_tokens):
            raise ValueError("landmark token counts must cover every source token")
        if not np.isfinite(key_scales).all() or not np.isfinite(value_scales).all():
            raise ValueError("landmark scales must be finite")
        if np.any(key_scales <= 0) or np.any(value_scales <= 0):
            raise ValueError("landmark scales must be positive")

    @property
    def retained_bytes(self) -> int:
        """Exact ndarray payload bytes, excluding container metadata."""

        return int(
            self.keys.nbytes
            + self.values.nbytes
            + self.key_scales.nbytes
            + self.value_scales.nbytes
            + self.token_counts.nbytes
        )

    @property
    def landmarks(self) -> int:
        return int(self.keys.shape[1])

    @property
    def kv_heads(self) -> int:
        return int(self.keys.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.keys.shape[2])

    def dequantized_keys(self) -> np.ndarray:
        return self.keys.astype(np.float32) * self.key_scales.astype(np.float32)[
            :, None, None
        ]

    def dequantized_values(self) -> np.ndarray:
        return self.values.astype(np.float32) * self.value_scales.astype(np.float32)[
            :, None, None
        ]


def balanced_token_buckets(tokens: int, landmarks: int) -> tuple[slice, ...]:
    """Partition all tokens into at most ``landmarks`` contiguous buckets."""

    count = int(tokens)
    limit = int(landmarks)
    if count < 1:
        raise ValueError("tokens must be positive")
    if limit < 1:
        raise ValueError("landmarks must be positive")
    buckets = min(count, limit)
    quotient, remainder = divmod(count, buckets)
    result: list[slice] = []
    cursor = 0
    for index in range(buckets):
        width = quotient + (1 if index < remainder else 0)
        result.append(slice(cursor, cursor + width))
        cursor += width
    return tuple(result)


def pool_kv_landmarks(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    landmarks: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool ``[KV heads, tokens, width]`` into fixed contiguous landmarks."""

    key_matrix = np.asarray(keys, dtype=np.float32)
    value_matrix = np.asarray(values, dtype=np.float32)
    if key_matrix.ndim != 3 or value_matrix.shape != key_matrix.shape:
        raise ValueError("K/V inputs must share [heads, tokens, width]")
    if not np.isfinite(key_matrix).all() or not np.isfinite(value_matrix).all():
        raise ValueError("K/V inputs must be finite")
    heads, tokens, width = key_matrix.shape
    buckets = balanced_token_buckets(tokens, landmarks)
    pooled_keys = np.zeros((heads, int(landmarks), width), dtype=np.float32)
    pooled_values = np.zeros_like(pooled_keys)
    counts = np.zeros(int(landmarks), dtype=np.uint16)
    for position, bucket in enumerate(buckets):
        count = int(bucket.stop - bucket.start)
        if count > np.iinfo(np.uint16).max:
            raise ValueError("one landmark bucket exceeds uint16 token capacity")
        pooled_keys[:, position, :] = key_matrix[:, bucket, :].mean(axis=1)
        pooled_values[:, position, :] = value_matrix[:, bucket, :].mean(axis=1)
        counts[position] = count
    return pooled_keys, pooled_values, counts


def _quantize_per_head(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 3:
        raise ValueError("values must have [heads, landmarks, width] shape")
    maxima = np.max(np.abs(matrix), axis=(1, 2))
    # A positive finite scale also gives all-zero heads a canonical code.
    minimum = np.float32(np.finfo(np.float16).tiny)
    scales32 = np.maximum(maxima / np.float32(127.0), minimum)
    scales = scales32.astype(np.float16)
    # Recompute with the retained scale so in-memory and serialized behavior match.
    retained = scales.astype(np.float32)
    quantized = np.rint(matrix / retained[:, None, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scales)


def compile_int8_kv_landmarks(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    landmarks: int = 8,
) -> tuple[Int8KVLandmarks, np.ndarray, np.ndarray]:
    """Pool and quantize K/V, returning FP32 centroids for matched ablations."""

    pooled_keys, pooled_values, counts = pool_kv_landmarks(
        keys,
        values,
        landmarks=landmarks,
    )
    quantized_keys, key_scales = _quantize_per_head(pooled_keys)
    quantized_values, value_scales = _quantize_per_head(pooled_values)
    code = Int8KVLandmarks(
        keys=quantized_keys,
        values=quantized_values,
        key_scales=key_scales,
        value_scales=value_scales,
        token_counts=np.ascontiguousarray(counts),
        source_tokens=int(np.asarray(keys).shape[1]),
    )
    return code, pooled_keys, pooled_values


def grouped_query_head_indices(query_heads: int, kv_heads: int) -> np.ndarray:
    """Return the standard GQA mapping from each query head to one KV head."""

    queries = int(query_heads)
    keys = int(kv_heads)
    if queries < 1 or keys < 1 or queries % keys:
        raise ValueError("query heads must be a positive multiple of KV heads")
    return np.repeat(np.arange(keys, dtype=np.int64), queries // keys)


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (
        np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)) + maximum
    ).squeeze(axis)


def score_landmark_keys(
    query_states: np.ndarray,
    keys: np.ndarray,
    token_counts: np.ndarray,
    *,
    key_group_indices: Sequence[int],
    scaling: float,
    head_vote_k: int = 4,
) -> tuple[float, np.ndarray]:
    """Score one candidate with the linker's length-normalized QK statistic.

    ``query_states`` is ``[query heads, query tokens, width]`` and ``keys`` is
    ``[KV heads, landmarks, width]``.  A bucket's token count enters as a log
    multiplicity, making each centroid approximate all tokens it represents.
    """

    queries = np.asarray(query_states, dtype=np.float32)
    landmarks = np.asarray(keys, dtype=np.float32)
    counts = np.asarray(token_counts, dtype=np.uint16)
    groups = np.asarray(tuple(key_group_indices), dtype=np.int64)
    if queries.ndim != 3 or landmarks.ndim != 3:
        raise ValueError("query and landmark keys must be rank-three arrays")
    if queries.shape[0] != len(groups) or queries.shape[2] != landmarks.shape[2]:
        raise ValueError("query heads and landmark keys are incompatible")
    if np.any(groups < 0) or np.any(groups >= landmarks.shape[0]):
        raise ValueError("key-group index lies outside the KV-head range")
    if counts.shape != (landmarks.shape[1],):
        raise ValueError("token counts do not match landmark count")
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("at least one populated landmark is required")
    vote_count = int(head_vote_k)
    if not 1 <= vote_count <= queries.shape[0]:
        raise ValueError("head_vote_k lies outside the query-head range")
    expanded = landmarks[groups][:, valid, :]
    logits = np.einsum("hqd,hld->hql", queries, expanded, optimize=True)
    logits *= np.float32(scaling)
    multiplicity = np.log(counts[valid].astype(np.float32))
    head_scores = (
        _logsumexp(logits + multiplicity[None, None, :], axis=-1)
        - math.log(int(counts[valid].sum(dtype=np.uint64)))
    ).mean(axis=-1)
    strongest = np.partition(head_scores, -vote_count)[-vote_count:]
    return float(strongest.mean()), np.ascontiguousarray(head_scores)


def score_int8_landmarks(
    query_states: np.ndarray,
    code: Int8KVLandmarks,
    *,
    key_group_indices: Sequence[int],
    scaling: float,
    head_vote_k: int = 4,
) -> tuple[float, np.ndarray]:
    """Dequantize and score one persistent landmark code."""

    return score_landmark_keys(
        query_states,
        code.dequantized_keys(),
        code.token_counts,
        key_group_indices=key_group_indices,
        scaling=scaling,
        head_vote_k=head_vote_k,
    )


def score_symmetric_int8_kv_landmarks(
    query_keys: np.ndarray,
    query_values: np.ndarray,
    code: Int8KVLandmarks,
    *,
    head_vote_k: int = 4,
) -> tuple[float, np.ndarray]:
    """Score separately encoded content through symmetric K/K and V/V cosine.

    The query readout is mean-pooled within each KV head.  Each candidate head
    is the token-count-weighted mean of its normalized landmarks.  Averaging K
    and V cosine uses both halves of the retained code while remaining free of
    candidate re-encoding and transformer weights at score time.
    """

    query_key_matrix = np.asarray(query_keys, dtype=np.float32)
    query_value_matrix = np.asarray(query_values, dtype=np.float32)
    if query_key_matrix.ndim != 3 or query_value_matrix.shape != query_key_matrix.shape:
        raise ValueError("query K/V inputs must share [KV heads, tokens, width]")
    if query_key_matrix.shape[0] != code.kv_heads or query_key_matrix.shape[2] != code.head_dim:
        raise ValueError("query K/V dimensions do not match the landmark code")
    vote_count = int(head_vote_k)
    if not 1 <= vote_count <= code.kv_heads:
        raise ValueError("head_vote_k lies outside the KV-head range")

    counts = code.token_counts.astype(np.float32)
    valid = counts > 0
    weights = counts[valid] / counts[valid].sum()

    def head_cosines(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        query_pooled = query.mean(axis=1)
        query_norms = np.linalg.norm(query_pooled, axis=-1, keepdims=True)
        query_pooled = query_pooled / np.maximum(query_norms, np.float32(1e-12))
        candidate_valid = candidates[:, valid, :]
        candidate_norms = np.linalg.norm(candidate_valid, axis=-1, keepdims=True)
        candidate_normalized = candidate_valid / np.maximum(
            candidate_norms, np.float32(1e-12)
        )
        candidate_pooled = np.einsum(
            "hld,l->hd", candidate_normalized, weights, optimize=True
        )
        candidate_pooled /= np.maximum(
            np.linalg.norm(candidate_pooled, axis=-1, keepdims=True),
            np.float32(1e-12),
        )
        return np.einsum("hd,hd->h", query_pooled, candidate_pooled, optimize=True)

    key_scores = head_cosines(query_key_matrix, code.dequantized_keys())
    value_scores = head_cosines(query_value_matrix, code.dequantized_values())
    head_scores = (key_scores + value_scores) * np.float32(0.5)
    strongest = np.partition(head_scores, -vote_count)[-vote_count:]
    return float(strongest.mean()), np.ascontiguousarray(head_scores)
