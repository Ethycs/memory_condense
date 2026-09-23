from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
)
from memory_condense.domain.discourse import EpisodeRepresentative, identity_sha256
from memory_condense.search.episodes.representative_prefilter import (
    EpisodeRepresentativeEmbedding,
    EpisodeRepresentativePrefilterPolicy,
    prefilter_episode_representatives,
    prefilter_prescored_episode_representatives,
)


EMBEDDING_IDENTITY = {
    "backend": "fixture",
    "model_id": "fixture-embedding",
    "dimension": 2,
}
QUERY_FEATURE_SHA256 = "a" * 64
EMBEDDING_IDENTITY_SHA256 = "b" * 64
CATALOG_RECEIPT_SHA256 = "c" * 64


def _identity(chunk_id: str, vector) -> str:
    raw = [0.0 if float(value) == 0.0 else float(value) for value in vector]
    return identity_sha256(
        {"method": "ordinary_embedding", "chunk_id": chunk_id, "vector": raw}
    )


def _embedding(
    episode_id: str,
    vector,
    *,
    chunk_id: str | None = None,
    identity: str | None = None,
) -> EpisodeRepresentativeEmbedding:
    chunk_id = chunk_id or f"chunk-{episode_id}"
    return EpisodeRepresentativeEmbedding(
        representative=EpisodeRepresentative(
            episode_id=episode_id,
            chunk_id=chunk_id,
            rank=0,
            vector_identity_sha256=identity or _identity(chunk_id, vector),
        ),
        vector=vector,
    )


def _fixture():
    population = tuple(
        AssociativeMemoryCandidate(
            episode_id=episode_id,
            text=text,
            score=float(5 - index),
            route="fixture",
        )
        for index, (episode_id, text) in enumerate(
            (
                ("a", "alpha evidence"),
                ("b", "beta evidence"),
                ("c", "gamma evidence"),
                ("d", "delta evidence"),
                ("e", "quasar evidence"),
            )
        )
    )
    vectors = {
        "a": (2.0, 0.0),
        "b": (0.8, 0.6),
        "c": (0.6, 0.8),
        "d": (0.0, 1.0),
        "e": (-1.0, 0.0),
    }
    embeddings = {
        episode_id: (_embedding(episode_id, vector),)
        for episode_id, vector in vectors.items()
    }
    return population, embeddings


def _run(
    *,
    mode="apply",
    cap=3,
    top_k=2,
    cutoff_margin=0.1,
    lexical_protection_threshold=None,
    query="target question",
    query_vector=(1.0, 0.0),
    protected_episode_ids=(),
    requires_complete_frontier=False,
    embedding_identity=EMBEDDING_IDENTITY,
    population=None,
    embeddings=None,
):
    default_population, default_embeddings = _fixture()
    return prefilter_episode_representatives(
        query,
        default_population if population is None else population,
        query_vector=query_vector,
        representative_embeddings=(
            default_embeddings if embeddings is None else embeddings
        ),
        protected_episode_ids=protected_episode_ids,
        requires_complete_frontier=requires_complete_frontier,
        embedding_identity=embedding_identity,
        policy=EpisodeRepresentativePrefilterPolicy(
            mode=mode,
            cap=cap,
            top_k=top_k,
            cutoff_margin=cutoff_margin,
            lexical_protection_threshold=lexical_protection_threshold,
        ),
    )


def _run_prescored(
    ranked_scores,
    *,
    mode="apply",
    cap=3,
    top_k=2,
    cutoff_margin=0.1,
    lexical_protection_threshold=None,
    query="target question",
    protected_episode_ids=(),
    requires_complete_frontier=False,
    population=None,
    query_feature_sha256=QUERY_FEATURE_SHA256,
    embedding_identity_sha256=EMBEDDING_IDENTITY_SHA256,
    descriptor_catalog_receipt_sha256=CATALOG_RECEIPT_SHA256,
):
    default_population, _ = _fixture()
    return prefilter_prescored_episode_representatives(
        query,
        default_population if population is None else population,
        ranked_scores=ranked_scores,
        query_feature_sha256=query_feature_sha256,
        embedding_identity_sha256=embedding_identity_sha256,
        descriptor_catalog_receipt_sha256=descriptor_catalog_receipt_sha256,
        protected_episode_ids=protected_episode_ids,
        requires_complete_frontier=requires_complete_frontier,
        policy=EpisodeRepresentativePrefilterPolicy(
            mode=mode,
            cap=cap,
            top_k=top_k,
            cutoff_margin=cutoff_margin,
            lexical_protection_threshold=lexical_protection_threshold,
        ),
    )


def test_apply_uses_max_cosine_and_stable_population_tie_break() -> None:
    population, embeddings = _fixture()
    embeddings["b"] = (
        _embedding("b", (0.0, 1.0), chunk_id="chunk-b-0"),
        EpisodeRepresentativeEmbedding(
            representative=EpisodeRepresentative(
                episode_id="b",
                chunk_id="chunk-b-1",
                rank=1,
                vector_identity_sha256=_identity("chunk-b-1", (2.0, 0.0)),
            ),
            vector=(2.0, 0.0),
        ),
    )
    result = _run(population=population, embeddings=embeddings)

    assert result.status == "applied"
    assert result.reason == "applied"
    assert result.output_ids == ("a", "b", "c")
    assert result.proposal_ids == result.output_ids
    assert len(result.candidates) <= 3
    assert len(result.candidates) >= 2
    assert result.semantic_receipt.episode_scores[:2] == (("a", 1.0), ("b", 1.0))
    assert result.semantic_receipt.observed_cutoff_margin == pytest.approx(0.6)
    # The persisted identity uses the original 2.0 vector; L2 normalization is
    # solely a scoring detail and therefore does not invalidate the row.
    assert result.semantic_receipt.scored_count == len(population)


def test_shadow_returns_proposal_but_preserves_exact_full_population() -> None:
    population, embeddings = _fixture()
    result = _run(mode="shadow", population=population, embeddings=embeddings)

    assert result.status == "shadow"
    assert result.reason == "shadow_proposal"
    assert result.population is population
    assert result.candidates is population
    assert result.output_ids == tuple(item.episode_id for item in population)
    assert result.proposal_ids == ("a", "b", "c")
    assert result.exhaustive is True


def test_disabled_and_complete_frontier_do_not_touch_vector_mapping() -> None:
    class Bomb(dict):
        def __getitem__(self, key):  # pragma: no cover - assertion is no call
            raise AssertionError(key)

    population, _ = _fixture()
    disabled = _run(mode="disabled", population=population, embeddings=Bomb())
    complete = _run(
        requires_complete_frontier=True,
        population=population,
        embeddings=Bomb(),
    )

    assert disabled.reason == "disabled"
    assert complete.reason == "requires_complete_frontier"
    assert disabled.candidates is population
    assert complete.candidates is population
    assert disabled.exhaustive and complete.exhaustive


def test_explicit_and_lexical_protection_are_unioned_without_reordering_rank() -> None:
    population, embeddings = _fixture()
    result = _run(
        cap=4,
        query="Find the quasar target",
        lexical_protection_threshold=0.3,
        protected_episode_ids=("d",),
        population=population,
        embeddings=embeddings,
    )

    assert result.output_ids == ("a", "b", "d", "e")
    assert result.semantic_receipt.explicit_protected_episode_ids == ("d",)
    assert result.semantic_receipt.lexical_protected_episode_ids == ("e",)


def test_protected_union_overflow_fails_open_in_original_order() -> None:
    population, embeddings = _fixture()
    result = _run(
        cap=2,
        top_k=2,
        protected_episode_ids=("e",),
        population=population,
        embeddings=embeddings,
    )

    assert result.status == "fail_open"
    assert result.reason == "protected_union_overflow"
    assert result.candidates is population
    assert result.proposal is population


def test_cutoff_must_be_strictly_greater_than_policy_margin() -> None:
    population, embeddings = _fixture()
    embeddings["a"] = (_embedding("a", (1.0, 0.0)),)
    for episode_id in ("b", "c", "d"):
        embeddings[episode_id] = (_embedding(episode_id, (0.0, 1.0)),)
    result = _run(
        cap=1,
        top_k=1,
        cutoff_margin=1.0,
        population=population,
        embeddings=embeddings,
    )

    assert result.reason == "uncertain_cutoff_margin"
    assert result.candidates is population
    assert result.semantic_receipt.observed_cutoff_margin == 1.0


def test_shadow_retains_uncertain_bounded_proposal_for_measurement() -> None:
    population, embeddings = _fixture()
    for episode_id in ("b", "c", "d"):
        embeddings[episode_id] = (_embedding(episode_id, (0.0, 1.0)),)
    result = _run(
        mode="shadow",
        cap=1,
        top_k=1,
        cutoff_margin=1.0,
        population=population,
        embeddings=embeddings,
    )

    assert result.status == "shadow"
    assert result.reason == "uncertain_cutoff_margin"
    assert result.candidates is population
    assert result.proposal_ids == ("a",)
    assert result.semantic_receipt.observed_cutoff_margin == 1.0


@pytest.mark.parametrize(
    ("query_vector", "reason"),
    (
        (None, "missing_query_vector"),
        ((), "missing_query_vector"),
        ((float("nan"), 0.0), "invalid_query_vector"),
        ((0.0, 0.0), "invalid_query_vector"),
    ),
)
def test_invalid_query_vectors_fail_open(query_vector, reason) -> None:
    population, embeddings = _fixture()
    result = _run(
        query_vector=query_vector,
        population=population,
        embeddings=embeddings,
    )

    assert result.reason == reason
    assert result.candidates is population


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_episode", "missing_representative_vector"),
        ("null_vector", "missing_representative_vector"),
        ("wrong_dimension", "dimension_mismatch"),
        ("nonfinite", "invalid_representative_vector"),
        ("zero", "invalid_representative_vector"),
        ("identity", "vector_identity_mismatch"),
    ),
)
def test_bad_representatives_fail_open(mutation: str, reason: str) -> None:
    population, embeddings = _fixture()
    if mutation == "missing_episode":
        del embeddings["c"]
    elif mutation == "null_vector":
        embeddings["c"] = (replace(embeddings["c"][0], vector=None),)
    elif mutation == "wrong_dimension":
        embeddings["c"] = (_embedding("c", (1.0, 0.0, 0.0)),)
    elif mutation == "nonfinite":
        row = embeddings["c"][0]
        embeddings["c"] = (replace(row, vector=(float("inf"), 0.0)),)
    elif mutation == "zero":
        row = embeddings["c"][0]
        embeddings["c"] = (replace(row, vector=(0.0, 0.0)),)
    else:
        row = embeddings["c"][0]
        bad = replace(row.representative, vector_identity_sha256="f" * 64)
        embeddings["c"] = (replace(row, representative=bad),)

    result = _run(population=population, embeddings=embeddings)

    assert result.status == "fail_open"
    assert result.reason == reason
    assert result.candidates is population
    assert result.output_ids == tuple(item.episode_id for item in population)


def test_unexpected_mapping_exception_is_a_sealed_fail_open_result() -> None:
    class Bomb(dict):
        def __getitem__(self, key):
            raise RuntimeError(f"must not escape: {key}")

    population, embeddings = _fixture()
    result = _run(population=population, embeddings=Bomb(embeddings))

    assert result.reason == "prefilter_exception"
    assert result.candidates is population
    assert result.semantic_receipt.receipt_sha256


def test_representative_mapping_requires_frozen_tuple_rows() -> None:
    population, embeddings = _fixture()
    embeddings["c"] = list(embeddings["c"])

    result = _run(population=population, embeddings=embeddings)

    assert result.reason == "invalid_representative_vector"
    assert result.candidates is population


def test_receipt_is_deterministic_and_timing_is_text_and_vector_free() -> None:
    population, embeddings = _fixture()
    first = _run(population=population, embeddings=embeddings)
    second = _run(population=population, embeddings=embeddings)

    assert first.semantic_receipt == second.semantic_receipt
    assert first.semantic_receipt.receipt_sha256 == second.semantic_receipt.receipt_sha256
    assert first.elapsed_ms >= 0.0
    timing_json = json.dumps(asdict(first.timing), sort_keys=True)
    receipt_json = json.dumps(asdict(first.semantic_receipt), sort_keys=True)
    assert "target question" not in timing_json + receipt_json
    assert "alpha evidence" not in timing_json + receipt_json
    assert "[2.0, 0.0]" not in timing_json + receipt_json


def test_policy_requires_cap_at_least_top_k_and_valid_thresholds() -> None:
    with pytest.raises(ValueError, match="cap must be at least top_k"):
        EpisodeRepresentativePrefilterPolicy(cap=2, top_k=3)
    with pytest.raises(ValueError, match="cutoff_margin"):
        EpisodeRepresentativePrefilterPolicy(cutoff_margin=float("nan"))
    with pytest.raises(ValueError, match="lexical_protection_threshold"):
        EpisodeRepresentativePrefilterPolicy(lexical_protection_threshold=1.1)


def test_prescored_entry_matches_vector_selection_and_binds_catalog() -> None:
    population, embeddings = _fixture()
    vector_result = _run(
        query="Find the quasar target",
        cap=4,
        protected_episode_ids=("d",),
        lexical_protection_threshold=0.3,
        population=population,
        embeddings=embeddings,
    )
    prescored = _run_prescored(
        vector_result.semantic_receipt.episode_scores,
        query="Find the quasar target",
        cap=4,
        protected_episode_ids=("d",),
        lexical_protection_threshold=0.3,
        population=population,
    )

    assert prescored.status == vector_result.status == "applied"
    assert prescored.reason == vector_result.reason == "applied"
    assert prescored.output_ids == vector_result.output_ids == ("a", "b", "d", "e")
    assert prescored.proposal_ids == vector_result.proposal_ids
    assert prescored.semantic_receipt.episode_scores == (
        vector_result.semantic_receipt.episode_scores
    )
    assert prescored.semantic_receipt.observed_cutoff_margin == (
        vector_result.semantic_receipt.observed_cutoff_margin
    )
    assert (
        prescored.semantic_receipt.descriptor_catalog_receipt_sha256
        == CATALOG_RECEIPT_SHA256
    )
    assert "descriptor_catalog_receipt_sha256" in (
        prescored.semantic_receipt.identity_payload()
    )
    assert "descriptor_catalog_receipt_sha256" not in (
        vector_result.semantic_receipt.identity_payload()
    )


@pytest.mark.parametrize(
    ("ranked_scores", "reason"),
    (
        (
            (("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.0)),
            "missing_descriptor_score",
        ),
        (
            (
                ("a", 1.0),
                ("b", 0.8),
                ("c", 0.6),
                ("d", 0.0),
                ("d", -1.0),
            ),
            "duplicate_descriptor_score",
        ),
        (
            (
                ("a", 1.0),
                ("b", float("nan")),
                ("c", 0.6),
                ("d", 0.0),
                ("e", -1.0),
            ),
            "invalid_descriptor_score",
        ),
        (
            (("b", 0.8), ("a", 1.0), ("c", 0.6), ("d", 0.0), ("e", -1.0)),
            "invalid_descriptor_score",
        ),
    ),
)
def test_prescored_malformed_missing_and_duplicate_scores_fail_open(
    ranked_scores,
    reason: str,
) -> None:
    population, _ = _fixture()
    result = _run_prescored(ranked_scores, population=population)

    assert result.status == "fail_open"
    assert result.reason == reason
    assert result.population is population
    assert result.proposal is population
    assert result.candidates is population


def test_prescored_shadow_retains_uncertain_proposal() -> None:
    population, _ = _fixture()
    result = _run_prescored(
        (("a", 1.0), ("b", 0.0), ("c", 0.0), ("d", 0.0), ("e", -1.0)),
        mode="shadow",
        cap=1,
        top_k=1,
        cutoff_margin=1.0,
        population=population,
    )

    assert result.status == "shadow"
    assert result.reason == "uncertain_cutoff_margin"
    assert result.candidates is population
    assert result.proposal_ids == ("a",)
    assert result.semantic_receipt.observed_cutoff_margin == 1.0


def test_prescored_digests_are_validated_fail_open() -> None:
    population, _ = _fixture()
    result = _run_prescored(
        (("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.0), ("e", -1.0)),
        query_feature_sha256="not-a-digest",
        population=population,
    )

    assert result.reason == "invalid_descriptor_binding"
    assert result.candidates is population


def test_prescored_missing_catalog_binding_fails_open() -> None:
    population, _ = _fixture()
    result = _run_prescored(
        (("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.0), ("e", -1.0)),
        descriptor_catalog_receipt_sha256=None,
        population=population,
    )

    assert result.status == "fail_open"
    assert result.reason == "invalid_descriptor_binding"
    assert result.population is population
    assert result.proposal is population
    assert result.candidates is population


def test_reordering_all_candidates_is_not_exhaustive() -> None:
    population, embeddings = _fixture()
    scrambled = (population[4], *population[:4])
    result = _run(
        cap=5,
        top_k=2,
        population=scrambled,
        embeddings=embeddings,
    )

    assert result.output_ids == ("a", "b", "c", "d", "e")
    assert set(result.output_ids) == set(item.episode_id for item in scrambled)
    assert result.exhaustive is False
    assert result.semantic_receipt.exhaustive is False
