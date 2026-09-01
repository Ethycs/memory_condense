from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.fast_em_fact_memory import episodic_neighborhood
from tools._routed_repair_prompts import RoutedRepairPromptError
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import canonical_json_bytes
from tools.matched_eval.query_expansion import preflight_query_expansion, run_query_expansion
from tools.matched_eval.query_fact_adapter import (
    QUERY_FACT_STAGE_ID,
    QueryFactAdapterError,
    build_query_fact_population,
    load_query_fact_population,
    parse_query_fact_compression,
    preflight_query_fact_population,
)

from tests.test_matched_eval_query_expansion import (
    _FakePartitionSearch,
    _StructuredClient,
    _candidate,
    _population,
    _valid_completion,
)


def _sealed(path: Path, payload: dict[str, object]) -> SealedArtifact:
    return SealedArtifact(
        path=path,
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )


def _arm(tmp_path: Path):
    source, namespace, query_population = _population(tmp_path)
    output = tmp_path / "query-arm"
    preflight = preflight_query_expansion(query_population, output_root=output)
    protected = source.rows[0].packet.protected_evidence[0]
    duplicate = _candidate(
        chunk_id="chunk-0",
        source_id=protected.source_id,
        text=protected.text,
        score=0.99,
    )
    cross_prefix = _candidate(
        chunk_id="cross-prefix-chunk",
        source_id="unrelated-history::episode-7",
        text="I planted rosemary and mint two weeks ago.",
        score=0.91,
    )
    result = run_query_expansion(
        query_population,
        output_root=output,
        retrievers_by_namespace={
            namespace.namespace_id: _FakePartitionSearch(
                namespace, (duplicate, cross_prefix)
            )
        },
        enable_provider=True,
        authorized_provider_calls=1,
        client=_StructuredClient(_valid_completion()),
        max_concurrency=1,
    )
    return source, query_population, preflight, result.run_artifact, output


def _build(source, query_population, preflight, run, *, max_prompt_tokens=8_000):
    return build_query_fact_population(
        source,
        query_preflight=preflight,
        query_run=run,
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_source_population_id=source.population_id,
        expected_query_preflight_sha256=preflight.sha256,
        expected_query_run_sha256=run.sha256,
        expected_query_population_id=query_population.population_id,
        expected_query_prompt_population_sha256=(
            query_population.prompt_population.prompt_population_sha256
        ),
        max_prompt_tokens=max_prompt_tokens,
    )


def test_projects_protected_s0_plus_exact_admitted_delta_and_routed_prompt(
    tmp_path: Path,
) -> None:
    source, query_population, preflight, run, output = _arm(tmp_path)

    population = _build(source, query_population, preflight, run)
    row = population.rows[0]
    root, delta = episodic_neighborhood(
        row.question,  # type: ignore[arg-type]
        stage_id=QUERY_FACT_STAGE_ID,
    )

    assert population.question_count == 1
    assert row.question.stage_ids == (
        source.rows[0].packet.stage_id,
        QUERY_FACT_STAGE_ID,
    )
    assert tuple(item.evidence_id for item in root) == tuple(
        item.evidence_id for item in source.rows[0].packet.protected_evidence
    )
    assert delta == row.admitted_delta
    assert row.dedup_excluded_ids
    assert set(row.dedup_excluded_ids) <= set(row.selected_before_dedup_ids)
    assert len(row.admitted_delta) == 1
    assert row.admitted_delta[0].source_id == "unrelated-history::episode-7"
    assert row.admitted_delta[0].text == (
        "I planted rosemary and mint two weeks ago."
    )
    assert row.compression_prompt.source_stage_id == QUERY_FACT_STAGE_ID
    assert row.compression_prompt.route_receipt_sha256 == row.route.receipt_sha256
    assert row.compression_prompt.prompt_token_proxy <= 8_000
    joined = "\n".join(message.content for message in row.compression_prompt.messages)
    assert "I planted rosemary and mint two weeks ago." in joined
    assert "Choice 0 was blue." not in joined

    compression = parse_query_fact_compression(
        row,
        '{"facts":[{"text":"Rosemary and mint were planted two weeks ago.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"planted rosemary and mint"}]}]}',
    )
    assert compression.neighborhood_evidence_ids == row.admitted_ids
    assert compression.facts[0].citations[0].source_id == (
        "unrelated-history::episode-7"
    )

    loaded = load_query_fact_population(
        tmp_path / "retrieval.json",
        query_preflight_path=output / "query-expansion-preflight.json",
        query_run_path=output / "query-expansion-run.json",
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_source_population_id=source.population_id,
        expected_query_preflight_sha256=preflight.sha256,
        expected_query_run_sha256=run.sha256,
        expected_query_population_id=query_population.population_id,
        expected_query_prompt_population_sha256=(
            query_population.prompt_population.prompt_population_sha256
        ),
        expected_question_count=1,
    )
    assert loaded.population_id == population.population_id
    assert loaded.preflight_identity_sha256 == population.preflight_identity_sha256


def test_preflight_is_zero_call_hash_bound_and_deterministic(tmp_path: Path) -> None:
    source, query_population, query_preflight, query_run, _output = _arm(tmp_path)
    first = _build(source, query_population, query_preflight, query_run)
    replay = _build(source, query_population, query_preflight, query_run)

    projection = preflight_query_fact_population(first)

    assert first.population_id == replay.population_id
    assert first.preflight_identity_sha256 == replay.preflight_identity_sha256
    assert projection == preflight_query_fact_population(replay)
    assert projection["query_preflight_sha256"] == query_preflight.sha256
    assert projection["query_run_sha256"] == query_run.sha256
    assert projection["retrieval_sha256"] == source.retrieval_sha256
    assert projection["source_population_id"] == source.population_id
    assert projection["query_population_id"] == query_population.population_id
    assert projection["dedup_policy"] == (
        "select_then_exact_s0_coordinate_dedup"
    )
    assert projection["source_prefix_filter_used"] is False
    assert projection["question_id_filter_used"] is False
    assert projection["known_history_filter_used"] is False
    assert projection["provider_calls"] == projection["new_provider_calls"] == 0
    assert projection["writes"] == 0
    assert projection["gold_loaded"] is False
    assert projection["compression_prompt_population_sha256"] == (
        first.compression_prompt_population.prompt_population_sha256
    )


def test_rejects_a_resealed_run_that_dedups_before_selection(tmp_path: Path) -> None:
    source, query_population, preflight, run, _output = _arm(tmp_path)
    changed = copy.deepcopy(run.payload)
    row = changed["questions"][0]
    excluded = set(row["dedup_excluded_candidate_ids"])
    row["selected_before_dedup_candidate_ids"] = [
        value for value in row["selected_before_dedup_candidate_ids"]
        if value not in excluded
    ]
    row["dedup_excluded_candidate_ids"] = []
    unsigned = dict(row)
    unsigned.pop("receipt_sha256")
    row["receipt_sha256"] = identity_sha256(unsigned)
    changed_run = _sealed(tmp_path / "resealed-query-run.json", changed)

    with pytest.raises(QueryFactAdapterError, match="select before S0 dedup"):
        _build(source, query_population, preflight, changed_run)


def test_rejects_resealed_candidate_with_changed_exact_provenance(
    tmp_path: Path,
) -> None:
    source, query_population, preflight, run, _output = _arm(tmp_path)
    changed = copy.deepcopy(run.payload)
    row = changed["questions"][0]
    row["admitted_candidates"][0]["source_id"] = "another-history::episode"
    unsigned = dict(row)
    unsigned.pop("receipt_sha256")
    row["receipt_sha256"] = identity_sha256(unsigned)
    changed_run = _sealed(tmp_path / "resealed-query-run.json", changed)

    with pytest.raises(QueryFactAdapterError, match="exact provenance"):
        _build(source, query_population, preflight, changed_run)


def test_hard_compression_cap_fails_before_any_provider_boundary(
    tmp_path: Path,
) -> None:
    source, query_population, preflight, run, _output = _arm(tmp_path)
    population = _build(source, query_population, preflight, run)
    observed = population.rows[0].compression_prompt.prompt_token_proxy

    with pytest.raises(RoutedRepairPromptError, match="exceeds its cap"):
        _build(
            source,
            query_population,
            preflight,
            run,
            max_prompt_tokens=observed - 1,
        )
