from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
    ValidationShardPreflight,
)
from tools.assay_episode_representative_prefilter import (
    _artifact_validation_preflight,
    _campaign_root_from_shard_root,
    _historical_final_stage_boundary_aggregates,
    _parser,
)


def _preflight(tmp_path: Path) -> ValidationShardPreflight:
    return ValidationShardPreflight(
        sample=SimpleNamespace(questions=()),
        shard_identity={},
        population_identity={},
        policy=SimpleNamespace(),
        sample_offset=0,
        shard_root=tmp_path / "shards" / "offset-000",
        qwen_prefix_model_dir=tmp_path / "prefix",
        qwen_choice_model_dir=tmp_path / "choice",
        retrieval_implementation_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        source_embedding_device="cuda",
    )


def test_campaign_root_requires_exact_locked_shard_layout(tmp_path: Path) -> None:
    shard = tmp_path / "campaign" / "shards" / "offset-000"

    assert _campaign_root_from_shard_root(shard, sample_offset=0) == (
        tmp_path / "campaign"
    ).resolve()
    with pytest.raises(ValueError, match="exact validation shard root"):
        _campaign_root_from_shard_root(
            tmp_path / "campaign" / "offset-000",
            sample_offset=0,
        )


def test_historical_preflight_rebinds_only_sealed_implementation(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path)
    rebound = _artifact_validation_preflight(
        preflight,
        {
            "retrieval_implementation_sha256": "c" * 64,
            "environment_lock_sha256": "b" * 64,
        },
    )

    assert rebound.retrieval_implementation_sha256 == "c" * 64
    assert rebound.environment_lock_sha256 == preflight.environment_lock_sha256
    assert rebound.population_identity is preflight.population_identity
    assert rebound.policy is preflight.policy
    with pytest.raises(ValueError, match="environment differs"):
        _artifact_validation_preflight(
            preflight,
            {
                "retrieval_implementation_sha256": "c" * 64,
                "environment_lock_sha256": "d" * 64,
            },
        )


def test_final_stage_boundary_aggregates_are_content_free_and_deterministic() -> None:
    rows = []
    for index in range(1, 11):
        rows.append(
            {
                "stages": [
                    {
                        "stage_id": "S3",
                        "stage_receipt": {
                            "prompt_token_proxy": index * 10,
                            "context_token_proxy": index * 5,
                        },
                        "evidence": [{"text": "not exported"}] * index,
                        "provider_messages": [
                            {"role": "user", "content": f"memory {index} ☃"}
                        ],
                    }
                ]
            }
        )

    result = _historical_final_stage_boundary_aggregates(rows)

    assert result["stage_id"] == "S3"
    assert result["question_count"] == 10
    assert result["p95_method"] == "nearest_rank"
    assert result["metrics"]["prompt_token_proxy"] == {
        "count": 10,
        "min": 10,
        "median": 55.0,
        "mean": 55.0,
        "p95": 100,
        "max": 100,
    }
    assert result["metrics"]["selected_evidence_count"]["max"] == 10
    assert "memory" not in repr(result)


def test_parser_defaults_to_first_locked_validation_shard(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--store-root",
            str(tmp_path / "shards" / "offset-000"),
            "--output",
            str(tmp_path / "assay.json"),
        ]
    )

    assert args.sample_offset == 0
    assert args.retrieval is None
