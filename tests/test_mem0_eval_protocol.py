from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.eval.sample_identity import canonical_sha256
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.eval.reproducibility import file_sha256
from tools.mem0_eval import bootstrap
from tools.mem0_eval import preflight
from tools.mem0_eval import protocol
from tools.mem0_eval.protocol import (
    Mem0ComparisonProtocolError,
    build_composite_add_batches,
    compose_raw_stress_record,
    count_official_add_requests,
)


def _record(
    question_id: str,
    *,
    session_id: str,
    date: str,
    turns: list[dict[str, str]],
) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": f"question {question_id}",
        "answer": "answer",
        "haystack_sessions": [turns],
        "haystack_session_ids": [session_id],
        "haystack_dates": [date],
        "answer_session_ids": [session_id],
    }


def _source_policy_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "src" / "memory_condense"
    selection = repo / "docs" / "selection.json"
    source.mkdir(parents=True)
    selection.parent.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pixi.lock").write_text("locked\n", encoding="utf-8")
    selection.write_text('{"selected":true}\n', encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    policy = tmp_path / "policy.json"
    dataset.write_text("[]\n", encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    payload = {
        "format": "memory-condense-retrieval-policy-v1",
        "status": "validation_frozen",
        "split": "validation",
        "claim_profile": "longmemeval-s-1m-100q-95-v1",
        "dataset_sha256": file_sha256(dataset),
        "split_manifest": split.name,
        "split_manifest_sha256": file_sha256(split),
        "implementation_sha256": implementation_sha256(source),
        "environment_lock_sha256": file_sha256(repo / "pixi.lock"),
        "selection_artifact_required": True,
        "selection_artifact": "docs/selection.json",
        "selection_artifact_sha256": file_sha256(selection),
        "evaluation": {
            "responder_model": "openai/codex_sdk/gpt-5.6-terra",
            "judge_model": "openai/codex_sdk/gpt-5.6-sol",
            "embedding_device": "cuda",
            "benchmark_format": "longmemeval",
            "use_judge": True,
            "provider_retries": 0,
            "max_provider_calls": 20,
            "max_prompt_tokens": 8000,
            "prompt_cap_semantics": (
                "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
            ),
            "prompt_token_proxy_identity": {
                "schema": "memory-condense-prompt-token-proxy-v1",
                "implementation": "tiktoken",
                "implementation_version": "0.13.0",
                "encoding": "cl100k_base",
                "vocabulary_sha256": (
                    "8cd4fc3b76f9fdaf9df7d14f20a41eda79ce45b3e9c5ae8f68b0a41a59c3a9c9"
                ),
                "chat_framing_tokens_per_message": 8,
                "chat_framing_tokens_fixed": 8,
            },
            "responder_output_token_reserve": 256,
            "recent_window": 4,
            "accuracy_target": 0.95,
            "min_target_questions": 100,
            "stress_context_tokens": 1_000_000,
            "stress_questions": 10,
            "stress_question_offset": 0,
            "max_samples": 1,
            "sample_offsets": list(range(0, 100, 10)),
        },
    }
    policy.write_text(json.dumps(payload), encoding="utf-8")
    return repo, dataset, split, policy


def _locked_population_fixture(
    tmp_path: Path,
    records: list[dict],
) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset.json"
    manifest = tmp_path / "split.json"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "format": "memory-condense-locked-benchmark-split-v1",
                "dataset_sha256": file_sha256(dataset),
                "salt": "mem0-snapshot-test",
                "splits": {"validation": len(records)},
                "algorithm": "stratified-largest-remainder-v1",
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def test_raw_stress_record_namespaces_sources_and_preserves_raw_turns():
    first_turns = [
        {"role": "assistant", "content": "first assistant"},
        {"role": "user", "content": "first user"},
    ]
    second_turns = [{"role": "user", "content": "singleton"}]
    combined = compose_raw_stress_record(
        [
            _record(
                "q1",
                session_id="shared",
                date="2024/02/02 (Fri) 10:00",
                turns=first_turns,
            ),
            _record(
                "q2",
                session_id="shared",
                date="2024/01/01 (Mon) 09:00",
                turns=second_turns,
            ),
        ],
        sample_id="mem0-shard",
    )

    assert combined["question_id"] == "mem0-shard"
    assert combined["format"] == "memory-condense-mem0-raw-history-bundle-v1"
    assert [row["source_sample_id"] for row in combined["records"]] == [
        "q1",
        "q2",
    ]
    assert combined["records"][0]["haystack_session_ids"] == ["q1::shared"]
    assert combined["records"][1]["haystack_session_ids"] == ["q2::shared"]
    assert combined["records"][0]["haystack_sessions"] == [first_turns]
    assert combined["records"][1]["haystack_sessions"] == [second_turns]


def test_add_batches_sort_within_record_but_never_globally_interleave_records():
    first = {
        "question_id": "q1",
        "haystack_sessions": [
            [{"role": "user", "content": "q1 late"}],
            [{"role": "user", "content": "q1 early"}],
        ],
        "haystack_session_ids": ["late", "early"],
        "haystack_dates": [
            "2024/02/02 (Fri) 10:00",
            "2024/01/01 (Mon) 09:00",
        ],
    }
    second = _record(
        "q2",
        session_id="earliest",
        date="2023/01/01 (Sun) 09:00",
        turns=[{"role": "user", "content": "q2 earliest"}],
    )

    batches = build_composite_add_batches([first, second])

    assert [batch.messages[0][1] for batch in batches] == [
        "q1 early",
        "q1 late",
        "q2 earliest",
    ]
    assert [batch.source for batch in batches] == [
        "q1::early",
        "q1::late",
        "q2::earliest",
    ]


def test_add_request_count_matches_consecutive_pairs_and_empty_pair_skip():
    record = _record(
        "q1",
        session_id="s1",
        date="2024/01/01 (Mon) 09:00",
        turns=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "four"},
            {"role": "user", "content": "five"},
        ],
    )

    counts = count_official_add_requests(record)

    assert counts.raw_pairs == 3
    assert counts.skipped_empty_pairs == 1
    assert counts.add_requests == 2
    assert counts.whitespace_only_pairs == 0


def test_whitespace_only_message_fails_before_protocols_can_diverge():
    record = _record(
        "q1",
        session_id="s1",
        date="2024/01/01 (Mon) 09:00",
        turns=[
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "reply"},
        ],
    )

    with pytest.raises(Mem0ComparisonProtocolError, match="whitespace-only"):
        count_official_add_requests(record)


def test_raw_stress_record_rejects_duplicate_source_records():
    record = _record(
        "q1",
        session_id="s1",
        date="2024/01/01 (Mon) 09:00",
        turns=[{"role": "user", "content": "one"}],
    )

    with pytest.raises(Mem0ComparisonProtocolError, match="duplicate source"):
        compose_raw_stress_record([record, record], sample_id="shard")


def test_locked_population_uses_one_snapshot_if_path_is_replaced_between_parses(
    tmp_path,
    monkeypatch,
):
    original = _record(
        "q-original",
        session_id="s-original",
        date="2024/01/01 (Mon) 09:00",
        turns=[{"role": "user", "content": "original fact"}],
    )
    replacement = _record(
        "q-replacement",
        session_id="s-replacement",
        date="2025/01/01 (Wed) 09:00",
        turns=[{"role": "user", "content": "replacement fact"}],
    )
    dataset, manifest = _locked_population_fixture(tmp_path, [original])
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(json.dumps([replacement]), encoding="utf-8")
    real_parse = protocol.parse_longmemeval
    parse_calls = 0

    def parse_then_replace_path(payload):
        nonlocal parse_calls
        parse_calls += 1
        normalized = real_parse(payload)
        # Poison the normalized parser's mutable decoded object as well as the
        # source path. Raw reconstruction must still decode the captured bytes.
        payload.clear()
        payload.append(replacement)
        replacement_path.replace(dataset)
        return normalized

    monkeypatch.setattr(protocol, "parse_longmemeval", parse_then_replace_path)

    population = protocol.load_locked_raw_population(
        benchmark_file=dataset,
        split_manifest=manifest,
    )

    assert parse_calls == 1
    assert [sample.sample_id for sample in population.validation] == ["q-original"]
    assert set(population.raw_by_id) == {"q-original"}
    assert population.raw_by_id["q-original"]["haystack_sessions"] == [
        [{"role": "user", "content": "original fact"}]
    ]
    assert json.loads(dataset.read_text(encoding="utf-8"))[0]["question_id"] == (
        "q-replacement"
    )


def test_locked_population_fails_before_parsing_if_replacement_precedes_snapshot(
    tmp_path,
    monkeypatch,
):
    original = _record(
        "q-original",
        session_id="s-original",
        date="2024/01/01 (Mon) 09:00",
        turns=[{"role": "user", "content": "original fact"}],
    )
    replacement = _record(
        "q-replacement",
        session_id="s-replacement",
        date="2025/01/01 (Wed) 09:00",
        turns=[{"role": "user", "content": "replacement fact"}],
    )
    dataset, manifest = _locked_population_fixture(tmp_path, [original])
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(json.dumps([replacement]), encoding="utf-8")
    replacement_path.replace(dataset)
    parse_called = False

    def forbidden_parse(_payload):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("unverified snapshot must not be parsed")

    monkeypatch.setattr(protocol, "parse_longmemeval", forbidden_parse)

    with pytest.raises(Mem0ComparisonProtocolError, match="snapshot SHA-256"):
        protocol.load_locked_raw_population(
            benchmark_file=dataset,
            split_manifest=manifest,
        )

    assert parse_called is False


def test_preflight_cross_checks_population_and_reports_honest_usage(monkeypatch):
    question = BenchmarkQuestion(
        question_id="q1",
        question="What?",
        answer="answer",
    )
    sample = BenchmarkSample(
        sample_id="context-stress-10",
        turns=[("user", "fact")],
        turn_source_ids=["q1::s1"],
        questions=[question],
    )
    raw = _record(
        "q1",
        session_id="s1",
        date="2024/01/01 (Mon) 09:00",
        turns=[{"role": "user", "content": "fact"}],
    )
    from memory_condense.eval.sample_identity import sample_sha256
    from tools.mem0_eval.protocol import RawStressShard

    raw_bundle = compose_raw_stress_record([raw], sample_id="raw")
    shard = RawStressShard(
        sample_offset=0,
        parsed_sample=sample,
        sample_sha256=sample_sha256(sample),
        history_sample_ids=("q1",),
        raw_history_bundle=raw_bundle,
        raw_history_bundle_sha256=canonical_sha256(raw_bundle),
        add_batches=build_composite_add_batches([raw]),
        add_counts=count_official_add_requests(raw),
    )
    plan = preflight.SourceValidationPlan(
        dataset_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        policy_manifest_sha256="d" * 64,
        implementation_sha256="e" * 64,
        environment_lock_sha256="f" * 64,
        sample_offsets=(0,),
        target_tokens=10,
        questions_per_shard=1,
        evaluation_identity={"responder_model": "responder"},
    )
    monkeypatch.setattr(preflight, "load_source_validation_plan", lambda **_: plan)
    monkeypatch.setattr(preflight, "build_raw_stress_shards", lambda **_: (shard,))
    monkeypatch.setattr(
        preflight,
        "shard_receipt",
        lambda _: {
            "sample_offset": 0,
            "transcript_tokens": 1,
            "raw_pairs": 1,
            "skipped_empty_pairs": 0,
            "mem0_add_requests": 1,
        },
    )
    monkeypatch.setattr(preflight, "tool_implementation_sha256", lambda: "2" * 64)
    monkeypatch.setattr(
        preflight,
        "load_locked_raw_population",
        lambda **_: type("Population", (), {"validation": (sample,)})(),
    )

    receipt = preflight.build_preflight_receipt(
        benchmark_file="dataset.json",
        split_manifest="split.json",
        policy_manifest="policy.json",
    )

    assert receipt["status"] == "provider_free_ready"
    assert receipt["totals"]["questions"] == 1
    assert receipt["totals"]["mem0_add_operations"] == 1
    assert receipt["totals"]["expected_logical_extraction_calls"] == 1
    assert receipt["totals"]["logical_extraction_calls_per_add"] == 1
    assert receipt["totals"]["answer_judge_provider_calls"] == 2
    assert receipt["totals"]["underlying_mem0_provider_calls"] is None
    assert receipt["supports_exact_source_provenance"] is False
    assert receipt["source_evaluation_identity"] == {
        "responder_model": "responder"
    }


def test_source_plan_binds_the_exact_scoring_and_prompt_identity(tmp_path):
    repo, dataset, split, policy = _source_policy_fixture(tmp_path)

    plan = preflight.load_source_validation_plan(
        benchmark_file=dataset,
        split_manifest=split,
        policy_manifest=policy,
        repository_root=repo,
    )

    assert plan.sample_offsets == tuple(range(0, 100, 10))
    assert plan.evaluation_identity["responder_model"] == (
        "openai/codex_sdk/gpt-5.6-terra"
    )
    assert plan.evaluation_identity["judge_model"] == (
        "openai/codex_sdk/gpt-5.6-sol"
    )
    assert plan.evaluation_identity["max_prompt_tokens"] == 8000
    assert plan.evaluation_identity["responder_output_token_reserve"] == 256
    assert plan.evaluation_identity["max_provider_calls_per_shard"] == 20


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("provider_retries", 1, "retries must be zero"),
        ("max_provider_calls", 19, "authorization is inconsistent"),
        ("use_judge", False, "must use the frozen judge"),
    ],
)
def test_source_plan_rejects_scoring_identity_drift(
    tmp_path,
    field,
    value,
    match,
):
    repo, dataset, split, policy = _source_policy_fixture(tmp_path)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["evaluation"][field] = value
    policy.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        preflight.load_source_validation_plan(
            benchmark_file=dataset,
            split_manifest=split,
            policy_manifest=policy,
            repository_root=repo,
        )


def test_isolated_bootstrap_hash_matches_frozen_source_fingerprint():
    assert bootstrap._tree_sha256(  # noqa: SLF001 - executable contract check
        Path(__file__).resolve().parents[1] / "src" / "memory_condense"
    ) == implementation_sha256()


def test_isolated_bootstrap_fails_before_launch_on_source_drift(tmp_path):
    source = tmp_path / "src" / "memory_condense"
    tool = tmp_path / "tools" / "mem0_eval"
    source.mkdir(parents=True)
    tool.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tool / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="frozen memory-condense source mismatch"):
        bootstrap.main(
            [
                "--source-root",
                str(source),
                "--tool-root",
                str(tool),
                "--expected-source-sha256",
                "0" * 64,
                "--expected-tool-sha256",
                bootstrap._tree_sha256(tool),  # noqa: SLF001
                "--module",
                "tools.mem0_eval.preflight",
            ]
        )


def test_isolated_bootstrap_sets_offline_guards_and_forwards_arguments(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source-bundle" / "src" / "memory_condense"
    tool = tmp_path / "tool-bundle" / "tools" / "mem0_eval"
    source.mkdir(parents=True)
    tool.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tool / "preflight.py").write_text("VALUE = 2\n", encoding="utf-8")
    source_digest = bootstrap._tree_sha256(source)  # noqa: SLF001
    tool_digest = bootstrap._tree_sha256(tool)  # noqa: SLF001
    launched: dict[str, object] = {}

    monkeypatch.setattr(
        bootstrap,
        "_deny_network",
        lambda: launched.setdefault("offline", True),
    )
    monkeypatch.setattr(bootstrap, "_verify_bootstrap_origin", lambda _root: None)
    monkeypatch.setattr(bootstrap, "_verify_isolated_runtime", lambda: None)
    monkeypatch.setattr(
        bootstrap,
        "_verify_import_resolution",
        lambda _source, _tool: None,
    )
    monkeypatch.setattr(
        bootstrap.runpy,
        "run_module",
        lambda module, **kwargs: launched.update(
            module=module,
            kwargs=kwargs,
            argv=list(sys.argv),
        ),
    )
    old_path = list(sys.path)
    try:
        assert bootstrap.main(
            [
                "--source-root",
                str(source),
                "--tool-root",
                str(tool),
                "--expected-source-sha256",
                source_digest,
                "--expected-tool-sha256",
                tool_digest,
                "--module",
                "tools.mem0_eval.preflight",
                "--",
                "--output",
                "receipt.json",
            ]
        ) == 0
    finally:
        sys.path[:] = old_path

    assert launched == {
        "offline": True,
        "module": "tools.mem0_eval.preflight",
        "kwargs": {"run_name": "__main__", "alter_sys": False},
        "argv": [
            "tools.mem0_eval.preflight",
            "--output",
            "receipt.json",
        ],
    }
    assert bootstrap.os.environ["MEM0_TELEMETRY"] == "false"
    assert bootstrap.os.environ["HF_HUB_OFFLINE"] == "1"
    assert bootstrap.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert bootstrap.os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert bootstrap.os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"


@pytest.mark.parametrize("mutated", ["source", "tool"])
def test_isolated_bootstrap_rechecks_both_frozen_trees_after_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
) -> None:
    source = tmp_path / "source-bundle" / "src" / "memory_condense"
    tool = tmp_path / "tool-bundle" / "tools" / "mem0_eval"
    source.mkdir(parents=True)
    tool.mkdir(parents=True)
    source_file = source / "module.py"
    tool_file = tool / "preflight.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    tool_file.write_text("VALUE = 2\n", encoding="utf-8")
    source_digest = bootstrap._tree_sha256(source)  # noqa: SLF001
    tool_digest = bootstrap._tree_sha256(tool)  # noqa: SLF001

    monkeypatch.setattr(bootstrap, "_deny_network", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_bootstrap_origin", lambda _root: None)
    monkeypatch.setattr(bootstrap, "_verify_isolated_runtime", lambda: None)
    monkeypatch.setattr(
        bootstrap,
        "_verify_import_resolution",
        lambda _source, _tool: None,
    )

    def mutate_tree(*_args: object, **_kwargs: object) -> None:
        target = source_file if mutated == "source" else tool_file
        target.write_text("VALUE = 3\n", encoding="utf-8")
        raise SystemExit(0)

    monkeypatch.setattr(bootstrap.runpy, "run_module", mutate_tree)
    match = (
        "memory-condense source changed during launch"
        if mutated == "source"
        else "Mem0 tool changed during launch"
    )
    old_path = list(sys.path)
    try:
        with pytest.raises(RuntimeError, match=match):
            bootstrap.main(
                [
                    "--source-root",
                    str(source),
                    "--tool-root",
                    str(tool),
                    "--expected-source-sha256",
                    source_digest,
                    "--expected-tool-sha256",
                    tool_digest,
                    "--module",
                    "tools.mem0_eval.preflight",
                ]
            )
    finally:
        sys.path[:] = old_path


def test_isolated_bootstrap_imports_preflight_against_exact_v3_source(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    frozen = tmp_path / "frozen-v3"
    frozen.mkdir()
    archived = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2",
            "src/memory_condense",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        archive.extractall(frozen, filter="data")

    source = frozen / "src" / "memory_condense"
    tool = repository / "tools" / "mem0_eval"
    source_digest = bootstrap._tree_sha256(source)  # noqa: SLF001
    assert source_digest == (
        "452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83"
    )
    tool_digest = bootstrap._tree_sha256(tool)  # noqa: SLF001
    for module, forwarded in (
        ("tools.mem0_eval.preflight", ["--", "--help"]),
        ("tools.mem0_eval.production_binding", []),
        ("tools.mem0_eval.report", []),
        ("tools.mem0_eval.compare", []),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(tool / "bootstrap.py"),
                "--source-root",
                str(source),
                "--tool-root",
                str(tool),
                "--expected-source-sha256",
                source_digest,
                "--expected-tool-sha256",
                tool_digest,
                "--module",
                module,
                *forwarded,
            ],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, f"{module}: {completed.stderr}"
        if module.endswith("preflight"):
            assert "Reconstruct the locked Mem0 comparison" in completed.stdout


def test_preflight_output_is_atomic_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, dataset, split, policy = _source_policy_fixture(tmp_path)
    output = tmp_path / "receipt.json"
    output.write_bytes(b"existing-receipt\n")
    monkeypatch.setattr(
        preflight,
        "build_preflight_receipt",
        lambda **_kwargs: {"status": "provider_free_ready"},
    )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        preflight.main(
            [
                "--benchmark-file",
                str(dataset),
                "--split-manifest",
                str(split),
                "--policy-manifest",
                str(policy),
                "--repository-root",
                str(repo),
                "--output",
                str(output),
            ]
        )

    assert output.read_bytes() == b"existing-receipt\n"
    assert not list(tmp_path.glob("*.staging"))


def test_preflight_rejects_output_inside_hashed_source_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, dataset, split, policy = _source_policy_fixture(tmp_path)
    output = repo / "src" / "memory_condense" / "receipt.json"
    called = False

    def forbidden_build(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("protected output must fail before reconstruction")

    monkeypatch.setattr(preflight, "build_preflight_receipt", forbidden_build)
    with pytest.raises(ValueError, match="protected source implementation root"):
        preflight.main(
            [
                "--benchmark-file",
                str(dataset),
                "--split-manifest",
                str(split),
                "--policy-manifest",
                str(policy),
                "--repository-root",
                str(repo),
                "--output",
                str(output),
            ]
        )

    assert called is False
    assert not output.exists()
