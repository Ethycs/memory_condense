from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
)
from memory_condense.persistence.db import Database
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_query_expansion_adapter as subject
from tools import confirmation_protected_s0_plane as protected
from tools.confirmation_contracts import SealedJson, publish_sealed_json
from tools.matched_eval.query_expansion import (
    PartitionRoutingReceipt,
    QuerySearchResult,
)
from tools.v4_population_firebreak.canonical import canonical_sha256
from tests.test_confirmation_protected_s0_plane import (
    _completion_fixture,
    _kwargs,
)
from tests.test_confirmation_s0_prompt_preflight import Fixture, _build_fixture


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(value: dict[str, object], key: str) -> dict[str, object]:
    return {**value, key: canonical_sha256(value)}


def _create_store(root: Path, index: int) -> tuple[Path, str, str, str, str]:
    store = root / "combined-store"
    store.mkdir(parents=True)
    database_path = store / "memory.db"
    source_id = f"partition-{index}::global-history"
    chunk_id = f"global-chunk-{index}"
    text = f"Global evidence for namespace {index}."
    database = Database(database_path)
    try:
        database.execute(
            "INSERT INTO turns(turn_id, role, text, source_id, created_at, ordinal) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"global-turn-{index}",
                "user",
                text,
                source_id,
                "2026-01-01T00:00:00Z",
                0,
            ),
        )
        database.execute(
            "INSERT INTO chunks(chunk_id, turn_id, text, start_char, end_char, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                f"global-turn-{index}",
                text,
                0,
                len(text),
                6,
            ),
        )
        database.commit()
    finally:
        database.close()
    index_path = store / "hnsw_index.bin"
    index_path.write_bytes(f"synthetic-index-{index}".encode("ascii"))
    return (
        store,
        file_sha256(database_path),
        file_sha256(index_path),
        source_id,
        chunk_id,
    )


def _attach_authenticated_stores(
    fixture: Fixture,
) -> tuple[Fixture, dict[str, Path], dict[str, tuple[str, str]]]:
    changed = copy.deepcopy(fixture.cumulative.payload)
    references = changed["namespace_checkpoints"]
    checkpoint_paths: dict[str, Path] = {}
    source_coordinates: dict[str, tuple[str, str]] = {}
    execution_root = fixture.root / "cumulative-execution"
    for index, reference in enumerate(references):
        store_id = reference["namespace_store_id"]
        namespace_id = reference["namespace_id"]
        namespace_root = execution_root / "namespaces" / store_id
        store, database_sha, index_sha, source_id, chunk_id = _create_store(
            namespace_root,
            index,
        )
        source_coordinates[store_id] = (source_id, chunk_id)
        identity = _digest(f"store-identity:{store_id}")
        compilation = _digest(f"compilation:{store_id}")
        receipt = CombinedCumulativeStoreReceipt(
            source_store_identity_sha256=identity,
            target_store_identity_sha256=identity,
            source_database_sha256=_digest(f"source-database:{store_id}"),
            target_database_sha256=database_sha,
            target_index_sha256=index_sha,
            retrieval_policy_sha256=_digest(f"retrieval-policy:{store_id}"),
            context_budget_sha256=_digest(f"context-budget:{store_id}"),
            training_query_batch_sha256=_digest(f"training:{store_id}"),
            held_out_query_batch_sha256=_digest(f"held-out:{store_id}"),
            compilation_receipt_sha256=compilation,
            artifact_id=f"artifact-{index}",
            snapshot_sha256=_digest(f"snapshot:{store_id}"),
            turn_count=1,
            chunk_count=1,
            causal_events=0,
            causal_graph_edges=0,
        )
        base_checkpoint = _digest(f"base-checkpoint:{store_id}")
        namespace_questions = [
            row["question"]
            for row in changed["questions"]
            if row["namespace_store_id"] == store_id
        ]
        execution = {
            "artifact_projection": {
                "combined_store_mode": "synthetic-authenticated-existing",
                "combined_store_relative_path": store.name,
                "retained_request_token_state_bytes": 0,
            },
            "base_checkpoint_sha256": base_checkpoint,
            "combined_store_receipt": asdict(receipt),
            "compilation_receipt_sha256": compilation,
            "format": cumulative.BACKEND_RESULT_FORMAT,
            "namespace_id": namespace_id,
            "namespace_store_id": store_id,
            "physical_provider_calls": 0,
            "questions": namespace_questions,
        }
        body = {
            "backend_identity_sha256": changed["backend_identity_sha256"],
            "base_backend_identity_sha256": _digest(f"base-backend:{store_id}"),
            "base_checkpoint_receipt_sha256": _digest(
                f"base-checkpoint-receipt:{store_id}"
            ),
            "base_checkpoint_sha256": base_checkpoint,
            "execution": execution,
            "format": cumulative.CHECKPOINT_FORMAT,
            "freeze_sha256": changed["freeze_sha256"],
            "gold_loaded": False,
            "namespace_id": namespace_id,
            "namespace_store_id": store_id,
            "namespace_work_receipt_sha256": reference[
                "namespace_work_receipt_sha256"
            ],
            "physical_provider_calls": 0,
            "preflight_sha256": changed["preflight_sha256"],
            "workset_identity_sha256": changed["workset_identity_sha256"],
        }
        checkpoint_path = execution_root / "checkpoints" / f"{store_id}.json"
        checkpoint, _ = publish_sealed_json(
            checkpoint_path,
            _seal(body, "checkpoint_receipt_sha256"),
        )
        checkpoint_paths[store_id] = checkpoint.path
        reference["checkpoint_sha256"] = checkpoint.sha256
        reference["checkpoint_receipt_sha256"] = checkpoint.payload[
            "checkpoint_receipt_sha256"
        ]
        for row in changed["questions"]:
            if row["namespace_store_id"] == store_id:
                row["namespace_checkpoint_sha256"] = checkpoint.sha256

    unsigned = dict(changed)
    unsigned.pop("merge_receipt_sha256")
    changed["merge_receipt_sha256"] = canonical_sha256(unsigned)
    merged, _ = publish_sealed_json(
        fixture.root / "cumulative-with-authenticated-stores.json",
        changed,
    )
    updated = Fixture(
        root=fixture.root,
        policy=fixture.policy,
        treatment=fixture.treatment,
        treatment_preflight=fixture.treatment_preflight,
        cumulative=merged,
        semantics=fixture.semantics,
    )
    return updated, checkpoint_paths, source_coordinates


def _context(
    tmp_path: Path,
    *,
    semantics: tuple[int, ...],
    namespace_sizes: tuple[int, ...],
    prefix: str,
):
    fixture = _build_fixture(
        tmp_path,
        semantics=semantics,
        id_prefix=prefix,
        namespace_sizes=namespace_sizes,
    )
    fixture, checkpoint_paths, coordinates = _attach_authenticated_stores(fixture)
    s0_prompt, s0_completion, lifecycle_sha, release_sha, _ = _completion_fixture(
        fixture
    )
    protected_inputs = _kwargs(
        fixture,
        s0_prompt,
        s0_completion,
        lifecycle_sha,
        release_sha,
    )
    protected_artifact, _, _ = protected.publish_protected_s0_answer_plane(
        fixture.root / "protected-s0.json",
        **protected_inputs,
    )
    context = subject.load_confirmation_query_expansion_context(
        protected_s0_plane_path=protected_artifact.path,
        expected_protected_s0_plane_sha256=protected_artifact.sha256,
        protected_s0_inputs=protected_inputs,
        namespace_checkpoint_paths_by_store_id=checkpoint_paths,
        include_s0_evidence=True,
    )
    return fixture, context, checkpoint_paths, coordinates


class _QueryCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        completion = json.dumps(
            {
                "queries": ["global evidence complete history"],
                "entities": ["global evidence"],
                "dates": [],
                "operators": ["timeline"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return SimpleNamespace(
            id=f"query-response-{number}",
            model="synthetic-terra-query",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=20,
                total_tokens=40,
            ),
        )


class _QueryClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_QueryCompletions())
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FrozenSearch:
    def __init__(self, namespace, *, source_id: str, chunk_id: str) -> None:
        self.namespace = namespace
        index = int(chunk_id.rsplit("-", 1)[1])
        text = f"Global evidence for namespace {index}."
        turn = Turn(
            turn_id=f"global-turn-{index}",
            role="user",
            text=text,
            source_id=source_id,
            created_at="2026-01-01T00:00:00Z",
        )
        self.hit = RetrievalResult(
            chunk=Chunk(
                chunk_id=chunk_id,
                turn_id=turn.turn_id,
                text=text,
                start_char=0,
                end_char=len(text),
                token_count=count_tokens(text),
            ),
            score=0.9,
            turn=turn,
            dense_score=0.9,
            lexical_score=0.8,
            route="synthetic-global-partition",
        )

    def search_many(self, queries, *, budget):
        del budget
        return tuple(
            QuerySearchResult(
                query_sha256=quote_sha256(query),
                hits=(self.hit,),
                routing_receipt=PartitionRoutingReceipt.create(
                    query=query,
                    namespace=self.namespace,
                    selected_partitions=(self.namespace.partition_ids[0],),
                    routed_source_count=1,
                    active_partition_scan_status="applied",
                    active_partition_scan_contract="synthetic-complete-scan-v1",
                    active_partition_exhaustive=True,
                ),
            )
            for query in queries
        )


def _retrievers(context, coordinates):
    by_store = {
        snapshot.namespace_store_id: snapshot for snapshot in context.namespace_snapshots
    }
    return {
        snapshot.namespace.namespace_id: _FrozenSearch(
            snapshot.namespace,
            source_id=coordinates[store_id][0],
            chunk_id=coordinates[store_id][1],
        )
        for store_id, snapshot in by_store.items()
    }


@pytest.mark.parametrize(
    ("semantics", "namespace_sizes"),
    [((5,), (1,)), ((9, 2, 7), (2, 1)), ((4, 1, 8, 3, 6), (1, 3, 1))],
)
def test_real_store_context_and_native_preflight_are_population_neutral(
    tmp_path: Path,
    semantics: tuple[int, ...],
    namespace_sizes: tuple[int, ...],
) -> None:
    _, context, _, _ = _context(
        tmp_path / f"n-{len(semantics)}",
        semantics=semantics,
        namespace_sizes=namespace_sizes,
        prefix=f"neutral-{len(semantics)}",
    )
    output_root = tmp_path / f"query-{len(semantics)}"
    preflight = subject.preflight_confirmation_query_expansion(
        context,
        output_root=output_root,
    )

    assert context.question_count == len(semantics)
    assert preflight.payload["logical_prompt_count"] == len(semantics)
    assert preflight.payload["question_count"] == len(semantics)
    assert preflight.payload["required_authorized_provider_calls"] == len(
        semantics
    )
    assert preflight.payload["query_population_id"] == (
        context.population.population_id
    )
    assert not (output_root / subject.query_expansion.CHECKPOINT_DIR_NAME).exists()
    assert len(context.store_dirs_by_namespace) == len(namespace_sizes)
    assert sorted(set(context.shard_offsets_by_question.values())) == [
        sum(namespace_sizes[:index]) for index in range(len(namespace_sizes))
    ]


def test_sealed_native_release_materializes_and_replays_existing_query_arm(
    tmp_path: Path,
) -> None:
    _, context, _, coordinates = _context(
        tmp_path / "run",
        semantics=(3, 1, 4, 2),
        namespace_sizes=(2, 2),
        prefix="execute",
    )
    output_root = tmp_path / "query-run"
    preflight = subject.preflight_confirmation_query_expansion(
        context,
        output_root=output_root,
    )
    with pytest.raises(
        subject.ConfirmationQueryExpansionError,
        match="exactly equal remaining",
    ):
        subject.approve_confirmation_query_expansion_provider_release(
            context,
            output_root=output_root,
            expected_query_preflight_sha256=preflight.sha256,
            approve_provider_release=True,
            authorized_provider_calls=context.question_count - 1,
        )
    release, created = (
        subject.approve_confirmation_query_expansion_provider_release(
            context,
            output_root=output_root,
            expected_query_preflight_sha256=preflight.sha256,
            approve_provider_release=True,
            authorized_provider_calls=context.question_count,
        )
    )
    assert created is True
    assert release.payload["required_authorized_provider_calls"] == (
        context.question_count
    )
    assert release.payload["provider_calls_during_release"] == 0

    client = _QueryClient()
    completion = subject.run_confirmation_query_expansion_provider(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=context.question_count,
        client=client,
    )
    assert completion.physical_provider_calls == context.question_count
    assert completion.checkpoint_hits == 0
    assert len(client.chat.completions.requests) == context.question_count
    assert client.close_calls == 1

    resumed = subject.run_confirmation_query_expansion_provider(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=0,
        client=None,
    )
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == context.question_count

    retrievers = _retrievers(context, coordinates)
    result = subject.materialize_confirmation_query_expansion(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        retrievers_by_namespace=retrievers,
    )

    assert result.physical_provider_calls == 0
    assert result.checkpoint_hits == context.question_count
    assert result.run_artifact.payload["question_count"] == context.question_count
    assert all(
        row["disposition"] == "added"
        for row in result.run_artifact.payload["questions"]
    )
    replay = subject.replay_confirmation_query_expansion(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        retrievers_by_namespace=retrievers,
        expected_run_sha256=result.run_artifact.sha256,
        expected_runtime_ledger_sha256=result.runtime_ledger_artifact.sha256,
    )
    assert replay.run_artifact.sha256 == result.run_artifact.sha256
    assert replay.runtime_ledger_artifact.sha256 == (
        result.runtime_ledger_artifact.sha256
    )


def test_missing_checkpoint_and_post_freeze_store_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "tamper",
        semantics=(0, 1, 2),
        id_prefix="tamper",
        namespace_sizes=(2, 1),
    )
    fixture, checkpoint_paths, _ = _attach_authenticated_stores(fixture)
    s0_prompt, s0_completion, lifecycle_sha, release_sha, _ = _completion_fixture(
        fixture
    )
    protected_inputs = _kwargs(
        fixture, s0_prompt, s0_completion, lifecycle_sha, release_sha
    )
    protected_artifact, _, _ = protected.publish_protected_s0_answer_plane(
        fixture.root / "protected-s0.json", **protected_inputs
    )
    missing = dict(checkpoint_paths)
    missing.pop(next(iter(missing)))
    with pytest.raises(subject.ConfirmationQueryExpansionError, match="exact cumulative"):
        subject.load_confirmation_query_expansion_context(
            protected_s0_plane_path=protected_artifact.path,
            expected_protected_s0_plane_sha256=protected_artifact.sha256,
            protected_s0_inputs=protected_inputs,
            namespace_checkpoint_paths_by_store_id=missing,
        )

    context = subject.load_confirmation_query_expansion_context(
        protected_s0_plane_path=protected_artifact.path,
        expected_protected_s0_plane_sha256=protected_artifact.sha256,
        protected_s0_inputs=protected_inputs,
        namespace_checkpoint_paths_by_store_id=checkpoint_paths,
    )
    (context.namespace_snapshots[0].store_dir / "hnsw_index.bin").write_bytes(
        b"tampered-index"
    )
    with pytest.raises(subject.ConfirmationQueryExpansionError, match="index changed"):
        context.revalidate_store_bytes()


def test_no_validation_constants_provider_sdk_or_cli_surface() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8").casefold()
    assert "expected_question_count" not in source
    assert "validation_question" not in source
    assert "def build_parser" not in source
    assert "argparse" not in source
    assert "import litellm" not in source
    assert "import openai" not in source
