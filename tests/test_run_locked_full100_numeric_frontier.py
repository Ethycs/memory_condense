from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools import run_locked_full100_numeric_frontier as lifecycle
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _index(path: Path):
    asked = datetime(2023, 5, 30, 21, 51, tzinfo=timezone.utc)
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for ordinal, (source, text, created) in enumerate(
        (
            (
                "garden::one",
                "I bought a peace lily and a succulent plant two weeks ago.",
                asked - timedelta(days=2),
            ),
            (
                "garden::two",
                "My snake plant, which I got last month, needs repotting.",
                asked - timedelta(days=1),
            ),
        )
    ):
        turn = transcript.append("user", text, source_id=source, created_at=created)
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{ordinal}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha("store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("database"),
            source_store_receipt_sha256=store_receipt,
        )
    return build_full_store_window_index(cache)


def _provider(question: str) -> dict:
    summaries = (
        "I bought a peace lily and a succulent plant two weeks ago.",
        "My snake plant, which I got last month, needs repotting.",
    )
    return {
        "dated_question": question,
        "protected_parent_fallback": {"prediction": "not consulted"},
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "format": "synthetic-typed-v1",
            "frontier": {"closed": False, "mode": "open", "truncated": True},
            "handles": [
                {"group_handle": f"G00{i}", "handle_id": f"H00{i}", "origin": "map"}
                for i in (1, 2)
            ],
            "items": [
                {
                    "content_coherence": "match",
                    "date": "2023-05-28T21:51:00+00:00",
                    "handle_ids": [f"H00{i}"],
                    "included": True,
                    "kind": "direct",
                    "relation": "authored_by_user;date_basis=source_created_at",
                    "status": "completed" if i == 1 else "unknown",
                    "summary": summary,
                    "supported_slot_ids": [],
                    "value_authority": "explicit",
                }
                for i, summary in enumerate(summaries, start=1)
            ],
            "operator_spec": {
                "answer_shape": "number",
                "comparison_mode": "none",
                "include_proposed": False,
                "operation": "count_or_aggregate",
                "query_timestamp": "2023/05/30 (Tue) 21:51",
                "required_slots": [],
                "requires_complete_frontier": True,
                "style": "numeric_reduce",
                "temporal_window_days": 31,
            },
        },
    }


def _inputs(index, provider: dict) -> lifecycle.VerifiedInputs:
    plan = {
        "ordinal": 53,
        "provider_input": provider,
        "provider_input_sha256": identity_sha256(provider),
        "question_id": "q53",
        "question_sha256": _sha("question"),
    }
    artifact = lambda label: SealedArtifact(Path(label), _sha(label), {})
    construction = artifact("construction")
    replay = artifact("construction-replay")
    full100 = SimpleNamespace(construction=construction, replay=replay)
    return lifecycle.VerifiedInputs(
        artifact("preflight"),
        artifact("answer-run"),
        artifact("answer-replay"),
        full100,
        (plan,),
        {53: index.cache.namespace_id},
    )


def test_materialization_streams_namespace_once_and_is_deterministic(tmp_path: Path) -> None:
    index = _index(tmp_path / "memory.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    inputs = _inputs(index, _provider(question))
    calls: list[str] = []

    def load(namespace_id: str):
        calls.append(namespace_id)
        return index, {"synthetic": True}

    first = lifecycle.build_materialization_payload(inputs, index_loader=load)
    second = lifecycle.build_materialization_payload(inputs, index_loader=load)

    assert first == second
    assert calls == [index.cache.namespace_id, index.cache.namespace_id]
    assert first["ordinals"] == [53]
    assert first["frontier_count"] == 1
    assert first["new_provider_calls"] == 0
    assert first["gold_loaded"] is False
    assert first["frontier_rows"][0]["bridge"]["physical_scan_exhaustive"] is True


def test_cli_has_no_ordinal_selector() -> None:
    parser = lifecycle.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"materialize", "replay"} <= set(choices)
    option_strings = {
        option
        for command in choices.values()
        for action in command._actions
        for option in action.option_strings
    }
    assert "--ordinal" not in option_strings
    assert "--ordinals" not in option_strings


def test_resident_loader_authenticates_shared_population_once(
    tmp_path: Path, monkeypatch
) -> None:
    namespace_ids = (_sha("namespace-0"), _sha("namespace-10"))
    receipt_ids = (_sha("receipt-0"), _sha("receipt-10"))
    database_ids = (_sha("database-0"), _sha("database-10"))
    index_ids = (_sha("index-0"), _sha("index-10"))
    namespaces = tuple(
        SimpleNamespace(
            namespace_id=namespace_id,
            combined_store_receipt_sha256=receipt_id,
        )
        for namespace_id, receipt_id in zip(
            namespace_ids, receipt_ids, strict=True
        )
    )
    question_ids = ("q0", "q10")
    prompts = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                packet=SimpleNamespace(question_id=question_id)
            ),
            namespace=namespace,
        )
        for question_id, namespace in zip(
            question_ids, namespaces, strict=True
        )
    )
    retrieval_sha = _sha("retrieval")
    preflight_sha = _sha("query-preflight")
    population = SimpleNamespace(
        source_population=SimpleNamespace(retrieval_sha256=retrieval_sha),
        namespaces=namespaces,
        rows=prompts,
    )
    retrieval = SealedArtifact(
        tmp_path / "retrieval.json",
        retrieval_sha,
        {
            "shards": [
                {
                    "combined_store_receipt": {
                        "receipt_sha256": receipt_id,
                        "target_database_sha256": database_id,
                        "target_index_sha256": index_id,
                    },
                    "combined_store_receipt_sha256": receipt_id,
                    "shard_offset": offset,
                }
                for offset, receipt_id, database_id, index_id in zip(
                    (0, 10),
                    receipt_ids,
                    database_ids,
                    index_ids,
                    strict=True,
                )
            ],
            "questions": [
                {"question_id": question_id, "shard_offset": offset}
                for question_id, offset in zip(
                    question_ids, (0, 10), strict=True
                )
            ],
        },
    )
    calls = {
        "population": 0,
        "retrieval": 0,
        "file_sha256": 0,
        "database": 0,
        "cache": 0,
        "index": 0,
    }

    def load_population(*_args, **_kwargs):
        calls["population"] += 1
        return population, SealedArtifact(
            tmp_path / "preflight.json", preflight_sha, {}
        )

    def read_retrieval(_path):
        calls["retrieval"] += 1
        return retrieval

    for offset in (0, 10):
        store = (
            tmp_path
            / "store"
            / "shards"
            / f"offset-{offset:03d}"
            / "combined-store"
        )
        store.mkdir(parents=True)
        (store / "memory.db").touch()
        (store / "hnsw_index.bin").touch()

    expected_file_sha = {
        str(
            tmp_path
            / "store"
            / "shards"
            / f"offset-{offset:03d}"
            / "combined-store"
            / filename
        ): digest
        for offset, database_id, index_id in zip(
            (0, 10), database_ids, index_ids, strict=True
        )
        for filename, digest in (
            ("memory.db", database_id),
            ("hnsw_index.bin", index_id),
        )
    }

    def file_sha256(path):
        calls["file_sha256"] += 1
        return expected_file_sha[str(path)]

    class FakeDatabase:
        def __init__(self, path, *, read_only):
            assert Path(path).name == "memory.db"
            assert read_only is True
            calls["database"] += 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def build_cache(_database, namespace, **kwargs):
        calls["cache"] += 1
        assert kwargs == {
            "source_database_sha256": database_ids[
                namespaces.index(namespace)
            ],
            "source_store_receipt_sha256": (
                namespace.combined_store_receipt_sha256
            ),
        }
        return SimpleNamespace(
            cache_receipt_sha256=_sha(
                f"cache-{namespaces.index(namespace)}"
            ),
            namespace_id=namespace.namespace_id,
        )

    def build_index(cache):
        calls["index"] += 1
        return SimpleNamespace(
            cache=cache,
            receipt_sha256=_sha(f"window-{cache.namespace_id}"),
            rows=(object(), object()),
            windows=(object(), object(), object()),
        )

    monkeypatch.setattr(
        lifecycle.resident_cli,
        "load_preflighted_query_expansion_population",
        load_population,
    )
    monkeypatch.setattr(lifecycle, "read_sealed_json", read_retrieval)
    monkeypatch.setattr(lifecycle.resident_cli, "file_sha256", file_sha256)
    monkeypatch.setattr(lifecycle, "Database", FakeDatabase)
    monkeypatch.setattr(lifecycle, "cache_namespace_partitions", build_cache)
    monkeypatch.setattr(lifecycle, "build_full_store_window_index", build_index)

    args = SimpleNamespace(
        retrieval=tmp_path / "retrieval.json",
        query_parent_output_root=tmp_path / "query-parent",
        expected_retrieval_sha256=retrieval_sha,
        expected_query_parent_preflight_sha256=preflight_sha,
        store_root=tmp_path / "store",
    )
    loader = lifecycle._resident_index_loader(args)
    loaded = [loader(namespace_id) for namespace_id in namespace_ids]

    assert calls == {
        "population": 1,
        "retrieval": 1,
        "file_sha256": 4,
        "database": 2,
        "cache": 2,
        "index": 2,
    }
    assert [index.cache.namespace_id for index, _ in loaded] == list(
        namespace_ids
    )
    assert [row["physical_content_row_count"] for _, row in loaded] == [2, 2]
    assert [row["physical_sentence_window_count"] for _, row in loaded] == [3, 3]


def test_v2_lifecycle_uses_new_artifact_names_and_root() -> None:
    assert lifecycle.FORMAT == "memory-condense-locked-full100-numeric-frontier-v2"
    assert lifecycle.MATERIALIZATION_NAME.endswith("-v2.json")
    assert lifecycle.REPLAY_NAME.endswith("-replay-v2.json")
    assert lifecycle.DEFAULT_OUTPUT_ROOT.name == "locked-full100-numeric-frontier-v2"


def test_operator_material_v3_profile_is_distinct_and_status_invariant(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "operator-material.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    provider = _provider(question)
    # The source compiler recognizes the first row as completed while this
    # provider-side translation deliberately uses the equally eligible
    # unknown state.
    provider["typed_evidence"]["items"][0]["status"] = "unknown"
    inputs = _inputs(index, provider)

    strict = lifecycle.build_materialization_payload(
        inputs,
        index_loader=lambda _namespace: (index, {"synthetic": True}),
    )
    successor = lifecycle.build_materialization_payload(
        inputs,
        index_loader=lambda _namespace: (index, {"synthetic": True}),
        policy_profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
    )

    assert strict["format"] == lifecycle.FORMAT
    assert strict["closed_count"] == 0
    assert successor["format"] == lifecycle.V3_FORMAT
    assert successor["closed_count"] == 1
    assert successor["frontier_rows"][0]["format"] == lifecycle.V3_ROW_FORMAT
    assert {
        row["status"]
        for row in successor["frontier_rows"][0]["bridge"]["census_atoms"]
    } == {"operator_eligible"}


def test_typed_loader_returns_frontiers_for_overlay(tmp_path: Path) -> None:
    index = _index(tmp_path / "source.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    payload = lifecycle.build_materialization_payload(
        _inputs(index, _provider(question)),
        index_loader=lambda _namespace: (index, {"synthetic": True}),
    )
    root = tmp_path / "sealed"
    materialization, _ = publish_sealed_json(
        root / lifecycle.MATERIALIZATION_NAME, payload
    )
    replay, _ = publish_sealed_json(root / lifecycle.REPLAY_NAME, payload)

    loaded, loaded_replay, by_ordinal = lifecycle.load_verified_numeric_frontiers(
        root, materialization.sha256, replay.sha256
    )

    assert loaded.sha256 == loaded_replay.sha256
    assert tuple(by_ordinal) == (53,)
    assert by_ordinal[53].projection() == payload["frontier_rows"][0]["bridge"]["frontier"]


def test_typed_loader_accepts_operator_material_v3_profile(tmp_path: Path) -> None:
    index = _index(tmp_path / "v3-source.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    payload = lifecycle.build_materialization_payload(
        _inputs(index, _provider(question)),
        index_loader=lambda _namespace: (index, {"synthetic": True}),
        policy_profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
    )
    root = tmp_path / "v3-sealed"
    materialization, _ = publish_sealed_json(
        root / lifecycle.V3_MATERIALIZATION_NAME, payload
    )
    replay, _ = publish_sealed_json(root / lifecycle.V3_REPLAY_NAME, payload)

    loaded, loaded_replay, by_ordinal = lifecycle.load_verified_numeric_frontiers(
        root,
        materialization.sha256,
        replay.sha256,
        policy_profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
    )

    assert loaded.sha256 == loaded_replay.sha256
    assert tuple(by_ordinal) == (53,)


def _refresh_payload_identity(payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    payload["identity_sha256"] = identity_sha256(unsigned)


def _publish_numeric_pair(root: Path, payload: dict, *, profile: str):
    selected = lifecycle.lifecycle_profile(profile)
    materialization, _ = publish_sealed_json(
        root / selected.materialization_name, payload
    )
    replay, _ = publish_sealed_json(root / selected.replay_name, payload)
    return materialization, replay


def test_typed_loader_rejects_unbound_namespace_lifecycle(tmp_path: Path) -> None:
    index = _index(tmp_path / "lifecycle-source.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    original = lifecycle.build_materialization_payload(
        _inputs(index, _provider(question)),
        index_loader=lambda _namespace: (index, {"synthetic": True}),
    )

    variants: list[tuple[str, dict]] = []
    missing = deepcopy(original)
    missing["namespace_lifecycle"] = []
    variants.append(("missing", missing))

    duplicate = deepcopy(original)
    duplicate["namespace_lifecycle"].append(
        deepcopy(duplicate["namespace_lifecycle"][0])
    )
    variants.append(("duplicate", duplicate))

    wrong_rows = deepcopy(original)
    row_lifecycle = wrong_rows["namespace_lifecycle"][0]
    row_lifecycle["numeric_row_receipt_sha256s"] = []
    lifecycle_body = dict(row_lifecycle)
    lifecycle_body.pop("receipt_sha256")
    row_lifecycle["receipt_sha256"] = identity_sha256(lifecycle_body)
    variants.append(("wrong-rows", wrong_rows))

    wrong_window = deepcopy(original)
    window_lifecycle = wrong_window["namespace_lifecycle"][0]
    window_lifecycle["window_index_receipt_sha256"] = _sha("wrong-window")
    lifecycle_body = dict(window_lifecycle)
    lifecycle_body.pop("receipt_sha256")
    window_lifecycle["receipt_sha256"] = identity_sha256(lifecycle_body)
    variants.append(("wrong-window", wrong_window))

    for label, payload in variants:
        _refresh_payload_identity(payload)
        materialization, replay = _publish_numeric_pair(
            tmp_path / label,
            payload,
            profile=lifecycle.STRICT_PROFILE,
        )
        with pytest.raises(lifecycle.LockedFull100NumericFrontierError):
            lifecycle.load_verified_numeric_frontiers(
                tmp_path / label,
                materialization.sha256,
                replay.sha256,
            )


def test_v3_loader_rejects_non_operator_material_census_status(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path / "v3-status-source.db")
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    payload = lifecycle.build_materialization_payload(
        _inputs(index, _provider(question)),
        index_loader=lambda _namespace: (index, {"synthetic": True}),
        policy_profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
    )
    payload["frontier_rows"][0]["bridge"]["census_atoms"][0][
        "status"
    ] = "completed"
    _refresh_payload_identity(payload)
    materialization, replay = _publish_numeric_pair(
        tmp_path / "v3-wrong-status",
        payload,
        profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
    )

    with pytest.raises(
        lifecycle.LockedFull100NumericFrontierError,
        match="operator-material census status changed",
    ):
        lifecycle.load_verified_numeric_frontiers(
            tmp_path / "v3-wrong-status",
            materialization.sha256,
            replay.sha256,
            policy_profile=lifecycle.OPERATOR_MATERIAL_PROFILE,
        )
