from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tools import assay_minilm_evidence_accuracy as assay


class _NoAnswerReadDict(dict[str, Any]):
    """Fixture mapping that makes accidental answer-text access observable."""

    def __getitem__(self, key: str) -> Any:
        if key == "answer":
            raise AssertionError("answer text must not be read")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "answer":
            raise AssertionError("answer text must not be read")
        return super().get(key, default)


def _coordinate_fixture() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    """Build the complete 200-question shape required by reconstruction."""

    pilot = assay._pilot()
    questions: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    identities: dict[str, dict[str, str]] = {}
    for index in range(200):
        question_id = f"q-{index:03d}"
        partition = "train" if index < 140 else "calibration" if index < 170 else "test"
        question_text = f"What happened to item {index}?"
        question_type = f"type-{index % 5}"
        question_date = f"2026/09/{index % 28 + 1:02d}"
        required_raw_source = f"answer-source-{index}"
        other_raw_source = f"other-source-{index}"
        required_source = pilot._opaque_id(
            "source", question_id, "0", required_raw_source
        )
        other_source = pilot._opaque_id(
            "source", question_id, "1", other_raw_source
        )
        turns = (
            (
                0,
                required_source,
                "2026/08/01",
                [
                    {"role": "user", "content": f"background for {index}", "has_answer": False},
                    {
                        "role": "assistant",
                        "content": f"decisive evidence for {index}",
                        "has_answer": True,
                    },
                ],
            ),
            (
                1,
                other_source,
                "2026/08/02",
                [
                    {"role": "user", "content": f"irrelevant material for {index}", "has_answer": False}
                ],
            ),
        )
        ids: dict[str, str] = {
            "required_source": required_source,
            "other_source": other_source,
        }
        sessions: list[list[dict[str, Any]]] = []
        for session_index, source_id, source_date, session in turns:
            sessions.append(session)
            for turn_index, turn in enumerate(session):
                text = str(turn["content"])
                memory_id = pilot._opaque_id(
                    "memory",
                    question_id,
                    str(session_index),
                    str(turn_index),
                    assay._sha256_bytes(text.encode("utf-8")),
                )
                memories.append(
                    {
                        "memory_id": memory_id,
                        "source_id": source_id,
                        "source_date": source_date,
                        "role": pilot._raw_role(turn["role"], turn_index),
                        "text": text,
                        "partition": partition,
                    }
                )
                if session_index == 0 and turn_index == 0:
                    ids["grade1"] = memory_id
                elif session_index == 0:
                    ids["grade2"] = memory_id
                else:
                    ids["grade0"] = memory_id
        questions.append(
            {
                "question_id": question_id,
                "question": question_text,
                "question_type": question_type,
                "question_date": question_date,
                "partition": partition,
                "candidate_memory_ids": [],
            }
        )
        records.append(
            _NoAnswerReadDict(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "question_type": question_type,
                    "question_date": question_date,
                    "answer": f"FORBIDDEN-ANSWER-{index}",
                    "answer_session_ids": [required_raw_source],
                    "haystack_session_ids": [required_raw_source, other_raw_source],
                    "haystack_dates": ["2026/08/01", "2026/08/02"],
                    "haystack_sessions": sessions,
                }
            )
        )
        identities[question_id] = ids
    return {"questions": questions, "memories": memories}, records, identities


def test_five_fold_assignment_is_deterministic_disjoint_and_balanced() -> None:
    plane = {
        "questions": [
            {"question_id": f"q-{index:03d}", "question_type": f"type-{index % 7}"}
            for index in range(200)
        ]
    }
    first = assay._fold_assignments(plane)
    second = assay._fold_assignments(
        {"questions": list(reversed(plane["questions"]))}
    )

    assert first == second
    assert len(first) == 200
    assert Counter(first.values()) == Counter(assay.FOLD_COUNTS)
    assert set(first) == {f"q-{index:03d}" for index in range(200)}
    fold_members = {
        fold: {question_id for question_id, assigned in first.items() if assigned == fold}
        for fold in assay.FOLD_COUNTS
    }
    assert set().union(*fold_members.values()) == set(first)
    for fold, members in fold_members.items():
        assert not members.intersection(
            *(other for name, other in fold_members.items() if name != fold)
        )


def test_gold_reconstruction_separates_exact_source_and_other_without_answer_text() -> None:
    plane, records, identities = _coordinate_fixture()
    coordinates = assay._reconstruct_gold_coordinates(plane, records)
    question_id = "q-000"
    ids = identities[question_id]
    gold = coordinates[question_id]

    assert gold.required_source_ids == (ids["required_source"],)
    assert gold.exact_memory_ids == (ids["grade2"],)
    assert "FORBIDDEN-ANSWER" not in repr(coordinates)

    question = assay.PreparedQuestion(
        question_id=question_id,
        question_type="type-0",
        original_partition="train",
        fold="fold-0",
        query="opaque query",
        candidate_ids=(ids["grade2"], ids["grade1"], ids["grade0"]),
        candidate_texts=("exact", "same answer session", "other"),
        candidate_source_ids=(
            ids["required_source"],
            ids["required_source"],
            ids["other_source"],
        ),
        required_source_ids=gold.required_source_ids,
        exact_memory_ids=gold.exact_memory_ids,
    )
    any_positive, by_source, exact = assay._label_indices(question)

    # grade 2 is exact annotated evidence; grade 1 shares its answer session;
    # grade 0 is outside every registered answer session.
    assert exact == (0,)
    assert any_positive == (0, 1)
    assert by_source == {ids["required_source"]: (0, 1)}
    assert 2 not in any_positive


def test_pure_lexical_frontier_is_partition_local_deterministic_and_exactly_96() -> None:
    partitions = ("train", "calibration", "test")
    memories = [
        {
            "memory_id": f"{partition}-m-{index:03d}",
            "partition": partition,
            "text": (
                f"needle-{partition} uniquely relevant"
                if index == 0
                else f"ordinary filler {partition} {index}"
            ),
        }
        for partition in partitions
        for index in range(100)
    ]
    questions = []
    for index in range(200):
        partition = "train" if index < 140 else "calibration" if index < 170 else "test"
        questions.append(
            {
                "question_id": f"q-{index:03d}",
                "question": f"Where is needle-{partition}?",
                "question_type": "fixture",
                "partition": partition,
            }
        )
    plane = {"questions": questions, "memories": memories}

    first = assay._lexical_frontiers(plane)
    second = assay._lexical_frontiers(plane)

    assert first == second
    assert len(first) == 200
    memory_partition = {row["memory_id"]: row["partition"] for row in memories}
    question_partition = {row["question_id"]: row["partition"] for row in questions}
    for question_id, frontier in first.items():
        partition = question_partition[question_id]
        assert len(frontier) == assay.CANDIDATES == 96
        assert len(set(frontier)) == 96
        assert {memory_partition[memory_id] for memory_id in frontier} == {partition}
        assert frontier[0] == f"{partition}-m-000"


def test_hit_at_k_loss_boundary_and_degenerate_all_positive_case() -> None:
    torch = pytest.importorskip("torch")
    boundary = torch.tensor([1.2, 1.0, 0.8], dtype=torch.float64, requires_grad=True)
    loss = assay._hit_at_k_loss(
        torch,
        boundary,
        (0,),
        (1, 2),
        k=1,
        tau=assay.SMOOTHMAX_TAU,
        margin=assay.RANK_MARGIN,
    )
    assert float(loss.detach()) == pytest.approx(math.log(2.0))
    loss.backward()
    assert boundary.grad is not None
    assert float(boundary.grad[0]) < 0.0
    assert float(boundary.grad[1]) > 0.0

    easy = assay._hit_at_k_loss(
        torch,
        torch.tensor([5.0, 1.0, 0.0]),
        (0,),
        (1, 2),
        k=1,
        tau=assay.SMOOTHMAX_TAU,
        margin=assay.RANK_MARGIN,
    )
    hard = assay._hit_at_k_loss(
        torch,
        torch.tensor([0.0, 1.0, 0.0]),
        (0,),
        (1, 2),
        k=1,
        tau=assay.SMOOTHMAX_TAU,
        margin=assay.RANK_MARGIN,
    )
    assert float(easy) < float(hard)
    assert float(
        assay._hit_at_k_loss(
            torch,
            torch.tensor([1.0, 2.0]),
            (0, 1),
            (),
            k=8,
            tau=assay.SMOOTHMAX_TAU,
            margin=assay.RANK_MARGIN,
        )
    ) == 0.0


def test_evidence_loss_applies_fixed_any_source_exact_terms_and_missing_exact_boundary() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.tensor(
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
        requires_grad=True,
    )
    total, components = assay._evidence_loss(
        torch,
        scores,
        any_positive_indices=(0, 1),
        positives_by_source={"source-a": (0,), "source-b": (1,), "missing": ()},
        exact_positive_indices=(0,),
    )
    expected = (
        components["any_at_8"]
        + 0.5 * components["source_balanced_at_8"]
        + 0.5 * components["exact_at_8"]
        + 0.25 * components["any_at_1"]
    )
    assert float(total.detach()) == pytest.approx(expected)
    assert components["exact_present"] == 1.0
    assert components["represented_required_sources"] == 2.0
    expected_source = torch.stack(
        (
            assay._hit_at_k_loss(
                torch,
                scores,
                (0,),
                (1, 2, 3, 4, 5, 6, 7, 8, 9),
                k=assay.TOP_K,
                tau=assay.SMOOTHMAX_TAU,
                margin=assay.RANK_MARGIN,
            ),
            assay._hit_at_k_loss(
                torch,
                scores,
                (1,),
                (0, 2, 3, 4, 5, 6, 7, 8, 9),
                k=assay.TOP_K,
                tau=assay.SMOOTHMAX_TAU,
                margin=assay.RANK_MARGIN,
            ),
        )
    ).mean()
    assert components["source_balanced_at_8"] == pytest.approx(
        float(expected_source.detach())
    )
    total.backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())

    no_exact_total, no_exact = assay._evidence_loss(
        torch,
        torch.tensor([0.2, 0.1, 0.0]),
        any_positive_indices=(0,),
        positives_by_source={"source-a": (0,)},
        exact_positive_indices=(),
    )
    assert float(no_exact_total) >= 0.0
    assert no_exact["exact_at_8"] == 0.0
    assert no_exact["exact_present"] == 0.0
    with pytest.raises(ValueError, match="reachable positive"):
        assay._evidence_loss(
            torch,
            torch.tensor([0.2, 0.1]),
            any_positive_indices=(),
            positives_by_source={},
            exact_positive_indices=(),
        )


def _metrics_question(question_id: str) -> assay.PreparedQuestion:
    candidate_ids = tuple(f"{question_id}-m-{index:02d}" for index in range(96))
    return assay.PreparedQuestion(
        question_id=question_id,
        question_type="fixture",
        original_partition="test",
        fold="fold-4",
        query="fixture query",
        candidate_ids=candidate_ids,
        candidate_texts=tuple(f"candidate {index}" for index in range(96)),
        candidate_source_ids=("required-source",) + ("other-source",) * 95,
        required_source_ids=("required-source",),
        exact_memory_ids=(candidate_ids[0],),
    )


def _ranking(*first: int) -> tuple[int, ...]:
    return (*first, *(index for index in range(96) if index not in set(first)))


def test_metrics_and_rescue_regression_accounting_use_any_source_at_8() -> None:
    cases = (
        ("rescue", _ranking(1, 2, 3, 4, 5, 6, 7, 8), _ranking(0)),
        ("regress", _ranking(0), _ranking(1, 2, 3, 4, 5, 6, 7, 8)),
        ("success", _ranking(0), _ranking(0)),
        ("miss", _ranking(1, 2, 3, 4, 5, 6, 7, 8), _ranking(1, 2, 3, 4, 5, 6, 7, 8)),
    )
    rows = []
    for question_id, baseline_order, candidate_order in cases:
        question = _metrics_question(question_id)
        rows.append(
            {
                "question_id": question_id,
                "candidate_ceiling": assay._candidate_ceiling(question),
                "arms": {
                    "lexical": assay._row_metrics(question, baseline_order),
                    "trained": assay._row_metrics(question, candidate_order),
                },
            }
        )

    comparison = assay._comparison(rows, baseline="lexical", candidate="trained")
    assert comparison == {
        "baseline": "lexical",
        "candidate": "trained",
        "rescued": 1,
        "regressed": 1,
        "net_marginal": 0,
        "unchanged_success": 1,
        "unchanged_miss": 1,
        "rescued_question_ids": ["rescue"],
        "regressed_question_ids": ["regress"],
    }

    metrics = assay._aggregate_metrics(rows, "trained")
    assert metrics["questions"] == 4
    assert metrics["any_source"]["hit_at_8"]["count"] == 2
    assert float.fromhex(metrics["any_source"]["hit_at_8"]["rate_hex"]) == 0.5
    assert metrics["exact_turn"]["labeled_questions"] == 4
    assert metrics["exact_turn"]["frontier_eligible_questions"] == 4
    assert metrics["exact_turn"]["hit_at_8_over_all"]["count"] == 2
    assert float.fromhex(metrics["exact_turn"]["hit_at_8_conditional_on_frontier_hex"]) == 0.5
    assert metrics["all_sources_at_8"]["count"] == 2
    assert float.fromhex(metrics["mean_source_recall_at_8_hex"]) == 0.5


def test_publish_is_canonical_hashed_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "accuracy.json"
    payload = {"z": [3, 2, 1], "a": {"unicode": "café"}}
    digest, sidecar = assay._publish(path, payload)
    expected = (assay._canonical_json(payload) + "\n").encode("utf-8")

    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        assay._publish(path, {"changed": True})
    assert path.read_bytes() == before


def test_run_uses_fresh_models_and_trains_only_outside_each_heldout_fold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    question_ids = tuple(f"q-{index:03d}" for index in range(200))
    folds = {
        question_id: f"fold-{index // 40}"
        for index, question_id in enumerate(question_ids)
    }
    questions = []
    coordinates = {}
    frontiers = {}
    for question_id in question_ids:
        candidate_ids = tuple(
            f"{question_id}-m-{index:02d}" for index in range(assay.CANDIDATES)
        )
        questions.append(
            assay.PreparedQuestion(
                question_id=question_id,
                question_type="fixture",
                original_partition="train",
                fold=folds[question_id],
                query="fixture query",
                candidate_ids=candidate_ids,
                candidate_texts=tuple(
                    f"candidate {index}" for index in range(assay.CANDIDATES)
                ),
                candidate_source_ids=("required-source",)
                + ("other-source",) * (assay.CANDIDATES - 1),
                required_source_ids=("required-source",),
                exact_memory_ids=(candidate_ids[0],),
            )
        )
        coordinates[question_id] = assay.GoldCoordinates(
            required_source_ids=("required-source",),
            exact_memory_ids=(candidate_ids[0],),
        )
        frontiers[question_id] = candidate_ids

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:  # pragma: no cover - guarded by is_available
            raise AssertionError("CPU fixture must not empty a CUDA cache")

    class _FakeTorch:
        cuda = _FakeCuda()

    class _FakeModel:
        def __init__(self, load_index: int) -> None:
            self.load_index = load_index
            self.trained = False

        def eval(self) -> None:
            return None

    loads: list[_FakeModel] = []

    class _FakePilot:
        @staticmethod
        def _load_student(
            _model_root: Path,
            *,
            device_name: str,
            verify_base: bool,
        ) -> tuple[Any, Any, _FakeModel, str, str]:
            assert device_name == "cpu"
            assert verify_base is True
            model = _FakeModel(len(loads))
            loads.append(model)
            return _FakeTorch(), object(), model, "cpu", "checkpoint-sha"

    monkeypatch.setattr(
        assay,
        "_load_plane_and_dataset",
        lambda _plane, _dataset: ({"fixture": True}, "plane-sha", [], "dataset-sha"),
    )
    monkeypatch.setattr(
        assay, "_reconstruct_gold_coordinates", lambda _plane, _records: coordinates
    )
    monkeypatch.setattr(assay, "_fold_assignments", lambda _plane: folds)
    monkeypatch.setattr(assay, "_lexical_frontiers", lambda _plane: frontiers)
    monkeypatch.setattr(
        assay,
        "_prepare_questions",
        lambda _plane, _coordinates, _folds, _frontiers: questions,
    )
    monkeypatch.setattr(
        assay,
        "_base_inventory",
        lambda _root: ([{"name": "model.safetensors", "bytes": 1, "sha256": "a"}], "inventory-sha"),
    )
    monkeypatch.setattr(assay, "_pilot", lambda: _FakePilot())
    monkeypatch.setattr(
        assay,
        "_forward_scores",
        lambda _torch, _tokenizer, _model, _device, _question, *, training: None,
    )

    scored: list[tuple[int, bool, str]] = []

    def fake_score_question(
        _torch: Any,
        _tokenizer: Any,
        model: _FakeModel,
        _device: Any,
        question: assay.PreparedQuestion,
    ) -> tuple[tuple[float, ...], float]:
        scored.append((model.load_index, model.trained, question.question_id))
        return tuple(
            float(assay.CANDIDATES - index)
            for index in range(assay.CANDIDATES)
        ), 1.0

    trained_on: dict[int, tuple[str, ...]] = {}

    def fake_train_fold(
        _torch: Any,
        _tokenizer: Any,
        model: _FakeModel,
        _device: Any,
        training_questions: list[assay.PreparedQuestion],
        *,
        seed: int,
    ) -> tuple[list[dict[str, Any]], float, int, int]:
        assert seed == assay.BASE_SEED + model.load_index
        trained_on[model.load_index] = tuple(
            question.question_id for question in training_questions
        )
        model.trained = True
        return [], 0.0, len(training_questions), 0

    monkeypatch.setattr(assay, "_score_question", fake_score_question)
    monkeypatch.setattr(assay, "_train_fold", fake_train_fold)
    monkeypatch.setattr(
        assay,
        "_state_sha256",
        lambda model: f"trained-state-{model.load_index}",
    )

    payload = assay._run(
        argparse.Namespace(
            plane=tmp_path / "plane.json",
            dataset=tmp_path / "oracle.json",
            minilm_model_dir=tmp_path / "minilm",
            device="cpu",
        )
    )

    assert len(loads) == 5
    assert len({id(model) for model in loads}) == 5
    assert len(trained_on) == 5
    for fold_index in range(5):
        held_out = {
            question_id
            for question_id in question_ids
            if folds[question_id] == f"fold-{fold_index}"
        }
        training = set(trained_on[fold_index])
        assert len(held_out) == 40
        assert len(training) == 160
        assert training == set(question_ids) - held_out
        assert training.isdisjoint(held_out)
        assert {
            question_id
            for load_index, trained, question_id in scored
            if load_index == fold_index and not trained
        } == held_out
        assert {
            question_id
            for load_index, trained, question_id in scored
            if load_index == fold_index and trained
        } == held_out
    assert len(scored) == 5 * 40 * 2
    assert len(payload["rows"]) == 200
    assert [report["heldout_questions"] for report in payload["folds"]] == [40] * 5
