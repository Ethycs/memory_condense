from __future__ import annotations

import json

import pytest
import torch

from memory_condense.cav_probe import (
    bootstrap_stability,
    evaluate_cav,
    fit_mean_difference_cav,
    load_concept_datasets,
    random_label_control,
)


def test_mean_difference_cav_separates_heldout_examples() -> None:
    train_positive = torch.tensor([[3.0, 0.1], [2.5, -0.1], [4.0, 0.0]])
    train_negative = torch.tensor([[-3.0, 0.0], [-2.5, 0.2], [-4.0, -0.2]])
    test_positive = torch.tensor([[2.0, 0.5], [5.0, -1.0]])
    test_negative = torch.tensor([[-2.0, -0.5], [-5.0, 1.0]])

    cav = fit_mean_difference_cav(train_positive, train_negative)
    metrics = evaluate_cav(cav, test_positive, test_negative)

    assert torch.allclose(cav.vector, torch.tensor([1.0, 0.0]), atol=0.05)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["score_margin"] > 0


def test_bootstrap_is_stable_for_a_clear_direction() -> None:
    positive = torch.tensor([[3.0, 0.0], [2.0, 0.1], [4.0, -0.1], [3.5, 0.2]])
    negative = -positive

    stability = bootstrap_stability(positive, negative, repeats=16, seed=4)

    assert stability["mean_cosine"] > 0.99


def test_random_label_control_does_not_recover_true_direction() -> None:
    positive = torch.tensor([[3.0, 0.0], [2.0, 0.1], [4.0, -0.1], [3.5, 0.2]])
    negative = -positive

    control = random_label_control(
        positive,
        negative,
        positive,
        negative,
        repeats=32,
        seed=7,
    )

    assert 0.20 <= control["mean_balanced_accuracy"] <= 0.80


def test_dataset_loader_requires_balanced_material(tmp_path) -> None:
    path = tmp_path / "concepts.json"
    path.write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "name": "constraint",
                        "train": {"positive": ["one"], "negative": ["a", "b"]},
                        "test": {"positive": ["c", "d"], "negative": ["e", "f"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two train positive"):
        load_concept_datasets(path)
