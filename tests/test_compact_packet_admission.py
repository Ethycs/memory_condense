import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools.assay_hot_compact_packet_reduced30 import admit_units


def test_complete_exchange_brings_its_raw_reference_even_when_that_reference_ranks_last():
    units = [
        {"label": "G1", "block": "<G1> exact cited raw 🐕", "dependencies": []},
        {"label": "G2", "block": "<G2> unrelated " * 100, "dependencies": []},
        {"label": "E1", "block": "<E1>\n<REF G1>\n<A1 owner=G1> complete reply", "dependencies": ["G1"]},
    ]
    context, audit = admit_units(units, {"G1": -10, "G2": 2, "E1": 3}, max_context_tokens=80)
    assert audit["selected_labels"] == ["G1", "E1"]
    assert audit["omitted_labels"] == ["G2"]
    assert context == units[0]["block"] + "\n\n" + units[2]["block"]
    assert audit["frontier_closed"] is False
    assert count_tokens(context) <= 80


def test_exchange_is_never_partly_admitted_when_its_dependency_cannot_fit():
    units = [
        {"label": "G1", "block": "large backing " * 100, "dependencies": []},
        {"label": "G2", "block": "small independent evidence", "dependencies": []},
        {"label": "E1", "block": "<REF G1>", "dependencies": ["G1"]},
    ]
    context, audit = admit_units(units, {"G1": -1, "G2": 0, "E1": 5}, max_context_tokens=30)
    assert context == units[1]["block"]
    assert audit["selected_labels"] == ["G2"]


@pytest.mark.parametrize("dependency", ["G99", "E1"])
def test_foreign_or_cyclic_dependencies_are_rejected(dependency):
    units = [{"label": "G1", "block": "raw", "dependencies": []},
             {"label": "E1", "block": "exchange", "dependencies": [dependency]}]
    with pytest.raises(ValueError, match="dependency"):
        admit_units(units, {"G1": 0, "E1": 1})


def test_primary_cap_and_ties_are_stable_without_claiming_complete_retrieval():
    units = [{"label": f"G{i}", "block": str(i), "dependencies": []} for i in range(3)]
    _, audit = admit_units(units, {u["label"]: 1 for u in units}, max_primary_units=1)
    assert audit["selected_labels"] == ["G0"]
    assert audit["omitted_labels"] == ["G1", "G2"]


def test_score_population_and_finite_values_are_required():
    units = [{"label": "G1", "block": "raw", "dependencies": []}]
    with pytest.raises(ValueError, match="population"):
        admit_units(units, {})
    with pytest.raises(ValueError, match="nonfinite"):
        admit_units(units, {"G1": float("nan")})
