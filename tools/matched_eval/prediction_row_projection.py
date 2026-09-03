"""Gold-free projection of a sealed typed prediction result row.

The historical helper name described its eventual consumer and therefore
looked like a judge capability to static firebreaks.  This module names the
actual operation: it validates and narrows an already materialized prediction
row.  It has no answer, reference, dataset, scorer, or provider input.
"""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    assert_gold_blind,
    require_sha256,
    require_text,
)


PREDICTION_ROW_FORMAT = "memory-condense-typed-memory-final-arm-v1-judge-row-v1"


def prediction_row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, label-free public projection of one result row."""

    value = {
        "changed_from_parent": row.get("changed_from_parent"),
        "dated_question_sha256": row.get("dated_question_sha256"),
        "format": PREDICTION_ROW_FORMAT,
        "ordinal": row.get("ordinal"),
        "parent_prediction_sha256": row.get("parent_prediction_sha256"),
        "prediction": row.get("prediction"),
        "prediction_sha256": row.get("prediction_sha256"),
        "prediction_source": row.get("prediction_source"),
        "question_id": row.get("question_id"),
        "question_sha256": row.get("question_sha256"),
        "route_id": row.get("route_id"),
        "source_row_sha256": row.get("source_row_sha256"),
    }
    require_text(value["prediction"], "prediction")
    require_text(value["prediction_source"], "prediction source")
    require_text(value["question_id"], "prediction question ID")
    require_text(value["route_id"], "prediction route")
    for key in (
        "dated_question_sha256",
        "parent_prediction_sha256",
        "prediction_sha256",
        "question_sha256",
        "source_row_sha256",
    ):
        require_sha256(value[key], f"prediction {key}")
    if type(value["ordinal"]) is not int or type(value["changed_from_parent"]) is not bool:
        raise ValueError("prediction row scalar changed")
    assert_gold_blind(value, path="typed_final_prediction_row")
    return value


__all__ = ["PREDICTION_ROW_FORMAT", "prediction_row_projection"]
