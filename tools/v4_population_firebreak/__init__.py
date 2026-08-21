"""Provider-free population and evaluator-firebreak verification.

Scoring objects are deliberately resolved lazily.  Treatment-only consumers
must not acquire the evaluator schema merely by importing this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .analysis import (
    export_analysis_treatment_input,
    verify_analysis_treatment_input,
)
from .canonical import FirebreakError
from .treatment import (
    AnalysisTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
    load_analysis_treatment_input,
)
from .verifier import (
    PRODUCTION_LOCK,
    ExpectedPopulationLock,
    verify_evaluator_firebreak,
)

__all__ = [
    "AnalysisScoringLabel",
    "AnalysisTreatmentInput",
    "ExpectedPopulationLock",
    "FirebreakError",
    "PRODUCTION_LOCK",
    "TreatmentQuestion",
    "TreatmentSample",
    "export_analysis_scoring_label",
    "export_analysis_treatment_input",
    "load_analysis_scoring_label",
    "load_analysis_treatment_input",
    "verify_analysis_treatment_input",
    "verify_evaluator_firebreak",
]


_LAZY_SCORING_EXPORTS = frozenset(
    {
        "AnalysisScoringLabel",
        "export_analysis_scoring_label",
        "load_analysis_scoring_label",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_SCORING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    scoring = import_module(f"{__name__}.scoring")
    value = getattr(scoring, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
