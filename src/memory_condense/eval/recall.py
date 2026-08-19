"""Retrieval-recall compatibility facade.

Recall asks a provider-free question: can the benchmark's gold answer reach
the final prompt?  Contracts, pure lexical transformations, context assembly,
measurement workflows, and reporting now live in focused modules.  Their
historical imports remain available here with canonical object identity.
"""

from __future__ import annotations

from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.recall_measurement import (
    measure_sample,
    run_recall,
)
from memory_condense.eval.recall_models import (
    DEFAULT_HORIZONS_TURNS,
    AnswerValueCoverage,
    QuestionRecall,
    RecallReport,
)
from memory_condense.eval.recall_reporting import print_recall_report
