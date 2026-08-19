"""Compatibility facade for query-conditioned coverage selection.

Implementation lives in focused compiler, contract, feature, INI, and prefix
pipeline modules. Existing imports and monkeypatch seams remain available here.
"""

from memory_condense.search.selectors.coverage_models import (
    CandidateAssignment,
    CompletionFn,
    CoverageScoreProvider,
    CoverageSelectionReport,
)
from memory_condense.search.selectors.evidence_features import (
    _VENUE_QUERY_RE,
    _canonical_answer_object_key,
    _energy_softmax,
    _normalized_event_key,
    _normalized_scalars,
    _normalized_transport,
    _optional_probability,
    _source_id,
    _surface_value_evidence,
    _timestamp_key,
)
from memory_condense.search.selectors.ini_coverage_selector import (
    QueryConditionedCoverageSelector,
    _ASSIGNMENT_COLUMNS,
    _SYSTEM_PROMPT,
    _clean_ini_field,
    _decode_assignment_rows,
    _extract_json_object,
    _parse_assignment,
)
from memory_condense.search.selectors.prefix_models import (
    _PrefixAssignment,
    _PrefixEventCluster,
)
from memory_condense.search.selectors.prefix_selector import QwenPrefixCoverageSelector
from memory_condense.search.selectors.set_program import (
    SetOperator,
    SetOrdering,
    SetProgram,
    SetQuantifier,
    _is_first_person_current_possessed_scalar,
    _required_evidence_role,
    compile_set_program,
)

__all__ = [
    "CandidateAssignment",
    "CompletionFn",
    "CoverageScoreProvider",
    "CoverageSelectionReport",
    "QueryConditionedCoverageSelector",
    "QwenPrefixCoverageSelector",
    "SetOperator",
    "SetOrdering",
    "SetProgram",
    "SetQuantifier",
    "compile_set_program",
]
