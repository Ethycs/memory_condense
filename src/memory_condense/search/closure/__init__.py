"""Source-grounded episodic discourse closure."""

from memory_condense.search.closure.compiler import (
    compile_query_program,
    extract_subject_terms,
    infer_intent,
)
from memory_condense.search.closure.engine import (
    EvidenceClosureEngine,
    close_evidence,
)
from memory_condense.search.closure.store import EvidenceClosureStore

__all__ = [
    "EvidenceClosureEngine",
    "EvidenceClosureStore",
    "close_evidence",
    "compile_query_program",
    "extract_subject_terms",
    "infer_intent",
]
