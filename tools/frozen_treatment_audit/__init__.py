"""Provider-free structural audit for the frozen v3 treatment campaign.

The package intentionally does not import :mod:`memory_condense`.  It reads
the frozen implementation through Git objects and treats benchmark reports
and cache artifacts as untrusted inputs.  It proves report lineage, exact
prompt reconstruction, source coordinates, ordinary-file integrity, and
schema/storage invariants.  It does *not* authenticate provider or judge
execution, independently verify factual accuracy, replay retrieval, or prove
the semantics of arbitrary vector/ANN bytes.
"""

from .audit import AuditError, audit_frozen_treatment

__all__ = ["AuditError", "audit_frozen_treatment"]
