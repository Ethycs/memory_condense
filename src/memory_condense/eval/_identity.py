"""One shared validator family for eval receipt identities.

Case policy (deliberate): SHA-256 digests are lowercase-normalized on
validation, matching the domain validator.  ``hashlib`` hexdigests only ever
emit lowercase, so widening the former reject-uppercase copies to
normalize-lowercase is behavior-preserving for every digest this pipeline
produces.
"""

from __future__ import annotations

from memory_condense.domain._discourse_identity import (
    _sha256 as sha256_digest,
)
from memory_condense.domain._discourse_identity import exact_int

__all__ = ["exact_int", "sha256_digest"]
