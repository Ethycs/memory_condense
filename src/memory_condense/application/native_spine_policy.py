"""Serving defaults validated on the persisted native summary-memory path."""
from types import MappingProxyType


CAP8_LIMITS = MappingProxyType({
    'ancestor_hops': 2,
    'context_seed_limit': 4,
    'lexical_reserve': 0,
    'max_additions': 16,
    'max_context_tokens': 2048,
    'max_direct': 8,
    'max_raw_spans': 128,
    'protected_direct': 0,
    'user_completion_atoms': 8,
})
