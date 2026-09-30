"""Local model-API proxy: the client shows its full transcript, the model sees ours.

``units`` groups a request's conversation into pair-safe units, ``policy``
chooses what to keep and what one memory block replaces, ``wire`` detects the
format (Anthropic Messages / Responses) and rewrites the body, ``server``
forwards and streams. Works under claude.ai and ChatGPT logins because auth
headers are forwarded verbatim.
"""

from .policy import (
    CondenseRequest,
    Condenser,
    ContextPolicy,
    PassthroughPolicy,
    Plan,
    RecencyWindowPolicy,
    extractive_digest,
)
from .server import ProxyConfig, Upstream, make_server
from .units import Unit, group, check, estimate_tokens
from .wire import Report, detect, rewrite

__all__ = [
    "CondenseRequest",
    "Condenser",
    "ContextPolicy",
    "PassthroughPolicy",
    "Plan",
    "ProxyConfig",
    "RecencyWindowPolicy",
    "Report",
    "Unit",
    "Upstream",
    "check",
    "detect",
    "estimate_tokens",
    "extractive_digest",
    "group",
    "make_server",
    "rewrite",
]
