"""Detect the wire format of a request body and rewrite its conversation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import units as U
from .policy import ContextPolicy, apply


@dataclass
class Report:
    format: str | None
    policy: str
    rewritten: bool
    reason: str = ""
    units_total: int = 0
    units_kept: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def detect(path: str, body: dict) -> U.Format | None:
    if path.endswith("/v1/messages") or path.endswith("/v1/messages/count_tokens"):
        return "anthropic" if isinstance(body.get("messages"), list) else None
    if path.endswith("/responses"):
        return "responses" if isinstance(body.get("input"), list) else None
    return None


def rewrite(path: str, body: dict, policy: ContextPolicy, locator=None) -> tuple[dict | None, Report]:
    """Return (rewritten body, report); body is None when nothing changed.

    ``locator`` (a ``bridge.SessionLocator``) lets the policy hand the memory
    system a handle to the originating on-disk transcript.
    """
    fmt = detect(path, body)
    if fmt is None:
        return None, Report(format=None, policy=policy.name, rewritten=False, reason="not an inference body")
    if fmt == "responses" and body.get("previous_response_id"):
        return None, Report(format=fmt, policy=policy.name, rewritten=False, reason="server-side state (previous_response_id)")
    key = "messages" if fmt == "anthropic" else "input"
    conversation = body[key]
    units = U.group(fmt, conversation)
    session = None
    if locator is not None:
        from .bridge import locate_for_units

        try:
            session = locate_for_units(locator, units)
        except Exception:  # locating is best-effort; never block inference on it
            session = None
    plan = policy.plan(units, fmt, session)
    report = Report(
        format=fmt,
        policy=policy.name,
        rewritten=False,
        units_total=len(units),
        units_kept=sum(plan.keep),
        tokens_before=U.estimate_tokens(conversation),
        tokens_after=U.estimate_tokens(conversation),
    )
    if not plan.changed:
        report.reason = "within budget"
        return None, report
    new_conversation = apply(fmt, units, plan)
    problems = U.check(fmt, new_conversation)
    if problems:
        # Safety net: never send a structurally invalid conversation.
        report.reason = "rewrite would break pairing: " + "; ".join(problems[:3])
        return None, report
    report.rewritten = True
    report.tokens_after = U.estimate_tokens(new_conversation)
    report.reason = f"elided {len(plan.dropped)} units"
    if session is not None:
        report.reason += f"; original transcript {getattr(session, 'session_id', '?')[:8]}"
    return {**body, key: new_conversation}, report
