"""Live-session bridge: find active Claude/Codex sessions, mmap them, serve views.

Pipeline: ``handles`` (which processes hold which transcripts open) ->
``discovery`` (session refs with liveness) -> ``mmap_view`` (seamless
read-only mapping that follows appends) -> ``virtual_fs`` (transformed
mirror driven by the live set).
"""

from .discovery import (
    SessionKind,
    SessionRef,
    discover_sessions,
    find_session,
    live_sessions,
)
from .handles import HandleEvent, HandleSnapshot, HandleTracker, OpenHandle, snapshot, writers_of
from .mmap_view import MappedSession, Record
from .virtual_fs import Mirror, RecordTransform, VirtualSession, projfs_available

__all__ = [
    "HandleEvent",
    "HandleSnapshot",
    "HandleTracker",
    "MappedSession",
    "Mirror",
    "OpenHandle",
    "Record",
    "RecordTransform",
    "SessionKind",
    "SessionRef",
    "VirtualSession",
    "discover_sessions",
    "find_session",
    "live_sessions",
    "projfs_available",
    "snapshot",
    "writers_of",
]
