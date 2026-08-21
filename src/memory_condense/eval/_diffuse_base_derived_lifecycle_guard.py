"""Transitive operation guard for the held derived-store lifecycle."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Any
import weakref

from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_operation_guard,
)


def freeze_derived_lifecycle_guard(
    namespace: dict[str, Any],
    *,
    primitive_namespace: dict[str, Any],
    native_namespace: dict[str, Any],
    state_op: Callable[[object], Any],
    revoke_op: Callable[[object, Any], None],
    registration_state_op: Callable[
        [object], tuple[Path, tuple[Any, ...]]
    ] | None,
    emergency_abandon_op: Callable[[object], Path],
    emergency_registration_op: Callable[[object], Path],
    raw_close: Callable[[int], object],
    kernel32: object | None,
    error_type: type[Exception],
) -> Callable[..., tuple[Callable[..., object], ...]]:
    """Freeze every mutable stdlib/native seam reached after callbacks."""

    os_names = (
        "close", "dup", "fstat", "fsencode", "fsync", "fspath", "ftruncate",
        "lseek", "open", "read", "scandir", "stat", "strerror",
        "unlink", "write", "name", "path", "SEEK_SET", "EEXIST",
        "O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY", "O_RDONLY",
        "O_NONBLOCK", "O_RDWR", "O_CREAT", "O_EXCL",
    )
    ctypes_names = (
        "addressof", "byref", "cast", "create_string_buffer",
        "create_unicode_buffer", "get_errno", "get_last_error", "memmove",
        "POINTER", "sizeof", "WinError",
    )
    dependencies: list[tuple[object, str]] = [
        *((os, name) for name in os_names),
        *((os.path, name) for name in (
            "abspath", "normcase", "normpath",
        )),
        *((stat, name) for name in ("S_IFMT", "S_ISDIR", "S_ISREG")),
        *((ctypes, name) for name in ctypes_names if hasattr(ctypes, name)),
        (Path, "mkdir"),
        (hashlib, "sha256"),
        (secrets, "token_hex"),
        (weakref, "finalize"),
        (weakref, "ref"),
        (weakref.finalize, "detach"),
        (errno, "EEXIST"),
        (threading.Event, "__init__"),
        (threading.Event, "set"),
        (threading.Event, "wait"),
    ]
    if os.name == "nt":
        from ctypes import wintypes

        dependencies.extend(
            (wintypes, name)
            for name in (
                "BOOL", "DWORD", "FILETIME", "HANDLE", "LPCVOID",
                "LPCWSTR", "LPVOID", "LPWSTR", "WCHAR",
            )
        )
        if kernel32 is None:
            raise RuntimeError("Windows lifecycle guard requires kernel32")
        dependencies.extend(
            (kernel32, name)
            for name in (
                "CloseHandle", "CreateFileW", "DuplicateHandle",
                "FlushFileBuffers", "GetCurrentProcess",
                "GetFileInformationByHandle", "GetFinalPathNameByHandleW",
                "ReadFile", "SetEndOfFile", "SetFileInformationByHandle",
                "SetFilePointerEx", "WriteFile",
            )
        )
    return freeze_operation_guard(
        namespace,
        primitive_namespace=primitive_namespace,
        additional_namespaces=(("derived-native", native_namespace),),
        state_op=state_op,
        revoke_op=revoke_op,
        registration_state_op=registration_state_op,
        emergency_abandon_op=emergency_abandon_op,
        emergency_registration_op=emergency_registration_op,
        raw_close=raw_close,
        windows=os.name == "nt",
        error_type=error_type,
        attribute_dependencies=tuple(dependencies),
    )


__all__ = ["freeze_derived_lifecycle_guard"]
