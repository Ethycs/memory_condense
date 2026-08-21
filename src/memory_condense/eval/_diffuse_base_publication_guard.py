"""Closure-owned integrity guard for diffuse publication operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import FunctionType
from typing import Any


def _make_fingerprinter() -> Callable[[object], object]:
    immutable = (type(None), bool, int, float, complex, str, bytes)

    def closure_value(value: object, active: set[int]) -> object:
        if isinstance(value, immutable):
            return value
        if isinstance(value, (tuple, frozenset)):
            return fingerprint(value, active)
        if getattr(value, "__code__", None) is not None:
            return fingerprint(value, active)
        return ("identity", type(value), id(value))

    def fingerprint(value: object, active: set[int] | None = None) -> object:
        if active is None:
            active = set()
        marker = id(value)
        if marker in active:
            return ("cycle", marker)
        if isinstance(value, immutable):
            return value
        active.add(marker)
        try:
            return project(value, active)
        finally:
            active.remove(marker)

    def project(value: object, active: set[int]) -> object:
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (
                        (fingerprint(key, active), fingerprint(item, active))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            )
        if isinstance(value, (set, frozenset)):
            return tuple(sorted((fingerprint(item, active) for item in value), key=repr))
        if isinstance(value, (list, tuple)):
            return tuple(fingerprint(item, active) for item in value)
        code = getattr(value, "__code__", None)
        if code is not None:
            closure = getattr(value, "__closure__", None) or ()
            cells = []
            for cell in closure:
                try:
                    item = cell.cell_contents
                except ValueError:
                    cells.append(("empty-cell",))
                else:
                    cells.append(closure_value(item, active))
            return (
                "callable",
                id(value),
                id(code),
                fingerprint(getattr(value, "__defaults__", None), active),
                fingerprint(getattr(value, "__kwdefaults__", None), active),
                tuple(cells),
            )
        if hasattr(value, "argtypes") and hasattr(value, "restype"):
            return (
                "ctypes-callable",
                id(value),
                fingerprint(getattr(value, "argtypes", None), active),
                fingerprint(getattr(value, "restype", None), active),
                fingerprint(getattr(value, "errcheck", None), active),
            )
        return ("identity", type(value), id(value))

    return fingerprint


def freeze_namespace_guard(
    namespace: dict[str, Any],
    *,
    error_type: type[Exception],
    label: str,
    exclude: tuple[str, ...] = (),
) -> Callable[[], None]:
    """Return a closure-owned callable/code/default namespace fingerprint."""

    fingerprint = _make_fingerprinter()
    expected = tuple(
        (name, fingerprint(value))
        for name, value in namespace.items()
        if not name.startswith("__") and name not in exclude
    )

    def assert_intact() -> None:
        changed = [
            name
            for name, value in expected
            if fingerprint(namespace.get(name)) != value
        ]
        if changed:
            raise error_type(f"{label} was rebound: " + ", ".join(changed))

    return assert_intact


def freeze_callable_guard(
    expected_callable: Callable[..., Any],
    *,
    error_type: type[Exception],
    label: str,
) -> Callable[[Callable[..., Any]], None]:
    """Return an unexposed recursive fingerprint check for one callable."""

    fingerprint = _make_fingerprinter()
    expected = fingerprint(expected_callable)

    def assert_current(current: Callable[..., Any]) -> None:
        if fingerprint(current) != expected:
            raise error_type(f"{label} was rebound")

    return assert_current


def freeze_operation_guard(
    namespace: dict[str, Any],
    *,
    primitive_namespace: dict[str, Any],
    state_op: Callable[[object], Any],
    revoke_op: Callable[[object, Any], None],
    raw_close: Callable[[int], object],
    windows: bool,
    error_type: type[Exception],
    attribute_dependencies: tuple[tuple[object, str], ...] = (),
    additional_namespaces: tuple[tuple[str, dict[str, Any]], ...] = (),
    registration_state_op: Callable[
        [object], tuple[Path, tuple[Any, ...]]
    ] | None = None,
    emergency_abandon_op: Callable[[object], Path] | None = None,
    emergency_registration_op: Callable[[object], Path] | None = None,
) -> Callable[
    [],
    tuple[Callable[[], None], Callable[[object], Path]]
    | tuple[
        Callable[[], None],
        Callable[[object], Path],
        Callable[[object], Path],
    ],
]:
    """Freeze both filesystem namespaces and an emergency revoke closure."""

    def clone_cell(value: object):
        def carry() -> object:
            return value

        return carry.__closure__[0]  # type: ignore[index]

    def clone_function(function: Callable[..., Any]) -> Callable[..., Any]:
        closure = tuple(
            clone_cell(cell.cell_contents) for cell in (function.__closure__ or ())
        )
        copied = FunctionType(
            function.__code__,
            dict(function.__globals__),
            function.__name__,
            function.__defaults__,
            closure or None,
        )
        copied.__kwdefaults__ = function.__kwdefaults__
        return copied

    stable_state_op = clone_function(state_op)
    stable_revoke_op = clone_function(revoke_op)
    stable_registration_state_op = (
        clone_function(registration_state_op)
        if registration_state_op is not None
        else None
    )
    stable_emergency_abandon_op = (
        clone_function(emergency_abandon_op)
        if emergency_abandon_op is not None
        else None
    )
    stable_emergency_registration_op = (
        clone_function(emergency_registration_op)
        if emergency_registration_op is not None
        else None
    )
    if windows:
        import ctypes
        from ctypes import wintypes

        emergency_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        stable_raw_close = emergency_kernel32.CloseHandle
        stable_raw_close.argtypes = (wintypes.HANDLE,)
        stable_raw_close.restype = wintypes.BOOL
    else:
        stable_raw_close = raw_close
    fingerprint = _make_fingerprinter()
    expected = {
        name: value
        for name, value in namespace.items()
        if not name.startswith("__")
    }
    primitive_expected = {
        name: value
        for name, value in primitive_namespace.items()
        if not name.startswith("__")
    }

    expected_fingerprints = tuple(
        (name, fingerprint(value)) for name, value in expected.items()
    )
    primitive_fingerprints = tuple(
        (name, fingerprint(value)) for name, value in primitive_expected.items()
    )
    additional_fingerprints = tuple(
        (
            label,
            additional,
            tuple(
                (name, fingerprint(value))
                for name, value in additional.items()
                if not name.startswith("__")
            ),
        )
        for label, additional in additional_namespaces
    )
    attribute_fingerprints = tuple(
        (owner, name, fingerprint(getattr(owner, name, None)))
        for owner, name in attribute_dependencies
    )

    def acquire() -> (
        tuple[Callable[[], None], Callable[[object], Path]]
        | tuple[
            Callable[[], None],
            Callable[[object], Path],
            Callable[[object], Path],
        ]
    ):
        def assert_intact() -> None:
            changed = [
                name
                for name, value in expected_fingerprints
                if fingerprint(namespace.get(name)) != value
            ]
            changed.extend(
                f"primitive:{name}"
                for name, value in primitive_fingerprints
                if fingerprint(primitive_namespace.get(name)) != value
            )
            for label, additional, expected_values in additional_fingerprints:
                changed.extend(
                    f"{label}:{name}"
                    for name, value in expected_values
                    if fingerprint(additional.get(name)) != value
                )
            changed.extend(
                f"attribute:{name}"
                for owner, name, value in attribute_fingerprints
                if fingerprint(getattr(owner, name, None)) != value
            )
            if changed:
                raise error_type(
                    "publication operation boundary was rebound: "
                    + ", ".join(changed)
                )

        def emergency_abandon(owner: object) -> Path:
            if stable_emergency_abandon_op is not None:
                return stable_emergency_abandon_op(owner)
            state = stable_state_op(owner)
            handles = [*state.held]
            if state.store_child is not None:
                handles.append(state.store_child)
            handles.append(state.root)
            if state.marker is not None:
                handles.append(state.marker)
            handles.extend(state.parent_chain)
            seen: set[int] = set()
            failure: BaseException | None = None
            for entry in reversed(handles):
                if entry.handle in seen:
                    continue
                try:
                    result = stable_raw_close(entry.handle)
                    if windows and not result:
                        raise error_type(
                            "cannot close quarantined publication handle"
                        )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                seen.add(entry.handle)
            try:
                stable_revoke_op(owner, state)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"publication revoke also failed: {exc!r}"
                    )
            if failure is not None:
                raise failure
            return state.path

        def emergency_discard_registration(clone: object) -> Path:
            if stable_emergency_registration_op is not None:
                return stable_emergency_registration_op(clone)
            if stable_registration_state_op is None:
                raise TypeError("registration cleanup is unavailable")
            path, entries = stable_registration_state_op(clone)
            seen: set[int] = set()
            failure: BaseException | None = None
            for entry in reversed(entries):
                if entry.handle in seen:
                    continue
                try:
                    result = stable_raw_close(entry.handle)
                    if windows and not result:
                        raise error_type(
                            "cannot close quarantined registration handle"
                        )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                seen.add(entry.handle)
            if failure is not None:
                raise failure
            return path

        assert_intact()
        if (
            stable_registration_state_op is not None
            or stable_emergency_registration_op is not None
        ):
            return assert_intact, emergency_abandon, emergency_discard_registration
        return assert_intact, emergency_abandon

    return acquire


__all__ = [
    "freeze_callable_guard",
    "freeze_namespace_guard",
    "freeze_operation_guard",
]
