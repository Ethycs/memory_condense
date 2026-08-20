"""Exact CPython-ABI-bound identities for replay-provider operations.

The v2 code digest owns an explicit, length-framed wire format. It binds the
current CPython implementation, cache tag, exact major/minor/micro policy, and
bytecode magic number. It intentionally elides source-location metadata.

This is an *operational code identity* within that exact ABI and exact owned
provider scope, not a claim of full Python semantic equivalence. Traceback,
frame, and source-location observability plus the values or resolution of
dynamic dependencies (globals, imports, descriptors, and external state) are
out of scope. A replay provider must bind relevant dynamic state separately in
``declared_identity``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import struct
import sys
from collections.abc import Mapping
from types import CodeType, FunctionType, MethodType
from typing import Literal, TypeAlias


OPERATIONAL_CODE_IDENTITY_SCHEMA = (
    "memory-condense-cpython-abi-location-elided-operational-code-v2"
)
OPERATIONAL_CODE_IDENTITY_VERSION = 2
PROVIDER_IDENTITY_SCHEMA = (
    "memory-condense-diffuse-replay-provider-operational-identity-v2"
)
PROVIDER_IDENTITY_VERSION = 2
CPYTHON_VERSION_POLICY = "exact-major-minor-micro-v1"

HISTORICAL_PROVIDER_IDENTITY_V1_KEYS = frozenset(
    {
        "implementation_type",
        "implementation",
        "python_code_sha256",
        "declared_identity",
    }
)
PROVIDER_IDENTITY_V2_KEYS = frozenset(
    {
        "schema",
        "version",
        "implementation_type_module",
        "implementation_type_qualname",
        "implementation_module",
        "implementation_qualname",
        "operational_code_schema",
        "operational_code_version",
        "operational_code_sha256",
        "python_abi",
        "excluded_observability",
        "declared_identity",
    }
)
EXCLUDED_OBSERVABILITY = (
    "traceback-frame-source-location",
    "dynamic-dependency-resolution-and-state",
)

# This selects the in-harness builder; it is never serialized and never lets a
# callable provide its own digest. Analysis requires exact direct class
# ownership, so subclasses cannot inherit the opt-in silently.
_OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER = object()

_BEHAVIOR_CODE_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_name",
    "co_qualname",
    "co_exceptiontable",
)
_LOCATION_CODE_FIELDS = frozenset(
    {
        "co_filename",
        "co_firstlineno",
        "co_linetable",
        "co_lnotab",
        "co_lines",
        "co_positions",
    }
)
_KNOWN_CODE_ATTRIBUTES = frozenset(_BEHAVIOR_CODE_FIELDS) | _LOCATION_CODE_FIELDS

_JsonScalar: TypeAlias = None | bool | int | float | str
_ClosedJson: TypeAlias = _JsonScalar | list["_ClosedJson"] | dict[str, "_ClosedJson"]


def current_cpython_abi_identity() -> dict[str, object]:
    """Return the exact ABI policy bound by the operational-code schema."""

    implementation = _detached_unicode(
        sys.implementation.name,
        label="Python implementation name",
    )
    if implementation != "cpython":
        raise RuntimeError("operational code identity requires CPython")
    cache_tag = _detached_unicode(
        sys.implementation.cache_tag,
        label="Python implementation cache tag",
    )
    if not cache_tag:
        raise RuntimeError("CPython cache tag must be non-empty")
    version = sys.version_info
    components = (version.major, version.minor, version.micro)
    if any(type(item) is not int or item < 0 for item in components):
        raise RuntimeError("CPython version components are invalid")
    magic = importlib.util.MAGIC_NUMBER
    if type(magic) is not bytes or not magic:
        raise RuntimeError("CPython bytecode magic number is unavailable")
    return {
        "implementation_name": implementation,
        "cache_tag": cache_tag,
        "version_policy": CPYTHON_VERSION_POLICY,
        "version": list(components),
        "bytecode_magic_number_hex": magic.hex(),
    }


def operational_code_sha256(code: CodeType) -> str:
    """Hash location-elided operations under the exact current CPython ABI.

    Equality is meaningful only for this schema and ABI. It does not establish
    full Python semantic equivalence because locations and dynamic dependencies
    are explicitly outside the identity boundary.
    """

    if type(code) is not CodeType:
        raise TypeError("code must be an exact CodeType")
    payload = _record(
        "operational-code-identity",
        (
            ("schema", _text(OPERATIONAL_CODE_IDENTITY_SCHEMA)),
            ("version", _integer(OPERATIONAL_CODE_IDENTITY_VERSION)),
            ("python_abi", _python_abi_wire()),
            ("code", _code(code)),
        ),
    )
    return hashlib.sha256(payload).hexdigest()


def build_provider_identity_v2(
    value: object,
    declared_identity: Mapping[str, object],
    *,
    label: str = "provider",
) -> dict[str, object]:
    """Build one detached, closed-JSON v2 identity from actual callable code.

    The callable cannot author its digest. Defaults, keyword defaults, and
    closures are rejected because this schema does not inspect their values.
    Builtins are rejected because they expose no exact CPython ``CodeType``.
    """

    label = _detached_unicode(label, label="identity label")
    if not label.strip():
        raise ValueError("label must be a non-empty string")
    if not callable(value):
        raise TypeError(f"{label} must be callable")
    if not isinstance(declared_identity, Mapping):
        raise TypeError(f"{label} declared identity must be a mapping")

    target = _python_target(value, label=label)
    _reject_unbound_callable_state(target, label=label)
    type_module, type_qualname = _qualified_components(type(value))
    target_module, target_qualname = _qualified_components(target)
    identity: dict[str, object] = {
        "schema": PROVIDER_IDENTITY_SCHEMA,
        "version": PROVIDER_IDENTITY_VERSION,
        "implementation_type_module": type_module,
        "implementation_type_qualname": type_qualname,
        "implementation_module": target_module,
        "implementation_qualname": target_qualname,
        "operational_code_schema": OPERATIONAL_CODE_IDENTITY_SCHEMA,
        "operational_code_version": OPERATIONAL_CODE_IDENTITY_VERSION,
        "operational_code_sha256": operational_code_sha256(target.__code__),
        "python_abi": current_cpython_abi_identity(),
        "excluded_observability": list(EXCLUDED_OBSERVABILITY),
        "declared_identity": _closed_json_mapping(
            declared_identity,
            label=f"{label} declared identity",
        ),
    }
    return _closed_json_mapping(identity, label=f"{label} v2 identity")


def classify_provider_identity_body(
    value: object,
) -> Literal["historical-v1", "operational-v2"]:
    """Discriminate the two exact replay-provider identity wire schemas."""

    if type(value) is not dict:
        raise ValueError("verified-base provider identity must be an object")
    keys = frozenset(value)
    if keys == HISTORICAL_PROVIDER_IDENTITY_V1_KEYS:
        return "historical-v1"
    if keys != PROVIDER_IDENTITY_V2_KEYS:
        raise ValueError("verified-base provider has an unsupported identity schema")
    value = _closed_json_mapping(
        value,
        label="verified-base provider v2 identity",
    )
    if (
        type(value.get("schema")) is not str
        or value["schema"] != PROVIDER_IDENTITY_SCHEMA
        or type(value.get("version")) is not int
        or value["version"] != PROVIDER_IDENTITY_VERSION
        or type(value.get("operational_code_schema")) is not str
        or value["operational_code_schema"] != OPERATIONAL_CODE_IDENTITY_SCHEMA
        or type(value.get("operational_code_version")) is not int
        or value["operational_code_version"] != OPERATIONAL_CODE_IDENTITY_VERSION
        or type(value.get("excluded_observability")) is not list
        or value["excluded_observability"] != list(EXCLUDED_OBSERVABILITY)
        or type(value.get("python_abi")) is not dict
        or value["python_abi"] != current_cpython_abi_identity()
        or type(value.get("declared_identity")) is not dict
    ):
        raise ValueError("verified-base provider v2 identity tags changed")
    _require_lower_hex(
        value.get("operational_code_sha256"),
        label="provider operational code SHA-256",
    )
    for key in (
        "implementation_type_module",
        "implementation_type_qualname",
        "implementation_module",
        "implementation_qualname",
    ):
        item = value.get(key)
        detached = _detached_unicode(item, label=f"provider {key}")
        if detached != item:
            raise ValueError(f"provider {key} is not canonical Unicode")
    normalized = _closed_json_mapping(
        value["declared_identity"],
        label="provider declared identity",
    )
    if normalized != value["declared_identity"]:
        raise ValueError("provider declared identity is not canonical closed JSON")
    return "operational-v2"


def _python_target(value: object, *, label: str) -> FunctionType:
    if type(value) is FunctionType:
        return value
    if type(value) is MethodType:
        target = value.__func__
    else:
        target = inspect.getattr_static(type(value), "__call__", None)
        if isinstance(target, (staticmethod, classmethod)):
            target = target.__func__
    if type(target) is not FunctionType or type(target.__code__) is not CodeType:
        raise TypeError(f"{label} must expose exact Python callable code")
    return target


def _reject_unbound_callable_state(target: FunctionType, *, label: str) -> None:
    if target.__defaults__ is not None:
        raise TypeError(f"{label} callable defaults are not supported")
    if target.__kwdefaults__ is not None:
        raise TypeError(f"{label} callable keyword defaults are not supported")
    if target.__closure__ is not None:
        raise TypeError(f"{label} callable closures are not supported")


def _qualified_components(value: object) -> tuple[str, str]:
    module = _detached_unicode(
        getattr(value, "__module__", None),
        label="Python callable module",
    )
    qualname = _detached_unicode(
        getattr(value, "__qualname__", None),
        label="Python callable qualname",
    )
    return module, qualname


def _python_abi_wire() -> bytes:
    abi = current_cpython_abi_identity()
    version = abi["version"]
    if type(version) is not list or len(version) != 3:  # pragma: no cover
        raise RuntimeError("CPython version identity changed")
    return _record(
        "python-abi",
        (
            ("implementation_name", _text(abi["implementation_name"])),
            ("cache_tag", _text(abi["cache_tag"])),
            ("version_policy", _text(abi["version_policy"])),
            (
                "version",
                _sequence("version", tuple(_integer(item) for item in version)),
            ),
            (
                "bytecode_magic_number_hex",
                _text(abi["bytecode_magic_number_hex"]),
            ),
        ),
    )


def _code(code: CodeType) -> bytes:
    _assert_code_schema(code)
    return _record(
        "code",
        (
            ("co_argcount", _integer(code.co_argcount)),
            ("co_posonlyargcount", _integer(code.co_posonlyargcount)),
            ("co_kwonlyargcount", _integer(code.co_kwonlyargcount)),
            ("co_nlocals", _integer(code.co_nlocals)),
            ("co_stacksize", _integer(code.co_stacksize)),
            ("co_flags", _integer(code.co_flags)),
            ("co_code", _binary(code.co_code)),
            ("co_consts", _constant_tuple(code.co_consts)),
            ("co_names", _string_tuple(code.co_names)),
            ("co_varnames", _string_tuple(code.co_varnames)),
            ("co_freevars", _string_tuple(code.co_freevars)),
            ("co_cellvars", _string_tuple(code.co_cellvars)),
            ("co_name", _text(code.co_name)),
            ("co_qualname", _text(code.co_qualname)),
            ("co_exceptiontable", _binary(code.co_exceptiontable)),
        ),
    )


def _assert_code_schema(code: CodeType) -> None:
    attributes = {name for name in dir(code) if name.startswith("co_")}
    missing = set(_BEHAVIOR_CODE_FIELDS) - attributes
    unknown = attributes - _KNOWN_CODE_ATTRIBUTES
    if missing or unknown:
        detail = ", ".join(
            part
            for part in (
                f"missing={sorted(missing)!r}" if missing else "",
                f"unknown={sorted(unknown)!r}" if unknown else "",
            )
            if part
        )
        raise RuntimeError(f"unsupported Python CodeType schema: {detail}")


def _constant(value: object) -> bytes:
    value_type = type(value)
    if value is None:
        return _frame("none", b"")
    if value is Ellipsis:
        return _frame("ellipsis", b"")
    if value_type is bool:
        return _frame("bool", b"\x01" if value else b"\x00")
    if value_type is int:
        return _integer(value)
    if value_type is float:
        return _frame("float64", struct.pack(">d", value))
    if value_type is complex:
        return _frame("complex128", struct.pack(">dd", value.real, value.imag))
    if value_type is str:
        return _text(value)
    if value_type is bytes:
        return _binary(value)
    if value_type is tuple:
        return _sequence("tuple", tuple(_constant(item) for item in value))
    if value_type is frozenset:
        encoded = tuple(sorted(_constant(item) for item in value))
        return _sequence("frozenset", encoded)
    if value_type is CodeType:
        return _code(value)
    raise TypeError(
        "unsupported Python code constant type: "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _constant_tuple(values: tuple[object, ...]) -> bytes:
    return _sequence("constant-tuple", tuple(_constant(value) for value in values))


def _string_tuple(values: tuple[str, ...]) -> bytes:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError("CodeType name fields must be exact string tuples")
    return _sequence("string-tuple", tuple(_text(value) for value in values))


def _integer(value: int) -> bytes:
    if type(value) is not int:
        raise TypeError("operational identity integers must be exact ints")
    return _frame("int", str(value).encode("ascii"))


def _text(value: object) -> bytes:
    detached = _detached_unicode(value, label="operational identity string")
    return _frame("str", detached.encode("utf-8"))


def _binary(value: bytes) -> bytes:
    if type(value) is not bytes:
        raise TypeError("operational identity byte fields must be exact bytes")
    return _frame("bytes", value)


def _sequence(tag: str, values: tuple[bytes, ...]) -> bytes:
    return _frame(tag, _uint64(len(values)) + b"".join(values))


def _record(tag: str, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    payload = _uint64(len(fields)) + b"".join(
        _frame(name, value) for name, value in fields
    )
    return _frame(tag, payload)


def _frame(tag: str, payload: bytes) -> bytes:
    if type(tag) is not str:
        raise TypeError("operational identity tags must be exact strings")
    try:
        tag_bytes = tag.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - internal constants
        raise ValueError("operational identity tags must be ASCII") from exc
    return _uint64(len(tag_bytes)) + tag_bytes + _uint64(len(payload)) + payload


def _uint64(value: int) -> bytes:
    if not 0 <= value < 2**64:
        raise OverflowError("operational identity frame is too large")
    return value.to_bytes(8, byteorder="big", signed=False)


def _closed_json_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, _ClosedJson]:
    normalized = _closed_json(value, label=label, active=set())
    if type(normalized) is not dict:  # pragma: no cover - guarded by input type
        raise TypeError(f"{label} must be a mapping")
    return normalized


def _closed_json(value: object, *, label: str, active: set[int]) -> _ClosedJson:
    value_type = type(value)
    if value is None or value_type in (bool, int):
        return value  # type: ignore[return-value]
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite JSON numbers")
        return value
    if value_type is str:
        return _detached_unicode(value, label=label)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{label} must not contain cycles")
        active.add(marker)
        try:
            pairs: list[tuple[str, _ClosedJson]] = []
            for key, child in value.items():
                detached_key = _detached_unicode(key, label=f"{label} key")
                pairs.append(
                    (detached_key, _closed_json(child, label=label, active=active))
                )
            return dict(sorted(pairs))
        finally:
            active.remove(marker)
    if value_type in (list, tuple):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{label} must not contain cycles")
        active.add(marker)
        try:
            return [_closed_json(child, label=label, active=active) for child in value]
        finally:
            active.remove(marker)
    raise TypeError(
        f"{label} contains a non-JSON value of type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _detached_unicode(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    try:
        return value.encode("utf-8", errors="strict").decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{label} must contain valid Unicode") from exc


def _require_lower_hex(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


__all__ = [
    "CPYTHON_VERSION_POLICY",
    "EXCLUDED_OBSERVABILITY",
    "HISTORICAL_PROVIDER_IDENTITY_V1_KEYS",
    "OPERATIONAL_CODE_IDENTITY_SCHEMA",
    "OPERATIONAL_CODE_IDENTITY_VERSION",
    "PROVIDER_IDENTITY_SCHEMA",
    "PROVIDER_IDENTITY_V2_KEYS",
    "PROVIDER_IDENTITY_VERSION",
    "build_provider_identity_v2",
    "classify_provider_identity_body",
    "current_cpython_abi_identity",
    "operational_code_sha256",
]
