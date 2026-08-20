from __future__ import annotations

import ast
import copy
import importlib.util
import json
import marshal
import subprocess
import sys
from pathlib import Path
from types import FunctionType, MappingProxyType

import pytest

import memory_condense.eval.diffuse_longmemeval_analysis as analysis_module
from memory_condense.eval import _diffuse_replay_provider_identity as identity_module
from memory_condense.eval._diffuse_replay_provider_identity import (
    CPYTHON_VERSION_POLICY,
    EXCLUDED_OBSERVABILITY,
    OPERATIONAL_CODE_IDENTITY_SCHEMA,
    OPERATIONAL_CODE_IDENTITY_VERSION,
    PROVIDER_IDENTITY_SCHEMA,
    PROVIDER_IDENTITY_VERSION,
    build_provider_identity_v2,
    classify_provider_identity_body,
    current_cpython_abi_identity,
    operational_code_sha256,
)


def _compile_provider(source: str, filename: str = "provider.py") -> FunctionType:
    namespace: dict[str, object] = {"__name__": "provider_identity_fixture"}
    exec(compile(source, filename, "exec"), namespace)
    provider = namespace["provider"]
    assert type(provider) is FunctionType
    return provider


def _digest(source: str) -> str:
    return operational_code_sha256(_compile_provider(source).__code__)


def test_digest_ignores_checkout_and_recursive_source_locations() -> None:
    compact = (
        "def provider(value):\n"
        "    def nested(item):\n"
        "        return item + 1\n"
        "    return nested(value)\n"
    )
    shifted = (
        "# checkout-local comment\n"
        "\n"
        "\n"
        "def provider(value):\n"
        "    # an inner comment also moves nested code\n"
        "\n"
        "    def nested(item):\n"
        "        # position-only detail\n"
        "        return item + 1\n"
        "\n"
        "    return nested(value)\n"
    )
    first = _compile_provider(compact, "C:/checkout-a/provider.py")
    second = _compile_provider(shifted, "Z:/checkout-b/provider.py")

    assert first.__code__.co_filename != second.__code__.co_filename
    assert first.__code__.co_firstlineno != second.__code__.co_firstlineno
    assert first.__code__.co_linetable != second.__code__.co_linetable
    assert operational_code_sha256(first.__code__) == operational_code_sha256(
        second.__code__
    )
    assert build_provider_identity_v2(first, {"control": "same"}) == (
        build_provider_identity_v2(second, {"control": "same"})
    )


def test_digest_excludes_only_explicit_position_metadata() -> None:
    provider = _compile_provider("def provider(value):\n    return value + 1\n")
    code = provider.__code__
    relocated = code.replace(
        co_filename="Q:/relocated/provider.py",
        co_firstlineno=code.co_firstlineno + 500,
        co_linetable=b"",
    )

    assert operational_code_sha256(code) == operational_code_sha256(relocated)


@pytest.mark.parametrize(
    ("left", "right", "changed_field"),
    [
        (
            "def provider(value):\n    return value + 1\n",
            "def provider(value):\n    return value - 1\n",
            "opcode",
        ),
        (
            "def provider(value):\n    return value + 1\n",
            "def provider(value):\n    return value + 2\n",
            "constant",
        ),
        (
            "def provider(value):\n    return normalize(value)\n",
            "def provider(value):\n    return transform(value)\n",
            "global-name",
        ),
        (
            "def provider(value):\n    return value\n",
            "def provider(value, /):\n    return value\n",
            "signature",
        ),
        (
            (
                "def provider(value):\n"
                "    def nested(item):\n"
                "        return item + 1\n"
                "    return nested(value)\n"
            ),
            (
                "def provider(value):\n"
                "    def nested(item):\n"
                "        return item + 2\n"
                "    return nested(value)\n"
            ),
            "nested-code",
        ),
        (
            (
                "def provider(value):\n"
                "    try:\n"
                "        return int(value)\n"
                "    except ValueError:\n"
                "        return 0\n"
            ),
            (
                "def provider(value):\n"
                "    try:\n"
                "        return int(value)\n"
                "    except TypeError:\n"
                "        return 0\n"
            ),
            "exception-behavior",
        ),
    ],
)
def test_digest_changes_with_executable_behavior(
    left: str,
    right: str,
    changed_field: str,
) -> None:
    del changed_field
    assert _digest(left) != _digest(right)


def test_digest_binds_exception_table_bytes_directly() -> None:
    provider = _compile_provider(
        "def provider(value):\n"
        "    try:\n"
        "        return int(value)\n"
        "    except ValueError:\n"
        "        return 0\n"
    )
    code = provider.__code__
    assert code.co_exceptiontable
    changed = code.replace(
        co_exceptiontable=(
            code.co_exceptiontable[:-1]
            + bytes((code.co_exceptiontable[-1] ^ 1,))
        )
    )

    assert code.co_code == changed.co_code
    assert code.co_consts == changed.co_consts
    assert operational_code_sha256(code) != operational_code_sha256(changed)


def test_constant_wire_format_retains_exact_types_and_boundaries() -> None:
    code = _compile_provider("def provider():\n    return None\n").__code__
    integer_tuple = code.replace(co_consts=(None, (1,)))
    boolean_tuple = code.replace(co_consts=(None, (True,)))
    float_tuple = code.replace(co_consts=(None, (1.0,)))
    frozen_set = code.replace(co_consts=(None, frozenset({1})))
    first_boundaries = code.replace(co_consts=(None, ("ab", "c")))
    second_boundaries = code.replace(co_consts=(None, ("a", "bc")))

    digests = {
        operational_code_sha256(item)
        for item in (integer_tuple, boolean_tuple, float_tuple, frozen_set)
    }
    assert len(digests) == 4
    assert operational_code_sha256(first_boundaries) != operational_code_sha256(
        second_boundaries
    )


def test_frozenset_constant_order_is_canonical() -> None:
    code = _compile_provider("def provider():\n    return None\n").__code__
    first = code.replace(co_consts=(None, frozenset(("alpha", "beta", 3))))
    second = code.replace(co_consts=(None, frozenset((3, "beta", "alpha"))))

    assert operational_code_sha256(first) == operational_code_sha256(second)


def test_provider_identity_is_explicit_deterministic_and_closed_json() -> None:
    class Provider:
        def analysis_identity_payload(self) -> object:
            raise AssertionError("v2 must not invoke an identity callback")

        def __call__(self, value: int) -> int:
            return value + 1

    declaration = {
        "z-control": (1, True, None),
        "a-control": MappingProxyType({"ratio": 0.5, "name": "stable"}),
    }
    first = build_provider_identity_v2(Provider(), declaration)
    second = build_provider_identity_v2(Provider(), declaration)

    assert first == second
    assert first == json.loads(
        json.dumps(
            first,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    assert first["schema"] == PROVIDER_IDENTITY_SCHEMA
    assert first["version"] == PROVIDER_IDENTITY_VERSION == 2
    assert first["operational_code_schema"] == OPERATIONAL_CODE_IDENTITY_SCHEMA
    assert (
        first["operational_code_version"]
        == OPERATIONAL_CODE_IDENTITY_VERSION
        == 2
    )
    assert first["python_abi"] == current_cpython_abi_identity()
    assert first["excluded_observability"] == list(EXCLUDED_OBSERVABILITY)
    assert first["declared_identity"] == {
        "a-control": {"name": "stable", "ratio": 0.5},
        "z-control": [1, True, None],
    }
    assert set(first) == {
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
    assert classify_provider_identity_body(first) == "operational-v2"


def test_provider_identity_detaches_and_binds_declared_controls() -> None:
    def provider(value: int) -> int:
        return value + 1

    nested = ["first"]
    declaration = {"controls": nested}
    first = build_provider_identity_v2(provider, declaration)
    nested.append("later mutation")
    second = build_provider_identity_v2(provider, declaration)

    assert first["declared_identity"] == {"controls": ["first"]}
    assert first != second


@pytest.mark.parametrize(
    "declaration",
    [
        {"binary": b"not-json"},
        {"set": {"not-json"}},
        {"nan": float("nan")},
        {1: "non-string-key"},
        {"surrogate": "\ud800"},
    ],
)
def test_provider_identity_rejects_non_closed_json_values(
    declaration: dict[object, object],
) -> None:
    def provider() -> None:
        return None

    with pytest.raises((TypeError, ValueError)):
        build_provider_identity_v2(provider, declaration)  # type: ignore[arg-type]


def test_provider_identity_rejects_cycles_and_non_python_callables() -> None:
    def provider() -> None:
        return None

    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="must not contain cycles"):
        build_provider_identity_v2(provider, cycle)
    with pytest.raises(TypeError, match="exact Python callable code"):
        build_provider_identity_v2(len, {})


def test_exact_cpython_abi_fields_are_explicit_and_digest_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abi = current_cpython_abi_identity()
    assert abi == {
        "implementation_name": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version_policy": CPYTHON_VERSION_POLICY,
        "version": list(sys.version_info[:3]),
        "bytecode_magic_number_hex": importlib.util.MAGIC_NUMBER.hex(),
    }
    baseline = _digest("def provider(value):\n    return value + 1\n")
    changed = copy.deepcopy(abi)
    changed["bytecode_magic_number_hex"] = "00000000"
    monkeypatch.setattr(
        identity_module,
        "current_cpython_abi_identity",
        lambda: copy.deepcopy(changed),
    )
    assert _digest("def provider(value):\n    return value + 1\n") != baseline


def test_callable_components_are_unambiguous_and_unicode_validated() -> None:
    first = _compile_provider("def provider(value):\n    return value\n")
    second = _compile_provider("def provider(value):\n    return value\n")
    first.__module__, first.__qualname__ = "a.", "b"
    second.__module__, second.__qualname__ = "a", ".b"

    first_identity = build_provider_identity_v2(first, {})
    second_identity = build_provider_identity_v2(second, {})
    assert first.__module__ + "." + first.__qualname__ == (
        second.__module__ + "." + second.__qualname__
    )
    assert first_identity != second_identity
    assert first_identity["implementation_module"] == "a."
    assert first_identity["implementation_qualname"] == "b"

    first.__module__ = "\ud800"
    with pytest.raises(ValueError, match="valid Unicode"):
        build_provider_identity_v2(first, {})
    first.__module__, first.__qualname__ = "valid", "\ud800"
    with pytest.raises(ValueError, match="valid Unicode"):
        build_provider_identity_v2(first, {})

    class Provider:
        def __call__(self, value: int) -> int:
            return value

    Provider.__module__ = "\ud800"
    with pytest.raises(ValueError, match="valid Unicode"):
        build_provider_identity_v2(Provider(), {})


def test_code_and_mapping_keys_require_exact_valid_unicode() -> None:
    provider = _compile_provider("def provider(value):\n    return value\n")
    invalid_code = provider.__code__.replace(co_name="\ud800")
    with pytest.raises(ValueError, match="valid Unicode"):
        operational_code_sha256(invalid_code)
    with pytest.raises(ValueError, match="valid Unicode"):
        build_provider_identity_v2(provider, {"\ud800": "value"})

    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="exact string"):
        build_provider_identity_v2(
            provider,
            {StringSubclass("control"): "value"},
        )


def test_provider_identity_rejects_defaults_keyword_defaults_and_closures() -> None:
    def positional_default(value: int = 1) -> int:
        return value

    def keyword_default(*, value: int = 1) -> int:
        return value

    captured = 1

    def closure(value: int) -> int:
        return value + captured

    with pytest.raises(TypeError, match="callable defaults"):
        build_provider_identity_v2(positional_default, {})
    with pytest.raises(TypeError, match="keyword defaults"):
        build_provider_identity_v2(keyword_default, {})
    with pytest.raises(TypeError, match="callable closures"):
        build_provider_identity_v2(closure, {})


def test_identity_schema_discriminator_rejects_unknown_and_mixed_bodies() -> None:
    def provider(value: int) -> int:
        return value

    current = build_provider_identity_v2(provider, {})
    legacy = {
        "implementation_type": "builtins.function",
        "implementation": "fixture.provider",
        "python_code_sha256": "0" * 64,
        "declared_identity": {},
    }
    assert classify_provider_identity_body(legacy) == "historical-v1"

    wrong_tag = copy.deepcopy(current)
    wrong_tag["schema"] = "unknown-v2"
    with pytest.raises(ValueError, match="v2 identity tags changed"):
        classify_provider_identity_body(wrong_tag)
    mixed = copy.deepcopy(current)
    mixed["python_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unsupported identity schema"):
        classify_provider_identity_body(mixed)
    incomplete = copy.deepcopy(current)
    incomplete.pop("python_abi")
    with pytest.raises(ValueError, match="unsupported identity schema"):
        classify_provider_identity_body(incomplete)


@pytest.mark.parametrize("field", ["python_abi", "excluded_observability"])
def test_identity_discriminator_rejects_nested_equality_imposters(
    field: str,
) -> None:
    def provider(value: int) -> int:
        return value

    class EqualityImposter:
        def __eq__(self, other: object) -> bool:
            del other
            return True

    identity = build_provider_identity_v2(provider, {})
    if field == "python_abi":
        identity["python_abi"]["cache_tag"] = EqualityImposter()
    else:
        identity["excluded_observability"][0] = EqualityImposter()
    with pytest.raises(TypeError, match="non-JSON value"):
        classify_provider_identity_body(identity)


def test_analysis_v2_marker_selects_harness_builder_and_rejects_forgery() -> None:
    class Provider:
        __memory_condense_operational_identity_v2__ = (
            identity_module._OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER
        )

        def analysis_identity_payload(self) -> dict[str, object]:
            return {
                "operational_code_sha256": "0" * 64,
                "control": "declared-only",
            }

        def __call__(self, value: int) -> int:
            return value + 1

    payload = analysis_module.analysis_callable_identity_payload(
        Provider(),
        "provider",
    )
    assert classify_provider_identity_body(payload) == "operational-v2"
    assert payload["operational_code_sha256"] != "0" * 64
    assert payload["declared_identity"] == {
        "control": "declared-only",
        "operational_code_sha256": "0" * 64,
    }


def test_analysis_rejects_unknown_and_inherited_v2_markers() -> None:
    class Unknown:
        __memory_condense_operational_identity_v2__ = object()

        def __call__(self, value: int) -> int:
            return value

    with pytest.raises(ValueError, match="unsupported identity version marker"):
        analysis_module.analysis_callable_identity_payload(Unknown(), "provider")

    class Marked:
        __memory_condense_operational_identity_v2__ = (
            identity_module._OPERATIONAL_PROVIDER_IDENTITY_V2_MARKER
        )

        def analysis_identity_payload(self) -> dict[str, object]:
            return {"control": "owned"}

        def __call__(self, value: int) -> int:
            return value

    class Inherited(Marked):
        pass

    with pytest.raises(TypeError, match="own its v2 identity marker directly"):
        analysis_module.analysis_callable_identity_payload(Inherited(), "provider")


def test_unmarked_analysis_callable_retains_exact_historical_v1_schema() -> None:
    def provider(value: int) -> int:
        return value + 1

    payload = analysis_module.analysis_callable_identity_payload(
        provider,
        "provider",
    )
    assert set(payload) == {
        "implementation_type",
        "implementation",
        "python_code_sha256",
        "declared_identity",
    }
    assert classify_provider_identity_body(payload) == "historical-v1"


def test_digest_does_not_use_whole_code_marshal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = Path(identity_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "marshal" not in imported_roots

    def forbidden(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("marshal.dumps must not be used")

    monkeypatch.setattr(marshal, "dumps", forbidden)
    assert len(_digest("def provider(value):\n    return value + 1\n")) == 64


def test_provider_identity_module_is_a_cold_import() -> None:
    forbidden = {
        "torch",
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "safetensors",
    }
    script = (
        "import sys; "
        "import memory_condense.eval._diffuse_replay_provider_identity; "
        f"print(sorted({forbidden!r} & set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]", result.stdout
