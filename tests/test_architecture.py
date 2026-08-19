"""Structural invariants that documentation used to assert and nothing checked.

`docs/03 - Architecture/00` states its axioms as things "validated by grep".
A grep in a document is validated exactly once, on the day someone runs it.
These tests run on every commit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "memory_condense"

#: SDKs that talk to a hosted model. Importing one at module scope makes the
#: whole package depend on a provider and pay its import cost.
_LLM_SDKS = {"litellm", "anthropic", "openai", "google", "cohere", "mistralai"}

#: The two places a provider binding is allowed to live. `eval/` is the harness
#: (it is *supposed* to call models); `llm_provider` is the single seam that
#: binds one for the core — and even there the import is inside a function.
_BINDING_ALLOWED = {"application/llm_provider.py"}


def _core_modules() -> list[Path]:
    """Every core module: `src/memory_condense/**.py` excluding `eval/`."""
    return sorted(
        p
        for p in SRC.rglob("*.py")
        if "eval" not in p.relative_to(SRC).parts
    )


def test_source_modules_are_grouped_by_responsibility():
    """Implementation modules belong to packages, not the project root."""
    assert {path.name for path in SRC.glob("*.py")} == {"__init__.py"}
    assert {
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {
        "application",
        "associations",
        "domain",
        "eval",
        "ingest",
        "interfaces",
        "modeling",
        "persistence",
        "search",
        "tooling",
    }
    assert {path.name for path in (SRC / "search").glob("*.py")} == {
        "__init__.py"
    }
    assert {
        path.name
        for path in (SRC / "search").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {"closure", "episodes", "indexes", "packing", "selectors"}


_WORKFLOW_FACADES = {
    "application/condenser.py": 800,
    "associations/association_store.py": 200,
    "associations/head_memory.py": 200,
    "eval/__main__.py": 400,
    "eval/campaign.py": 250,
    "eval/mem0_adapter.py": 250,
    "eval/recall.py": 200,
    "search/indexes/retrieval.py": 200,
    "search/packing/context_packer.py": 300,
    "search/selectors/coverage_selector.py": 200,
}


@pytest.mark.parametrize("relative_path,max_lines", _WORKFLOW_FACADES.items())
def test_decomposed_facades_remain_small(relative_path, max_lines):
    """Compatibility modules orchestrate; they do not regrow implementations."""

    path = SRC / relative_path
    assert len(path.read_text(encoding="utf-8").splitlines()) <= max_lines


def test_source_modules_remain_reviewably_sized():
    """Keep objects, transformations, and stateful workflows reviewable.

    Line count is only a coarse regression tripwire, not an architecture
    metric. The generous ceiling catches the former 1,500--4,400-line
    monoliths while leaving normal cohesive modules alone.
    """

    oversized = {
        path.relative_to(SRC).as_posix(): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in SRC.rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 1_300
    }
    assert not oversized, f"split oversized source modules: {oversized}"


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level import names only — imports inside a function body are fine."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # body, not walk: module scope only
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: p.name)
def test_no_core_module_imports_an_llm_sdk_at_module_scope(path):
    """Axiom 1: provider SDKs stay out of the core package's import graph.

    `llm_provider` binds a provider on purpose, but does it with a local
    import inside the function, so `import memory_condense` still costs
    nothing and needs no credentials.
    """
    offending = _module_scope_imports(path) & _LLM_SDKS
    assert not offending, (
        f"{path.name} imports {sorted(offending)} at module scope. "
        "Move it inside the function that needs it."
    )


def _any_scope_imports(path: Path) -> set[str]:
    """Import names anywhere in the file, including inside function bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_llm_provider_is_the_only_core_module_that_binds_a_provider():
    """A second binding seam would defeat the point of having one.

    Checked over the AST rather than the raw text: `extractor.py` names litellm
    in a docstring while importing nothing, and a grep-based version of this
    test failed on that — punishing the module for documenting its own contract.
    """
    binding = {
        path.relative_to(SRC).as_posix()
        for path in _core_modules()
        if _any_scope_imports(path) & _LLM_SDKS
    }
    assert binding <= _BINDING_ALLOWED, (
        f"unexpected provider imports in {sorted(binding - _BINDING_ALLOWED)}"
    )


def test_importing_the_package_does_not_import_an_llm_sdk():
    """The end-to-end version of the axiom, in one assertion."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, memory_condense; "
            f"print(sorted({_LLM_SDKS!r} & set(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_importing_an_eval_identity_helper_does_not_import_an_llm_sdk():
    """Read-only provenance helpers must not initialize provider machinery."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from memory_condense.eval.reproducibility import implementation_sha256; "
            f"print(sorted({_LLM_SDKS!r} & set(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_decay_kernel_has_exactly_one_implementation():
    """The bug that made decay decorative: two copies of the exponential.

    `ranking.recency_score` duplicated `decay.effective_energy`'s arithmetic
    and had drifted from it. Anything computing `0.5 ** (elapsed / half_life)`
    outside `decay.py` is re-opening that.
    """
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in _core_modules()
        if path.relative_to(SRC).as_posix() != "domain/decay.py"
        and "0.5 ** (" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{offenders} compute a decay curve directly; call decay.decay_factor"
    )


def test_root_facade_is_lazy_and_resolves_canonical_objects():
    """The public facade stays stable without pre-importing executable modules."""
    import importlib
    import memory_condense

    assert len(memory_condense.__all__) == 56
    assert set(memory_condense.__all__) <= set(dir(memory_condense))

    namespace: dict[str, object] = {}
    exec("from memory_condense import *", namespace)
    assert set(memory_condense.__all__) <= set(namespace)

    for name, (module_name, attribute_name) in memory_condense._EXPORTS.items():
        assert getattr(memory_condense, name) is getattr(
            importlib.import_module(module_name),
            attribute_name,
        )

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(memory_condense, "not_a_public_symbol")


@pytest.mark.parametrize(
    "module_name",
    [
        "memory_condense.modeling.qwen_prefix",
        "memory_condense.associations.head_memory",
    ],
)
def test_executable_module_is_not_preimported_by_package(module_name):
    """``python -m`` targets must not trigger runpy's pre-import warning."""
    import subprocess
    import sys

    code = (
        "import sys, memory_condense; "
        f"print({module_name!r} in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stdout
