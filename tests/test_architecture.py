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
_BINDING_ALLOWED = {"llm_provider.py"}


def _core_modules() -> list[Path]:
    """Every core module: `src/memory_condense/**.py` excluding `eval/`."""
    return sorted(
        p
        for p in SRC.rglob("*.py")
        if "eval" not in p.relative_to(SRC).parts
    )


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
        path.name
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


def test_decay_kernel_has_exactly_one_implementation():
    """The bug that made decay decorative: two copies of the exponential.

    `ranking.recency_score` duplicated `decay.effective_energy`'s arithmetic
    and had drifted from it. Anything computing `0.5 ** (elapsed / half_life)`
    outside `decay.py` is re-opening that.
    """
    offenders = [
        path.name
        for path in _core_modules()
        if path.name != "decay.py" and "0.5 ** (" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{offenders} compute a decay curve directly; call decay.decay_factor"
    )
