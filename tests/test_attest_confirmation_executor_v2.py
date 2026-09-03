from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools import attest_confirmation_executor_v1 as v1
from tools import attest_confirmation_executor_v2 as subject


HEAD = "a" * 40
TREE = "b" * 40


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _boundary_payload(root: Path) -> dict[str, Any]:
    entrypoint = root / subject.PRODUCTION_ENTRYPOINT
    entrypoint_sha = v1._sha256(entrypoint.read_bytes())  # noqa: SLF001
    executor_files = [
        {
            "bytes": entrypoint.stat().st_size,
            "path": subject.PRODUCTION_ENTRYPOINT,
            "sha256": entrypoint_sha,
        }
    ]
    locks = [{"bytes": 4, "path": "pixi.lock", "sha256": "c" * 64}]
    body = {
        "format": v1.FORMAT,
        "status": v1.STATUS,
        "frozen_policy": {
            "path": v1.POLICY_MANIFEST_RELATIVE,
            "sha256": v1.POLICY_MANIFEST_SHA256,
        },
        "executor_git": {
            "head_commit_sha1": HEAD,
            "git_tree_sha1": TREE,
            "worktree_clean_before_attestation": True,
        },
        "executor_inventory": {
            "format": v1.INVENTORY_FORMAT,
            "executor_files": executor_files,
            "dependency_locks": locks,
        },
    }
    return {**body, "attestation_identity_sha256": v1.identity_sha256(body)}


def _fixture(tmp_path: Path, *, entrypoint: str = "VALUE = 1\n") -> tuple[Path, Path]:
    _write(tmp_path / subject.PRODUCTION_ENTRYPOINT, entrypoint)
    _write(tmp_path / "tests/test_fake.py", "def test_ok(): assert True\n")
    _write(tmp_path / "pixi.lock", "lock")
    receipt_path = tmp_path / "eval_results/offline.json"
    receipt = subject._test_receipt_body(  # noqa: SLF001
        head_commit_sha1=HEAD,
        git_tree_sha1=TREE,
        test_files=("tests/test_fake.py",),
        passed_count=subject.MINIMUM_OFFLINE_TEST_COUNT,
    )
    v1.publish_sealed_json(receipt_path, receipt)
    return tmp_path, receipt_path


def _compile(root: Path, receipt: Path) -> dict[str, Any]:
    return subject.compile_confirmation_executor_attestation_v2(
        repository_root=root,
        offline_test_receipt_path=receipt,
        executor_files=(subject.PRODUCTION_ENTRYPOINT,),
        offline_test_files=("tests/test_fake.py",),
        prediction_firebreak_files=(subject.PRODUCTION_ENTRYPOINT,),
        boundary_compiler=lambda **_: _boundary_payload(root),
    )


def test_v2_readiness_is_narrow_provider_free_and_self_sealed(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    payload = _compile(root, receipt)

    assert payload["format"] == subject.FORMAT
    assert payload["status"] == subject.STATUS
    assert payload["release_scope"] == {
        "may_open_sanitized_treatment": True,
        "may_open_gold": False,
        "may_call_provider": False,
    }
    assert payload["provider_accounting"]["physical_provider_calls"] == 0
    assert payload["safety"]["remaining_executable_parent_stages_in_order"] == []
    body = dict(payload)
    seal = body.pop("attestation_identity_sha256")
    assert seal == v1.identity_sha256(body)


def test_v2_binds_inventory_lock_entrypoint_and_test_receipt(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    apparatus = _compile(root, receipt)["apparatus"]

    assert apparatus["executor_file_set_sha256"] == v1.identity_sha256(
        apparatus["executor_files"]
    )
    assert apparatus["dependency_lock_sha256"] == v1.identity_sha256(
        apparatus["dependency_locks"]
    )
    assert apparatus["production_entrypoint_path"] == subject.PRODUCTION_ENTRYPOINT
    assert apparatus["offline_test_count"] == subject.MINIMUM_OFFLINE_TEST_COUNT
    assert apparatus["offline_test_receipt_path"] == "eval_results/offline.json"


def test_v2_rejects_test_receipt_from_another_commit(tmp_path: Path) -> None:
    root, _ = _fixture(tmp_path)
    receipt_path = root / "eval_results/foreign.json"
    receipt = subject._test_receipt_body(  # noqa: SLF001
        head_commit_sha1="d" * 40,
        git_tree_sha1=TREE,
        test_files=("tests/test_fake.py",),
        passed_count=subject.MINIMUM_OFFLINE_TEST_COUNT,
    )
    v1.publish_sealed_json(receipt_path, receipt)

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="another executor"):
        _compile(root, receipt_path)


def test_v2_rejects_gold_or_judge_import_in_prediction_entrypoint(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint="from tools.confirmation_gold_judge_scaffold import main\n",
    )
    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="judge/gold"):
        _compile(root, receipt)


def test_v2_rejects_indirect_gold_import_in_prediction_closure(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path, entrypoint="from tools import helper\n")
    _write(
        root / "tools/helper.py",
        "from tools.confirmation_sol_judge_lifecycle import run\n",
    )
    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="judge/gold"):
        _compile(root, receipt)


def test_v2_rejects_judge_import_from_executed_package_initializer(
    tmp_path: Path,
) -> None:
    root, receipt = _fixture(tmp_path, entrypoint="from tools.pkg import helper\n")
    _write(
        root / "tools/pkg/__init__.py",
        "from tools.confirmation_gold_judge_scaffold import run\n",
    )
    _write(root / "tools/pkg/helper.py", "VALUE = 1\n")

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="judge/gold"):
        _compile(root, receipt)


def test_v2_rejects_literal_dynamic_judge_import(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint=(
            "import importlib\n"
            "def run():\n"
            "    return importlib.import_module("
            "'tools.confirmation_sol_judge_lifecycle')\n"
        ),
    )

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="judge/gold"):
        _compile(root, receipt)


def test_v2_rejects_unresolved_dynamic_import(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint=(
            "from importlib import import_module\n"
            "def run(module_name):\n"
            "    return import_module(module_name)\n"
        ),
    )

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="dynamic import"):
        _compile(root, receipt)


def test_v2_resolves_package_relative_dynamic_import(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path, entrypoint="from tools import pkg\n")
    _write(
        root / "tools/pkg/__init__.py",
        "from importlib import import_module\n"
        "boundary = import_module(f'{__name__}.gold')\n",
    )
    _write(root / "tools/pkg/gold.py", "VALUE = 1\n")

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="judge/gold"):
        _compile(root, receipt)


def test_v2_rejects_imported_reference_loader_callable(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint="from tools.helper import load_reference_answers\n",
    )
    _write(
        root / "tools/helper.py",
        "def load_reference_answers(path): return path\n",
    )

    with pytest.raises(subject.ConfirmationExecutorReadinessError, match="callable"):
        _compile(root, receipt)


def test_v2_does_not_confuse_preference_with_reference(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint="from tools.helper import select_profile_preference_evidence\n",
    )
    _write(
        root / "tools/helper.py",
        "def select_profile_preference_evidence(value): return value\n",
    )

    _compile(root, receipt)


def test_v2_allows_closed_inert_lazy_facade(tmp_path: Path) -> None:
    root, receipt = _fixture(
        tmp_path,
        entrypoint="from tools.pkg import safe\n",
    )
    _write(
        root / "tools/pkg/__init__.py",
        "from importlib import import_module as _import_module\n"
        "_EXPORTS = {'value': ('tools.pkg.safe', 'VALUE')}\n"
        "def __getattr__(name):\n"
        "    module_name, attribute_name = _EXPORTS[name]\n"
        "    return getattr(_import_module(module_name), attribute_name)\n",
    )
    _write(root / "tools/pkg/safe.py", "VALUE = 1\n")

    _compile(root, receipt)


def test_prediction_firebreak_root_excludes_treatment_exporter() -> None:
    assert subject.PREDICTION_FIREBREAK_FILES == {subject.PRODUCTION_ENTRYPOINT}
    assert "tools/export_confirmation_treatment_v5_r3.py" in (
        subject.DEFAULT_EXECUTOR_FILES
    )


def test_v2_rejects_tampered_offline_receipt(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(v1.ConfirmationExecutorAttestationError):
        _compile(root, receipt)


def test_cli_has_no_treatment_gold_or_provider_release_option() -> None:
    parser = subject.build_parser()
    options = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    for choice in parser._subparsers._group_actions[0].choices.values():  # noqa: SLF001
        options.update(
            option
            for action in choice._actions  # noqa: SLF001
            for option in action.option_strings
        )
    assert "--treatment" not in options
    assert "--gold" not in options
    assert "--enable-provider" not in options
    assert "--authorized-provider-calls" not in options
