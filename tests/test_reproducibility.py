from pathlib import Path

from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
)


def test_file_and_implementation_hashes_change_with_content(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first_file = file_sha256(source)
    first_tree = implementation_sha256(package)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert file_sha256(source) != first_file
    assert implementation_sha256(package) != first_tree


def test_environment_lock_hashes_pixi_lock(tmp_path: Path):
    lock = tmp_path / "pixi.lock"
    lock.write_text("locked", encoding="utf-8")

    assert environment_lock_sha256(tmp_path) == file_sha256(lock)
