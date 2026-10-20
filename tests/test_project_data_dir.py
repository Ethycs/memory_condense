"""One resolver decides which store is this project's.

The proxy, the MCP server and the status resolver each used to answer this
separately and disagreed, so `memory_reset` erased a different store than
`recall` read. These tests pin the shared contract and the two properties
that make it safe to share: the project-root walk, and the stdlib-only import
cost the statusline pays on every refresh.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_condense.control import status as control_status
from memory_condense.interfaces import mcp_server, proxy_server
from memory_condense.runtime import project
from memory_condense.runtime.config import user_directory


@pytest.fixture
def repo(tmp_path):
    """A project root with a marker and a nested working directory."""
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "src" / "pkg"
    deep.mkdir(parents=True)
    return tmp_path, deep


class TestResolutionOrder:
    def test_explicit_env_var_wins_over_everything(self, repo):
        root, deep = repo
        env = {"MEMORY_CONDENSE_DATA_DIR": str(root / "explicit"),
               "CLAUDE_PROJECT_DIR": str(root / "ignored")}
        assert project.project_data_dir(env, deep) == root / "explicit"

    def test_client_supplied_project_dir_beats_inference(self, repo):
        root, deep = repo
        env = {"CLAUDE_PROJECT_DIR": str(root / "elsewhere")}
        assert project.project_data_dir(env, deep) == root / "elsewhere" / ".memory_condense"

    def test_a_blank_env_var_does_not_count_as_set(self, repo):
        root, deep = repo
        for key in ("MEMORY_CONDENSE_DATA_DIR", "CLAUDE_PROJECT_DIR"):
            assert project.project_data_dir({key: "   "}, deep) == root / ".memory_condense"

    def test_no_project_falls_back_to_the_session_home(self, tmp_path):
        assert project.project_data_dir({}, tmp_path) == user_directory() / "data"


class TestProjectRootWalk:
    def test_a_subdirectory_shares_the_project_store(self, repo):
        root, deep = repo
        assert project.project_data_dir({}, deep) == project.project_data_dir({}, root)

    def test_pyproject_is_a_marker_for_a_non_git_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        assert project.project_root(deep, {}) == tmp_path

    def test_git_anywhere_above_beats_a_nearer_pyproject(self, tmp_path):
        """A monorepo subpackage shares its repository's memory."""
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "packages" / "leaf"
        sub.mkdir(parents=True)
        (sub / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        assert project.project_root(sub, {}) == tmp_path

    def test_a_worktree_git_file_is_its_own_project(self, tmp_path):
        """`.git` is a file in a worktree, so each worktree keeps its own store."""
        (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
        assert project.project_root(tmp_path, {}) == tmp_path

    def test_unreachable_working_directory_is_not_fatal(self, monkeypatch):
        """A deleted cwd must degrade, not raise — every surface calls this.

        Windows refuses to remove the current directory, so the condition is
        provoked at its source rather than by deleting a real one.
        """
        def boom():
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(project.Path, "cwd", staticmethod(boom))
        assert project.project_root(None, {}) is None
        assert project.project_data_dir({}, None) == user_directory() / "data"


class TestLegacyLocalStore:
    def test_an_existing_local_store_is_adopted(self, tmp_path):
        (tmp_path / ".memory_condense").mkdir()
        assert project.project_data_dir({}, tmp_path) == tmp_path / ".memory_condense"

    def test_adoption_never_creates_the_directory(self, tmp_path):
        assert project.project_data_dir({}, tmp_path) == user_directory() / "data"
        assert not (tmp_path / ".memory_condense").exists()

    def test_a_project_root_still_wins_over_a_local_store(self, tmp_path):
        """Inside a project, the root's store is the project's store."""
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "sub"
        deep.mkdir()
        (deep / ".memory_condense").mkdir()
        assert project.project_data_dir({}, deep) == tmp_path / ".memory_condense"


class TestEverySurfaceAgrees:
    def test_proxy_and_mcp_server_resolve_the_same_directory(self, repo, monkeypatch):
        root, deep = repo
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.chdir(deep)
        assert Path(proxy_server.default_data_dir()) == mcp_server._data_dir() == root / ".memory_condense"

    def test_the_status_resolver_searches_where_the_proxy_writes(self, repo, monkeypatch):
        root, deep = repo
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.chdir(deep)
        written = control_status.status_path_for(proxy_server.default_data_dir())
        assert written in control_status.candidate_status_files({}, deep)

    def test_a_statusline_in_a_subdirectory_finds_the_projects_document(self, repo):
        root, deep = repo
        written = control_status.status_path_for(root / ".memory_condense")
        control_status.write_status(written, control_status.empty_status())
        assert control_status.locate_status_file({}, deep) == written

    def test_legacy_memory_data_dir_still_steers_the_proxy(self, repo, monkeypatch):
        root, _ = repo
        monkeypatch.delenv("MEMORY_CONDENSE_DATA_DIR", raising=False)
        monkeypatch.setenv("MEMORY_DATA_DIR", str(root / "legacy"))
        assert Path(proxy_server.default_data_dir()) == root / "legacy"

    def test_the_canonical_name_wins_over_the_legacy_one(self, repo, monkeypatch):
        root, _ = repo
        monkeypatch.setenv("MEMORY_CONDENSE_DATA_DIR", str(root / "canonical"))
        monkeypatch.setenv("MEMORY_DATA_DIR", str(root / "legacy"))
        assert Path(proxy_server.default_data_dir()) == root / "canonical"


class TestStatuslineImportCost:
    """`control/status.py` is imported on every statusline refresh.

    `runtime.config` pulls `platform` (and `_wmi` on Windows) for a measured
    ~23 ms, which is why its import is deferred to the last candidate. Sharing
    the resolver must not quietly undo that, so this is asserted out of
    process — in-process `sys.modules` proves nothing once another test has
    already imported the heavy module.
    """

    def _modules_after(self, code: str) -> set[str]:
        script = (
            "import sys\n"
            f"{code}\n"
            "print('@@'.join(m for m in sys.modules "
            "if m == 'platform' or m.endswith('runtime.config')))\n"
        )
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        return {m for m in out.stdout.strip().split("@@") if m}

    def test_importing_the_status_module_stays_stdlib_only(self):
        assert self._modules_after("import memory_condense.control.status") == set()

    def test_resolving_a_found_document_never_imports_runtime_config(self, repo):
        root, deep = repo
        written = control_status.status_path_for(root / ".memory_condense")
        control_status.write_status(written, control_status.empty_status())
        heavy = self._modules_after(
            "from pathlib import Path\n"
            "from memory_condense.control.status import locate_status_file\n"
            f"assert locate_status_file({{}}, Path(r'{deep}')) is not None\n"
        )
        assert heavy == set(), f"statusline hot path pulled in {heavy}"


class TestDeploymentContractIsNotTheDevelopmentOne:
    """The released command must keep its absolute store, not infer a project.

    `cli.py` injects `--data-dir user_directory()/'data'` for the installed
    `memory-condense proxy`, and the Docker image relies on it: it sets
    `XDG_DATA_HOME=/data` so `user_directory()` resolves onto the mounted
    volume "with no argument juggling", and `packaging/RELEASE_NOTES.md` tells
    upgraders to restart with the same absolute `--data-dir` and back it up.
    The project-root resolver must never reach that path — a deployed proxy
    that followed the client's cwd into a user's repository would scatter the
    store it is supposed to keep on the volume.
    """

    def _installed_argv(self, monkeypatch, extra=()):
        import memory_condense.cli as cli
        from memory_condense.interfaces import proxy_server as ps

        captured = {}
        monkeypatch.setattr(ps, "main", lambda argv: captured.setdefault("argv", argv) and 0)
        cli.main(["proxy", *extra])
        return captured["argv"]

    def test_the_installed_proxy_passes_an_explicit_absolute_data_dir(self, monkeypatch, repo):
        _, deep = repo
        monkeypatch.chdir(deep)  # inside a project: must be ignored
        argv = self._installed_argv(monkeypatch)
        assert "--data-dir" in argv
        passed = Path(argv[argv.index("--data-dir") + 1])
        assert passed == user_directory() / "data"
        assert passed.is_absolute()
        assert ".memory_condense" not in passed.parts

    def test_the_deployment_store_follows_the_data_volume(self, monkeypatch, tmp_path):
        """The container mounts the store by environment, so this must stay env-driven.

        `user_directory()` reads the environment on every call -- that is the
        whole mechanism behind `XDG_DATA_HOME=/data` in the image. Asserted by
        changing the variable for this platform's branch, never by reloading
        the module: a reload swaps the function object while other modules keep
        the one they already imported.
        """
        key = "LOCALAPPDATA" if os.name == "nt" else "XDG_DATA_HOME"
        monkeypatch.setenv(key, str(tmp_path / "volume"))
        assert user_directory() == tmp_path / "volume" / "memory_condense"
        assert user_directory() / "data" == tmp_path / "volume" / "memory_condense" / "data"

    def test_a_user_supplied_data_dir_still_overrides_the_injected_default(self, monkeypatch, tmp_path):
        argv = self._installed_argv(monkeypatch, extra=["--data-dir", str(tmp_path / "chosen")])
        assert argv[-2:] == ["--data-dir", str(tmp_path / "chosen")]


class TestOneProjectOneStore:
    """Inside a project the answer is the root, never the directory contents.

    A store left in a subdirectory by the old `Path.cwd()` rule is NOT found
    from the root. That is a deliberate migration, not an oversight: adopting
    it would make which store a project uses depend on what some subdirectory
    happens to contain, so neither a user nor this code could answer the
    question without going and looking. Called out in the release notes
    instead. Adoption stays only for the no-project case, where there is no
    root to prefer and nothing to be ambiguous against.
    """

    def _project(self, tmp_path, *, sub_store, root_store):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        if sub_store:
            (sub / ".memory_condense").mkdir()
        if root_store:
            (tmp_path / ".memory_condense").mkdir()
        return sub

    def test_a_store_in_a_subdirectory_does_not_hijack_the_project(self, tmp_path):
        sub = self._project(tmp_path, sub_store=True, root_store=False)
        assert project.project_data_dir({}, sub) == tmp_path / ".memory_condense"

    def test_the_project_store_is_used_when_both_exist(self, tmp_path):
        sub = self._project(tmp_path, sub_store=True, root_store=True)
        assert project.project_data_dir({}, sub) == tmp_path / ".memory_condense"

    def test_a_fresh_project_uses_its_root(self, tmp_path):
        sub = self._project(tmp_path, sub_store=False, root_store=False)
        assert project.project_data_dir({}, sub) == tmp_path / ".memory_condense"
        assert not (sub / ".memory_condense").exists()
