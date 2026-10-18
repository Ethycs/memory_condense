"""`memory-condense proxy --print-env` and the installed command's dispatch.

``--print-env`` is answered before the ``proxy`` early-dispatch, so the server
is never *started*; it does read ``DEFAULT_UPSTREAMS`` out of the server module
rather than restating it, which is why these tests let that import happen.
``TestImportLightDispatch`` pins the property that actually matters: the cheap
paths (`--help`, `route`, `ctl`) still pull in no starlette. The other
subcommands are checked only for routing, with their targets replaced by
recording stubs.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from memory_condense import cli

PROXY_SERVER = "memory_condense.interfaces.proxy_server"


# ---------------------------------------------------------------------------
# --print-env
# ---------------------------------------------------------------------------


class TestPrintEnv:
    def test_default_preset_is_claude(self, capsys):
        assert cli.main(["proxy", "--print-env"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("Claude Code:  set ANTHROPIC_BASE_URL=http://127.0.0.1:8787\n")
        assert "memory-condense proxy --port 8787" in out
        assert "--openai-base-url" not in out
        assert "ANTHROPIC_AUTH_TOKEN" in out

    def test_claude_custom_port(self, capsys):
        assert cli.main(["proxy", "--print-env", "--preset", "claude", "--port", "9999"]) == 0
        out = capsys.readouterr().out
        assert "ANTHROPIC_BASE_URL=http://127.0.0.1:9999" in out
        assert "memory-condense proxy --port 9999" in out

    def test_codex_chatgpt_names_upstream_flag(self, capsys):
        assert cli.main(["proxy", "--print-env", "--preset", "codex-chatgpt"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("Codex: start the proxy with: memory-condense proxy --port 8788 "
                              "--openai-base-url https://chatgpt.com/backend-api/codex\n")
        assert 'base_url = "http://127.0.0.1:8788/v1"' in out
        assert "requires_openai_auth = true" in out
        assert "env_key" not in out

    def test_codex_api_uses_builtin_upstream(self, capsys):
        assert cli.main(["proxy", "--print-env", "--preset", "codex-api"]) == 0
        out = capsys.readouterr().out
        assert "memory-condense proxy --port 8788\n" in out
        assert "--openai-base-url" not in out
        assert 'base_url = "http://127.0.0.1:8788/v1"' in out
        assert "requires_openai_auth = false" in out
        assert 'env_key = "OPENAI_API_KEY"' in out

    @pytest.mark.parametrize("preset", ["codex-chatgpt", "codex-api"])
    def test_codex_model_provider_precedes_table(self, capsys, preset):
        assert cli.main(["proxy", "--print-env", "--preset", preset]) == 0
        lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
        key = lines.index('model_provider = "local_proxy"')
        table = lines.index("[model_providers.local_proxy]")
        assert key < table
        # Nothing between the top-level key and the table but whitespace.
        assert all(not line for line in lines[key + 1:table])
        assert lines[table + 1:].count("[model_providers.local_proxy]") == 0

    def test_codex_snippet_is_valid_toml(self, capsys):
        tomllib = pytest.importorskip("tomllib")
        assert cli.main(["proxy", "--print-env", "--preset", "codex-chatgpt"]) == 0
        lines = capsys.readouterr().out.splitlines()
        snippet = "\n".join(line.strip() for line in lines[2:])
        doc = tomllib.loads(snippet)
        assert doc["model_provider"] == "local_proxy"
        assert doc["model_providers"]["local_proxy"]["base_url"] == "http://127.0.0.1:8788/v1"

    def test_print_env_anywhere_in_args(self, capsys):
        assert cli.main(["proxy", "--port", "8788", "--preset", "codex-api", "--print-env"]) == 0
        assert "requires_openai_auth = false" in capsys.readouterr().out

    def test_unknown_preset_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["proxy", "--print-env", "--preset", "gemini"])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_default_port_table(self):
        assert cli.DEFAULT_PORTS == {"claude": 8787, "codex-chatgpt": 8788, "codex-api": 8788}
        assert set(cli.PRESETS) == set(cli.DEFAULT_PORTS)


# ---------------------------------------------------------------------------
# Dispatch of the other subcommands is unchanged
# ---------------------------------------------------------------------------


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    # `from pkg import mod` prefers the attribute on an already-imported package.
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        monkeypatch.setattr(parent, child, module, raising=False)
    return module


class TestDispatch:
    def test_proxy_without_print_env_reaches_server_with_prefix(self, monkeypatch, tmp_path):
        seen = []
        _stub_module(monkeypatch, PROXY_SERVER, main=lambda argv: seen.append(list(argv)) or 7)
        from memory_condense.runtime import config

        monkeypatch.setattr(config, "user_directory", lambda: tmp_path)
        assert cli.main(["proxy", "--port", "8788", "--openai-base-url", "https://x"]) == 7
        assert seen == [["--mode", "augment", "--data-dir", str(tmp_path / "data"),
                         "--port", "8788", "--openai-base-url", "https://x"]]

    def test_route_forwards_rest(self, monkeypatch):
        seen = []
        _stub_module(monkeypatch, "memory_condense.interfaces.tactic_cli",
                     main=lambda argv: seen.append(list(argv)) or 0)
        assert cli.main(["route", "--top", "3"]) == 0
        assert seen == [["--top", "3"]]

    def test_ctl_forwards_rest(self, monkeypatch):
        seen = []
        _stub_module(monkeypatch, "memory_condense.control.cli",
                     main=lambda argv: seen.append(list(argv)) or 0)
        assert cli.main(["ctl", "status", "--json"]) == 0
        assert seen == [["status", "--json"]]

    def test_setup_and_doctor(self, monkeypatch):
        calls = []
        _stub_module(monkeypatch, "memory_condense.runtime.setup",
                     setup=lambda **kw: calls.append(("setup", kw)) or 0,
                     doctor=lambda **kw: calls.append(("doctor", kw)) or 0)
        assert cli.main(["setup", "--reuse", "/cache"]) == 0
        assert cli.main(["doctor", "--verify-models"]) == 0
        assert calls == [("setup", {"assets_dir": None, "reuse": "/cache"}),
                         ("doctor", {"assets_dir": None, "verify": True})]

    def test_no_command_is_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# No hand-copied upstreams, and no starlette on the cheap paths
# ---------------------------------------------------------------------------


class TestUpstreamsComeFromTheProxy:
    """`--print-env` must not restate `DEFAULT_UPSTREAMS` from memory."""

    def test_presets_agree_with_the_running_proxy(self):
        from memory_condense.interfaces.proxy_server import DEFAULT_UPSTREAMS

        for preset, provider in cli._PRESET_PROVIDERS.items():
            assert cli._proxy_default_upstream(preset) == DEFAULT_UPSTREAMS[provider]
        # The two presets that ride the proxy's defaults must compare equal to
        # them, so no redundant override flag is printed.
        for preset in ("claude", "codex-api"):
            assert cli._same_upstream(cli.PRESETS[preset], cli._proxy_default_upstream(preset))
        assert not cli._same_upstream(
            cli.PRESETS["codex-chatgpt"], cli._proxy_default_upstream("codex-chatgpt")
        )

    def test_v1_spellings_are_equivalent(self):
        # `proxy_server.upstream_url` strips a leading /v1 from the path when the
        # base ends in /v1, so the comparison has to treat them as one place.
        assert cli._same_upstream("https://api.openai.com", "https://api.openai.com/v1")
        assert cli._same_upstream("https://api.openai.com/v1/", "https://api.openai.com/")
        assert not cli._same_upstream("https://api.openai.com", "https://chatgpt.com/backend-api/codex")

    def test_moved_default_changes_the_emitted_command(self, monkeypatch):
        from memory_condense.interfaces import proxy_server

        monkeypatch.setitem(proxy_server.DEFAULT_UPSTREAMS, "openai", "https://elsewhere.test")
        assert "--openai-base-url https://api.openai.com" in cli.proxy_command("codex-api", 8788)

    def test_a_moved_anthropic_default_names_the_anthropic_flag(self, monkeypatch):
        """The old code would have emitted `--openai-base-url` for `claude`."""
        from memory_condense.interfaces import proxy_server

        monkeypatch.setitem(proxy_server.DEFAULT_UPSTREAMS, "anthropic", "https://elsewhere.test")
        assert cli.proxy_command("claude", 8787) == (
            "memory-condense proxy --port 8787 --anthropic-base-url https://api.anthropic.com"
        )


class TestImportLightDispatch:
    """The cheap paths must not drag in starlette/httpx.

    Run out of process: the `--print-env` tests above import the proxy server
    on purpose, so this interpreter's `sys.modules` proves nothing.
    """

    @staticmethod
    def _heavy_modules_after(*argv):
        code = (
            "import sys\n"
            "from memory_condense import cli\n"
            f"try:\n    cli.main({list(argv)!r})\nexcept SystemExit:\n    pass\n"
            "heavy = sorted(m for m in sys.modules\n"
            "               if m.split('.')[0] in {'starlette', 'httpx'}\n"
            "               or m.endswith('proxy_server'))\n"
            # Sentinel-prefixed: `--help` writes its own usage text to stdout.
            "print('@@HEAVY@@', ' '.join(heavy))\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        marker = [ln for ln in done.stdout.splitlines() if ln.startswith("@@HEAVY@@")]
        assert len(marker) == 1, done.stdout
        return marker[0].split()[1:]

    def test_help_stays_light(self):
        assert self._heavy_modules_after("--help") == []

    def test_route_dispatch_stays_light(self):
        assert self._heavy_modules_after("route", "--help") == []

    def test_ctl_dispatch_stays_light(self):
        assert self._heavy_modules_after("ctl", "--help") == []

    def test_cli_module_import_alone_stays_light(self):
        assert self._heavy_modules_after() == []
