# Codex transport and Windows startup verification

## Scope and baseline

Baseline: upstream `541300b8a2a7acd0df95d0d981c06f672a2bafec` (Responses memory
support). At preparation, the fetched default branch still matched that commit.
No CONTRIBUTING, AGENTS or PR template was present in that upstream tree.
The existing package/installer workflow is the release verification authority.

This change adds Codex client compatibility around the existing memory engine,
an optional windowless Windows login task, and explicit installed-asset reuse
for optional MCP. Feature relationships and configuration are documented in
[the client/startup guide](../02%20-%20Implementation/32%20-%20Codex%20Client%20and%20Windows%20Startup.md).

The unpublished custom-adapter migration utility, old startup scripts, personal
Codex configuration, raw conversations, request captures, logs, model assets,
environments and generated package artifacts are excluded. No memory-format
migration or automatic client-configuration rewrite is introduced.

## Verification

Environment: Windows 11 x64, Python 3.12, NVIDIA RTX 4090. A separate test
environment and application installation were used. The existing working
installation, Codex configuration, task and conversation store were not updated.

| Check | Observed result and limits |
| --- | --- |
| Upstream CI-selected tests plus Codex/startup/MCP regressions | Final full run: **524 passed, 1 existing POSIX-executable-bit skip**, 53.26 seconds, `eval_results/pr-final-ci-tests-short.xml` |
| Focused final startup run | **16 passed**, including configuration rollback and setup gating, after adding the bounded wait for asynchronous task stop |
| Python wheel and source distribution | `python -m build --outdir dist/python` succeeded |
| Shared noarch conda package | `pixi build --output-dir dist` succeeded using the upstream build workflow |
| Both release bundle builders/verifiers | Windows zip and Linux tarball built; member digests verified. Linux archive verification on Windows is not Linux runtime execution |
| Independent Windows installation | Release installer completed in a separate environment; installed-artifact smoke imported packaged modules without checkout/research imports |
| Native doctor | `doctor --verify-models` verified 69 asset files, CUDA, PyTorch, FastEmbed, ONNX Runtime and nvCOMP |
| Real Windows task | Registered under current-user InteractiveToken/Limited, without elevation on this host; action was the installed `pythonw.exe`, with a user-specific logon trigger and no shell |
| Actual windowless process | Listener owned by `pythonw`; MainWindowHandle was zero. Health returned HTTP 200 in observe mode and after reconfiguration in augment mode |
| Task lifecycle | Status, stop, reconfiguration, restart and removal exercised on a disposable task/port; the working task/listener remained unchanged |
| Complete installer startup option | `install-proxy.ps1 -StartAtLogin -AssetsDir <verified cache> -ProxyArguments <Codex options>` completed in an independent directory containing spaces. Setup configuration was redirected to that installation; existing client configuration was not rewritten |
| Lint/syntax | Ruff `--select F` on changed Python files, PowerShell parser checks on both scripts, and `git diff --check` passed |
| Privacy | Only source/tests/general documentation are proposed; private state and raw verification artifacts remain outside the change |

The first full run under Windows' default long TEMP path had two
`test_summary_api_fallback` failures at nested receipt creation, with 520 passed
and 1 skipped. The paths exceeded MAX_PATH. Rerunning with a shorter owned test
root passed. A checkout-relative `.pytest-ci` was still too long on this host,
so CI uses `--basetemp="${{ runner.temp }}/memory-tests"` outside the checkout.
No production storage/path
algorithm was changed to hide that failure. A preexisting Starlette/httpx test
client deprecation warning remains.

The working Codex desktop chat previously completed a real terminal call and
its continuation through these compatibility modules. That is separate live
evidence from the new startup test: no second GPU-resident memory engine or
real-provider conversation was started in the new installation. The new
startup task's health check alone does not prove a generation or recall result.
Full logout/login, a computer reboot, another Windows account/device, live
Anthropic credentials, Linux runtime and hosted CI are not locally verified.

All six changed compatibility runtime modules have identical SHA-256 hashes in
this source, the independent installed artifact and the previously verified
working installation. The added startup module is present in the new artifact.
Package metadata exposes both `mcp` and `proxy` extras. The built conda SHA-256 is
`86741622c84212d3ffbf7cd01c9fd2d266de480536bdb75322e940a2fe8c5b34`.

Both disposable tasks were removed after testing; their ports were released.
The original working service remained healthy in augment mode, with no pending
events or failed sessions at the final inspection. Personal path/account/chat
identifiers and credential patterns were checked across all proposed changed
files, with no matches. No raw receipts, model files or generated artifacts are
included in the PR. Hosted CI and release publication remain pending.

## Release requirements and limits

Build the package and both platform bundles through the existing workflow.
Windows must ship the matching `manage-startup.ps1` beside its installer, both
inside the archive and as a loose release asset. New login startup is opt-in;
upgrades must stop the previous process and retain the same absolute data path.
Provider/model tool support and client transport configuration remain required.

No consumer ChatGPT transport integration, automatic cross-conversation memory,
immediate streaming, general multimodal memory, or universal future-client
compatibility is claimed. Live provider and full reboot results must be stated
separately from deterministic contracts and simulated task boundaries.
