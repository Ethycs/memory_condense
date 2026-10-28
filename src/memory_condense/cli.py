"""Installed command for provisioning and running the memory proxy."""
import argparse
import sys

# Client presets for `proxy --print-env`: which upstream each client expects
# the proxy to forward to. Carried over from the retired context_proxy CLI.
# `codex-api` is written without the `/v1` segment so it compares equal to the
# proxy's own OpenAI default (the request path already carries `/v1`).
PRESETS = {
    'claude': 'https://api.anthropic.com',
    'codex-chatgpt': 'https://chatgpt.com/backend-api/codex',
    'codex-api': 'https://api.openai.com',
}
DEFAULT_PORTS = {'claude': 8787, 'codex-chatgpt': 8788, 'codex-api': 8788}
# Which of the proxy's two upstream slots a preset's URL would occupy. The URLs
# themselves are read from the proxy rather than restated here, so `--print-env`
# cannot drift away from DEFAULT_UPSTREAMS.
_PRESET_PROVIDERS = {'claude': 'anthropic', 'codex-chatgpt': 'openai', 'codex-api': 'openai'}


def _proxy_default_upstream(preset):
    """The upstream the running proxy already defaults to for ``preset``.

    Imported inside the function on purpose: ``proxy_server`` drags in starlette
    and httpx, and this module stays import-light so `--help`, `route` and `ctl`
    start fast. `--print-env` is a terminal-only path where the cost is invisible.
    """
    from memory_condense.interfaces.proxy_server import DEFAULT_UPSTREAMS

    return DEFAULT_UPSTREAMS[_PRESET_PROVIDERS[preset]]


def _same_upstream(a, b):
    """Whether two base URLs make the proxy forward to the same place.

    ``proxy_server.upstream_url`` drops a leading ``/v1`` from the request path
    when the base already ends in ``/v1``, so the two spellings are equivalent
    there and must be equivalent here too — otherwise `--print-env` emits a
    redundant (or, if the defaults move, a wrong) ``--openai-base-url``.
    """
    def normalise(url):
        trimmed = url.rstrip('/')
        return trimmed[:-3].rstrip('/') if trimmed.endswith('/v1') else trimmed

    return normalise(a) == normalise(b)


def proxy_command(preset, port):
    """The `memory-condense proxy` invocation that serves ``preset`` on ``port``."""
    command = f'memory-condense proxy --port {port}'
    if not _same_upstream(PRESETS[preset], _proxy_default_upstream(preset)):
        # The proxy's override flags are named after the provider slot.
        command += f' --{_PRESET_PROVIDERS[preset]}-base-url {PRESETS[preset]}'
    return command


def client_instructions(preset, port):
    url = f'http://127.0.0.1:{port}'
    if preset == 'claude':
        return (
            f'Claude Code:  set ANTHROPIC_BASE_URL={url}\n'
            f'  Start the proxy with: {proxy_command(preset, port)}\n'
            '  Leave ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN unset to keep your claude.ai login;\n'
            '  the OAuth capability rides in anthropic-beta, which this proxy forwards verbatim.\n'
            '  Optional: CLAUDE_CODE_ATTRIBUTION_HEADER=0 (the system array is forwarded untouched, so not required).'
        )
    auth = 'true' if preset == 'codex-chatgpt' else 'false'
    return (
        f'Codex: start the proxy with: {proxy_command(preset, port)}\n'
        '  Then add to ~/.codex/config.toml (model_provider is a top-level key, above any table)\n'
        '  model_provider = "local_proxy"\n'
        '\n'
        '  [model_providers.local_proxy]\n'
        '  name = "local context proxy"\n'
        f'  base_url = "{url}/v1"\n'
        '  wire_api = "responses"\n'
        f'  requires_openai_auth = {auth}\n'
        + ('  env_key = "OPENAI_API_KEY"\n' if preset == 'codex-api' else '')
        + '  # Note: with OPENAI_API_KEY exported Codex ignores custom providers under ChatGPT login.'
    )


def print_env(args):
    """`proxy --print-env`: print client configuration without starting the server."""
    parser = argparse.ArgumentParser(prog='memory-condense proxy', add_help=False)
    parser.add_argument('--print-env', action='store_true')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='claude')
    parser.add_argument('--port', type=int)
    parsed, _ = parser.parse_known_args(args)
    print(client_instructions(parsed.preset, parsed.port or DEFAULT_PORTS[parsed.preset]))
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'route':
        from memory_condense.interfaces.tactic_cli import main as route
        return route(args[1:])
    if args and args[0] == 'proxy' and '--print-env' in args:
        return print_env(args[1:])
    if args and args[0] == 'proxy':
        from memory_condense.interfaces.proxy_server import main as proxy
        from memory_condense.runtime.config import user_directory
        # Installed command enables memory; the legacy module CLI stays observe.
        return proxy(['--mode','augment','--data-dir',str(user_directory()/'data'),*args[1:]])
    if args and args[0] == 'ctl':
        from memory_condense.control.cli import main as ctl
        return ctl(args[1:])
    from memory_condense.control.cli import HelpfulParser
    parser = HelpfulParser(prog='memory-condense', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('proxy', help='Run the memory-enabled provider proxy (--print-env [--preset claude|codex-chatgpt|codex-api] [--port N] prints client setup)')
    commands.add_parser('route', help='Prioritize memory tactics with the existing CPU BGE model')
    commands.add_parser('ctl', help='Control plane: status | save | load | reset | load-transcript | handoff')
    setup_parser = commands.add_parser('setup', help='Download pinned model assets or register an existing cache')
    location = setup_parser.add_mutually_exclusive_group()
    location.add_argument('--assets-dir', help='Download destination (default: per-user application data)')
    location.add_argument('--reuse', help='Verify and use an existing asset root without copying or downloading')
    doctor_parser = commands.add_parser('doctor', help='Check local assets, dependencies, and CUDA')
    doctor_parser.add_argument('--assets-dir')
    doctor_parser.add_argument('--verify-models', action='store_true', help='Check every pinned model/runtime file hash')
    if not args:
        parser.print_help()
        return 0
    parsed = parser.parse_args(args)
    try:
        from memory_condense.runtime import setup as provision
        if parsed.command == 'setup':
            return provision.setup(assets_dir=parsed.assets_dir, reuse=parsed.reuse)
        return provision.doctor(assets_dir=parsed.assets_dir, verify=parsed.verify_models)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
