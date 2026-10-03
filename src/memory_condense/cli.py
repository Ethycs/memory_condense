"""Installed command for provisioning and running the memory proxy."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'proxy':
        from memory_condense.interfaces.proxy_server import main as proxy
        from memory_condense.runtime.config import user_directory
        # Installed command enables memory; the legacy module CLI stays observe.
        return proxy(['--mode','augment','--data-dir',str(user_directory()/'data'),*args[1:]])
    parser = argparse.ArgumentParser(prog='memory-condense', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('proxy', help='Run the memory-enabled provider proxy')
    setup_parser = commands.add_parser('setup', help='Download pinned model assets or register an existing cache')
    location = setup_parser.add_mutually_exclusive_group()
    location.add_argument('--assets-dir', help='Download destination (default: per-user application data)')
    location.add_argument('--reuse', help='Verify and use an existing asset root without copying or downloading')
    doctor_parser = commands.add_parser('doctor', help='Check local assets, dependencies, and CUDA')
    doctor_parser.add_argument('--assets-dir')
    doctor_parser.add_argument('--verify-models', action='store_true', help='Check every pinned model/runtime file hash')
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
