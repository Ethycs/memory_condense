"""CPU-only tactic classification, with a persistent JSON-lines interface."""
import argparse
from dataclasses import asdict
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='memory-condense route',description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--query',help='Classify one question')
    mode.add_argument('--jsonl',action='store_true',help='Keep one encoder resident; read JSON objects with query/recent from stdin')
    parser.add_argument('--recent',action='append',default=[],help='Recent message or summary, oldest first; repeat as needed')
    parser.add_argument('--assets-dir',help='Existing assets root; no model downloads are performed')
    parser.add_argument('--classifier',help='Explicit sealed classifier artifact instead of the packaged head')
    args = parser.parse_args(argv)
    if args.jsonl and args.recent:
        parser.error('--recent belongs with --query; JSONL requests carry their own recent array')
    # Read the runtime's shared one-model adapter; Qwen, Llama and API clients
    # are never constructed by this command.
    from memory_condense.application.tactic_router import BGETacticRouter
    from memory_condense.runtime.config import RuntimeAssets
    from memory_condense.runtime.embedding import FastEmbedBGE
    encoder = None
    try:
        encoder = FastEmbedBGE(assets=RuntimeAssets.resolve(args.assets_dir))
        router = BGETacticRouter(encoder,args.classifier)
        if not args.jsonl:
            print(json.dumps(asdict(router.prioritize(args.query,recent=args.recent)),ensure_ascii=False),flush=True)
            return 0
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if (not isinstance(value,dict) or set(value)-{'query','recent'}
                        or not isinstance(value.get('recent',[]),list)):
                    raise ValueError('Expected query and optional recent text array')
                result = asdict(router.prioritize(value.get('query'),recent=value.get('recent',[])))
            except (ValueError,TypeError,KeyError) as exc:
                result = {'error':str(exc)}
            print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(1,str(exc)+'\n')
    finally:
        if encoder is not None:
            encoder.close()


if __name__ == '__main__':
    raise SystemExit(main())
