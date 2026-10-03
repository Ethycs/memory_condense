"""One frontier answer+summary call; validate capture and reuse without loading GPUs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.inline_memory import generate_inline
from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_gateway_reader import GatewayReaderRuntime
from tools.engineering_research_memory import Compiler
from tools.run_hot_reduced30_answer_judge import _completion_client


def finalize(root):
    exchange = read(root/'exchange.json')
    event, output = (ChatEvent(**exchange[key]) for key in ('input', 'output'))
    metadata, response = output.metadata, output.metadata['response']
    if 'inline_memory' not in metadata:
        report = dict(accepted=False, fallback=metadata['inline_generation']['error'],
            provider_calls=1, usage=response.get('usage'), elapsed_s=response['elapsed_s'])
        save(root/'report.json', report)
        raise ValueError('Provider returned an answer but its inline memory required fallback')
    class NoGeneration:
        def call(self, *args, **kwargs):
            raise AssertionError('Accepted inline summaries must not need another raw summary call')
    compiler = Compiler(root, 'inline-smoke')
    compiler.gateway = NoGeneration()
    atoms = compiler.atoms([e.row('inline-smoke') for e in (event, output)],
                           'inline-smoke', '2026-09-30T00:00:00+00:00')
    save(root/'compiled-atoms.json', dict(atoms=[asdict(atom) for atom in atoms]))
    report = dict(accepted=True, provider_calls=1, extra_raw_summary_calls=0, atoms=len(atoms),
        roles=[atom.spans[0].role for atom in atoms], model=response.get('response_model'),
        elapsed_s=response['elapsed_s'], usage=response.get('usage'),
        user_summary=metadata['inline_memory']['summaries']['user']['summary'],
        assistant_summary=metadata['inline_memory']['summaries']['assistant']['summary'],
        answer=response['content'],
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='One synthetic protocol check; buffered delivery; no accuracy or speed comparison')
    save(root/'report.json', report)
    emit(**report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gateway', default='https://central-dev.zt:4000/v1')
    parser.add_argument('--model', default='codex_sdk/gpt-5.6-sol')
    parser.add_argument('--report-only', action='store_true', help='Finalize saved output without provider calls')
    args = parser.parse_args()
    if args.report_only:
        finalize(args.output)
        return
    if args.output.exists():
        raise ValueError('Use a new output directory; never repeat an ambiguous generation')
    user = ('Plan a reversible deployment of migration-482 to cobalt-731. Do not execute it. '
            'Identify checks still needed. Answer in at most six short bullets.')
    messages = [dict(role='system', content='Assist with the engineering task using the supplied evidence.'),
                dict(role='user', content='Evidence: staging only; rollback tag orchid-927. '
                     'No deployment or tests have been run.\n\nCurrent request:\n' + user)]
    files = ['src/memory_condense/application/inline_memory.py',
             'src/memory_condense/domain/inline_memory.py',
             'src/memory_condense/application/chat_io.py',
             'tools/engineering_research_memory.py', 'tools/engineering_research_gateway_reader.py',
             'tools/probe_inline_memory.py']
    save(args.output/'plan.json', dict(model=args.model, gateway=args.gateway,
        provider_calls=1, max_tokens=1536, input=user, messages=messages,
        purpose='Synthetic engineering protocol smoke check, not an accuracy or latency benchmark',
        implementation={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}))
    runtime = GatewayReaderRuntime(args.output/'runtime', gateway=args.gateway,
                                   reader_model=args.model, judge_model=args.model)
    runtime.remote = _completion_client('LITELLM_KEY', args.gateway).with_options(timeout=120, max_retries=0)
    try:
        response = generate_inline(runtime.call, messages, user_text=user, scope='inline-smoke', max_tokens=1536)
    finally:
        runtime.remote.close()
        runtime.remote = None
    event = ChatEvent('smoke-user', 'user', user)
    metadata = response.capture(event, 'smoke-answer', [])
    output = ChatEvent('smoke-answer', 'assistant', response.response['content'], metadata=dict(
        io=dict(input_event_id=event.event_id, packet_id=None, packet_ids=[]),
        response=response.response, **metadata))
    save(args.output/'exchange.json', dict(input=asdict(event), output=asdict(output)))
    finalize(args.output)


if __name__ == '__main__':
    main()
