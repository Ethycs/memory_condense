"""Chronological memory for the build replay, using the existing spine components.

Initial fragment summaries use Sol. Qwen receives typed summary merge requests
and user-summary attention windows only. GPU phases run separately from model
gateway phases and from the five timed accuracy evaluations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from memory_condense.domain.schemas import Turn
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search import native_spine_summary as raw_summary
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges, UserSpineExchange
from memory_condense.search.episodes.parent_budgeted_spine_hierarchy import build_parent_budgeted_spine_hierarchy
from memory_condense.search.native_spine_merges import neutral_messages, neutral_key
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.spine_summary import parse_spine_summary
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools import native_spine_build_replay as replay
from tools import native_spine_five100 as evaluation
from tools.build_spine_corpus_hierarchy import ScalarAttentionCache
from tools.compile_native_spine_attention import cache_method


RAW_SYSTEM = raw_summary.SYSTEM + ' Prefer at most 32 words per summary and one short support quotation.'


def fragments_for(rows):
    body = {'turns': [{'role': r['role'], 'text': r['text']} for r in rows]}
    return raw_summary.fragment_body(body)


def fragment_key(fragment):
    return identity_sha256({'role': fragment.role, 'turn_text_sha256': fragment.turn_text_sha256,
        'start_char': fragment.start_char, 'end_char': fragment.end_char,
        'text_sha256': quote_sha256(fragment.text), 'system_sha256': quote_sha256(RAW_SYSTEM)})


def summarize_raw(root, step):
    plan, rows, folder = replay.state(root, step)
    fragments = fragments_for(rows)
    pending = [f for f in fragments if not (root / 'atomic-summaries' / f'{fragment_key(f)}.json').exists()]
    batches = raw_summary.pack_batches(pending, max_atoms=8, prompt_cap=6500)
    replay.publish(folder / 'raw-summary-preflight.json', {'history_sha256': identity_sha256(rows),
        'pending_fragments': len(pending), 'batch_count': len(batches), 'model': evaluation.MODEL,
        'raw_inputs_to_qwen': False, 'raw_summarizer': 'Sol', 'implementation_sha256': evaluation.digest(__file__)})

    def one(batch):
        messages = raw_summary.summary_messages(batch)
        messages[0]['content'] = RAW_SYSTEM
        key = identity_sha256(messages)
        prefix = root / 'raw-summary-journal' / key
        request = replay.publish(prefix.with_suffix('.request.json'), {'messages': messages,
            'model': evaluation.MODEL, 'fragment_keys': [fragment_key(f) for f in batch], 'max_tokens': 4096})
        if prefix.with_suffix('.response.json').exists():
            response = replay.load(prefix.with_suffix('.response.json'))
        else:
            with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                handle.write(request.sha256 + '\n')
            with closing(evaluation.authoring._completion_client('LITELLM_KEY', evaluation.current.frozen.GATEWAY)) as client:
                result = client.chat.completions.create(model=evaluation.MODEL, messages=messages,
                    max_tokens=4096, temperature=0, timeout=180)
            choice, = result.choices
            response = replay.publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                'content': choice.message.content, 'finish_reason': choice.finish_reason, 'response_model': result.model})
        if response.payload['request_sha256'] != request.sha256 or response.payload['finish_reason'] != 'stop':
            raise ValueError('raw summary generation did not finish correctly')
        values = raw_summary.parse_summaries(response.payload['content'], batch)
        for fragment, value in zip(batch, values, strict=True):
            replay.publish(root / 'atomic-summaries' / f'{fragment_key(fragment)}.json', {
                'key': fragment_key(fragment), 'summary': value['summary'], 'support': value['support'],
                'response': evaluation.binding(response), 'role': fragment.role,
                'span_text_sha256': quote_sha256(fragment.text), 'turn_text_sha256': fragment.turn_text_sha256})
        return len(batch)

    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, batch) for batch in batches]):
            completed += future.result()
            evaluation.emit(phase='build_raw_summaries', step=step, completed=completed, required=len(pending))


def atoms_for(root, rows):
    turns = [Turn(turn_id=r['turn_id'], source_id=replay.SOURCE, role=r['role'], text=r['text'],
                  created_at=datetime.fromisoformat(replay.DATE)) for r in rows]
    atoms = []
    for fragment in fragments_for(rows):
        cached = replay.load(root / 'atomic-summaries' / f'{fragment_key(fragment)}.json')
        if (cached.payload['key'] != fragment_key(fragment)
                or cached.payload['span_text_sha256'] != quote_sha256(fragment.text)
                or any(q not in fragment.text for q in cached.payload['support'])):
            raise ValueError('raw summary is not bound to this exact fragment')
        span = RawSectionSpan.from_turn(turns[fragment.turn_ordinal], start_char=fragment.start_char, end_char=fragment.end_char)
        atoms.append(SectionSummary('replay-atom-' + span.receipt_sha256, replay.SOURCE,
            cached.payload['summary'], (span,), cached.sha256))
    return tuple(atoms)


class JournaledQwen:
    def __init__(self, root):
        self.root = root

    def __call__(self, request):
        messages = neutral_messages(request)
        key = neutral_key(request)
        prefix = self.root / 'qwen-summary-journal' / key
        saved_request = replay.publish(prefix.with_suffix('.request.json'), {'messages': messages,
            'request': asdict(request), 'raw_inputs': False, 'model': 'qwen3-8b', 'max_tokens': 768})
        if prefix.with_suffix('.response.json').exists():
            response = replay.load(prefix.with_suffix('.response.json'))
        else:
            with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                handle.write(saved_request.sha256 + '\n')
            with closing(evaluation.authoring._completion_client('LITELLM_KEY', evaluation.current.frozen.GATEWAY)) as client:
                result = client.chat.completions.create(model='qwen3-8b', messages=messages,
                    max_tokens=768, temperature=0, extra_body={'enable_thinking': False}, timeout=180)
            choice, = result.choices
            response = replay.publish(prefix.with_suffix('.response.json'), {'request_sha256': saved_request.sha256,
                'content': choice.message.content, 'finish_reason': choice.finish_reason, 'response_model': result.model})
        if response.payload['request_sha256'] != saved_request.sha256 or response.payload['finish_reason'] != 'stop':
            raise ValueError('Qwen summary merge did not finish normally')
        return parse_spine_summary(response.payload['content'], request)


def exchanges(root, step):
    plan, rows, folder = replay.state(root, step)
    atoms = atoms_for(root, rows)
    compiled = compile_user_spine_exchanges(atoms, summarize=ReusingSpineSummarizer(JournaledQwen(root)),
        summarizer_identity='replay-qwen-summary-only-v1', max_channel_tokens=128, max_prompt_tokens=2048)
    result = replay.publish(folder / 'exchanges.json', {'history_sha256': identity_sha256(rows),
        'atoms': [a.identity_payload() for a in atoms], 'exchanges': [asdict(e) for e in compiled],
        'raw_inputs_to_qwen': False, 'input_kind': 'typed precompiled role-bound summaries only'})
    evaluation.emit(phase='build_exchanges', step=step, count=len(compiled), sha256=result.sha256)


def restore_exchanges(root, step):
    plan, rows, folder = replay.state(root, step)
    artifact = replay.load(folder / 'exchanges.json')
    if artifact.payload['history_sha256'] != identity_sha256(rows):
        raise ValueError('exchange hierarchy includes a different chronological prefix')
    compiled = tuple(UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e['section'])))
                     for e in artifact.payload['exchanges'])
    return rows, folder, artifact, compiled


def attention(root, step):
    rows, folder, artifact, compiled = restore_exchanges(root, step)
    method = cache_method(root / 'attention-cache')
    cache = ScalarAttentionCache(root / 'attention-cache', method)
    start, receipts = 0, []
    while start < len(compiled):
        end = min(start + 8, len(compiled))
        texts = tuple(e.user_spine or 'Unowned prelude.' for e in compiled[start:end])
        signal = cache.score_sequence(texts)
        receipts.append({'start': start, 'end': end, 'texts': list(texts), 'receipt_sha256': signal.receipt.receipt_sha256})
        if end == len(compiled):
            break
        start = end - 1
    replay.publish(folder / 'attention.json', {'exchanges': evaluation.binding(artifact),
        'method': evaluation.binding(method), 'windows': receipts, 'raw_inputs_to_qwen': False})
    evaluation.emit(phase='build_summary_attention', step=step, windows=len(receipts))


def hierarchy(root, step):
    rows, folder, artifact, compiled = restore_exchanges(root, step)
    prepared = replay.load(folder / 'attention.json')
    method = evaluation.bound(prepared.payload['method'])
    cache = ScalarAttentionCache(root / 'attention-cache', method)
    # Refuse any unprepared GPU work in this summary-generation phase.
    for window in prepared.payload['windows']:
        key = identity_sha256({'preflight_sha256': method.sha256, 'texts': window['texts']})
        replay.load(root / 'attention-cache' / 'attention' / f'{key}.json')
    built = build_parent_budgeted_spine_hierarchy(compiled, scorer=cache,
        summarize=ReusingSpineSummarizer(JournaledQwen(root)), summarizer_identity='replay-qwen-summary-only-v1',
        leaf_token_cap=512, max_leaf_exchanges=2, max_exchange_channel_tokens=128,
        max_parent_channel_tokens=512, window_exchange_cap=8, max_prompt_tokens=2048)
    index = built.summary_index()
    replay.publish(folder / 'hierarchy.json', {'exchanges': evaluation.binding(artifact),
        'history_sha256': identity_sha256(rows), 'index_json': index.to_json(),
        'attention': evaluation.binding(prepared), 'raw_inputs_to_qwen': False,
        'splits': [asdict(s) for s in built.splits], 'root_section_ids': list(built.root_section_ids)})
    evaluation.emit(phase='build_hierarchy', step=step, sections=len(index.sections))


def install(root, step):
    plan, rows, folder = replay.state(root, step)
    prepared = replay.load(folder / 'hierarchy.json')
    if prepared.payload['history_sha256'] != identity_sha256(rows):
        raise ValueError('cannot install future or different transcript content')
    hierarchy = SectionSummaryIndex.from_json(prepared.payload['index_json'])
    atoms = SectionSummaryIndex(atoms_for(root, rows))
    started = time.perf_counter()
    app_path = root / 'memory'
    with closing(evaluation.EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with evaluation.MemoryCondenser(app_path, embedder=encoder, auto_extract=False) as app:
            current = app.transcript.get_all()
            if len(current) > len(rows) or any((t.turn_id, t.role, t.text) !=
                    (r['turn_id'], r['role'], r['text']) for t, r in zip(current, rows)):
                raise ValueError('persisted replay history is not the exact past prefix')
            new = [(r['role'], r['text'], replay.SOURCE, datetime.fromisoformat(replay.DATE), r['turn_id'])
                   for r in rows[len(current):]]
            for start in range(0, len(new), 32):
                app.ingest_many(new[start:start + 32])
            matrix = np.asarray(encoder.embed_queries([s.summary for s in atoms.sections]), dtype=np.float32)
            identity = evaluation.summary_embedding_identity(encoder)
            snapshot = app.install_native_spine(atoms, hierarchy, matrix, embedding_identity=identity)
        parents = evaluation.project_parent_users(hierarchy)
        texts = [s.summary for s in parents.sections]
        parent_matrix = np.asarray(encoder.embed_queries(texts), dtype=np.float32)
        parent_matrix /= np.linalg.norm(parent_matrix, axis=1, keepdims=True)
        version = folder / 'parent-users.sqlite'
        receipt = evaluation.parent_store.publish(version, hierarchy=hierarchy, matrix=parent_matrix, native_receipt=snapshot)
        target = app_path / evaluation.parent_store.FILENAME
        # Each previous parent index is retained in its step folder. Copy the
        # new immutable version to a temporary local file, then replace active.
        temporary = app_path / 'next-parent-users.sqlite'
        temporary.resolve().relative_to(app_path.resolve())
        target.resolve().relative_to(app_path.resolve())
        shutil.copyfile(version, temporary)
        os.replace(temporary, target)
        with evaluation.ParentUserMemoryCondenser(app_path, embedder=encoder, auto_extract=False, read_only=True) as app:
            if app.native_spine_receipt() != snapshot or app.native_parent_user_receipt() != receipt:
                raise ValueError('replay memory failed close/reopen verification')
            order = evaluation.current.presentation.renderer.TranscriptOrder(app.transcript.get_all())
            prompt = plan.payload['prompts'][step]['text']
            question = {'retrieval_query': prompt, 'prompt_question': '[Question asked at 2026/08/16 (Sun) 23:59] ' + prompt}
            policy = evaluation.bound(evaluation.read_sealed_json(evaluation.PREVIOUS).payload['context_policy']).payload
            result = app.retrieve_native_spine(question['retrieval_query'], question['prompt_question'], **policy)
            rendered = evaluation.current.presentation.renderer.render_user_spine_sections(result.hydration, order)
            for section in result.hydration.sections:
                for evidence in section.evidence:
                    turn = app.transcript.get_turn(evidence.span.turn_id)
                    if turn.text[evidence.span.start_char:evidence.span.end_char] != evidence.text:
                        raise ValueError('build evidence differs from exact stored original text')
            replay.publish(folder / 'memory-context.json', {'history_sha256': identity_sha256(rows),
                'history_turns': len(rows), 'newly_ingested_turns': len(new), 'snapshot': snapshot,
                'parent_snapshot': receipt, 'text': rendered.text, 'rendered': rendered.identity_payload(),
                'hydration': result.hydration.identity_payload(), 'routing': result.routing.identity_payload(),
                'original_future_assistant_responses_used': False, 'setup_and_ingest_s': time.perf_counter() - started})
    evaluation.emit(phase='build_memory_ready', step=step, history_turns=len(rows),
                    newly_ingested_turns=len(new), evidence_tokens=count_tokens(rendered.text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('raw', 'exchanges', 'attention', 'hierarchy', 'install'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, choices=range(8))
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase in ('raw', 'exchanges', 'hierarchy') and not args.enable_provider:
        parser.error('this phase requires --enable-provider')
    {'raw': summarize_raw, 'exchanges': exchanges, 'attention': attention,
     'hierarchy': hierarchy, 'install': install}[args.phase](args.root.resolve(), args.step)
