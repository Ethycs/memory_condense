"""Replay saved continuation atoms against the initial archive, without models."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import sqlite3
import time

from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_memory import Compiler, storage_rows
from memory_condense.search.section_summary import SectionSummary


def main():
    source = Path('eval_results/chat-io-complete-turn-20260929-r2')
    saved = Path('eval_results/chat-io-batch12-20260929-r3')
    root = Path('eval_results/recall-summary-reuse-20260929-r2')
    if root.exists():
        raise ValueError('Use a fresh output directory')
    shutil.copytree(source/'cache', root/'cache')
    path = source/'live/store/memory/native-spine-live-v1.sqlite'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        originals = tuple(SectionSummary.from_dict(json.loads(r[0]))
                          for r in db.execute("SELECT payload FROM sections WHERE kind='atomic'"))
    initial = read(saved/'report.json')['initial']['events']
    events = read(saved/'final-events.json')['events'][initial:]
    actor = read(saved/'actor.json')
    rows = storage_rows([dict(turn_id=e['event_id'], role=e['role'], text=e['text'],
                              metadata=e['metadata']) for e in events])
    compiler = Compiler(root, 'replay', report=lambda **_: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved copied evidence must not require raw generation')
    compiler.gateway.call = forbidden
    started = time.perf_counter()
    atoms = compiler.atoms(rows, actor['source']['family'], actor['source']['export_timestamp'], original_atoms=originals)
    aliases = [read(p) for p in (root/'cache/recall-aliases').glob('*.json')]
    spans = [s for a in atoms for s in a.spans]
    checks = dict(all_events_covered={s.turn_id for s in spans}=={r['turn_id'] for r in rows},
        ten_new_receipts_reused=len(aliases)==10,
        all_145_new_receipt_references_reused=sum(len(a['reused_sections']) for a in aliases)==145,
        exact_receipt_text_retained=all(
            (parts := [s for s in spans if s.turn_id==r['turn_id']])
            and parts[0].start_char==0 and parts[-1].end_char==len(r['text'])
            and all(a.end_char==b.start_char for a,b in zip(parts,parts[1:]))
            for r in rows if r.get('metadata',{}).get('_chat',{}).get('kind')=='recall'),
        deterministic_replay=compiler.atoms(rows, actor['source']['family'], actor['source']['export_timestamp'],
                                          original_atoms=originals)==atoms)
    result = dict(checks=checks, all_passed=all(checks.values()), elapsed_s=time.perf_counter()-started,
        events=len(rows), atoms=len(atoms), reused_packets=len(aliases),
        reused_references=sum(len(a['reused_sections']) for a in aliases), raw_generation_calls=0,
        preserved_historical_cached_packets=2, preserved_historical_cached_references=28,
        saved_workload=str(saved), original_archive=str(path),
        full_pipeline_or_latency_benchmark=False)
    save(root/'atoms.json', dict(atoms=[asdict(a) for a in atoms]))
    save(root/'report.json', result)
    emit(**result)


if __name__ == '__main__':
    main()
