"""Verify a finished chat-cycle store in a new process without model calls."""
from contextlib import closing
from pathlib import Path
import sqlite3

from memory_condense.persistence import native_spine_incremental_store as store
from memory_condense.persistence.transcript_store import TranscriptStore
from tools.engineering_research_gateway import read, save, emit


def verify(root, runtime_root):
    expected = read(root/'final-events.json')['events']
    expected_feedback = read(root/'run-plan.json')['live_exchanges']
    packet = read(root/'post-drain-recall.json')
    memory = runtime_root/'store'/'memory'
    with closing(sqlite3.connect((memory/'memory.db').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        turns = TranscriptStore(db).get_all()
        raw_integrity = db.execute('PRAGMA quick_check').fetchone()[0]
    state = store.load(memory/store.FILENAME, turns=turns)
    with closing(sqlite3.connect((runtime_root/'chat'/'chat-events.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        journal = db.execute('SELECT event_id,role,text FROM events ORDER BY sequence').fetchall()
        committed, target = db.execute('SELECT committed,target FROM ingestion_state WHERE id=1').fetchone()
        pending_feedback = db.execute('SELECT COUNT(*) FROM feedback WHERE applied=0').fetchone()[0]
        applied_new_feedback = db.execute("SELECT COUNT(*) FROM feedback WHERE packet_id LIKE 'batch12-%' AND applied=1").fetchone()[0]
        journal_integrity = db.execute('PRAGMA quick_check').fetchone()[0]
    checks = dict(
        exact_raw_events=[(t.turn_id,t.role,t.text) for t in turns]==[
            (e['event_id'],'system' if e['role'] in ('source','tool') else e['role'],e['text']) for e in expected],
        exact_journal_events=journal==[(e['event_id'],e['role'],e['text']) for e in expected],
        same_native_snapshot=state.native.receipt==packet['snapshot'],
        same_parent_snapshot=state.parent_receipt==packet['parent_snapshot'],
        all_events_committed=committed==len(expected) and target is None,
        no_pending_feedback=pending_feedback==0,
        all_new_feedback_applied=applied_new_feedback==expected_feedback,
        raw_integrity=raw_integrity=='ok', journal_integrity=journal_integrity=='ok')
    return dict(runtime_root=str(runtime_root),checks=checks,all_passed=all(checks.values()),
        events=len(turns),snapshot=state.native.receipt,parent_snapshot=state.parent_receipt,
        generation_calls=0,raw_history_reingestions=0,read_only=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--runtime-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=args.runtime_root or Path(read(args.root/'run-plan.json')['runtime_root'])
    result=verify(args.root,runtime)
    save(args.output,result)
    emit(**result)
    if not result['all_passed']:
        raise SystemExit(1)
