"""Profile only the saved 48-turn suffix using exact already computed vectors."""
from datetime import datetime
from pathlib import Path
import cProfile
import json
import pstats
import shutil
import sqlite3
import time
import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._discourse_identity import quote_sha256
from tools.engineering_research_gateway import read,save,emit
from tools.engineering_research_memory import storage_rows


def main(root,manual_checkpoint=False,cache_mib=None):
    root.mkdir(exist_ok=False)
    source=Path('eval_results/chat-io-complete-turn-20260929-r2')
    saved=Path('eval_results/chat-io-batch12-20260930-r6')
    shutil.copytree(source/'live/store/memory',root/'memory')
    with sqlite3.connect((saved/'live/store/memory/memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        vectors={quote_sha256(text):np.frombuffer(blob,dtype=np.float32).tolist()
                 for text,blob in db.execute('SELECT text,embedding FROM chunks WHERE embedding IS NOT NULL')}
    class Encoder:
        dim=1024
        def embed_chunks(self,chunks):
            return [c.model_copy(update={'embedding':vectors[quote_sha256(c.text)]}) for c in chunks]
    actor=read(saved/'actor.json')
    rows=storage_rows([dict(turn_id=e['event_id'],role=e['role'],text=e['text'])
        for e in read(saved/'final-events.json')['events'][-48:]])
    results=[]
    with MemoryCondenser(root/'memory',embedder=Encoder(),auto_extract=False) as app:
        if cache_mib is not None:
            app._db.connection.execute(f'PRAGMA cache_size={-cache_mib*1024}')
        if manual_checkpoint:
            app._db.connection.execute('PRAGMA wal_autocheckpoint=0')
        synchronous=app._db.connection.execute('PRAGMA synchronous').fetchone()[0]
        for i in (0,24):
            records=[(r['role'],r['text'],actor['source']['family'],
                      datetime.fromisoformat(actor['source']['export_timestamp']),r['turn_id']) for r in rows[i:i+24]]
            profiler=cProfile.Profile()
            start=time.perf_counter()
            profiler.runcall(app.ingest_many,records)
            elapsed=time.perf_counter()-start
            profiler.dump_stats(str(root/f'batch-{i//24}.prof'))
            with (root/f'batch-{i//24}.txt').open('w',encoding='utf-8') as out:
                pstats.Stats(profiler,stream=out).sort_stats('cumtime').print_stats(22)
            checkpoint_s=0
            checkpoint=None
            if manual_checkpoint:
                start=time.perf_counter()
                checkpoint=app._db.connection.execute('PRAGMA wal_checkpoint(FULL)').fetchone()
                checkpoint_s=time.perf_counter()-start
            results.append(dict(batch=i//24,wall_s=elapsed,checkpoint_s=checkpoint_s,checkpoint=checkpoint))
            emit(**results[-1])
    save(root/'report.json',dict(results=results,embedding_vectors_reused=True,
        profiling_overhead_included=True,full_history_ingestions=0,original_store_mutated=False,
        manual_checkpoint=manual_checkpoint,synchronous=synchronous,cache_mib=cache_mib))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('eval_results/sub50-raw-profile-20260930-r1'))
    parser.add_argument('--manual-checkpoint',action='store_true')
    parser.add_argument('--cache-mib',type=int)
    args=parser.parse_args()
    main(args.root,args.manual_checkpoint,args.cache_mib)
