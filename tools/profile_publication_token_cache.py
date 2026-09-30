"""Compare bounded token-count caches on the same authenticated saved snapshot."""
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import shutil
import sqlite3
import time

from memory_condense.domain import _tokenizer as tokens
from memory_condense.persistence import native_spine_incremental_store as store
from memory_condense.persistence.transcript_store import TranscriptStore
from tools.engineering_research_gateway import save,emit


def main():
    source=Path('eval_results/chat-io-batch12-20260930-r12/live/store/memory')
    root=Path('eval_results/publication-token-cache-20260930-r1')
    root.mkdir(exist_ok=False)
    with closing(sqlite3.connect((source/'memory.db').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        turns=TranscriptStore(db).get_all()
    uncached=tokens._count_short_text.__wrapped__
    results=[]
    for size in (4096,8192):
        tokens._count_short_text=lru_cache(maxsize=size)(uncached)
        path=root/f'snapshot-{size}.sqlite'
        shutil.copyfile(source/store.FILENAME,path)
        state=store.load(path,turns=turns)
        before=tokens._count_short_text.cache_info()
        started=time.perf_counter()
        actual=store.publish(path,atomic_index=state.native.semantic.hierarchy,hierarchy=state.native.hierarchy,
            matrix=state.native.semantic._dense._matrix,projection=state.parents.hierarchy,
            parent_matrix=state.parents._dense._matrix,embedding_identity=state.native.semantic.embedding_identity,
            turns=turns,previous=state)
        elapsed=time.perf_counter()-started
        after=tokens._count_short_text.cache_info()
        row=dict(entries=size,publish_s=elapsed,hits=after.hits-before.hits,misses=after.misses-before.misses,
                 identical_manifest=actual.manifest==state.manifest)
        results.append(row)
        emit(**row)
    save(root/'report.json',dict(results=results,generation_calls=0,raw_history_reingestions=0))


if __name__=='__main__': main()
