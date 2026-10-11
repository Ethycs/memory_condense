"""Query-time decision/fix annotations over evidence already hydrated exactly.

Chronology and co-membership are structural links, never inferred supersession.
Only existing memory revisions with both quoted endpoints delivered are exposed.
No model, extraction, database mutation or extra raw read occurs here.
"""
from dataclasses import dataclass
import json

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import MemoryStatus

GUIDANCE = (
    'DECISION AND FIX HISTORY\n'
    'Use the question to reconstruct relevant decisions, attempted fixes, their rationale, '
    'outcomes and remaining work. Cite the D evidence labels or original turn IDs. '
    'Distinguish proposed, attempted, reported implemented, tested and released; missing '
    'verification means unverified, not failed. User statements record user assertions; '
    'assistant reports do not establish successful execution. Tool observations support '
    'only the operation actually observed. Mention timestamps are not effective dates. '
    'Episode membership and follows links describe transcript structure, not causation '
    'or revision. A stored_revision link records an existing memory revision, not proof '
    'of truth. For a revision inferred from the passages, cite both sides and explain '
    'the explicit correction; otherwise leave conflicting claims unresolved. Scope '
    'conclusions to the retrieved evidence; this is not a complete inventory.\n'
)


@dataclass(frozen=True)
class StoredRevision:
    predecessor: str
    successor: str
    before: tuple[tuple[str, str], ...]
    after: tuple[tuple[str, str], ...]


def snapshot_revisions(store):
    """Capture through the canonical store API on the writer's snapshot boundary."""
    if store is None:
        return ()
    revisions = []
    for old in store.list_items(status=MemoryStatus.SUPERSEDED):
        for new in store.successors(old.mem_id):
            if new.status is MemoryStatus.DELETED:
                continue
            revisions.append(StoredRevision(old.mem_id, new.mem_id,
                tuple((p.turn_id, p.quote) for p in old.provenance),
                tuple((p.turn_id, p.quote) for p in new.provenance)))
    return tuple(revisions)


def annotate(references, events, *, revisions=(), max_tokens=2048):
    """Label only delivered spans; unresolved/missing endpoints create no edge."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('Decision history requires a positive annotation budget')
    by_id = {e.event_id:e for e in events}
    positions = {e.event_id:i for i,e in enumerate(events)}
    episodes, active = {}, {}
    for e in events:
        source = e.metadata.get('source_id')
        if e.role == 'user' or source not in active:
            active[source] = e.event_id
        episodes[e.event_id] = active[source]
    ordered = sorted(enumerate(references), key=lambda pair:(
        positions[pair[1]['span']['turn_id']], pair[1]['span']['start_char'], pair[0]))
    labels = {index:f'D{n:03}' for n,(index,_) in enumerate(ordered,1)}
    rows, previous, passages, episode_labels = [], {}, {}, {}
    for index, reference in ordered:
        s = reference['span']
        e = by_id[s['turn_id']]
        text = e.text[s['start_char']:s['end_char']]
        label = labels[index]
        episode = (s['source_id'], episodes[e.event_id])
        episode_labels.setdefault(episode, f'E{len(episode_labels)+1:03}')
        row = dict(id=label, turn=e.event_id, mentioned_at=s['created_at'], role=e.role,
                   episode=episode_labels[episode], follows=previous.get(episode),
                   attribution=('recalled_copy' if not reference.get('independent',True) else
                                {'user':'user_statement','assistant':'assistant_report',
                                 'tool':'tool_observation'}.get(e.role,'source_statement')))
        previous[episode] = label
        passages[label] = (reference, text)
        rows.append(row)
    # A quote must be fully delivered, not merely somewhere in the same turn.
    def endpoints(provenance):
        return [label for label,(r,text) in passages.items() if r.get('independent',True)
                and any(tid==r['span']['turn_id'] and quote and quote in text for tid,quote in provenance)]
    links = []
    for revision in revisions:
        before, after = endpoints(revision.before), endpoints(revision.after)
        if before and after and set(before).isdisjoint(after):
            links.append(dict(kind='stored_revision', before=before, after=after,
                              predecessor=revision.predecessor, successor=revision.successor))
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(',',':'))
    included, admitted_links = [], []
    def render():
        note = (f'Annotations: {len(included)}/{len(rows)} delivered spans; '
                f'stored revision links: {len(admitted_links)}/{len(links)} with delivered support.\n')
        # Original turn IDs already appear on exact passages; avoid repeating
        # long hashes in the model's annotation table. Audit rows retain them.
        visible = [{k:v for k,v in row.items() if k not in ('turn','role')} for row in included]
        return GUIDANCE+note+'\n'.join(compact(row) for row in [*visible,*admitted_links])
    # Prioritize actual revision links, then whole chronological annotations.
    if count_tokens(render()) > max_tokens:
        return '', labels, dict(annotations=[], revision_links=[], omitted_annotations=len(rows),
                               omitted_revision_links=len(links), token_count=0, max_tokens=max_tokens)
    for link in links:
        admitted_links.append(link)
        if count_tokens(render()) > max_tokens:
            admitted_links.pop()
    for row in rows:
        included.append(row)
        if count_tokens(render()) > max_tokens:
            included.pop()
    text = render()
    receipt = dict(format='decision-fix-history-v1', annotations=included,
        revision_links=admitted_links, omitted_annotations=len(rows)-len(included),
        omitted_revision_links=len(links)-len(admitted_links), token_count=count_tokens(text),
        max_tokens=max_tokens, exhaustive=False, raw_reads=0, provider_calls=0)
    receipt['receipt_sha256'] = identity_sha256(receipt)
    return text, labels, receipt
