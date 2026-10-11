"""One optional summary-only Qwen weighting of the existing native IO update."""
from types import SimpleNamespace


def summary_weights(app, packet, chunks, router, attention, policy):
    from memory_condense.associations.consolidation import ConsolidationNode, context_activations
    eligible = {r['section_id'] for r in packet.references
                if r.get('independent') and r.get('learning_eligible',True)}
    by_turn = {}
    for atom in router.atoms.values():
        if atom.section_id in eligible or atom.spans[0].turn_id==packet.input_event_id:
            by_turn.setdefault(atom.spans[0].turn_id,[]).append(atom)
    input_summaries = by_turn.get(packet.input_event_id,())
    if not input_summaries or not callable(getattr(attention,'experiment',None)):
        return None,dict(status='no_input_summary' if not input_summaries else 'unavailable_attention',
                         model_passes=0,raw_reads=0)
    query = ' '.join(a.summary for a in input_summaries)
    candidates = []
    for chunk in chunks:
        row = app._db.execute('SELECT turn_id,start_char,end_char FROM chunks WHERE chunk_id=?',(chunk,)).fetchone()
        if row is None or row[0]==packet.input_event_id:
            continue
        texts = [a.summary for a in by_turn.get(row[0],())
                 if a.spans[0].start_char<row[2] and a.spans[0].end_char>row[1]]
        if texts:
            candidates.append(SimpleNamespace(section_id=ConsolidationNode.chunk(chunk).key,
                                              summary=' '.join(texts)))
    scores,receipt = attention.experiment('qwen_consolidation',query,tuple(candidates),
        policy=policy,remaining_passes=policy.max_qwen_passes)
    if receipt['status']!='selected':
        return None,receipt
    baseline = context_activations([],chunks)
    allowed = {node.key:node for node in baseline}
    for key,value in scores:
        if key not in allowed or not 0<value<=1:
            raise ValueError('Qwen learning escaped original delivered chunks')
        baseline[allowed[key]] = value
    return baseline,receipt
