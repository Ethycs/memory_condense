"""Bounded summary-only adapters for the resident compressed Qwen prefix."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path


def cav_bundle(filename, *, layer, checkpoint_sha256=None):
    """A local bundle must bind training checkpoint, layer and both file hashes."""
    path = Path(filename).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('layer') != layer or type(value.get('layer')) is not int:
        raise ValueError('CAV bundle layer differs from experimental Qwen layer')
    checkpoint = value.get('checkpoint_sha256','')
    if len(checkpoint)!=64 or any(c not in '0123456789abcdef' for c in checkpoint):
        raise ValueError('CAV bundle requires the training checkpoint hash')
    if checkpoint_sha256 is not None and checkpoint != checkpoint_sha256:
        raise ValueError('CAV training checkpoint differs from resident Qwen')
    result = dict(value)
    for name in ('report','vectors'):
        target = (path.parent/value[name]).resolve()
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        if digest != value.get(name+'_sha256'):
            raise ValueError('CAV artifact hash mismatch: '+name)
        result[name] = target
    report = json.loads(result['report'].read_text(encoding='utf-8-sig'))
    if report.get('checkpoint_sha256') != checkpoint:
        raise ValueError('CAV training report does not bind the declared checkpoint')
    if report.get('runtime_dtype') != 'float16':
        raise ValueError('CAV training precision differs from resident FP16 Qwen')
    result['bundle_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run(module, query, summaries, *, encoder, policy, remaining_passes):
    """Caller holds the resident GPU lock, stream, and inference context.

    Layer four is the default: it has a complete residual, while the shipping
    layer-five coverage path deliberately omits its final MLP. No second model
    is loaded and no transformer state survives this call.
    """
    from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker, numbered_memory_block
    from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
    p = policy
    receipt = dict(status='no_candidates',model_passes=0,raw_reads=0,
                   retained_transformer_token_state_bytes=0,attention_layer=p.qwen_layer,
                   input_kind='stored_summaries',search_space='bounded_atomic_summary_forest')
    if not summaries:
        return [],receipt
    needed = 2 if module=='qwen_summary_routing' else 1
    if remaining_passes < needed:
        return [],dict(receipt,status='pass_budget')
    # Never silently truncate the query or an individual summary. Reserve a
    # small margin for nested-pass aliases/tokenizer boundary differences.
    if len(encoder.tokenizer.encode(query))>256:
        return [],dict(receipt,status='query_budget')
    selected = []
    cap = min(8,p.candidate_limit)
    if module=='live_head_memory':
        cap = min(cap,remaining_passes-1)
    for summary in summaries:
        proposal = [*selected,summary]
        block,_,_ = numbered_memory_block([s.summary for s in proposal])
        if len(encoder.tokenizer.encode(block+f'[New memory] {query}\n'))<=p.workspace_tokens-32:
            selected.append(summary)
        if len(selected)>=cap:
            break
    if not selected or cap<1:
        return [],dict(receipt,status='workspace_budget')
    aliases = {f'summary-{i}':s for i,s in enumerate(selected)}
    candidates = [AssociativeMemoryCandidate(key,s.summary,route='section_summary')
                  for key,s in aliases.items()]
    receipt.update(candidate_count=len(candidates),omitted_candidates=len(summaries)-len(candidates))
    bank = None
    if module=='cav_associations':
        from memory_condense.associations.cav_memory import CAVBank
        bundle = cav_bundle(p.cav_bundle,layer=p.qwen_layer,
                            checkpoint_sha256=encoder.checkpoint_identity.checkpoint_sha256)
        bank = CAVBank.load(bundle['report'],bundle['vectors'],layer=p.qwen_layer,device=encoder.device)
        if (bank.vectors.shape[1]!=encoder.config.hidden_size
                or not encoder._torch.isfinite(bank.vectors).all()
                or not encoder._torch.isfinite(bank.thresholds).all()):
            raise ValueError('CAV vectors differ from resident hidden size or contain nonfinite values')
        receipt['cav_bundle_sha256'] = bundle['bundle_sha256']
    linker = QwenMemoryLinker(encoder,layer=p.qwen_layer,cav_bank=bank,
                             max_candidates=8,max_workspace_tokens=p.workspace_tokens)
    receipt['model'] = qwen_linker_identity(linker,strict=bank is None)
    # Enforce and count real prefix executions, including failed attempts. The
    # shared GPU lock prevents any other recall or compiler from seeing this
    # temporary counter. All hooks and this wrapper are removed on every exit.
    forward = encoder._forward
    own_forward = encoder.__dict__.get('_forward')
    had_forward = '_forward' in encoder.__dict__
    def counted(**inputs):
        if receipt['model_passes'] >= remaining_passes:
            raise RuntimeError('Experimental Qwen pass budget exhausted')
        receipt['model_passes'] += 1
        return forward(**inputs)
    encoder._forward = counted
    try:
        if module=='qwen_summary_routing':
            from memory_condense.search.section_attention import route_summary_hierarchy
            from memory_condense.search.section_routing import SectionSummaryIndex
            plan = route_summary_hierarchy(query,SectionSummaryIndex(selected),linker=linker,
                max_sections=min(p.additions,len(selected)),max_depth=1)
            scores = [(r.section.section_id,r.score) for r in plan.routes]
            receipt['attention_receipt'] = plan.attention_receipt.identity_payload()
        elif module=='live_head_memory':
            from memory_condense.associations.qwen_live_memory import QwenLiveHeadMemory
            workspace = QwenLiveHeadMemory(encoder,layer=p.qwen_layer,
                association_candidates=0,max_items=cap)
            for c in candidates:
                workspace.write(c.episode_id,c.text)
            result = workspace.retrieve(query,top_k=min(p.additions,len(candidates)),hops=p.graph_hops)
            scores = [(aliases[h.episode_id].section_id,max(1e-12,h.score)) for h in result.hits]
            del result,workspace
        else:
            result = linker.link(query,candidates,top_k=None)
            hits = [replace(h,episode_id=aliases[h.episode_id].section_id) for h in result.hits]
            if module=='qwen_consolidation':
                from memory_condense.associations.consolidation import qwen_head_activations
                scores = [(node.key,value) for node,value in qwen_head_activations(hits).items()]
            else:
                scores = []
                for h in hits:
                    score = max(0.,h.qk_score)
                    if module=='cav_associations':
                        values = h.metadata.get('cav_signature',())
                        score += math.log1p(sum(max(0.,v) for v in values)/max(1,len(values)))
                    scores.append((h.episode_id,max(1e-12,score)))
                scores.sort(key=lambda x:(-x[1],x[0]))
            receipt['workspace_tokens'] = result.workspace_tokens
        receipt['status'] = 'selected'
        return scores,receipt
    except (MemoryError,RuntimeError) as exc:
        # Optional GPU arms may fail without breaking an otherwise valid chat.
        # Report a failure; never count this as a working/empty experimental arm.
        return [],dict(receipt,status='model_error',error_type=type(exc).__name__)
    finally:
        if had_forward:
            encoder._forward = own_forward
        else:
            del encoder._forward
