"""Source validation and legacy embedding placement."""
import json
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.modeling.embedding import EmbeddingService

def repair_raw_support(content, fragments):
    """Repair quote escaping only; never invent support or modify summaries.

Every retained candidate must occur literally within its attributed fragment.
Unsupported quotations may be dropped only when model-selected exact support
remains. Strict native parsing still validates the complete resulting response.
"""
    value = json.loads(content)
    repairs = []
    if not isinstance(value, dict) or not isinstance(value.get('atoms'), list) or len(value['atoms']) != len(fragments):
        return content, repairs
    for row, fragment in zip(value['atoms'], fragments, strict=True):
        if not isinstance(row, dict) or not isinstance(row.get('support'), list):
            continue
        kept = []
        original = row['support']
        for quote in original:
            if not isinstance(quote, str):
                continue
            candidates = [quote]
            if len(quote) > 2 and quote[0] == quote[-1] == '"':
                candidates.append(quote[1:-1])
            for _ in range(2):
                candidates.extend(json.dumps(q, ensure_ascii=False)[1:-1] for q in tuple(candidates))
            match = next((q for q in candidates if q.strip() and q in fragment.text and count_tokens(q) <= 32), None)
            if match and match not in kept:
                kept.append(match)
        if kept and kept != original:
            repairs.append({'label': row.get('label'), 'original_support': original, 'exact_support': kept,
                            'fragment_sha256': quote_sha256(fragment.text)})
            row['support'] = kept
    return json.dumps(value, ensure_ascii=False), repairs

class StagedEmbedding(EmbeddingService):
    """Keep verified weights in RAM between GPU phases without reloading files."""
    def _load_model(self):
        model = super()._load_model()
        if next(model.parameters()).device.type != 'cuda':
            model.to('cuda')
        return model

    def park(self):
        if self._model is not None:
            self._model.to('cpu')
            import torch
            torch.cuda.empty_cache()
