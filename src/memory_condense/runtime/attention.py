"""Pinned summary-only attention and its durable scalar cache."""
from pathlib import Path
import hashlib
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.modeling.qwen_prefix import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, expected_prefix_checkpoint_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.runtime.artifacts import publish_sealed_json, read_sealed_json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FILES = ("runtime/attention.py", "modeling/qwen_prefix.py", "associations/qwen_memory_linker.py", "search/episodes/qwen_episode_signal.py", "search/episodes/surprise_models.py", "domain/_tokenizer.py")
def digest(name):
    return hashlib.sha256((PACKAGE_ROOT/name).read_bytes()).hexdigest()

def user_windows(exchanges):
    rows = tuple(exchanges)
    if not rows or any(type(e) is not UserSpineExchange for e in rows):
        raise TypeError("attention preparation requires typed, complete source exchanges")
    if len({e.section.source_id for e in rows}) != 1:
        raise ValueError("attention windows cannot cross source occurrences")
    SectionSummaryIndex(tuple(e.section for e in rows))
    texts = tuple(e.user_spine if e.user_spine is not None else "Unowned prelude." for e in rows)
    if any(count_tokens(t) > 128 for t in texts):
        raise ValueError("a user summary would be truncated by attention")
    windows, start = [], 0
    while start < len(rows):
        end = min(len(rows), start+8)
        windows.append({"start_exchange": start, "end_exchange": end, "texts": list(texts[start:end])})
        if end == len(rows):
            break
        start = end-1
    return windows

def cache_method(root, *, host_embeddings=None, versioned=False):
    # This identity is independent of a corpus/snapshot or occurrence timestamp.
    payload = {
        "model_id": DEFAULT_MODEL_ID, "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": expected_prefix_checkpoint_sha256(6),
        "device": "cuda", "dtype": "float16", "prefix_layers": 6, "attention_layer": 5,
        "head_vote_k": 4, "max_input_spans": 8, "span_token_cap": 128,
        "linker_max_candidates": 8, "linker_max_workspace_tokens": 4096,
        "owned_runtime_binding": True, "raw_inputs_to_qwen": False,
        "implementation": {name: digest(name) for name in FILES},
    }
    if host_embeddings is not None:
        payload['host_embeddings'] = bool(host_embeddings)
    name = 'method-' + identity_sha256(payload) + '.json' if versioned else 'method.json'
    return publish_sealed_json(root / name, payload)[0]

class ScalarAttentionCache:
    max_spans = 8
    span_token_cap = 128

    def __init__(self, root, preflight):
        self.root, self.preflight = root, preflight
        self.scorer = None
        self.values = {}

    def score_sequence(self, texts):
        from memory_condense.search.episodes.surprise_models import AttentionHeadSurpriseReceipt, ScoredSurpriseSequence
        key = identity_sha256({"preflight_sha256": self.preflight.sha256, "texts": list(texts)})
        if key not in self.values:
            path = self.root / "attention" / (key + ".json")
            if path.exists():
                artifact = read_sealed_json(path)
                row = artifact.payload
                if row["preflight_sha256"] != self.preflight.sha256:
                    raise ValueError("attention cache belongs to another compilation")
                signal = ScoredSurpriseSequence(row["scores"], row["similarities"], AttentionHeadSurpriseReceipt(**row["receipt"]))
            else:
                if self.scorer is None:
                    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
                    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
                    from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
                    print("Loading local Qwen for query-independent user-summary attention...", flush=True)
                    from memory_condense.runtime.config import RuntimeAssets
                    encoder = Qwen3PrefixEncoder(RuntimeAssets.resolve().qwen,
                        layers=6, device="cuda", dtype="float16")
                    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
                    self.scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=128)
                signal = self.scorer.score_sequence(texts)
                publish_sealed_json(path, {"preflight_sha256": self.preflight.sha256,
                    "scores": signal.scores, "similarities": signal.similarities,
                    "receipt": signal.receipt.identity_payload()})
            signal.validate_inputs(texts)
            self.values[key] = signal
        return self.values[key]

class LocalAttention(ScalarAttentionCache):
    def load_scorer(self):
        from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
        from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
        from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
        from memory_condense.runtime.config import RuntimeAssets
        model = getattr(self, 'model_dir', None) or RuntimeAssets.resolve().qwen
        encoder = Qwen3PrefixEncoder(model, layers=6, device='cuda', dtype='float16',
                                    host_embeddings=getattr(self, 'host_embeddings', False))
        self.scorer = QwenAttentionHeadSurpriseScorer(
            QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096),
            max_spans=8, span_token_cap=128)

    def score_sequence(self, texts):
        key = identity_sha256({'preflight_sha256': self.preflight.sha256, 'texts': list(texts)})
        if key not in self.values and not (self.root / 'attention' / (key + '.json')).exists() and self.scorer is None:
            self.load_scorer()
        return super().score_sequence(texts)
