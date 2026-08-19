"""Bounded Qwen prefix-transport signals for episodic event formation."""

from __future__ import annotations

import hashlib
import inspect
import marshal
import math
import re
import sys
import threading
from pathlib import Path
from types import CodeType
from typing import Any, Sequence

import numpy as np

from memory_condense.domain._tokenizer import (
    tokenizer_proxy_identity,
    truncate_to_tokens_lossless,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256

from .surprise_models import (
    EPISODIC_SURPRISE_PROBE,
    AttentionHeadSurpriseReceipt,
    ScoredSurpriseSequence,
    adjacent_cosine_change,
    exact_integer,
    input_sequence_sha256,
    score_sequence_sha256,
    similarity_matrix_sha256,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QwenAttentionHeadSurpriseScorer:
    """Transient Qwen prefix-head change and cohesion signal.

    A frozen neutral probe makes span vectors independent of the later user
    query. Every bounded linker workspace returns normalized OV transport
    signatures, which are reduced immediately to scalar cosine similarities.
    The adapter retains only the linker, immutable integer caps, and a lock
    serializing calls made through this scorer instance.
    """

    __slots__ = (
        "linker",
        "max_spans",
        "span_token_cap",
        "probe_token_cap",
        "max_transport_dimension",
        "_lock",
    )

    def __init__(
        self,
        linker: Any,
        *,
        max_spans: int = 256,
        span_token_cap: int = 64,
        probe_token_cap: int = 96,
        max_transport_dimension: int = 8192,
    ) -> None:
        if not callable(getattr(linker, "inspect_coverage", None)):
            raise TypeError("linker must expose inspect_coverage")
        self.linker = linker
        self.max_spans = exact_integer(max_spans, "max_spans", minimum=1)
        self.span_token_cap = exact_integer(
            span_token_cap,
            "span_token_cap",
            minimum=1,
        )
        self.probe_token_cap = exact_integer(
            probe_token_cap,
            "probe_token_cap",
            minimum=1,
        )
        self.max_transport_dimension = exact_integer(
            max_transport_dimension,
            "max_transport_dimension",
            minimum=1,
        )
        self._lock = threading.Lock()
        _qwen_linker_identity(linker)

    def score(
        self,
        previous_text: str | None,
        current_text: str,
        *,
        previous_embedding: Sequence[float] | None = None,
        current_embedding: Sequence[float] | None = None,
    ) -> float:
        """Score one pair through the same first-class sequence path."""

        if previous_text is None:
            return 0.0
        result = self.score_sequence(
            (str(previous_text), str(current_text)),
            embeddings=(previous_embedding, current_embedding),
        )
        return result.scores[1]

    def score_sequence(
        self,
        texts: Sequence[str],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> ScoredSurpriseSequence:
        with self._lock:
            return self._score_sequence_once(texts, embeddings=embeddings)

    def _score_sequence_once(
        self,
        texts: Sequence[str],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> ScoredSurpriseSequence:
        rows = tuple(str(text) for text in texts)
        if len(rows) > self.max_spans:
            raise MemoryError(
                f"surprise sequence has {len(rows)} spans, above the hard cap "
                f"of {self.max_spans}"
            )
        if embeddings is not None and len(tuple(embeddings)) != len(rows):
            raise ValueError("embeddings must align one-for-one with texts")
        identity = _qwen_linker_identity(self.linker)
        implementation_sha256 = _attention_head_implementation_sha256(self.linker)
        similarities, stats = _head_transport_similarities(
            self.linker,
            rows,
            span_token_cap=self.span_token_cap,
            probe_token_cap=self.probe_token_cap,
            max_transport_dimension=self.max_transport_dimension,
        )
        if _qwen_linker_identity(self.linker) != identity:
            raise RuntimeError("Qwen linker identity changed during surprise scoring")
        if (
            _attention_head_implementation_sha256(self.linker)
            != implementation_sha256
        ):
            raise RuntimeError("episode surprise implementation changed during scoring")
        scores = (
            ()
            if not rows
            else (
                0.0,
                *(
                    adjacent_cosine_change(similarities[index - 1][index])
                    for index in range(1, len(rows))
                ),
            )
        )
        receipt = AttentionHeadSurpriseReceipt(
            **identity,
            implementation_sha256=implementation_sha256,
            tokenizer_proxy_sha256=identity_sha256(
                dict(_active_tokenizer_proxy_identity()())
            ),
            neutral_probe_sha256=quote_sha256(EPISODIC_SURPRISE_PROBE),
            max_input_spans=self.max_spans,
            span_token_cap=self.span_token_cap,
            probe_token_cap=self.probe_token_cap,
            max_transport_dimension=self.max_transport_dimension,
            input_spans=len(rows),
            workspace_batches=stats["workspace_batches"],
            forward_passes=stats["forward_passes"],
            inspected_spans=stats["inspected_spans"],
            transport_dimension=stats["transport_dimension"],
            similarity_scalar_pairs=len(rows) * max(0, len(rows) - 1) // 2,
            max_workspace_candidates=stats["max_workspace_candidates"],
            max_workspace_tokens=stats["max_workspace_tokens"],
            total_workspace_tokens=stats["total_workspace_tokens"],
            input_sequence_sha256=input_sequence_sha256(rows),
            score_sequence_sha256=score_sequence_sha256(scores),
            similarity_matrix_sha256=similarity_matrix_sha256(similarities),
        )
        return ScoredSurpriseSequence(scores, similarities, receipt)


def _head_transport_similarities(
    linker: Any,
    texts: tuple[str, ...],
    *,
    span_token_cap: int,
    probe_token_cap: int,
    max_transport_dimension: int,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, int]]:
    if not texts:
        return (), {
            "workspace_batches": 0,
            "forward_passes": 0,
            "inspected_spans": 0,
            "transport_dimension": 0,
            "max_workspace_candidates": 0,
            "max_workspace_tokens": 0,
            "total_workspace_tokens": 0,
        }
    from memory_condense.associations.head_memory import (
        AssociativeMemoryCandidate,
    )

    truncated_texts = tuple(
        _lossless_proxy_prefix(text, span_token_cap) for text in texts
    )
    candidates = tuple(
        AssociativeMemoryCandidate(
            episode_id=f"span-{index}-{quote_sha256(text)[:16]}",
            text=truncated_texts[index],
            route="episode_head_surprise",
        )
        for index, text in enumerate(texts)
    )
    probe = _lossless_proxy_prefix(EPISODIC_SURPRISE_PROBE, probe_token_cap)
    batch_limit = exact_integer(
        getattr(linker, "max_candidates", -1),
        "linker.max_candidates",
        minimum=1,
    )
    linker_token_cap = exact_integer(
        getattr(linker, "max_workspace_tokens", -1),
        "linker.max_workspace_tokens",
        minimum=1,
    )
    vectors: dict[str, np.ndarray] = {}
    cursor = 0
    workspace_batches = 0
    forward_passes = 0
    inspected_spans = 0
    max_workspace_candidates = 0
    max_workspace_tokens = 0
    total_workspace_tokens = 0
    width: int | None = None
    while cursor < len(candidates):
        batch = candidates[cursor : cursor + batch_limit]
        linked = linker.inspect_coverage(probe, batch)
        try:
            retained_state = exact_integer(
                getattr(linked, "retained_transformer_state_bytes", 0),
                "retained_transformer_state_bytes",
                minimum=0,
            )
            if retained_state != 0:
                raise RuntimeError("Qwen linker retained transformer state")
            if getattr(linked, "past_key_values", None) is not None:
                raise RuntimeError("Qwen linker returned a K/V cache")
            consumed = exact_integer(
                getattr(linked, "workspace_candidates", -1),
                "workspace_candidates",
                minimum=1,
            )
            tokens = exact_integer(
                getattr(linked, "workspace_tokens", -1),
                "workspace_tokens",
                minimum=1,
            )
            passes = exact_integer(
                getattr(linked, "passes", -1),
                "passes",
                minimum=1,
            )
            inspections = exact_integer(
                getattr(linked, "total_candidate_inspections", -1),
                "total_candidate_inspections",
                minimum=1,
            )
            if consumed > len(batch) or inspections != consumed:
                raise RuntimeError("Qwen linker reported invalid workspace coverage")
            if tokens > linker_token_cap:
                raise RuntimeError("Qwen linker exceeded its workspace token cap")
            accepted = batch[:consumed]
            expected_ids = {candidate.episode_id for candidate in accepted}
            hits = tuple(getattr(linked, "hits", ()))
            hit_ids = [str(getattr(hit, "episode_id", "")) for hit in hits]
            if len(hit_ids) != len(set(hit_ids)) or set(hit_ids) != expected_ids:
                raise RuntimeError("Qwen linker returned incomplete transport hits")
            by_id = {str(hit.episode_id): hit for hit in hits}
            for candidate in accepted:
                hit = by_id[candidate.episode_id]
                qk = float(getattr(hit, "qk_score"))
                ov = float(getattr(hit, "ov_transport"))
                if not math.isfinite(qk) or not math.isfinite(ov):
                    raise ValueError("Qwen linker returned non-finite QK/OV scores")
                vector = _active_transport_normalizer()(
                    getattr(hit, "transport_signature", None),
                    max_dimension=max_transport_dimension,
                )
                if vector is None:
                    raise RuntimeError("Qwen linker omitted a finite OV signature")
                if width is None:
                    width = int(vector.size)
                elif vector.size != width:
                    raise RuntimeError("Qwen OV signatures have inconsistent width")
                vectors[candidate.episode_id] = vector
            cursor += consumed
            workspace_batches += 1
            forward_passes += passes
            inspected_spans += inspections
            max_workspace_candidates = max(max_workspace_candidates, consumed)
            max_workspace_tokens = max(max_workspace_tokens, tokens)
            total_workspace_tokens += tokens
        finally:
            del linked
    ordered = tuple(vectors[candidate.episode_id] for candidate in candidates)
    matrix = [[0.0] * len(ordered) for _ in ordered]
    for left in range(len(ordered)):
        matrix[left][left] = 1.0
        for right in range(left + 1, len(ordered)):
            similarity = float(np.dot(ordered[left], ordered[right]))
            similarity = max(-1.0, min(1.0, similarity))
            matrix[left][right] = similarity
            matrix[right][left] = similarity
    scalar_matrix = tuple(tuple(row) for row in matrix)
    return scalar_matrix, {
        "workspace_batches": workspace_batches,
        "forward_passes": forward_passes,
        "inspected_spans": inspected_spans,
        "transport_dimension": int(width or 0),
        "max_workspace_candidates": max_workspace_candidates,
        "max_workspace_tokens": max_workspace_tokens,
        "total_workspace_tokens": total_workspace_tokens,
    }


def _active_transport_normalizer() -> Any:
    facade = sys.modules.get("memory_condense.search.episodes.surprise")
    return getattr(
        facade,
        "_normalized_transport_signature",
        _normalized_transport_signature,
    )


def _active_tokenizer_proxy_identity() -> Any:
    facade = sys.modules.get("memory_condense.search.episodes.surprise")
    return getattr(facade, "tokenizer_proxy_identity", tokenizer_proxy_identity)


def _normalized_transport_signature(
    value: Any,
    *,
    max_dimension: int,
) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
        dimensions = getattr(value, "dim", None)
        if callable(dimensions) and int(dimensions()) != 1:
            raise ValueError("Qwen OV signature must be one-dimensional")
        element_count = getattr(value, "numel", None)
        if callable(element_count) and int(element_count()) > max_dimension:
            raise MemoryError("Qwen OV signature exceeds its transport-dimension cap")
        value = value.float().cpu().numpy()
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("Qwen OV signature must be one-dimensional")
    if raw.size > max_dimension:
        raise MemoryError("Qwen OV signature exceeds its transport-dimension cap")
    if raw.size == 0:
        return None
    working = raw.astype(np.float64, copy=False)
    if not np.isfinite(working).all():
        return None
    scale = float(np.max(np.abs(working)))
    if not math.isfinite(scale) or scale == 0.0:
        return None
    scaled = working / scale
    scaled_norm = float(np.linalg.norm(scaled))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        return None
    if scale <= 1e-12 / scaled_norm:
        return None
    normalized = scaled / scaled_norm
    vector = normalized.astype(np.float32)
    post_norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
    if (
        not np.isfinite(vector).all()
        or not math.isfinite(post_norm)
        or not math.isclose(post_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    ):
        return None
    return vector


def _lossless_proxy_prefix(text: str, max_tokens: int) -> str:
    """Return a nonempty exact prefix for nonempty source text."""

    cap = exact_integer(max_tokens, "max_tokens", minimum=1)
    return truncate_to_tokens_lossless(str(text), cap)


def _qwen_linker_identity(linker: Any) -> dict[str, str | int | bool]:
    encoder = getattr(linker, "encoder", None)
    checkpoint = getattr(encoder, "checkpoint_identity", None)
    values: dict[str, str | int | bool] = {
        "model_id": str(getattr(checkpoint, "model_id", "")).strip(),
        "model_revision": str(
            getattr(checkpoint, "model_revision", "")
        ).strip(),
        "checkpoint_sha256": str(
            getattr(checkpoint, "checkpoint_sha256", "")
        ),
        "device": str(getattr(encoder, "device", "")).strip(),
        "dtype": str(getattr(encoder, "dtype_name", "")).strip(),
        "prefix_layers": exact_integer(
            getattr(encoder, "layers", -1),
            "encoder.layers",
            minimum=1,
        ),
        "attention_layer": exact_integer(
            getattr(linker, "layer", -1),
            "linker.layer",
            minimum=0,
        ),
        "head_vote_k": exact_integer(
            getattr(linker, "head_vote_k", -1),
            "linker.head_vote_k",
            minimum=1,
        ),
        "linker_implementation": (
            f"{type(linker).__module__}.{type(linker).__qualname__}"
        ),
        "owned_runtime_binding": _owned_qwen_runtime_binding(linker),
        "linker_max_candidates": exact_integer(
            getattr(linker, "max_candidates", -1),
            "linker.max_candidates",
            minimum=1,
        ),
        "linker_max_workspace_tokens": exact_integer(
            getattr(linker, "max_workspace_tokens", -1),
            "linker.max_workspace_tokens",
            minimum=1,
        ),
    }
    if any(
        not str(values[name]).strip()
        for name in ("model_id", "model_revision", "device", "dtype")
    ):
        raise ValueError("Qwen linker lacks complete checkpoint/runtime identity")
    if _SHA256_RE.fullmatch(str(values["checkpoint_sha256"])) is None:
        raise ValueError("Qwen linker checkpoint identity is not a SHA-256 digest")
    if int(values["attention_layer"]) >= int(values["prefix_layers"]):
        raise ValueError("Qwen linker attention layer lies outside its prefix")
    if getattr(linker, "cav_bank", None) is not None:
        raise ValueError("attention surprise requires the coverage linker without CAV")
    return values


def _owned_qwen_receipt_matches(
    scorer: Any,
    receipt: AttentionHeadSurpriseReceipt | None,
) -> bool:
    """Revalidate one receipt against the exact live owned scorer runtime."""

    if type(scorer) is not QwenAttentionHeadSurpriseScorer:
        return False
    if type(receipt) is not AttentionHeadSurpriseReceipt:
        return False
    try:
        identity = _qwen_linker_identity(scorer.linker)
        if identity.get("owned_runtime_binding") is not True:
            return False
        if any(getattr(receipt, name) != value for name, value in identity.items()):
            return False
        if receipt.implementation_sha256 != _attention_head_implementation_sha256(
            scorer.linker
        ):
            return False
        tokenizer_identity = _active_tokenizer_proxy_identity()
        if receipt.tokenizer_proxy_sha256 != identity_sha256(
            dict(tokenizer_identity())
        ):
            return False
        if receipt.neutral_probe_sha256 != quote_sha256(EPISODIC_SURPRISE_PROBE):
            return False
        return (
            receipt.max_input_spans == scorer.max_spans
            and receipt.span_token_cap == scorer.span_token_cap
            and receipt.probe_token_cap == scorer.probe_token_cap
            and receipt.max_transport_dimension == scorer.max_transport_dimension
            and receipt.retained_signal_transformer_state_bytes == 0
            and receipt.evidence_sequence_sha256 is not None
        )
    except Exception:
        # This is an attestation boundary: malformed or changing live state can
        # only remove the owned-runtime qualification, never create it.
        return False


def _owned_qwen_runtime_binding(linker: Any) -> bool:
    """Observe exact owned types, state shape, and unshadowed public method."""

    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder

    if type(linker) is not QwenMemoryLinker:
        return False
    encoder = getattr(linker, "encoder", None)
    if type(encoder) is not Qwen3PrefixEncoder:
        return False
    if set(vars(linker)) != {
        "encoder",
        "layer",
        "cav_bank",
        "max_candidates",
        "max_workspace_tokens",
        "max_neighbors_per_episode",
        "head_vote_k",
    }:
        return False
    if set(vars(encoder)) != {
        "model_dir",
        "layers",
        "model_id",
        "model_revision",
        "checkpoint_identity",
        "checkpoint_sha256",
        "_torch",
        "_apply_rotary_pos_emb",
        "device",
        "dtype",
        "dtype_name",
        "config",
        "model",
        "tokenizer",
        "loaded_parameter_names",
    }:
        return False
    inspection = getattr(linker, "inspect_coverage", None)
    function = getattr(inspection, "__func__", None)
    if not (
        getattr(inspection, "__self__", None) is linker
        and function is QwenMemoryLinker.inspect_coverage
    ):
        return False
    if (
        getattr(function, "__module__", None)
        != "memory_condense.associations.qwen_memory_linker"
        or getattr(function, "__qualname__", None)
        != "QwenMemoryLinker.inspect_coverage"
    ):
        return False
    function_source = inspect.getsourcefile(function)
    class_source = inspect.getsourcefile(QwenMemoryLinker)
    if function_source is None or class_source is None:
        return False
    return Path(function_source).resolve() == Path(class_source).resolve()


def _attention_head_implementation_sha256(linker: Any) -> str:
    """Hash every signal/receipt source plus live executed callables."""

    import memory_condense.associations.qwen_memory_linker as linker_module
    import memory_condense.domain._tokenizer as tokenizer_module
    import memory_condense.modeling.qwen_prefix as prefix_module

    root = Path(__file__).parent
    modules = (
        ("memory_condense.search.episodes.surprise", root / "surprise.py"),
        (
            "memory_condense.search.episodes.surprise_models",
            root / "surprise_models.py",
        ),
        (
            "memory_condense.search.episodes.surprise_controls",
            root / "surprise_controls.py",
        ),
        (__name__, Path(__file__)),
        (linker_module.__name__, Path(str(linker_module.__file__))),
        (prefix_module.__name__, Path(str(prefix_module.__file__))),
        (tokenizer_module.__name__, Path(str(tokenizer_module.__file__))),
    )
    digest = hashlib.sha256()
    for module_name, path in modules:
        if not path.is_file():
            raise RuntimeError(
                f"cannot identify attention-head implementation source: {module_name}"
            )
        name = module_name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    _update_callable_digest(digest, getattr(linker, "inspect_coverage", None))
    _update_callable_digest(digest, _active_transport_normalizer())
    _update_callable_digest(digest, _active_tokenizer_proxy_identity())
    return digest.hexdigest()


def _update_callable_digest(digest: Any, callable_value: Any) -> None:
    function = getattr(callable_value, "__func__", callable_value)
    code = getattr(function, "__code__", None)
    if code is None:
        raise RuntimeError("signal callable has no Python code identity")
    callable_name = (
        f"{getattr(function, '__module__', '')}."
        f"{getattr(function, '__qualname__', '')}"
    ).encode("utf-8")
    callable_code = _canonical_callable_code(
        code,
        stable_filename=callable_name.decode("utf-8"),
    )
    digest.update(len(callable_name).to_bytes(4, "big"))
    digest.update(callable_name)
    digest.update(len(callable_code).to_bytes(8, "big"))
    digest.update(callable_code)


def _canonical_callable_code(
    code: CodeType,
    *,
    stable_filename: str,
) -> bytes:
    """Marshal live code without checkout-specific absolute filenames."""

    constants = tuple(
        (
            _canonical_code_object(value, stable_filename=stable_filename)
            if isinstance(value, CodeType)
            else value
        )
        for value in code.co_consts
    )
    normalized = code.replace(
        co_filename=str(stable_filename),
        co_consts=constants,
    )
    return marshal.dumps(normalized)


def _canonical_code_object(
    code: CodeType,
    *,
    stable_filename: str,
) -> CodeType:
    constants = tuple(
        (
            _canonical_code_object(value, stable_filename=stable_filename)
            if isinstance(value, CodeType)
            else value
        )
        for value in code.co_consts
    )
    return code.replace(
        co_filename=str(stable_filename),
        co_consts=constants,
    )


__all__ = ["QwenAttentionHeadSurpriseScorer"]
