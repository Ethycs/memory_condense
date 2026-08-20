from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
    make_atom_id,
    quote_sha256,
)
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3PrefixEncoder,
    QwenPrefixCheckpointIdentity,
    _qwen_prefix_execution_gate,
)
from memory_condense.search.fusion import FusionCaps
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomFeatureCaps,
    QwenAtomFeatureProviderReceipt,
)
import memory_condense.search.fusion.qwen_features as qwen_features


class FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"
    pad_token_id = 0
    eos_token_id = 510
    bos_token_id = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __len__(self) -> int:
        return 512

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.calls.append((text, dict(kwargs)))
        token_ids = [ord(character) % 500 + 1 for character in text]
        if kwargs.get("truncation"):
            token_ids = token_ids[: int(kwargs["max_length"])]
        return {"input_ids": token_ids}


class FakeModel:
    def modules(self):
        return (self,)

    def forward(self):
        return None


class FakeConfig:
    pass


class FakeTorch:
    __version__ = "fake-provider-free"


class FakeLeaseTensor:
    pass


class FakeLeaseTorch:
    Tensor = FakeLeaseTensor


class _Device:
    def __init__(self, value: str) -> None:
        self.value = value
        self.type, _, index = value.partition(":")
        self.index = int(index) if index else None

    def __str__(self) -> str:
        return self.value


class FakeTensor:
    forbidden_host_calls = 0

    def __init__(
        self,
        data: list[list[float]],
        *,
        device: str = "cuda:0",
        dtype: str = "torch.float32",
        requires_grad: bool = False,
        grad_fn=None,
        parent=None,
        row_slice: slice | None = None,
    ) -> None:
        self.data = data
        self.device = _Device(device)
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self._parent = parent
        self._row_slice = row_slice

    @property
    def shape(self) -> tuple[int, int]:
        width = len(self.data[0]) if self.data else 0
        return (len(self.data), width)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("fake tensor supports row slices only")
        return FakeTensor(
            self.data[item],
            device=str(self.device),
            dtype=self.dtype,
            parent=self,
            row_slice=item,
        )

    def copy_(self, other) -> None:
        copied = [list(row) for row in other.data]
        if self._parent is None or self._row_slice is None:
            self.data[:] = copied
        else:
            self._parent.data[self._row_slice] = copied
            self.data = self._parent.data[self._row_slice]

    def cpu(self):
        type(self).forbidden_host_calls += 1
        raise AssertionError("full feature CPU transfer is forbidden")

    def numpy(self):
        type(self).forbidden_host_calls += 1
        raise AssertionError("full feature NumPy conversion is forbidden")

    def tolist(self):
        type(self).forbidden_host_calls += 1
        raise AssertionError("full feature tolist conversion is forbidden")


class _FakeScalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self):
        return self

    def item(self) -> bool:
        return self.value


class FakeTensorTorch:
    Tensor = FakeTensor

    @staticmethod
    def empty(shape, *, device, dtype):
        rows, width = shape
        return FakeTensor(
            [[0.0] * width for _ in range(rows)],
            device=device,
            dtype=str(dtype),
        )

    @staticmethod
    def isfinite(tensor):
        import math

        return _FakeScalar(all(math.isfinite(value) for row in tensor.data for value in row))

    @staticmethod
    def allclose(left, right, *, atol, rtol):
        import math

        return all(
            math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
            for left_row, right_row in zip(left.data, right.data, strict=True)
            for a, b in zip(left_row, right_row, strict=True)
        )


class FakeFeatureEncoder:
    dtype = "torch.float32"

    def __init__(self, *, batch_bias: float = 0.0, fault: str | None = None) -> None:
        self.calls: list[tuple[tuple[int, ...], ...]] = []
        self.batch_bias = batch_bias
        self.fault = fault

    def _encode_selected_layer_final_readout(self, rows, *, layer, _gate_token):
        del layer, _gate_token
        exact_rows = tuple(tuple(row) for row in rows)
        self.calls.append(exact_rows)
        result = FakeTensor(
            [
                [float(sum(row)) + self.batch_bias * len(exact_rows), float(len(row)), float(row[-1]), 1.0]
                for row in exact_rows
            ]
        )
        if len(self.calls) == 1:
            if self.fault == "foreign":
                return object()
            if self.fault == "rows":
                result.data = result.data[:-1]
            elif self.fault == "width":
                result.data[0].append(0.0)
            elif self.fault == "device":
                result.device = _Device("cuda:1")
            elif self.fault == "dtype":
                result.dtype = "torch.float16"
            elif self.fault == "nan":
                result.data[0][0] = float("nan")
            elif self.fault == "grad":
                result.requires_grad = True
            elif self.fault == "grad_fn":
                result.grad_fn = object()
        return result


class PrimitiveTensor:
    def __init__(self, data, *, dtype="torch.int64", device="cpu") -> None:
        self.data = data
        self.dtype = dtype
        self.device = _Device(device)
        self._base = None

    @property
    def shape(self):
        def dimensions(value):
            if not isinstance(value, list):
                return ()
            return (len(value), *dimensions(value[0])) if value else (0,)

        return dimensions(self.data)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            row, column = key
            exact = value.data if isinstance(value, PrimitiveTensor) else value
            if isinstance(column, slice):
                if isinstance(exact, list):
                    self.data[row][column] = exact
                else:
                    start, stop, step = column.indices(len(self.data[row]))
                    for index in range(start, stop, step):
                        self.data[row][index] = exact
                return
        self.data[key] = value.data if isinstance(value, PrimitiveTensor) else value

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            batch_indices, readout_indices, _columns = key
            return PrimitiveTensor(
                [
                    self.data[batch][readout_indices.data[offset]]
                    for offset, batch in enumerate(batch_indices.data)
                ],
                dtype=self.dtype,
                device=str(self.device),
            )
        raise TypeError("unsupported primitive tensor index")

    def to(self, device):
        self.device = _Device(str(device))
        return self

    def clone(self):
        import copy

        return PrimitiveTensor(copy.deepcopy(self.data), dtype=self.dtype, device=str(self.device))


class _PrimitiveContext:
    def __init__(self, calls: list[tuple[str, object]], name: str, detail: object) -> None:
        self.calls = calls
        self.name = name
        self.detail = detail

    def __enter__(self):
        self.calls.append((self.name, self.detail))

    def __exit__(self, *_args):
        return False


class PrimitiveTorch:
    Tensor = PrimitiveTensor
    long = "torch.int64"

    def __init__(self) -> None:
        self.context_calls: list[tuple[str, object]] = []

    def full(self, shape, value, *, dtype, device):
        rows, width = shape
        return PrimitiveTensor([[value] * width for _ in range(rows)], dtype=dtype, device=device)

    def zeros(self, shape, *, dtype, device):
        rows, width = shape
        return PrimitiveTensor([[0] * width for _ in range(rows)], dtype=dtype, device=device)

    def empty(self, size, *, dtype, device):
        return PrimitiveTensor([0] * size, dtype=dtype, device=device)

    def tensor(self, values, *, dtype, device):
        return PrimitiveTensor(list(values), dtype=dtype, device=device)

    def arange(self, size, *, dtype, device):
        return PrimitiveTensor(list(range(size)), dtype=dtype, device=str(device))

    def inference_mode(self):
        return _PrimitiveContext(self.context_calls, "inference", True)

    def autocast(self, *, device_type, enabled):
        return _PrimitiveContext(self.context_calls, "autocast", (device_type, enabled))


class PrimitiveHandle:
    def __init__(self, layer) -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.hook = None
        self.layer.removed += 1


class PrimitiveLayer:
    def __init__(self) -> None:
        self.hook = None
        self.removed = 0

    def register_forward_hook(self, hook):
        self.hook = hook
        return PrimitiveHandle(self)


class PrimitiveModel:
    def __init__(self, layer: PrimitiveLayer, *, fail: bool = False) -> None:
        self.layers = (layer,)
        self.embed_tokens = SimpleNamespace(num_embeddings=128)
        self.fail = fail
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("primitive model failure")
        input_ids = kwargs["input_ids"]
        residual = PrimitiveTensor(
            [
                [[token * 10.0, token * 10.0 + 1.0] for token in row]
                for row in input_ids.data
            ],
            dtype="torch.float32",
            device=str(input_ids.device),
        )
        self.layers[0].hook(self.layers[0], (), residual)
        return SimpleNamespace()


def _atom(index: int, text: str) -> EvidenceAtom:
    span = EvidenceSpan(
        chunk_id=f"qwen-feature-chunk-{index}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=index,
        source_id="qwen-feature-fixture",
        turn_id=f"turn-{index}",
        role="user",
    )
    return EvidenceAtom(
        atom_id=make_atom_id(span),
        span=span,
        text=text,
        label=f"fixture-{index}",
    )


def _packet_and_plan(
    *,
    query: str = "Where is the relation?",
    texts: tuple[str, ...] = (
        "Alpha evidence remains exact and intentionally longer.",
        "Beta evidence shares the selected bundle.",
        "Gamma evidence completes packet order.",
    ),
) -> tuple[EvidencePacket, ClosurePlan]:
    atoms = tuple(_atom(index + 1, text) for index, text in enumerate(texts))
    bundles = (
        EvidenceBundle(
            bundle_id="bundle-qwen-feature",
            atom_ids=tuple(atom.atom_id for atom in atoms),
            obligation_ids=("answer",),
            unit_ids=("unit-qwen-feature",),
            relation_ids=("relation-qwen-feature",),
            required=True,
            utility=1.0,
        ),
    )
    program = QueryProgram(
        query=query,
        intent="relate",
        subject_terms=("relation",),
        obligations=(
            EvidenceObligation(
                obligation_id="answer",
                kind="answer_fact",
                required=True,
                weight=1.0,
            ),
        ),
    )
    plan = ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(max_bundles=8, beam_width=16),
        snapshot=DiscourseSnapshot(
            max_turn_ordinal=len(atoms),
            chunk_count=len(atoms),
            graph_revision=1,
            schema_version=1,
            artifact_ids=("qwen-feature-artifact",),
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        ),
        seeds=(),
        atoms=atoms,
        bundles=bundles,
        obligation_results=(
            ObligationResult(
                obligation_id="answer",
                status="satisfied",
                unit_ids=("unit-qwen-feature",),
                relation_ids=("relation-qwen-feature",),
                bundle_ids=("bundle-qwen-feature",),
            ),
        ),
        visited_episode_ids=(),
        visited_unit_ids=("unit-qwen-feature",),
        visited_relation_ids=("relation-qwen-feature",),
        stopping_reason="complete",
        complete_claimed=True,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="fixture_scope",
                subject_id="qwen-feature-fixture",
                requested_limit=len(atoms),
                returned_count=len(atoms),
                exhaustive=True,
            ),
        ),
        artifact_id="qwen-feature-artifact",
    )
    context = "qwen feature packed context"
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=quote_sha256(context),
        selected_bundle_ids=tuple(bundle.bundle_id for bundle in bundles),
        selected_atom_ids=tuple(atom.atom_id for atom in atoms),
        dropped_bundle_reasons={},
        context_token_proxy=4,
        max_context_token_proxy=32,
        tokenizer_identity="fixture-tokenizer",
        stopping_reason="complete",
        complete_claimed=True,
    )
    return EvidencePacket(context, atoms, bundles, receipt), plan


def _fusion_caps(**updates: int) -> FusionCaps:
    values = {
        "max_atoms": 3,
        "max_hidden_dim": 4,
        "max_hyperedges": 2,
        "max_topology_links": 3,
    }
    values.update(updates)
    return FusionCaps(**values)


def _feature_caps(**updates: object) -> QwenAtomFeatureCaps:
    values: dict[str, object] = {
        "max_row_tokens": 64,
        "max_query_tail_tokens": 48,
        "max_rows_per_forward": 2,
        "max_workspace_tokens": 128,
        "max_evidence_characters": 256,
        "max_query_characters": 128,
    }
    values.update(updates)
    return QwenAtomFeatureCaps(**values)


def _provider_receipt() -> QwenAtomFeatureProviderReceipt:
    return QwenAtomFeatureProviderReceipt(
        implementation_sha256="b" * 64,
        model_id="qwen-fixture",
        model_revision="fixture-revision",
        checkpoint_sha256="a" * 64,
        verified_files_sha256="c" * 64,
        tokenizer_identity_sha256="d" * 64,
        retained_layers=2,
        output_layer=1,
        hidden_dim=4,
        device="cuda:0",
        execution_dtype="torch.float32",
        prompt_template_sha256="e" * 64,
    )


def _fake_encoder(tokenizer: FakeTokenizer) -> Qwen3PrefixEncoder:
    encoder = object.__new__(Qwen3PrefixEncoder)
    checkpoint = QwenPrefixCheckpointIdentity(
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        checkpoint_sha256="a" * 64,
        verified_files=("config.json",),
    )
    values = {
        "model_dir": Path("fake-qwen-prefix"),
        "layers": 2,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_identity": checkpoint,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "_torch": FakeTorch,
        "_apply_rotary_pos_emb": lambda *args: args,
        "device": SimpleNamespace(type="cuda", index=0),
        "dtype": "fake-float32",
        "dtype_name": "float32",
        "config": FakeConfig(),
        "model": FakeModel(),
        "tokenizer": tokenizer,
        "loaded_parameter_names": frozenset(),
    }
    for name, value in values.items():
        setattr(encoder, name, value)
    return encoder


def test_feature_caps_are_sealed_and_allow_dynamic_workspace_splitting() -> None:
    caps = QwenAtomFeatureCaps(
        max_row_tokens=128,
        max_query_tail_tokens=64,
        max_rows_per_forward=4,
        max_workspace_tokens=256,
    )

    assert len(caps.caps_sha256) == 64
    assert caps.max_evidence_characters == 4096
    assert caps.max_query_characters == 2048
    with pytest.raises(ValueError, match="below max_row_tokens"):
        QwenAtomFeatureCaps(max_row_tokens=64, max_query_tail_tokens=64)
    with pytest.raises(ValueError, match="one maximum-length row"):
        QwenAtomFeatureCaps(max_row_tokens=513, max_workspace_tokens=512)
    with pytest.raises(ValueError, match="finite non-negative"):
        QwenAtomFeatureCaps(batch_invariance_atol=float("nan"))
    for invalid in ("0.001", 0, True, float("inf"), -0.001):
        with pytest.raises(ValueError, match="finite non-negative"):
            QwenAtomFeatureCaps(batch_invariance_rtol=invalid)


def test_row_builder_preserves_query_readout_and_truncates_only_evidence_prefix() -> None:
    packet, plan = _packet_and_plan()
    tokenizer = FakeTokenizer()
    caps = _feature_caps()

    rows = qwen_features._build_qwen_atom_rows(
        tokenizer,
        packet.atoms,
        plan.query_program.query,
        caps,
    )

    prefix = tuple(ord(char) % 500 + 1 for char in "[Evidence]\n")
    tail_text = f"\n[Question]\n{plan.query_program.query}\n[Readout]"
    tail = tuple(ord(char) % 500 + 1 for char in tail_text)
    budget = caps.max_row_tokens - len(prefix) - len(tail)
    assert len(rows) == len(packet.atoms)
    for index, row in enumerate(rows):
        evidence = tuple(ord(char) % 500 + 1 for char in packet.atoms[index].text)
        assert row.token_ids == (*prefix, *evidence[:budget], *tail)
        assert row.receipt.readout_end_index == len(row.token_ids) - 1
        assert row.receipt.evidence_tokens_admitted == budget
        assert row.receipt.evidence_tokens_observed == budget + 1
        assert row.receipt.evidence_truncated is True
    assert all(call[1]["add_special_tokens"] is False for call in tokenizer.calls)
    bounded = [kwargs for _text, kwargs in tokenizer.calls if kwargs.get("truncation")]
    assert bounded[0]["max_length"] == caps.max_query_tail_tokens + 1
    assert all(int(kwargs["max_length"]) <= caps.max_row_tokens for kwargs in bounded)


@pytest.mark.parametrize(
    "query,texts,match",
    [
        ("q" * 9, ("short",), "max_query_characters"),
        ("short", ("e" * 9,), "max_evidence_characters"),
    ],
)
def test_raw_character_caps_reject_before_tokenizer(
    query: str,
    texts: tuple[str, ...],
    match: str,
) -> None:
    packet, plan = _packet_and_plan(query=query, texts=texts)
    tokenizer = FakeTokenizer()
    caps = _feature_caps(max_query_characters=8, max_evidence_characters=8)

    with pytest.raises(MemoryError, match=match):
        qwen_features._build_qwen_atom_rows(tokenizer, packet.atoms, query, caps)

    assert tokenizer.calls == []


def test_real_fake_executor_and_receipt_are_exhaustive_resident_and_text_free() -> None:
    packet, plan = _packet_and_plan()
    caps = _fusion_caps()
    feature_caps = _feature_caps(
        batch_invariance_atol=0.0,
        batch_invariance_rtol=0.0,
    )
    provider_receipt = _provider_receipt()
    tokenizer = FakeTokenizer()
    qwen_features._preflight_packet(
        packet,
        plan,
        caps,
        feature_caps,
        hidden_dim=provider_receipt.hidden_dim,
    )
    qwen_features._validate_packet_plan(packet, plan)
    inputs = qwen_features._capture_operation_inputs(packet, plan, caps, feature_caps)
    rows = qwen_features._build_qwen_atom_rows(
        tokenizer,
        inputs.atom_values,
        inputs.query,
        inputs.feature_caps,
    )
    batches = qwen_features._batch_rows(rows, inputs.feature_caps)
    encoder = FakeFeatureEncoder()
    FakeTensor.forbidden_host_calls = 0
    execution = qwen_features._execute_feature_batches(
        encoder=encoder,
        torch=FakeTensorTorch,
        output_layer=1,
        provider_receipt=provider_receipt,
        feature_caps=inputs.feature_caps,
        rows=rows,
        batches=batches,
        gate_token=object(),
    )
    assert execution.primary_forward_count == 2
    assert execution.batch_invariance_forward_count == 0
    assert len(encoder.calls) == 2
    assert encoder.calls == [
        tuple(row.token_ids for row in rows[:2]),
        tuple(row.token_ids for row in rows[2:]),
    ]
    assert FakeTensor.forbidden_host_calls == 0
    encoded = canonical_json(
        {
            "provider": provider_receipt.identity_payload(),
            "rows": [row.receipt.identity_payload() for row in rows],
            "batches": list(batches),
        }
    )
    assert plan.query_program.query not in encoded
    assert all(atom.text not in encoded for atom in packet.atoms)
    assert provider_receipt.exclusive_synchronous_ownership_verified is False
    assert provider_receipt.loaded_parameter_content_attested is False
    assert provider_receipt.loaded_tokenizer_content_attested is False
    assert provider_receipt.general_concurrency_safe is False


def test_input_validation_rejects_packet_tampering_before_tokenization() -> None:
    packet, plan = _packet_and_plan()
    tokenizer = FakeTokenizer()
    object.__setattr__(plan.query_program, "query", "tampered query")

    with pytest.raises(ValueError, match="query program SHA-256"):
        qwen_features._validate_packet_plan(packet, plan)

    assert tokenizer.calls == []


def test_preflight_rejects_unsealed_caps_and_foreign_query_before_nested_access() -> None:
    packet, plan = _packet_and_plan()
    caps = _fusion_caps()
    feature_caps = _feature_caps()
    object.__setattr__(feature_caps, "max_query_characters", 10_000)
    with pytest.raises(ValueError, match="caps SHA-256"):
        qwen_features._preflight_packet(
            packet, plan, caps, feature_caps, hidden_dim=4
        )

    class ForeignQuery:
        @property
        def query(self):
            raise AssertionError("foreign query property must not execute")

    packet, plan = _packet_and_plan()
    object.__setattr__(plan, "query_program", ForeignQuery())
    with pytest.raises(TypeError, match="exact QueryProgram"):
        qwen_features._preflight_packet(
            packet, plan, caps, _feature_caps(), hidden_dim=4
        )


def test_provider_free_partition_diagnostic_rejects_batch_dependent_outputs() -> None:
    packet, plan = _packet_and_plan()
    tokenizer = FakeTokenizer()
    feature_caps = _feature_caps(
        batch_invariance_atol=0.0,
        batch_invariance_rtol=0.0,
    )
    rows = qwen_features._build_qwen_atom_rows(
        tokenizer,
        packet.atoms,
        plan.query_program.query,
        feature_caps,
    )
    encoder = FakeFeatureEncoder(batch_bias=1.0)
    mixed = encoder._encode_selected_layer_final_readout(
        tuple(row.token_ids for row in rows[:2]), layer=1, _gate_token=object()
    )
    singleton = encoder._encode_selected_layer_final_readout(
        (rows[0].token_ids,), layer=1, _gate_token=object()
    )
    assert not FakeTensorTorch.allclose(
        mixed[0:1],
        singleton,
        atol=feature_caps.batch_invariance_atol,
        rtol=feature_caps.batch_invariance_rtol,
    )


def test_provider_free_partition_diagnostic_is_exact_for_row_local_fake() -> None:
    packet, plan = _packet_and_plan()
    feature_caps = _feature_caps(
        batch_invariance_atol=0.0,
        batch_invariance_rtol=0.0,
    )
    rows = qwen_features._build_qwen_atom_rows(
        FakeTokenizer(), packet.atoms, plan.query_program.query, feature_caps
    )
    encoder = FakeFeatureEncoder()
    mixed = encoder._encode_selected_layer_final_readout(
        tuple(row.token_ids for row in rows[:2]), layer=1, _gate_token=object()
    )
    singleton = encoder._encode_selected_layer_final_readout(
        (rows[0].token_ids,), layer=1, _gate_token=object()
    )
    paired = encoder._encode_selected_layer_final_readout(
        (rows[0].token_ids, rows[0].token_ids), layer=1, _gate_token=object()
    )
    assert FakeTensorTorch.allclose(mixed[0:1], singleton, atol=0.0, rtol=0.0)
    assert FakeTensorTorch.allclose(singleton, paired[0:1], atol=0.0, rtol=0.0)
    assert FakeTensorTorch.allclose(singleton, paired[1:2], atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("foreign", "foreign tensor type"),
        ("rows", "wrong shape"),
        ("width", "wrong shape"),
        ("device", "left the provider CUDA device"),
        ("dtype", "changed execution dtype"),
        ("nan", "non-finite"),
        ("grad", "autograd graph"),
        ("grad_fn", "autograd graph"),
    ],
)
def test_real_fake_executor_rejects_malformed_outputs(
    fault: str,
    match: str,
) -> None:
    packet, plan = _packet_and_plan()
    feature_caps = _feature_caps()
    rows = qwen_features._build_qwen_atom_rows(
        FakeTokenizer(), packet.atoms, plan.query_program.query, feature_caps
    )
    batches = qwen_features._batch_rows(rows, feature_caps)
    FakeTensor.forbidden_host_calls = 0

    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        qwen_features._execute_feature_batches(
            encoder=FakeFeatureEncoder(fault=fault),
            torch=FakeTensorTorch,
            output_layer=1,
            provider_receipt=_provider_receipt(),
            feature_caps=feature_caps,
            rows=rows,
            batches=batches,
            gate_token=object(),
        )

    assert FakeTensor.forbidden_host_calls == 0


def test_feature_lease_is_discard_only_single_use_and_not_copyable() -> None:
    import copy
    import pickle

    lease = qwen_features._QwenFeatureLease(FakeLeaseTensor(), FakeLeaseTorch)

    with pytest.raises(TypeError, match="exact discard"):
        lease._discard_once(lambda _value: None)
    assert lease.consumed is False
    with pytest.raises(TypeError, match="copied"):
        copy.copy(lease)
    with pytest.raises(TypeError, match="deep-copied"):
        copy.deepcopy(lease)
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(lease)
    lease._discard_once(qwen_features._DiscardFeatures())
    assert lease.closed and lease.consumed
    with pytest.raises(RuntimeError, match="already consumed"):
        lease._discard_once(qwen_features._DiscardFeatures())


def test_provider_gate_is_non_reentrant_and_cleans_up_after_exception() -> None:
    encoder = _fake_encoder(FakeTokenizer())

    with _qwen_prefix_execution_gate(encoder):
        with pytest.raises(RuntimeError, match="already has an active"):
            with _qwen_prefix_execution_gate(encoder):
                pass
    with pytest.raises(RuntimeError, match="boom"):
        with _qwen_prefix_execution_gate(encoder):
            raise RuntimeError("boom")
    with _qwen_prefix_execution_gate(encoder):
        pass


def test_provider_rejects_subclasses_and_preconstruction_helper_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="does not support subclassing"):
        class EvilProvider(qwen_features.QwenAtomFeatureProvider):
            pass

    monkeypatch.setattr(qwen_features, "_execute_feature_batches", lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match="owned Qwen feature implementation"):
        qwen_features.QwenAtomFeatureProvider(
            _fake_encoder(FakeTokenizer()),
            output_layer=1,
        )


def test_provider_rejects_stable_prefix_gate_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen_features.prefix_module,
        "_qwen_prefix_execution_gate",
        lambda _encoder: None,
    )
    with pytest.raises(RuntimeError, match="prefix execution seams"):
        qwen_features.QwenAtomFeatureProvider(
            _fake_encoder(FakeTokenizer()),
            output_layer=1,
        )


def test_provider_rejects_stable_prefix_type_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForeignEncoderType:
        _encode_selected_layer_final_readout = (
            qwen_features._PINNED_FINAL_READOUT_PRIMITIVE
        )

    monkeypatch.setattr(
        qwen_features.prefix_module,
        "Qwen3PrefixEncoder",
        ForeignEncoderType,
    )
    with pytest.raises(RuntimeError, match="prefix execution seams"):
        qwen_features.QwenAtomFeatureProvider(
            _fake_encoder(FakeTokenizer()),
            output_layer=1,
        )


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("_require_torch_stack", lambda: ()),
        ("expected_prefix_checkpoint_sha256", lambda *_args, **_kwargs: "0" * 64),
        ("_known_required_shards", lambda _layers: ("foreign.safetensors",)),
        ("_PREFIX_METADATA_FILES", ("foreign.json",)),
    ],
)
def test_provider_rejects_stable_prefix_authority_injection(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(qwen_features.prefix_module, name, replacement)
    with pytest.raises(RuntimeError, match="prefix execution seams"):
        qwen_features.QwenAtomFeatureProvider(
            _fake_encoder(FakeTokenizer()),
            output_layer=1,
        )


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("_PINNED_CHECKPOINT_SHA256_BY_LAYER", ("0" * 64,) * 36),
        ("_PINNED_VERIFIED_FILES_BY_LAYER", (("foreign.json",),) * 36),
    ],
)
def test_provider_rejects_pinned_manifest_table_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(qwen_features, name, replacement)
    with pytest.raises(RuntimeError, match="owned Qwen feature implementation"):
        qwen_features.QwenAtomFeatureProvider(
            _fake_encoder(FakeTokenizer()),
            output_layer=1,
        )


def test_owned_implementation_getter_ignores_live_container_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qwen_features._pinned_owned_implementation()
    monkeypatch.setattr(
        qwen_features,
        "_PINNED_OWNED_IMPLEMENTATION",
        qwen_features._OwnedImplementation(
            **{
                **{
                    name: getattr(original, name)
                    for name in original.__dataclass_fields__
                },
                "execute_feature_batches": lambda **_kwargs: None,
            }
        ),
    )
    assert qwen_features._pinned_owned_implementation() is original


def test_final_readout_primitive_requires_active_gate_and_bounds_vocabulary() -> None:
    encoder = object.__new__(Qwen3PrefixEncoder)
    encoder.device = SimpleNamespace(type="cuda", index=0)
    encoder.layers = 1
    encoder.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(num_embeddings=8),
    )

    with pytest.raises(RuntimeError, match="requires its active provider gate"):
        encoder._encode_selected_layer_final_readout(((1,),), layer=0, _gate_token=object())
    with _qwen_prefix_execution_gate(encoder) as token:
        with pytest.raises(ValueError, match="outside the embedding vocabulary"):
            encoder._encode_selected_layer_final_readout(
                ((8,),),
                layer=0,
                _gate_token=token,
            )


@pytest.mark.parametrize("fail", [False, True])
def test_final_readout_primitive_gathers_last_valid_token_and_cleans_hook(fail: bool) -> None:
    encoder = object.__new__(Qwen3PrefixEncoder)
    layer = PrimitiveLayer()
    model = PrimitiveModel(layer, fail=fail)
    primitive_torch = PrimitiveTorch()
    encoder.device = _Device("cuda:0")
    encoder.layers = 1
    encoder.model = model
    encoder.tokenizer = SimpleNamespace(pad_token_id=0)
    encoder.config = SimpleNamespace(hidden_size=2)
    encoder._torch = primitive_torch

    with _qwen_prefix_execution_gate(encoder) as token:
        if fail:
            with pytest.raises(RuntimeError, match="primitive model failure"):
                encoder._encode_selected_layer_final_readout(
                    ((1, 2, 3), (4,)), layer=0, _gate_token=token
                )
        else:
            features = encoder._encode_selected_layer_final_readout(
                ((1, 2, 3), (4,)), layer=0, _gate_token=token
            )
            assert features.data == [[30.0, 31.0], [40.0, 41.0]]
            assert model.kwargs["input_ids"].data == [[1, 2, 3], [4, 0, 0]]
            assert model.kwargs["attention_mask"].data == [[1, 1, 1], [1, 0, 0]]
            assert model.kwargs["use_cache"] is False
            assert model.kwargs["output_attentions"] is False
            assert model.kwargs["output_hidden_states"] is False
            assert ("inference", True) in primitive_torch.context_calls
            assert ("autocast", ("cuda", False)) in primitive_torch.context_calls
    assert layer.hook is None
    assert layer.removed == 1
