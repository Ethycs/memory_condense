from memory_condense.eval.local_qwen import (
    resolve_local_qwen_dtype,
    strip_qwen_thinking,
)


class _FakeCuda:
    def __init__(self, *, available: bool, major: int = 0) -> None:
        self.available = available
        self.major = major
        self.requested_device = None

    def is_available(self) -> bool:
        return self.available

    def get_device_capability(self, device=None):
        self.requested_device = device
        return self.major, 0


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    def __init__(self, *, available: bool, major: int = 0) -> None:
        self.cuda = _FakeCuda(available=available, major=major)


def test_strip_qwen_thinking_keeps_only_visible_answer() -> None:
    assert strip_qwen_thinking("<think>private work</think>\nBoston") == "Boston"


def test_strip_qwen_thinking_leaves_plain_answer_unchanged() -> None:
    assert strip_qwen_thinking("  SQLite  ") == "SQLite"


def test_auto_dtype_uses_native_fp16_on_pre_ampere_cuda() -> None:
    dtype, name = resolve_local_qwen_dtype(
        _FakeTorch(available=True, major=7),
        "auto",
    )

    assert (dtype, name) == ("fp16", "float16")


def test_auto_dtype_keeps_bfloat16_on_ampere_and_cpu() -> None:
    assert resolve_local_qwen_dtype(
        _FakeTorch(available=True, major=8),
        "auto",
    ) == ("bf16", "bfloat16")


def test_auto_dtype_honors_explicit_target_device() -> None:
    torch = _FakeTorch(available=True, major=7)

    assert resolve_local_qwen_dtype(
        torch,
        "auto",
        device="cpu",
    ) == ("bf16", "bfloat16")
    assert resolve_local_qwen_dtype(
        torch,
        "auto",
        device="cuda:1",
    ) == ("fp16", "float16")
    assert torch.cuda.requested_device == "cuda:1"
    assert resolve_local_qwen_dtype(
        _FakeTorch(available=False),
        "auto",
    ) == ("bf16", "bfloat16")
