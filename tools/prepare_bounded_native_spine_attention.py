"""Authenticate reused exchange outputs before the unchanged attention cache stage."""
import argparse
from pathlib import Path

from tools import compile_native_spine_attention as attention
from tools import compile_bounded_native_spine_exchanges as exchanges
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = "native-spine-bounded-attention-admission-v1"


def implementation():
    return {**attention.implementation(), **exchanges.implementation(),
            "tools/prepare_bounded_native_spine_attention.py": digest(__file__)}


def forbidden(*args, **kwargs):
    raise AssertionError("attention preparation cannot load the full Qwen model or generate summaries")


def prepare(exchange_root, root, cache_root, backend):
    backend.generate = backend.load = forbidden
    with _phase_lock(root, "native-reused-attention-admission"):
        plan = read_sealed_json(exchange_root/"preflight.json")
        if (plan.payload.get("producer_format") != exchanges.FORMAT
                or plan.payload["producer_implementation"] != exchanges.implementation()):
            raise ValueError("attention admission requires authenticated reused exchanges")
        result = exchanges.execute(exchange_root, backend, 0)
        if result.payload["complete_available_body_exchanges"] is not True:
            raise ValueError("expanded attention requires the complete prepared exchange population")
        preflight = attention.prepare(exchange_root, root, cache_root)
        admission, _ = publish_sealed_json(root/"producer-admission.json", {
            "format": FORMAT, "exchange_root": str(exchange_root.resolve()),
            "exchange_preflight_sha256": plan.sha256, "exchange_result_sha256": result.sha256,
            "attention_preflight_sha256": preflight.sha256, "implementation": implementation(),
            "new_generation_calls": 0, "raw_inputs_to_qwen": False,
            "existing_attention_method_unchanged": True,
        })
        print({"expanded_attention_admission_sha256": admission.sha256,
               "attention_preflight_sha256": preflight.sha256}, flush=True)
        return preflight


def validate_admission(root):
    admission = read_sealed_json(root/"producer-admission.json")
    p = admission.payload
    preflight = read_sealed_json(root/"preflight.json")
    source = Path(p["exchange_root"])
    plan = read_sealed_json(source/"preflight.json")
    result = read_sealed_json(source/"result.json")
    if (p["format"] != FORMAT or p["implementation"] != implementation()
            or p["raw_inputs_to_qwen"] is not False or p["new_generation_calls"] != 0
            or p["attention_preflight_sha256"] != preflight.sha256
            or p["exchange_preflight_sha256"] != plan.sha256
            or p["exchange_result_sha256"] != result.sha256
            or preflight.payload["exchange_result_sha256"] != result.sha256
            or result.payload["preflight_sha256"] != plan.sha256
            or result.payload["complete_available_body_exchanges"] is not True
            or plan.payload.get("producer_format") != exchanges.FORMAT
            or plan.payload["producer_implementation"] != exchanges.implementation()):
        raise ValueError("reused exchange attention admission changed")
    return admission


def execute(root):
    validate_admission(root)
    return attention.execute(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
            Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
        prepare(args.exchange_root, args.root, args.cache_root, backend)
    else:
        execute(args.root)
