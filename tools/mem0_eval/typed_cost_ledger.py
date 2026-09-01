"""Sealed cost-ledger schemas for the fair ``mem0-typed-v1`` epoch.

These records measure the complete write, read, and common-final paths.  They
are data contracts only: constructing or validating one never invokes Mem0,
an embedding model, a responder, or a judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)

from .prompt_pack import MEM0_TYPED_EPOCH


WRITE_LEDGER_FORMAT = "memory-condense-mem0-typed-write-cost-ledger-v1"
READ_LEDGER_FORMAT = "memory-condense-mem0-typed-read-cost-ledger-v1"
PROVIDER_STAGE_FORMAT = "memory-condense-common-provider-stage-cost-v1"
FINAL_LEDGER_FORMAT = "memory-condense-common-final-cost-ledger-v1"
EPOCH_LEDGER_FORMAT = "memory-condense-mem0-typed-epoch-cost-ledger-v1"
HARD_REQUEST_TOKEN_CAP = 8_000


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedEvalContractError(f"{label} must be a non-negative integer")
    return value


def _optional_count(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _count(value, label)


def _latency(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise MatchedEvalContractError(f"{label} must be finite and non-negative")
    return float(value)


def _closed_calls(attempted: int, completed: int, failed: int, label: str) -> None:
    for value, suffix in (
        (attempted, "attempted"),
        (completed, "completed"),
        (failed, "failed"),
    ):
        _count(value, f"{label} {suffix}")
    if attempted != completed + failed:
        raise MatchedEvalContractError(f"sealed {label} calls do not close")


@dataclass(frozen=True, slots=True)
class Mem0WriteCostLedger:
    population_identity_sha256: str
    add_attempted: int
    add_completed: int
    add_failed: int
    extraction_attempted: int
    extraction_completed: int
    extraction_failed: int
    extraction_raw_message_token_proxy: int
    extraction_provider_input_tokens: int | None
    extraction_provider_output_tokens: int | None
    extraction_usage_status: str
    embedding_operations: int
    embedding_input_token_proxy: int
    returned_memory_count: int
    persisted_memory_count: int
    persisted_storage_bytes: int
    add_latency_s: float
    extraction_latency_s: float
    embedding_latency_s: float
    storage_latency_s: float
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.population_identity_sha256, "Mem0 population identity")
        _closed_calls(self.add_attempted, self.add_completed, self.add_failed, "add")
        _closed_calls(
            self.extraction_attempted,
            self.extraction_completed,
            self.extraction_failed,
            "extraction",
        )
        if (
            self.add_attempted != self.extraction_attempted
            or self.add_completed != self.extraction_completed
            or self.add_failed != self.extraction_failed
        ):
            raise MatchedEvalContractError(
                "certified Mem0 write requires one extraction attempt per add"
            )
        for value, label in (
            (
                self.extraction_raw_message_token_proxy,
                "extraction raw-message token proxy",
            ),
            (self.embedding_operations, "embedding operations"),
            (self.embedding_input_token_proxy, "embedding input token proxy"),
            (self.returned_memory_count, "returned memories"),
            (self.persisted_memory_count, "persisted memories"),
            (self.persisted_storage_bytes, "persisted storage bytes"),
        ):
            _count(value, label)
        provider_in = _optional_count(
            self.extraction_provider_input_tokens,
            "extraction provider input tokens",
        )
        provider_out = _optional_count(
            self.extraction_provider_output_tokens,
            "extraction provider output tokens",
        )
        require_text(self.extraction_usage_status, "extraction usage status")
        if (provider_in is None) != (provider_out is None):
            raise MatchedEvalContractError(
                "extraction provider token fields must be jointly known or unknown"
            )
        if provider_in is None and "unavailable" not in self.extraction_usage_status:
            raise MatchedEvalContractError(
                "unknown extraction tokens need an explicit unavailable status"
            )
        for value, label in (
            (self.add_latency_s, "add latency"),
            (self.extraction_latency_s, "extraction latency"),
            (self.embedding_latency_s, "embedding latency"),
            (self.storage_latency_s, "storage latency"),
        ):
            _latency(value, label)
        if self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("write ledger retained transformer state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("write cost receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="mem0_write_cost")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "add_calls": {
                "attempted": self.add_attempted,
                "completed": self.add_completed,
                "failed": self.add_failed,
            },
            "embedding": {
                "input_token_proxy": self.embedding_input_token_proxy,
                "latency_s": float(self.embedding_latency_s),
                "operations": self.embedding_operations,
            },
            "extraction": {
                "attempted": self.extraction_attempted,
                "completed": self.extraction_completed,
                "failed": self.extraction_failed,
                "latency_s": float(self.extraction_latency_s),
                "provider_input_tokens": self.extraction_provider_input_tokens,
                "provider_output_tokens": self.extraction_provider_output_tokens,
                "raw_message_token_proxy": self.extraction_raw_message_token_proxy,
                "usage_status": self.extraction_usage_status,
            },
            "format": WRITE_LEDGER_FORMAT,
            "latency": {
                "add_s": float(self.add_latency_s),
                "storage_s": float(self.storage_latency_s),
            },
            "memory": {
                "persisted_count": self.persisted_memory_count,
                "returned_count": self.returned_memory_count,
                "storage_bytes": self.persisted_storage_bytes,
            },
            "population_identity_sha256": self.population_identity_sha256,
            "retained_transformer_token_state_bytes": 0,
            "typed_epoch": MEM0_TYPED_EPOCH,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class Mem0ReadCostLedger:
    retrieval_artifact_sha256: str
    search_attempted: int
    search_completed: int
    search_failed: int
    raw_memory_count: int
    raw_memory_token_proxy: int
    adapted_memory_count: int
    adapted_memory_token_proxy: int
    packed_memory_count: int
    packed_memory_token_proxy: int
    packed_full_prompt_token_proxy: int
    responder_output_token_reserve: int
    search_latency_s: float
    adaptation_latency_s: float
    packing_latency_s: float
    hard_request_token_cap: Literal[8000] = HARD_REQUEST_TOKEN_CAP
    prompt_budget_compliant: Literal[True] = True
    frontier_mode: Literal["bounded"] = "bounded"
    permits_absence_claims: Literal[False] = False
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.retrieval_artifact_sha256, "Mem0 retrieval artifact")
        _closed_calls(
            self.search_attempted,
            self.search_completed,
            self.search_failed,
            "search",
        )
        for value, label in (
            (self.raw_memory_count, "raw memory count"),
            (self.raw_memory_token_proxy, "raw memory token proxy"),
            (self.adapted_memory_count, "adapted memory count"),
            (self.adapted_memory_token_proxy, "adapted memory token proxy"),
            (self.packed_memory_count, "packed memory count"),
            (self.packed_memory_token_proxy, "packed memory token proxy"),
            (self.packed_full_prompt_token_proxy, "full packed prompt token proxy"),
            (self.responder_output_token_reserve, "responder output reserve"),
        ):
            _count(value, label)
        if not (
            self.packed_memory_count
            <= self.adapted_memory_count
            <= self.raw_memory_count
        ):
            raise MatchedEvalContractError("Mem0 read memory counts are not nested")
        if self.packed_memory_token_proxy > self.adapted_memory_token_proxy:
            raise MatchedEvalContractError("packed memory tokens exceed adapted tokens")
        if self.packed_full_prompt_token_proxy + self.responder_output_token_reserve > self.hard_request_token_cap:
            raise MatchedEvalContractError("Mem0 full request exceeds the hard 8k budget")
        for value, label in (
            (self.search_latency_s, "search latency"),
            (self.adaptation_latency_s, "adaptation latency"),
            (self.packing_latency_s, "packing latency"),
        ):
            _latency(value, label)
        if (
            self.hard_request_token_cap != HARD_REQUEST_TOKEN_CAP
            or self.prompt_budget_compliant is not True
            or self.frontier_mode != "bounded"
            or self.permits_absence_claims is not False
            or self.retained_transformer_token_state_bytes != 0
        ):
            raise MatchedEvalContractError("Mem0 read invariant changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("read cost receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="mem0_read_cost")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "adapted": {
                "memory_count": self.adapted_memory_count,
                "memory_token_proxy": self.adapted_memory_token_proxy,
            },
            "format": READ_LEDGER_FORMAT,
            "frontier_mode": "bounded",
            "hard_request_token_cap": HARD_REQUEST_TOKEN_CAP,
            "latency": {
                "adaptation_s": float(self.adaptation_latency_s),
                "packing_s": float(self.packing_latency_s),
                "search_s": float(self.search_latency_s),
            },
            "packed": {
                "full_prompt_token_proxy": self.packed_full_prompt_token_proxy,
                "memory_count": self.packed_memory_count,
                "memory_token_proxy": self.packed_memory_token_proxy,
                "responder_output_token_reserve": self.responder_output_token_reserve,
            },
            "permits_absence_claims": False,
            "prompt_budget_compliant": True,
            "raw": {
                "memory_count": self.raw_memory_count,
                "memory_token_proxy": self.raw_memory_token_proxy,
            },
            "retained_transformer_token_state_bytes": 0,
            "retrieval_artifact_sha256": self.retrieval_artifact_sha256,
            "search_calls": {
                "attempted": self.search_attempted,
                "completed": self.search_completed,
                "failed": self.search_failed,
            },
            "typed_epoch": MEM0_TYPED_EPOCH,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class CommonProviderStageCost:
    role: Literal["responder", "judge"]
    model_id: str
    logical_calls_attempted: int
    logical_calls_completed: int
    logical_calls_failed: int
    sdk_retry_attempts: int
    provider_input_tokens: int
    provider_output_tokens: int
    latency_s: float
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"responder", "judge"}:
            raise MatchedEvalContractError("provider stage role is invalid")
        require_text(self.model_id, "provider stage model")
        expected_fragment = "terra" if self.role == "responder" else "sol"
        if expected_fragment not in self.model_id.casefold():
            raise MatchedEvalContractError(
                f"common {self.role} stage must use {expected_fragment.title()}"
            )
        _closed_calls(
            self.logical_calls_attempted,
            self.logical_calls_completed,
            self.logical_calls_failed,
            f"{self.role} logical",
        )
        for value, label in (
            (self.sdk_retry_attempts, f"{self.role} SDK retries"),
            (self.provider_input_tokens, f"{self.role} provider input tokens"),
            (self.provider_output_tokens, f"{self.role} provider output tokens"),
        ):
            _count(value, label)
        _latency(self.latency_s, f"{self.role} latency")
        if self.retained_transformer_token_state_bytes != 0:
            raise MatchedEvalContractError("provider stage retained transformer state")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("provider-stage cost receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="common_provider_cost")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": PROVIDER_STAGE_FORMAT,
            "latency_s": float(self.latency_s),
            "logical_calls": {
                "attempted": self.logical_calls_attempted,
                "completed": self.logical_calls_completed,
                "failed": self.logical_calls_failed,
            },
            "model_id": self.model_id,
            "provider_input_tokens": self.provider_input_tokens,
            "provider_output_tokens": self.provider_output_tokens,
            "retained_transformer_token_state_bytes": 0,
            "role": self.role,
            "sdk_retry_attempts": self.sdk_retry_attempts,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class CommonFinalCostLedger:
    question_count: int
    responder: CommonProviderStageCost
    judge: CommonProviderStageCost
    max_full_responder_prompt_token_proxy: int
    responder_output_token_reserve: int
    max_full_judge_prompt_token_proxy: int
    judge_output_token_reserve: int
    hard_request_token_cap: Literal[8000] = HARD_REQUEST_TOKEN_CAP
    prompt_budget_compliant: Literal[True] = True
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _count(self.question_count, "common-final question count")
        if type(self.responder) is not CommonProviderStageCost or self.responder.role != "responder":
            raise TypeError("common-final responder cost is invalid")
        if type(self.judge) is not CommonProviderStageCost or self.judge.role != "judge":
            raise TypeError("common-final judge cost is invalid")
        if (
            self.responder.logical_calls_completed != self.question_count
            or self.judge.logical_calls_completed != self.question_count
        ):
            raise MatchedEvalContractError("common-final call counts do not cover questions")
        _count(
            self.max_full_responder_prompt_token_proxy,
            "max full responder prompt token proxy",
        )
        _count(self.responder_output_token_reserve, "responder output reserve")
        _count(
            self.max_full_judge_prompt_token_proxy,
            "max full judge prompt token proxy",
        )
        _count(self.judge_output_token_reserve, "judge output reserve")
        if (
            self.max_full_responder_prompt_token_proxy
            + self.responder_output_token_reserve
            > self.hard_request_token_cap
            or self.max_full_judge_prompt_token_proxy
            + self.judge_output_token_reserve
            > self.hard_request_token_cap
        ):
            raise MatchedEvalContractError(
                "common-final request exceeds the hard 8k budget"
            )
        if (
            self.hard_request_token_cap != HARD_REQUEST_TOKEN_CAP
            or self.prompt_budget_compliant is not True
            or self.retained_transformer_token_state_bytes != 0
        ):
            raise MatchedEvalContractError("common-final invariant changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("common-final cost receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="common_final_cost")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": FINAL_LEDGER_FORMAT,
            "hard_request_token_cap": HARD_REQUEST_TOKEN_CAP,
            "judge_receipt_sha256": self.judge.receipt_sha256,
            "judge_output_token_reserve": self.judge_output_token_reserve,
            "max_full_judge_prompt_token_proxy": (
                self.max_full_judge_prompt_token_proxy
            ),
            "max_full_responder_prompt_token_proxy": (
                self.max_full_responder_prompt_token_proxy
            ),
            "prompt_budget_compliant": True,
            "question_count": self.question_count,
            "responder_output_token_reserve": self.responder_output_token_reserve,
            "responder_receipt_sha256": self.responder.receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "typed_epoch": MEM0_TYPED_EPOCH,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class Mem0TypedEpochCostLedger:
    write: Mem0WriteCostLedger
    read: Mem0ReadCostLedger
    common_final: CommonFinalCostLedger
    population_identity_sha256: str
    retrieval_artifact_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.write) is not Mem0WriteCostLedger:
            raise TypeError("epoch write ledger must be exact")
        if type(self.read) is not Mem0ReadCostLedger:
            raise TypeError("epoch read ledger must be exact")
        if type(self.common_final) is not CommonFinalCostLedger:
            raise TypeError("epoch final ledger must be exact")
        require_sha256(self.population_identity_sha256, "epoch population identity")
        require_sha256(self.retrieval_artifact_sha256, "epoch retrieval artifact")
        if self.write.population_identity_sha256 != self.population_identity_sha256:
            raise MatchedEvalContractError("epoch/write population binding changed")
        if self.read.retrieval_artifact_sha256 != self.retrieval_artifact_sha256:
            raise MatchedEvalContractError("epoch/read retrieval binding changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("epoch cost receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="mem0_typed_epoch_cost")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "common_final_receipt_sha256": self.common_final.receipt_sha256,
            "format": EPOCH_LEDGER_FORMAT,
            "population_identity_sha256": self.population_identity_sha256,
            "read_receipt_sha256": self.read.receipt_sha256,
            "retrieval_artifact_sha256": self.retrieval_artifact_sha256,
            "typed_epoch": MEM0_TYPED_EPOCH,
            "write_receipt_sha256": self.write.receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


__all__ = [
    "CommonFinalCostLedger",
    "CommonProviderStageCost",
    "EPOCH_LEDGER_FORMAT",
    "FINAL_LEDGER_FORMAT",
    "HARD_REQUEST_TOKEN_CAP",
    "Mem0ReadCostLedger",
    "Mem0TypedEpochCostLedger",
    "Mem0WriteCostLedger",
    "PROVIDER_STAGE_FORMAT",
    "READ_LEDGER_FORMAT",
    "WRITE_LEDGER_FORMAT",
]
