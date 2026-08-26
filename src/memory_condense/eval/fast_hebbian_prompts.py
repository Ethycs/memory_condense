"""Provider-free matched S0/H1 prompts from a frozen Hebbian graph.

The control is exactly the protected S0 membership from the sealed fast
retrieval artifact.  H1 gets one reserved tail replacement from the external
same-turn co-access graph.  Both arms are freshly rendered through the same
compact catalog, then recounted with the deterministic chat-token proxy.

Only :class:`FastHebbianUniquePrompt` retains prompt text.  Every other public
value is a frozen, SHA-sealed, text-free receipt containing IDs, hashes, and
bounded scalar provenance.  No model, tokenizer state, gold answer, or judge
is used here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from memory_condense.associations.association_models import AssociationArtifact
from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.hebbian_retrieval import (
    HebbianExpansionReceipt,
    expand_hebbian_results_with_receipt,
)
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from memory_condense.eval.hebbian_derived_store import (
    DERIVED_STORE_FORMAT as HEBBIAN_DERIVED_STORE_FORMAT,
    MANIFEST_NAME as HEBBIAN_DERIVED_STORE_MANIFEST,
    HebbianDerivedStoreReceipt,
    load_hebbian_derived_store_receipt,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
    FastProviderMessage,
    FastRetrievalArtifact,
)
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result


FAST_HEBBIAN_PROMPT_POPULATION_FORMAT = (
    "memory-condense-fast-hebbian-s0-h1-prompts-v1"
)
FAST_HEBBIAN_QUESTION_RECEIPT_FORMAT = (
    "memory-condense-fast-hebbian-s0-h1-question-v1"
)
FAST_HEBBIAN_ARM_RECEIPT_FORMAT = (
    "memory-condense-fast-hebbian-s0-h1-arm-prompt-v1"
)
FAST_HEBBIAN_ALIAS_BINDING_FORMAT = (
    "memory-condense-fast-hebbian-s0-h1-alias-binding-v1"
)
FAST_HEBBIAN_CATALOG_FORMAT = "memory-condense-fast-hebbian-catalog-v1"
S0_STAGE_ID = STAGE_IDS[0]
ARM_IDS = ("base", "h1")
ABSOLUTE_MAX_PROMPT_TOKENS = 8_000
HEBBIAN_SLOTS = 1
MAX_SEED_CONCEPTS = 12
MAX_CANDIDATES = 32
HALF_LIFE_TURNS = 200.0
MIN_SCORE = 0.05
MAX_PROMPT_TOKEN_INCREASE = 0
RETAINED_REQUEST_TOKEN_STATE_BYTES = 0

FastHebbianEffectiveStatus = Literal[
    "replaced",
    "no_neighbor",
    "all_protected",
    "no_slot",
    "hydration_failed",
    "token_budget_rollback",
    "exact_prompt_budget_rollback",
]
_EFFECTIVE_STATUSES = frozenset(
    {
        "replaced",
        "no_neighbor",
        "all_protected",
        "no_slot",
        "hydration_failed",
        "token_budget_rollback",
        "exact_prompt_budget_rollback",
    }
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_HEADER = "Retrieved evidence catalog:"


class FastHebbianPromptValidationError(ValueError):
    """Raised when a matched H1 population cannot prove its inputs."""


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FastHebbianPromptValidationError(
            f"{label} must be an exact non-empty string"
        )
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FastHebbianPromptValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FastHebbianPromptValidationError(
            f"{label} must be an exact non-negative integer"
        )
    return value


def _unique_ids(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise FastHebbianPromptValidationError(f"{label} must be an exact tuple")
    result = tuple(_nonempty(item, f"{label} item") for item in value)
    if not allow_empty and not result:
        raise FastHebbianPromptValidationError(f"{label} must be non-empty")
    if len(result) != len(set(result)):
        raise FastHebbianPromptValidationError(f"{label} must contain unique IDs")
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FastHebbianAliasBinding:
    """Text-free stable alias for one chunk/source/content coordinate."""

    format: str
    alias: str
    chunk_id: str
    source_id: str
    content_sha256: str
    origin: Literal["s0", "hebbian_candidate"]

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_ALIAS_BINDING_FORMAT:
            raise FastHebbianPromptValidationError(
                "unsupported Hebbian alias-binding format"
            )
        _nonempty(self.alias, "alias")
        _nonempty(self.chunk_id, "chunk_id")
        _nonempty(self.source_id, "source_id")
        _digest(self.content_sha256, "content_sha256")
        if self.origin not in {"s0", "hebbian_candidate"}:
            raise FastHebbianPromptValidationError(
                "unsupported Hebbian alias-binding origin"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "alias": self.alias,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "content_sha256": self.content_sha256,
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class FastHebbianUniquePrompt:
    """The sole public row that retains full provider messages."""

    unique_prompt_ordinal: int
    messages_sha256: str
    context_sha256: str
    prompt_token_proxy: int
    messages: tuple[FastProviderMessage, ...]

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.unique_prompt_ordinal, "unique_prompt_ordinal")
        _digest(self.messages_sha256, "messages_sha256")
        _digest(self.context_sha256, "context_sha256")
        if type(self.prompt_token_proxy) is not int or self.prompt_token_proxy < 1:
            raise FastHebbianPromptValidationError(
                "prompt_token_proxy must be a positive integer"
            )
        if (
            type(self.messages) is not tuple
            or len(self.messages) != 2
            or any(type(item) is not FastProviderMessage for item in self.messages)
            or tuple(item.role for item in self.messages) != ("system", "user")
        ):
            raise FastHebbianPromptValidationError(
                "unique prompt must contain exact system/user messages"
            )
        mappings = self.as_mappings()
        if identity_sha256(mappings) != self.messages_sha256:
            raise FastHebbianPromptValidationError(
                "unique prompt messages do not match messages_sha256"
            )
        if count_chat_prompt_token_proxy(mappings) != self.prompt_token_proxy:
            raise FastHebbianPromptValidationError(
                "unique prompt token count does not recompute exactly"
            )

    def as_mappings(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"role": message.role, "content": message.content}
            for message in self.messages
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "unique_prompt_ordinal": self.unique_prompt_ordinal,
            "messages_sha256": self.messages_sha256,
            "context_sha256": self.context_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
        }


@dataclass(frozen=True, slots=True)
class FastHebbianArmPrompt(SealedIdentity):
    """Text-free logical arm pointer into the deduplicated prompt table."""

    _SEAL_FIELD = "arm_prompt_sha256"
    _SEAL_MISMATCH = "Hebbian arm-prompt seal does not match its contents"

    format: str
    logical_ordinal: int
    question_ordinal: int
    question_id: str
    stage_id: str
    arm_id: Literal["base", "h1"]
    chunk_ids: tuple[str, ...]
    alias_order: tuple[str, ...]
    context_sha256: str
    messages_sha256: str
    prompt_token_proxy: int
    hard_prompt_token_cap: int
    unique_prompt_ordinal: int
    retained_request_token_state_bytes: int = 0
    arm_prompt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_ARM_RECEIPT_FORMAT:
            raise FastHebbianPromptValidationError(
                "unsupported Hebbian arm-prompt format"
            )
        _exact_nonnegative_int(self.logical_ordinal, "logical_ordinal")
        _exact_nonnegative_int(self.question_ordinal, "question_ordinal")
        _exact_nonnegative_int(self.unique_prompt_ordinal, "unique_prompt_ordinal")
        _nonempty(self.question_id, "question_id")
        if self.stage_id != S0_STAGE_ID:
            raise FastHebbianPromptValidationError("Hebbian prompt arm must be S0")
        if self.arm_id not in ARM_IDS:
            raise FastHebbianPromptValidationError("unsupported Hebbian arm ID")
        chunks = _unique_ids(self.chunk_ids, "chunk_ids")
        aliases = _unique_ids(self.alias_order, "alias_order")
        if len(chunks) != len(aliases):
            raise FastHebbianPromptValidationError(
                "chunk_ids and alias_order must remain one-to-one"
            )
        _digest(self.context_sha256, "context_sha256")
        _digest(self.messages_sha256, "messages_sha256")
        if type(self.prompt_token_proxy) is not int or self.prompt_token_proxy < 1:
            raise FastHebbianPromptValidationError(
                "prompt_token_proxy must be a positive integer"
            )
        if (
            type(self.hard_prompt_token_cap) is not int
            or not 1 <= self.hard_prompt_token_cap <= ABSOLUTE_MAX_PROMPT_TOKENS
            or self.prompt_token_proxy > self.hard_prompt_token_cap
        ):
            raise FastHebbianPromptValidationError(
                "arm prompt exceeds its valid hard prompt-token cap"
            )
        if self.retained_request_token_state_bytes != 0:
            raise FastHebbianPromptValidationError(
                "Hebbian prompt arm retained request token state"
            )
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianQuestionPromptReceipt(SealedIdentity):
    """Sealed, text-free provenance for one matched S0/H1 question pair."""

    _SEAL_FIELD = "receipt_sha256"
    _SEAL_MISMATCH = "Hebbian question-prompt receipt does not match its contents"

    format: str
    question_ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    stage_id: str
    catalog_format: str
    retrieval_artifact_sha256: str
    source_store_receipt_sha256: str
    predecessor_receipt_sha256: str
    retrieval_receipt_sha256: str
    stage_receipt_sha256: str
    s0_evidence_projection_sha256: str
    history_receipt_sha256: str
    derived_store_receipt_sha256: str
    association_artifact_id: str
    association_artifact_sha256: str
    protected_chunk_ids: tuple[str, ...]
    s0_evidence_ids: tuple[str, ...]
    alias_bindings: tuple[FastHebbianAliasBinding, ...]
    expansion_receipt: HebbianExpansionReceipt
    effective_status: FastHebbianEffectiveStatus
    effective_h1_chunk_ids: tuple[str, ...]
    base_arm_prompt_sha256: str
    h1_arm_prompt_sha256: str
    base_messages_sha256: str
    h1_messages_sha256: str
    retained_request_token_state_bytes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_QUESTION_RECEIPT_FORMAT:
            raise FastHebbianPromptValidationError(
                "unsupported Hebbian question-receipt format"
            )
        _exact_nonnegative_int(self.question_ordinal, "question_ordinal")
        _nonempty(self.question_id, "question_id")
        if self.stage_id != S0_STAGE_ID:
            raise FastHebbianPromptValidationError(
                "Hebbian question receipt must bind S0"
            )
        if self.catalog_format != FAST_HEBBIAN_CATALOG_FORMAT:
            raise FastHebbianPromptValidationError(
                "unsupported canonical Hebbian catalog format"
            )
        for name in (
            "question_sha256",
            "dated_question_sha256",
            "retrieval_artifact_sha256",
            "source_store_receipt_sha256",
            "predecessor_receipt_sha256",
            "retrieval_receipt_sha256",
            "stage_receipt_sha256",
            "s0_evidence_projection_sha256",
            "history_receipt_sha256",
            "derived_store_receipt_sha256",
            "association_artifact_sha256",
            "base_arm_prompt_sha256",
            "h1_arm_prompt_sha256",
            "base_messages_sha256",
            "h1_messages_sha256",
        ):
            _digest(getattr(self, name), name)
        _nonempty(self.association_artifact_id, "association_artifact_id")
        protected = _unique_ids(self.protected_chunk_ids, "protected_chunk_ids")
        evidence = _unique_ids(self.s0_evidence_ids, "s0_evidence_ids")
        if len(protected) != len(evidence):
            raise FastHebbianPromptValidationError(
                "protected chunks and S0 evidence must remain one-to-one"
            )
        if type(self.alias_bindings) is not tuple or not self.alias_bindings:
            raise FastHebbianPromptValidationError(
                "alias_bindings must be a non-empty exact tuple"
            )
        if any(
            type(item) is not FastHebbianAliasBinding
            for item in self.alias_bindings
        ):
            raise FastHebbianPromptValidationError(
                "alias_bindings must contain exact binding rows"
            )
        if len({item.alias for item in self.alias_bindings}) != len(
            self.alias_bindings
        ) or len({item.chunk_id for item in self.alias_bindings}) != len(
            self.alias_bindings
        ):
            raise FastHebbianPromptValidationError(
                "alias bindings must be one-to-one by alias and chunk"
            )
        binding_by_chunk = {
            item.chunk_id: item for item in self.alias_bindings
        }
        if type(self.expansion_receipt) is not HebbianExpansionReceipt:
            raise FastHebbianPromptValidationError(
                "expansion_receipt must be an exact HebbianExpansionReceipt"
            )
        if self.expansion_receipt.artifact_id != self.association_artifact_id:
            raise FastHebbianPromptValidationError(
                "expansion receipt changed association artifact"
            )
        if self.expansion_receipt.base_chunk_ids != protected:
            raise FastHebbianPromptValidationError(
                "expansion receipt changed protected S0 coordinates"
            )
        expected_binding_ids = set(protected) | set(
            self.expansion_receipt.final_chunk_ids
        )
        if set(binding_by_chunk) != expected_binding_ids:
            raise FastHebbianPromptValidationError(
                "alias bindings changed the rendered base/candidate union"
            )
        if any(
            binding_by_chunk[chunk_id].origin != "s0"
            for chunk_id in protected
        ) or any(
            row.origin != "hebbian_candidate"
            for chunk_id, row in binding_by_chunk.items()
            if chunk_id not in set(protected)
        ):
            raise FastHebbianPromptValidationError(
                "alias binding origin changed S0/candidate membership"
            )
        if self.effective_status not in _EFFECTIVE_STATUSES:
            raise FastHebbianPromptValidationError(
                "unsupported effective Hebbian status"
            )
        effective_ids = _unique_ids(
            self.effective_h1_chunk_ids,
            "effective_h1_chunk_ids",
        )
        if self.effective_status == "exact_prompt_budget_rollback":
            if self.expansion_receipt.status != "replaced":
                raise FastHebbianPromptValidationError(
                    "exact-prompt rollback requires a proposed replacement"
                )
            if effective_ids != protected:
                raise FastHebbianPromptValidationError(
                    "exact-prompt rollback must restore byte-identical base membership"
                )
            if self.base_messages_sha256 != self.h1_messages_sha256:
                raise FastHebbianPromptValidationError(
                    "exact-prompt rollback must restore the base arm prompt"
                )
        else:
            if self.effective_status != self.expansion_receipt.status:
                raise FastHebbianPromptValidationError(
                    "effective status changed the expansion result"
                )
            if effective_ids != self.expansion_receipt.final_chunk_ids:
                raise FastHebbianPromptValidationError(
                    "effective H1 membership changed the expansion result"
                )
        if self.effective_status != "replaced" and (
            self.base_messages_sha256 != self.h1_messages_sha256
        ):
            raise FastHebbianPromptValidationError(
                "a no-op H1 outcome must reuse the byte-identical base prompt"
            )
        if self.retained_request_token_state_bytes != 0:
            raise FastHebbianPromptValidationError(
                "Hebbian question receipt retained request token state"
            )
        self._seal()


@dataclass(frozen=True, slots=True)
class FastHebbianPromptPopulation(SealedIdentity):
    """All logical pairs, unique prompt text, and sealed question receipts."""

    _SEAL_FIELD = "prompt_population_sha256"
    _SEAL_MISMATCH = "Hebbian prompt-population seal does not match its contents"

    format: str
    retrieval_artifact_sha256: str
    stage_id: str
    history_receipt_sha256: str
    derived_store_receipt_sha256: str
    association_artifact_id: str
    logical_prompt_count: int
    unique_prompt_count: int
    logical_prompts: tuple[FastHebbianArmPrompt, ...]
    unique_prompts: tuple[FastHebbianUniquePrompt, ...]
    question_receipts: tuple[FastHebbianQuestionPromptReceipt, ...]
    retained_request_token_state_bytes: int = 0
    prompt_population_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != FAST_HEBBIAN_PROMPT_POPULATION_FORMAT:
            raise FastHebbianPromptValidationError(
                "unsupported Hebbian prompt-population format"
            )
        _digest(self.retrieval_artifact_sha256, "retrieval_artifact_sha256")
        _digest(self.history_receipt_sha256, "history_receipt_sha256")
        _digest(self.derived_store_receipt_sha256, "derived_store_receipt_sha256")
        _nonempty(self.association_artifact_id, "association_artifact_id")
        if self.stage_id != S0_STAGE_ID:
            raise FastHebbianPromptValidationError(
                "Hebbian prompt population must contain only S0/H1"
            )
        _exact_nonnegative_int(self.logical_prompt_count, "logical_prompt_count")
        _exact_nonnegative_int(self.unique_prompt_count, "unique_prompt_count")
        if type(self.logical_prompts) is not tuple or any(
            type(item) is not FastHebbianArmPrompt for item in self.logical_prompts
        ):
            raise FastHebbianPromptValidationError(
                "logical_prompts must be an exact arm-prompt tuple"
            )
        if type(self.unique_prompts) is not tuple or any(
            type(item) is not FastHebbianUniquePrompt for item in self.unique_prompts
        ):
            raise FastHebbianPromptValidationError(
                "unique_prompts must be an exact unique-prompt tuple"
            )
        if type(self.question_receipts) is not tuple or any(
            type(item) is not FastHebbianQuestionPromptReceipt
            for item in self.question_receipts
        ):
            raise FastHebbianPromptValidationError(
                "question_receipts must be an exact receipt tuple"
            )
        if self.logical_prompt_count != len(self.logical_prompts):
            raise FastHebbianPromptValidationError("logical prompt count mismatch")
        if self.unique_prompt_count != len(self.unique_prompts):
            raise FastHebbianPromptValidationError("unique prompt count mismatch")
        if self.logical_prompt_count != 2 * len(self.question_receipts):
            raise FastHebbianPromptValidationError(
                "each question must contribute exactly base and H1 prompts"
            )
        if not self.question_receipts:
            raise FastHebbianPromptValidationError(
                "Hebbian prompt population must be non-empty"
            )
        if tuple(row.logical_ordinal for row in self.logical_prompts) != tuple(
            range(self.logical_prompt_count)
        ):
            raise FastHebbianPromptValidationError(
                "logical prompt ordinals must be contiguous"
            )
        if tuple(row.unique_prompt_ordinal for row in self.unique_prompts) != tuple(
            range(self.unique_prompt_count)
        ):
            raise FastHebbianPromptValidationError(
                "unique prompt ordinals must be contiguous"
            )
        if any(
            row.unique_prompt_ordinal >= self.unique_prompt_count
            for row in self.logical_prompts
        ):
            raise FastHebbianPromptValidationError(
                "logical prompt points outside unique prompt table"
            )
        if {row.unique_prompt_ordinal for row in self.logical_prompts} != set(
            range(self.unique_prompt_count)
        ):
            raise FastHebbianPromptValidationError(
                "unique prompt table contains an unreferenced or missing row"
            )
        if len({row.messages_sha256 for row in self.unique_prompts}) != len(
            self.unique_prompts
        ):
            raise FastHebbianPromptValidationError(
                "unique prompt table contains duplicate message hashes"
            )
        for arm in self.logical_prompts:
            unique = self.unique_prompts[arm.unique_prompt_ordinal]
            if (
                arm.messages_sha256 != unique.messages_sha256
                or arm.context_sha256 != unique.context_sha256
                or arm.prompt_token_proxy != unique.prompt_token_proxy
            ):
                raise FastHebbianPromptValidationError(
                    "logical prompt metadata changed its unique prompt row"
                )
        for index, receipt in enumerate(self.question_receipts):
            base = self.logical_prompts[index * 2]
            treatment = self.logical_prompts[index * 2 + 1]
            if (base.arm_id, treatment.arm_id) != ARM_IDS or (
                base.question_ordinal,
                base.question_id,
            ) != (receipt.question_ordinal, receipt.question_id) or (
                treatment.question_ordinal,
                treatment.question_id,
            ) != (receipt.question_ordinal, receipt.question_id):
                raise FastHebbianPromptValidationError(
                    "question receipt changed its ordered base/H1 prompt pair"
                )
            if (
                receipt.base_arm_prompt_sha256 != base.arm_prompt_sha256
                or receipt.h1_arm_prompt_sha256 != treatment.arm_prompt_sha256
                or receipt.base_messages_sha256 != base.messages_sha256
                or receipt.h1_messages_sha256 != treatment.messages_sha256
            ):
                raise FastHebbianPromptValidationError(
                    "question receipt changed its arm prompt pointers"
                )
            alias_by_chunk = {
                binding.chunk_id: binding.alias
                for binding in receipt.alias_bindings
            }
            if base.chunk_ids != receipt.protected_chunk_ids or (
                treatment.chunk_ids != receipt.effective_h1_chunk_ids
            ) or base.alias_order != tuple(
                alias_by_chunk[chunk_id] for chunk_id in base.chunk_ids
            ) or treatment.alias_order != tuple(
                alias_by_chunk[chunk_id] for chunk_id in treatment.chunk_ids
            ):
                raise FastHebbianPromptValidationError(
                    "question receipt changed arm membership or alias order"
                )
            if (
                treatment.prompt_token_proxy > base.prompt_token_proxy
                or treatment.hard_prompt_token_cap
                != base.hard_prompt_token_cap
            ):
                raise FastHebbianPromptValidationError(
                    "H1 prompt exceeded its matched base prompt budget"
                )
            if (
                receipt.retrieval_artifact_sha256
                != self.retrieval_artifact_sha256
                or receipt.history_receipt_sha256
                != self.history_receipt_sha256
                or receipt.derived_store_receipt_sha256
                != self.derived_store_receipt_sha256
                or receipt.association_artifact_id
                != self.association_artifact_id
                or receipt.stage_id != self.stage_id
            ):
                raise FastHebbianPromptValidationError(
                    "question receipt changed population-level provenance"
                )
        if self.retained_request_token_state_bytes != 0:
            raise FastHebbianPromptValidationError(
                "Hebbian prompt population retained request token state"
            )
        self._seal()

    @property
    def logical_message_population(
        self,
    ) -> tuple[tuple[dict[str, str], ...], ...]:
        return tuple(
            self.unique_prompts[row.unique_prompt_ordinal].as_mappings()
            for row in self.logical_prompts
        )


@dataclass(frozen=True, slots=True)
class _EvidenceRow:
    chunk_id: str
    source_id: str
    content: str
    origin: Literal["s0", "hebbian_candidate"]


def _association_payload(artifact: AssociationArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "model_id": artifact.model_id,
        "checkpoint_id": artifact.checkpoint_id,
        "prefix_layers": artifact.prefix_layers,
        "head_layer": artifact.head_layer,
        "cav_layer": artifact.cav_layer,
        "concept_names": list(artifact.concept_names),
        "head_count": artifact.head_count,
        "created_at": artifact.created_at,
        "metadata": dict(artifact.metadata),
    }


def _derived_inputs(
    derived_store_path: str | Path,
    *,
    artifact: FastRetrievalArtifact,
    association_artifact_id: str,
    history_receipt_sha256: str,
    derived_store_receipt_sha256: str,
) -> tuple[Path, HebbianDerivedStoreReceipt]:
    root_candidate = Path(derived_store_path)
    if root_candidate.is_symlink():
        raise FastHebbianPromptValidationError(
            "derived_store_path must not be a symbolic link"
        )
    root = root_candidate.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise FastHebbianPromptValidationError(
            "derived_store_path must be a regular directory"
        )
    database_candidate = root / "memory.db"
    manifest_candidate = root / HEBBIAN_DERIVED_STORE_MANIFEST
    if database_candidate.is_symlink() or manifest_candidate.is_symlink():
        raise FastHebbianPromptValidationError(
            "derived store files must not be symbolic links"
        )
    database_path = database_candidate.resolve(strict=True)
    manifest_path = manifest_candidate.resolve(strict=True)
    if (
        not database_path.is_file()
        or database_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise FastHebbianPromptValidationError(
            "derived store requires regular memory.db and manifest files"
        )
    for suffix in ("-wal", "-shm"):
        if database_path.with_name(database_path.name + suffix).exists():
            raise FastHebbianPromptValidationError(
                "derived store retained SQLite sidecars"
            )
    try:
        payload = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FastHebbianPromptValidationError(
            "derived-store manifest is not valid JSON"
        ) from exc
    if type(payload) is not dict:
        raise FastHebbianPromptValidationError(
            "derived-store manifest has a noncanonical shape"
        )
    if manifest_path.read_bytes() != _canonical_json_bytes(payload):
        raise FastHebbianPromptValidationError(
            "derived-store manifest is not canonical JSON"
        )
    receipt = load_hebbian_derived_store_receipt(payload)
    if receipt.receipt_sha256 != derived_store_receipt_sha256:
        raise FastHebbianPromptValidationError(
            "derived store does not match derived_store_receipt_sha256"
        )
    if receipt.format != HEBBIAN_DERIVED_STORE_FORMAT:
        raise FastHebbianPromptValidationError(
            "unsupported derived-store manifest format"
        )
    if receipt.history_receipt_sha256 != history_receipt_sha256:
        raise FastHebbianPromptValidationError(
            "derived store does not match history_receipt_sha256"
        )
    if receipt.source_store_receipt_sha256 != (
        artifact.combined_store_receipt_sha256
    ):
        raise FastHebbianPromptValidationError(
            "derived store changed the combined source-store receipt"
        )
    if receipt.association_artifact_id != association_artifact_id:
        raise FastHebbianPromptValidationError(
            "derived store changed the exact association artifact ID"
        )
    if receipt.retained_request_token_state_bytes != 0:
        raise FastHebbianPromptValidationError(
            "derived store retained request token state"
        )
    if receipt.derived_database_sha256 != file_sha256(database_path):
        raise FastHebbianPromptValidationError(
            "derived database digest changed after publication"
        )
    return database_path, receipt


def _messages(
    context: str,
    dated_question: str,
) -> tuple[FastProviderMessage, ...]:
    return (
        FastProviderMessage(role="system", content=QA_SYSTEM_PROMPT),
        FastProviderMessage(
            role="user",
            content=QA_USER_TEMPLATE.format(
                context=context,
                question=dated_question,
            ),
        ),
    )


def _message_mappings(
    messages: tuple[FastProviderMessage, ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"role": message.role, "content": message.content}
        for message in messages
    )


def _catalog(
    rows: tuple[_EvidenceRow, ...],
    alias_by_chunk: Mapping[str, str],
) -> str:
    return _CATALOG_HEADER + "\n" + "\n".join(
        f"[{alias_by_chunk[row.chunk_id]}] {row.content}" for row in rows
    )


def _source_id(result: RetrievalResult) -> str:
    source_id = result.durable_source_id.strip()
    if not source_id or len(result.source_hints) > 1:
        raise FastHebbianPromptValidationError(
            f"chunk {result.chunk.chunk_id!r} has ambiguous source provenance"
        )
    return source_id


def _base_rows(
    evidence: tuple[FastEvidence, ...],
    chunk_ids: tuple[str, ...],
    hydrated: tuple[RetrievalResult, ...],
) -> tuple[_EvidenceRow, ...]:
    if not (len(evidence) == len(chunk_ids) == len(hydrated)):
        raise FastHebbianPromptValidationError(
            "S0 evidence, protected chunk IDs, and hydrated chunks must be one-to-one"
        )
    rows: list[_EvidenceRow] = []
    for index, (projection, chunk_id, result) in enumerate(
        zip(evidence, chunk_ids, hydrated, strict=True)
    ):
        if result.chunk.chunk_id != chunk_id:
            raise FastHebbianPromptValidationError(
                f"S0 coordinate {index} hydrated the wrong chunk"
            )
        source_id = _source_id(result)
        if source_id != projection.source_id:
            raise FastHebbianPromptValidationError(
                f"S0 coordinate {index} changed durable source provenance"
            )
        rows.append(
            _EvidenceRow(
                chunk_id=chunk_id,
                source_id=source_id,
                content=projection.text,
                origin="s0",
            )
        )
    return tuple(rows)


def _s0_evidence_projection_sha256(
    evidence: tuple[FastEvidence, ...],
    chunk_ids: tuple[str, ...],
) -> str:
    """Recompute the predecessor's exact ordered excerpt/chunk pair seal."""

    if len(evidence) != len(chunk_ids):
        raise FastHebbianPromptValidationError(
            "S0 evidence and protected chunks must be one-to-one"
        )
    return identity_sha256(
        {
            "protected_excerpts": [
                {
                    "chunk_id": chunk_id,
                    "source_id": row.source_id,
                    "text_sha256": quote_sha256(row.text),
                }
                for chunk_id, row in zip(chunk_ids, evidence, strict=True)
            ],
            "admitted_atoms": [],
        }
    )


def _alias_bindings(
    rows_by_chunk: Mapping[str, _EvidenceRow],
) -> tuple[FastHebbianAliasBinding, ...]:
    ordered = sorted(
        rows_by_chunk.values(),
        key=lambda row: (row.source_id, row.chunk_id),
    )
    bindings = tuple(
        FastHebbianAliasBinding(
            format=FAST_HEBBIAN_ALIAS_BINDING_FORMAT,
            # The alias is a function of this row only.  A treatment-only
            # candidate therefore cannot renumber or otherwise perturb the
            # base prompt.
            alias="H"
            + identity_sha256(
                {
                    "format": FAST_HEBBIAN_ALIAS_BINDING_FORMAT,
                    "chunk_id": row.chunk_id,
                    "source_id": row.source_id,
                }
            )[:10].upper(),
            chunk_id=row.chunk_id,
            source_id=row.source_id,
            content_sha256=quote_sha256(row.content),
            origin=row.origin,
        )
        for row in ordered
    )
    if len({row.alias for row in bindings}) != len(bindings):
        raise FastHebbianPromptValidationError(
            "stable Hebbian alias digest collision"
        )
    return bindings


def build_fast_hebbian_prompt_population(
    artifact: FastRetrievalArtifact,
    derived_store_path: str | Path,
    *,
    association_artifact_id: str,
    history_receipt_sha256: str,
    derived_store_receipt_sha256: str,
) -> FastHebbianPromptPopulation:
    """Build the complete matched S0/H1 population without a provider or LLM."""

    if type(artifact) is not FastRetrievalArtifact:
        raise TypeError("artifact must be an exact FastRetrievalArtifact")
    _digest(artifact.raw_sha256, "artifact.raw_sha256")
    _digest(history_receipt_sha256, "history_receipt_sha256")
    _digest(derived_store_receipt_sha256, "derived_store_receipt_sha256")
    association_artifact_id = _nonempty(
        association_artifact_id,
        "association_artifact_id",
    )
    if artifact.retained_request_token_state_bytes != 0:
        raise FastHebbianPromptValidationError(
            "retrieval artifact retained request token state"
        )
    if tuple(artifact.stage_ids) != STAGE_IDS:
        raise FastHebbianPromptValidationError(
            "retrieval artifact changed canonical cumulative stage IDs"
        )
    database_path, derived_receipt = _derived_inputs(
        derived_store_path,
        artifact=artifact,
        association_artifact_id=association_artifact_id,
        history_receipt_sha256=history_receipt_sha256,
        derived_store_receipt_sha256=derived_store_receipt_sha256,
    )

    unique_prompts: list[FastHebbianUniquePrompt] = []
    unique_by_sha: dict[str, tuple[int, tuple[FastProviderMessage, ...]]] = {}
    logical_prompts: list[FastHebbianArmPrompt] = []
    question_receipts: list[FastHebbianQuestionPromptReceipt] = []
    hydrated_cache: dict[str, RetrievalResult | None] = {}

    with Database(database_path, read_only=True) as database:
        if database.current_turn() != artifact.turn_count:
            raise FastHebbianPromptValidationError(
                "derived store turn coordinate changed the retrieval population"
            )
        associations = AssociationStore(database)
        association_artifact = associations.get_artifact(association_artifact_id)
        if association_artifact is None:
            raise FastHebbianPromptValidationError(
                "exact association artifact ID is absent from derived store"
            )
        association_sha = identity_sha256(
            _association_payload(association_artifact)
        )
        if derived_receipt.association_artifact_sha256 != association_sha:
            raise FastHebbianPromptValidationError(
                "derived manifest changed association artifact identity"
            )
        graph_stats = associations.hebbian_stats(association_artifact_id)
        if (
            int(graph_stats["nodes"]) != derived_receipt.graph_nodes
            or int(graph_stats["edges"]) != derived_receipt.graph_edges
            or int(graph_stats["event_receipts"])
            != derived_receipt.graph_event_receipts
            or int(graph_stats["retained_request_token_state_bytes"]) != 0
        ):
            raise FastHebbianPromptValidationError(
                "derived graph counts changed after publication"
            )

        def hydrate(chunk_id: str, **kwargs: object) -> RetrievalResult | None:
            cached = hydrated_cache.get(chunk_id)
            if chunk_id not in hydrated_cache:
                cached = hydrate_chunk_result(database, chunk_id, score=0.0)
                hydrated_cache[chunk_id] = cached
            if cached is None:
                return None
            score = float(kwargs.get("score", 0.0))
            return cached.model_copy(update={"score": score})

        seen_questions: set[str] = set()
        for question in artifact.questions:
            if question.question_id in seen_questions:
                raise FastHebbianPromptValidationError(
                    "retrieval artifact contains duplicate question IDs"
                )
            seen_questions.add(question.question_id)
            if question.retained_request_token_state_bytes != 0:
                raise FastHebbianPromptValidationError(
                    "retrieval question retained request token state"
                )
            if quote_sha256(question.question) != question.question_sha256 or (
                quote_sha256(question.dated_question)
                != question.dated_question_sha256
            ):
                raise FastHebbianPromptValidationError(
                    "retrieval question text changed its sealed digest"
                )
            stage = question.stage(S0_STAGE_ID)
            protected = _unique_ids(
                question.protected_chunk_ids,
                "question.protected_chunk_ids",
            )
            if len(protected) != len(stage.evidence):
                raise FastHebbianPromptValidationError(
                    "protected chunk count must exactly match S0 evidence count"
                )
            s0_projection_sha = _s0_evidence_projection_sha256(
                stage.evidence,
                protected,
            )
            if s0_projection_sha != stage.evidence_projection_sha256:
                raise FastHebbianPromptValidationError(
                    "protected chunk order does not match the sealed S0 "
                    "evidence projection"
                )
            base_results_list: list[RetrievalResult] = []
            for chunk_id in protected:
                result = hydrate(chunk_id, score=0.0)
                if result is None:
                    raise FastHebbianPromptValidationError(
                        f"protected S0 chunk is absent: {chunk_id}"
                    )
                base_results_list.append(result)
            base_results = tuple(base_results_list)
            base_rows = _base_rows(stage.evidence, protected, base_results)

            expanded, expansion_receipt = expand_hebbian_results_with_receipt(
                base_results,
                association_artifact_id,
                store=associations,
                hydrate=hydrate,
                now_turn=database.current_turn(),
                k=len(base_results),
                hebbian_slots=HEBBIAN_SLOTS,
                max_seed_concepts=MAX_SEED_CONCEPTS,
                max_candidates=MAX_CANDIDATES,
                half_life_turns=HALF_LIFE_TURNS,
                min_score=MIN_SCORE,
                lexical_protection_threshold=None,
                max_prompt_token_increase=MAX_PROMPT_TOKEN_INCREASE,
            )
            if expansion_receipt.retained_request_token_state_bytes != 0:
                raise FastHebbianPromptValidationError(
                    "Hebbian expansion retained request token state"
                )

            rows_by_chunk = {row.chunk_id: row for row in base_rows}
            candidate_rows: list[_EvidenceRow] = []
            for result in expanded:
                chunk_id = result.chunk.chunk_id
                row = rows_by_chunk.get(chunk_id)
                if row is None:
                    row = _EvidenceRow(
                        chunk_id=chunk_id,
                        source_id=_source_id(result),
                        content=result.chunk.text,
                        origin="hebbian_candidate",
                    )
                    rows_by_chunk[chunk_id] = row
                candidate_rows.append(row)
            if tuple(row.chunk_id for row in candidate_rows) != (
                expansion_receipt.final_chunk_ids
            ):
                raise FastHebbianPromptValidationError(
                    "rendered H1 rows changed expansion membership order"
                )

            aliases = _alias_bindings(rows_by_chunk)
            alias_by_chunk = {row.chunk_id: row.alias for row in aliases}
            hard_cap = min(
                stage.max_prompt_token_proxy,
                ABSOLUTE_MAX_PROMPT_TOKENS,
            )
            if type(hard_cap) is not int or hard_cap < 1:
                raise FastHebbianPromptValidationError(
                    "S0 hard prompt-token cap must be positive"
                )

            base_context = _catalog(base_rows, alias_by_chunk)
            base_messages = _messages(base_context, question.dated_question)
            base_mappings = _message_mappings(base_messages)
            base_tokens = count_chat_prompt_token_proxy(base_mappings)
            if base_tokens > hard_cap:
                raise FastHebbianPromptValidationError(
                    "canonical S0 base prompt exceeds the hard prompt-token cap "
                    f"for {question.question_id}: {base_tokens} > {hard_cap}"
                )

            candidate_tuple = tuple(candidate_rows)
            treatment_context = _catalog(candidate_tuple, alias_by_chunk)
            treatment_messages = _messages(
                treatment_context,
                question.dated_question,
            )
            treatment_mappings = _message_mappings(treatment_messages)
            treatment_tokens = count_chat_prompt_token_proxy(treatment_mappings)
            effective_status: FastHebbianEffectiveStatus = (
                expansion_receipt.status
            )
            effective_results = tuple(expanded)
            if expansion_receipt.status == "replaced" and (
                treatment_tokens > base_tokens or treatment_tokens > hard_cap
            ):
                effective_status = "exact_prompt_budget_rollback"
                effective_results = base_results
                treatment_context = base_context
                treatment_messages = base_messages
                treatment_mappings = base_mappings
                treatment_tokens = base_tokens
            if treatment_tokens > base_tokens or treatment_tokens > hard_cap:
                raise AssertionError(
                    "matched Hebbian treatment budget invariant failed"
                )

            arm_shas: list[str] = []
            for arm_id, results, context, messages, mappings, prompt_tokens in (
                (
                    "base",
                    base_results,
                    base_context,
                    base_messages,
                    base_mappings,
                    base_tokens,
                ),
                (
                    "h1",
                    effective_results,
                    treatment_context,
                    treatment_messages,
                    treatment_mappings,
                    treatment_tokens,
                ),
            ):
                messages_sha = identity_sha256(mappings)
                context_sha = quote_sha256(context)
                existing = unique_by_sha.get(messages_sha)
                if existing is None:
                    unique_ordinal = len(unique_prompts)
                    unique_by_sha[messages_sha] = (unique_ordinal, messages)
                    unique_prompts.append(
                        FastHebbianUniquePrompt(
                            unique_prompt_ordinal=unique_ordinal,
                            messages_sha256=messages_sha,
                            context_sha256=context_sha,
                            prompt_token_proxy=prompt_tokens,
                            messages=messages,
                        )
                    )
                else:
                    unique_ordinal, prior_messages = existing
                    if prior_messages != messages:
                        raise RuntimeError("provider prompt SHA-256 collision")
                    prior = unique_prompts[unique_ordinal]
                    if (
                        prior.context_sha256 != context_sha
                        or prior.prompt_token_proxy != prompt_tokens
                    ):
                        raise RuntimeError(
                            "identical provider prompt changed prompt metadata"
                        )
                arm = FastHebbianArmPrompt(
                    format=FAST_HEBBIAN_ARM_RECEIPT_FORMAT,
                    logical_ordinal=len(logical_prompts),
                    question_ordinal=question.ordinal,
                    question_id=question.question_id,
                    stage_id=S0_STAGE_ID,
                    arm_id=arm_id,
                    chunk_ids=tuple(result.chunk.chunk_id for result in results),
                    alias_order=tuple(
                        alias_by_chunk[result.chunk.chunk_id] for result in results
                    ),
                    context_sha256=context_sha,
                    messages_sha256=messages_sha,
                    prompt_token_proxy=prompt_tokens,
                    hard_prompt_token_cap=hard_cap,
                    unique_prompt_ordinal=unique_ordinal,
                    retained_request_token_state_bytes=0,
                )
                logical_prompts.append(arm)
                arm_shas.append(arm.arm_prompt_sha256)

            question_receipts.append(
                FastHebbianQuestionPromptReceipt(
                    format=FAST_HEBBIAN_QUESTION_RECEIPT_FORMAT,
                    question_ordinal=question.ordinal,
                    question_id=question.question_id,
                    question_sha256=question.question_sha256,
                    dated_question_sha256=question.dated_question_sha256,
                    stage_id=S0_STAGE_ID,
                    catalog_format=FAST_HEBBIAN_CATALOG_FORMAT,
                    retrieval_artifact_sha256=artifact.raw_sha256,
                    source_store_receipt_sha256=(
                        artifact.combined_store_receipt_sha256
                    ),
                    predecessor_receipt_sha256=(
                        question.predecessor_receipt_sha256
                    ),
                    retrieval_receipt_sha256=question.retrieval_receipt_sha256,
                    stage_receipt_sha256=stage.stage_receipt_sha256,
                    s0_evidence_projection_sha256=s0_projection_sha,
                    history_receipt_sha256=history_receipt_sha256,
                    derived_store_receipt_sha256=(
                        derived_store_receipt_sha256
                    ),
                    association_artifact_id=association_artifact_id,
                    association_artifact_sha256=association_sha,
                    protected_chunk_ids=protected,
                    s0_evidence_ids=stage.evidence_ids,
                    alias_bindings=aliases,
                    expansion_receipt=expansion_receipt,
                    effective_status=effective_status,
                    effective_h1_chunk_ids=tuple(
                        result.chunk.chunk_id for result in effective_results
                    ),
                    base_arm_prompt_sha256=arm_shas[0],
                    h1_arm_prompt_sha256=arm_shas[1],
                    base_messages_sha256=logical_prompts[-2].messages_sha256,
                    h1_messages_sha256=logical_prompts[-1].messages_sha256,
                    retained_request_token_state_bytes=0,
                )
            )

    return FastHebbianPromptPopulation(
        format=FAST_HEBBIAN_PROMPT_POPULATION_FORMAT,
        retrieval_artifact_sha256=artifact.raw_sha256,
        stage_id=S0_STAGE_ID,
        history_receipt_sha256=history_receipt_sha256,
        derived_store_receipt_sha256=derived_store_receipt_sha256,
        association_artifact_id=association_artifact_id,
        logical_prompt_count=len(logical_prompts),
        unique_prompt_count=len(unique_prompts),
        logical_prompts=tuple(logical_prompts),
        unique_prompts=tuple(unique_prompts),
        question_receipts=tuple(question_receipts),
        retained_request_token_state_bytes=0,
    )


__all__ = [
    "ABSOLUTE_MAX_PROMPT_TOKENS",
    "ARM_IDS",
    "FAST_HEBBIAN_ALIAS_BINDING_FORMAT",
    "FAST_HEBBIAN_ARM_RECEIPT_FORMAT",
    "FAST_HEBBIAN_CATALOG_FORMAT",
    "FAST_HEBBIAN_PROMPT_POPULATION_FORMAT",
    "FAST_HEBBIAN_QUESTION_RECEIPT_FORMAT",
    "HEBBIAN_SLOTS",
    "MAX_CANDIDATES",
    "MAX_PROMPT_TOKEN_INCREASE",
    "MAX_SEED_CONCEPTS",
    "MIN_SCORE",
    "HALF_LIFE_TURNS",
    "RETAINED_REQUEST_TOKEN_STATE_BYTES",
    "S0_STAGE_ID",
    "FastHebbianAliasBinding",
    "FastHebbianArmPrompt",
    "FastHebbianEffectiveStatus",
    "FastHebbianPromptPopulation",
    "FastHebbianPromptValidationError",
    "FastHebbianQuestionPromptReceipt",
    "FastHebbianUniquePrompt",
    "build_fast_hebbian_prompt_population",
]
