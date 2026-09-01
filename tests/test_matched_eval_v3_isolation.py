from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_population import _publish, _retrieval
from tools.matched_eval import judging, live
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import EvidenceItem, FactItem, LinkItem, MemoryPacket
from tools.matched_eval.population import (
    MatchedPopulationError,
    load_s0_population,
    select_s0_population,
)
from tools.matched_eval.renderer import (
    RENDERER_ID,
    SYSTEM_POLICY_SHA256,
    V3_RENDERED_PROMPT_FORMAT,
    V3_RENDERER_ID,
    V3_SLOT_ORDER,
    V3_SYSTEM_POLICY,
    V4_RENDERED_PROMPT_FORMAT,
    V4_RENDERER_ID,
    V4_SLOT_ORDER,
    MatchedRendererError,
    render_memory_packet,
    render_memory_packet_v3,
    render_memory_packet_v4,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SELECTED_ORDINALS = (1, 3)


def _evidence(evidence_id: str, source_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(evidence_id, source_id, text, count_tokens(text))


def _simple_packet() -> MemoryPacket:
    return MemoryPacket(
        question_id="q-1",
        question_sha256=SHA_A,
        dated_question="[2026-08-26]\nWhat color did I choose?",
        dated_question_sha256=SHA_B,
        stage_id="S0",
        protected_evidence=(
            _evidence("e-root", "turn-1", "I chose blue."),
        ),
    )


def _rich_packet() -> MemoryPacket:
    fact_text = "The latest choice is navy."
    link_text = "The confirmation updates the earlier choice."
    return MemoryPacket(
        question_id="question-private-id",
        question_sha256=SHA_A,
        dated_question="[2026-08-26]\nWhat color is my current choice?",
        dated_question_sha256=SHA_B,
        stage_id="LATEST",
        protected_evidence=(
            _evidence(
                "evidence-private-root",
                "source-private-root",
                "User: I chose blue.",
            ),
        ),
        admitted_evidence=(
            _evidence(
                "evidence-private-added",
                "source-private-added",
                "User: Later I confirmed navy.",
            ),
        ),
        facts=(
            FactItem(
                "fact-private-id",
                fact_text,
                ("evidence-private-root", "evidence-private-added"),
                count_tokens(fact_text),
            ),
        ),
        links=(
            LinkItem(
                "link-private-id",
                link_text,
                ("evidence-private-added", "evidence-private-root"),
                count_tokens(link_text),
            ),
        ),
        answer_operators=(
            ("operator-private-id", "Prefer the latest explicit user update."),
        ),
        applied_stage_ids=("S1", "EM", "CAV", "LATEST"),
    )


def test_v3_addition_does_not_change_pinned_v2_prompt_bytes() -> None:
    before = render_memory_packet(_simple_packet())
    render_memory_packet_v3(_simple_packet())
    after = render_memory_packet(_simple_packet())

    assert before == after
    assert before.renderer_id == RENDERER_ID
    assert SYSTEM_POLICY_SHA256 == (
        "11e5e2dbaa51a6ad9638087a103f6fa84b061bbafbd851d4d953d9564f30876d"
    )
    assert before.messages_sha256 == (
        "ce935bcca403b6e31403e5ac6336ecc91883eb28c04b76447515000650f3433c"
    )
    assert before.prompt_id == (
        "edd49c755f456000e3665117b46536431c40a23ac3484b663a3c1a2423ebb1fa"
    )
    assert before.messages[-1]["content"] == (
        "Dated question:\n[2026-08-26]\nWhat color did I choose?\n\n"
        "Protected raw evidence:\n\n"
        '[P001] evidence_id="e-root" source_id="turn-1"\nI chose blue.'
    )


def test_v3_prompt_is_question_last_cued_and_uses_private_local_aliases() -> None:
    packet = _rich_packet()
    prompt = render_memory_packet_v3(packet)

    assert prompt.renderer_id == V3_RENDERER_ID
    assert prompt.format == V3_RENDERED_PROMPT_FORMAT
    assert prompt.messages[0]["content"] == V3_SYSTEM_POLICY
    assert tuple(slot.slot_id for slot in prompt.slots) == V3_SLOT_ORDER

    user = prompt.messages[-1]["content"]
    assert user.endswith(
        "Question: [2026-08-26]\nWhat color is my current choice?\n"
        "Short answer:"
    )
    assert "[P1] User: I chose blue." in user
    assert "[A1] User: Later I confirmed navy." in user
    assert "[F1 <- P1,A1] The latest choice is navy." in user
    assert (
        "[L1: A1,P1] The confirmation updates the earlier choice." in user
    )
    assert "[O1] Prefer the latest explicit user update." in user

    assert [row.projection() for row in prompt.alias_receipt] == [
        {
            "alias": "P1",
            "kind": "protected_evidence",
            "item_id": "evidence-private-root",
            "source_id": "source-private-root",
        },
        {
            "alias": "A1",
            "kind": "admitted_evidence",
            "item_id": "evidence-private-added",
            "source_id": "source-private-added",
        },
        {"alias": "F1", "kind": "fact", "item_id": "fact-private-id"},
        {"alias": "L1", "kind": "link", "item_id": "link-private-id"},
        {
            "alias": "O1",
            "kind": "answer_operator",
            "item_id": "operator-private-id",
        },
    ]

    provider_text = "\n".join(message["content"] for message in prompt.messages)
    for private_id in (
        "question-private-id",
        "evidence-private-root",
        "source-private-root",
        "evidence-private-added",
        "source-private-added",
        "fact-private-id",
        "link-private-id",
        "operator-private-id",
    ):
        assert private_id not in provider_text
    assert "evidence_id=" not in provider_text
    assert "source_id=" not in provider_text


def test_v3_raw_policy_and_post_dedup_fact_source_aliases() -> None:
    assert render_memory_packet_v3(_simple_packet()).messages[0]["content"] == (
        QA_SYSTEM_PROMPT
    )
    fact_text = "The selected source says the current value is 12."
    packet = MemoryPacket(
        question_id="q-post-dedup",
        question_sha256=SHA_A,
        dated_question="[2026-08-26]\nWhat is the current value?",
        dated_question_sha256=SHA_B,
        stage_id="EM",
        protected_evidence=(
            _evidence("protected-id", "turn-protected", "Earlier value: 10."),
        ),
        facts=(
            FactItem(
                "fact-id",
                fact_text,
                ("selected-before-dedup-id",),
                count_tokens(fact_text),
            ),
        ),
    )
    prompt = render_memory_packet_v3(packet)
    assert "[F1 <- X1] " + fact_text in prompt.messages[-1]["content"]
    assert prompt.alias_receipt[1].projection() == {
        "alias": "X1",
        "kind": "fact_source",
        "item_id": "selected-before-dedup-id",
    }
    assert "selected-before-dedup-id" not in prompt.messages[-1]["content"]

    link_text = "An invalid external link."
    with pytest.raises(MatchedRendererError, match="outside the rendered packet"):
        render_memory_packet_v3(
            MemoryPacket(
                question_id="q-bad-link",
                question_sha256=SHA_A,
                dated_question="[2026-08-26]\nWhat is linked?",
                dated_question_sha256=SHA_B,
                stage_id="CAV",
                links=(
                    LinkItem(
                        "link-id",
                        link_text,
                        ("missing-evidence-id",),
                        count_tokens(link_text),
                    ),
                ),
            )
        )


def test_v4_compact_question_sandwich_is_uniform_and_gold_blind() -> None:
    packet = _simple_packet()
    prompt = render_memory_packet_v4(packet)

    assert prompt.renderer_id == V4_RENDERER_ID
    assert prompt.format == V4_RENDERED_PROMPT_FORMAT
    assert prompt.messages[0]["content"] == QA_SYSTEM_PROMPT
    assert tuple(slot.slot_id for slot in prompt.slots) == (
        "question_preview",
        "protected_raw_evidence",
        "dated_question",
    )
    assert tuple(
        slot_id for slot_id in V4_SLOT_ORDER if slot_id in {
            slot.slot_id for slot in prompt.slots
        }
    ) == tuple(slot.slot_id for slot in prompt.slots)
    user = prompt.messages[-1]["content"]
    assert user.startswith(
        "Question preview: [2026-08-26]\nWhat color did I choose?"
    )
    assert "\n\nRetrieved excerpts from the conversation history:\n" in user
    assert "[P1] I chose blue." in user
    assert user.endswith(
        "Question: [2026-08-26]\nWhat color did I choose?\nShort answer:"
    )
    assert user.count(packet.dated_question) == 2
    assert "e-root" not in user and "turn-1" not in user
    assert prompt.alias_receipt[0].item_id == "e-root"


def test_v3_subset_keeps_original_source_ordinals(tmp_path: Path) -> None:
    retrieval = _publish(tmp_path, _retrieval(5))
    full = load_s0_population(
        retrieval,
        expected_question_count=5,
        renderer_id=V3_RENDERER_ID,
    )
    subset = select_s0_population(full, SELECTED_ORDINALS)

    assert [row.ordinal for row in subset.rows] == [1, 3]
    assert [row.packet.question_id for row in subset.rows] == ["q-1", "q-3"]
    assert subset.renderer_id == V3_RENDERER_ID
    assert subset.snapshot.population_identity_sha256 == (
        full.snapshot.population_identity_sha256
    )
    assert subset.snapshot.snapshot_id != full.snapshot.snapshot_id
    assert subset.prompt_population.logical_prompt_count == 2
    assert subset.prompt_population.unique_prompt_count == 2

    with pytest.raises(MatchedPopulationError, match="sorted unique"):
        select_s0_population(full, (3, 1))


def _run_v3_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, live.S0V2AnswerRunResult]:
    retrieval = _publish(tmp_path, _retrieval(5))
    output = tmp_path / "matched-v3"
    terra = _FakeClient(output / live.CHECKPOINT_DIR_NAME, delay_s=0)
    monkeypatch.setenv("MATCHED_V3_TEST_KEY", "test-key")
    monkeypatch.setattr(live, "_make_provider_client", lambda *_args: terra)
    result = live.run_s0_v3_answers(
        retrieval_path=retrieval,
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=2,
        api_key_env="MATCHED_V3_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )
    return retrieval, output, result


def test_fake_v3_subset_run_replay_and_verified_load_are_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, output, result = _run_v3_subset(tmp_path, monkeypatch)

    assert result.physical_provider_calls == 2
    assert result.checkpoint_hits == 0
    assert result.answer_artifact.payload["format"] == live.V3_ANSWER_RUN_FORMAT
    assert result.answer_artifact.payload["arm_label"] == live.V3_ARM_LABEL
    assert result.answer_artifact.payload["renderer_id"] == V3_RENDERER_ID
    assert result.answer_artifact.payload["gold_loaded"] is False
    assert [
        row["ordinal"] for row in result.answer_artifact.payload["questions"]
    ] == [1, 3]
    assert (output / live.V3_PREFLIGHT_NAME).is_file()
    assert not (output / live.PREFLIGHT_NAME).exists()

    replay = live.replay_s0_v3_answers(
        retrieval_path=retrieval,
        output_root=output,
        expected_run_sha256=result.answer_artifact.sha256,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )
    assert replay.run_sha256 == replay.replay_sha256 == result.answer_artifact.sha256
    assert [row.ordinal for row in replay.rows] == [1, 3]

    verified = live.load_verified_s0_v3_answer_plane(
        output / live.ANSWER_RUN_NAME,
        output / live.ANSWER_REPLAY_NAME,
        expected_run_sha256=result.answer_artifact.sha256,
        retrieval_path=retrieval,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )
    assert verified.renderer_id == V3_RENDERER_ID
    assert verified.rows == replay.rows
    assert [row.ordinal for row in verified.rows] == [1, 3]
    assert read_sealed_json(
        output / live.RUNTIME_LEDGER_REPLAY_NAME
    ).sha256 == result.runtime_ledger_artifact.sha256


class _SolCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            id=f"sol-{len(self.requests)}",
            model="codex_sdk/gpt-5.6-sol-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="CORRECT - equivalent."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )


class _SolClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _SolCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        pass


def _synthetic_gold_loader(
    *, answer_plane: live.VerifiedS0V2AnswerPlane, **_kwargs
):
    rows = []
    for answer in answer_plane.rows:
        question = f"What was choice {answer.ordinal}?"
        dated = (
            f"[Question asked at 2026/08/{answer.ordinal + 1:02d}]\n{question}"
        )
        rows.append(
            judging._GoldRow(
                ordinal=answer.ordinal,
                question_id=answer.question_id,
                question=question,
                question_sha256=quote_sha256(question),
                dated_question=dated,
                dated_question_sha256=quote_sha256(dated),
                reference=answer.prediction,
                reference_sha256=quote_sha256(answer.prediction),
                category="synthetic",
            )
        )
    return tuple(rows), identity_sha256(
        [
            {
                "ordinal": row.ordinal,
                "reference_sha256": row.reference_sha256,
            }
            for row in rows
        ]
    )


def test_fake_v3_judge_scores_only_the_original_ordinal_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, output, answer = _run_v3_subset(tmp_path, monkeypatch)
    live.replay_s0_v3_answers(
        retrieval_path=retrieval,
        output_root=output,
        expected_run_sha256=answer.answer_artifact.sha256,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )

    sol = _SolClient()
    monkeypatch.setattr(judging, "_load_gold", _synthetic_gold_loader)
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: sol)
    monkeypatch.setenv("MATCHED_V3_TEST_KEY", "test-key")

    result = judging.run_s0_v3_judge(
        answer_run_path=output / live.ANSWER_RUN_NAME,
        answer_replay_path=output / live.ANSWER_REPLAY_NAME,
        expected_answer_run_sha256=answer.answer_artifact.sha256,
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=2,
        answer_checkpoint_dir=output / live.CHECKPOINT_DIR_NAME,
        api_key_env="MATCHED_V3_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )

    assert result.physical_provider_calls == 2
    assert result.correct == 2
    assert len(sol.completions.requests) == 2
    assert result.judge_artifact.payload["format"] == judging.V3_JUDGE_FORMAT
    assert [
        row["ordinal"] for row in result.judge_artifact.payload["questions"]
    ] == [1, 3]
    assert result.score_ledger_artifact.payload["aggregate"] == {
        "baseline_correct": None,
        "candidate_correct": 2,
        "net_marginal": None,
        "regressed": None,
        "rescued": None,
    }
    assert result.score_ledger_artifact.payload["row_count"] == 2

    replay = judging.replay_s0_v3_judge(
        answer_run_path=output / live.ANSWER_RUN_NAME,
        answer_replay_path=output / live.ANSWER_REPLAY_NAME,
        expected_answer_run_sha256=answer.answer_artifact.sha256,
        expected_judge_sha256=result.judge_artifact.sha256,
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        answer_checkpoint_dir=output / live.CHECKPOINT_DIR_NAME,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=5,
        selected_ordinals=SELECTED_ORDINALS,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 2
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256
    assert replay.score_ledger_artifact.sha256 == (
        result.score_ledger_artifact.sha256
    )
