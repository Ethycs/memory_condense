from __future__ import annotations

from memory_condense.associations.transition_policy import (
    CausalTransitionPolicy,
    TransitionCandidate,
)


def _candidate(
    destination: str,
    attention: tuple[float, ...],
    deltas: tuple[tuple[float, ...], ...] = (),
):
    return TransitionCandidate(
        destination_id=destination,
        base_score=0.5,
        head_attention=attention,
        head_cav_deltas=deltas,
    )


def test_feedback_is_delayed_and_changes_the_next_ranking() -> None:
    policy = CausalTransitionPolicy(transition_weight=1.0, prior_mass=0.1)
    candidates = (
        _candidate("right", (1.0, 0.0)),
        _candidate("wrong", (0.0, 1.0)),
    )

    before = policy.propose(
        source_id="current",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(0.0, 0.0),
        candidates=candidates,
        turn=10,
        top_k=2,
    )
    # Merely proposing cannot train on its own prediction.
    assert policy.snapshot()["heads"] == []

    feedback = policy.observe(
        before,
        actual_destination_id="right",
        actual_next_role="assistant",
        next_cav=(1.0, 0.0),
    )
    assert feedback.target_was_selected is True

    after = policy.propose(
        source_id="current",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(1.0, 0.0),
        candidates=candidates,
        turn=12,
        top_k=2,
    )
    assert after.selected[0].candidate.destination_id == "right"
    assert after.selected[0].score > after.selected[1].score
    assert after.selected[0].head_gates[0] > after.selected[0].head_gates[1]


def test_projected_ov_alignment_controls_head_reward() -> None:
    policy = CausalTransitionPolicy(prior_mass=0.1)
    decision = policy.propose(
        source_id="s1",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(0.0, 0.0),
        candidates=(
            _candidate(
                "s2",
                (1.0, 1.0),
                ((1.0, 0.0), (-1.0, 0.0)),
            ),
        ),
        turn=1,
    )

    policy.observe(
        decision,
        actual_destination_id="s2",
        actual_next_role="assistant",
        next_cav=(1.0, 0.0),
    )
    gates = policy.head_gates("user", "assistant", head_count=2, turn=3)
    assert gates[0] > gates[1]


def test_cav_only_feedback_does_not_need_an_exact_destination() -> None:
    policy = CausalTransitionPolicy(prior_mass=0.1)
    decision = policy.propose(
        source_id="s1",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(0.0, 0.0),
        candidates=(
            _candidate(
                "candidate",
                (1.0, 1.0),
                ((1.0, 0.0), (-1.0, 0.0)),
            ),
        ),
        turn=1,
    )

    feedback = policy.observe(
        decision,
        actual_destination_id=None,
        actual_next_role="assistant",
        next_cav=(1.0, 0.0),
    )

    assert feedback.actual_destination_id is None
    gates = policy.head_gates("user", "assistant", head_count=2, turn=3)
    assert gates[0] > 1.0 > gates[1]


def test_recent_cav_velocity_can_rerank_the_next_transition() -> None:
    policy = CausalTransitionPolicy(
        transition_weight=0.0,
        velocity_weight=1.0,
    )
    decision = policy.propose(
        source_id="s1",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(1.0, 0.0),
        cav_velocity=(1.0, 0.0),
        candidates=(
            TransitionCandidate(
                destination_id="wrong",
                base_score=0.6,
                head_attention=(1.0,),
                head_cav_deltas=((-1.0, 0.0),),
            ),
            TransitionCandidate(
                destination_id="continuation",
                base_score=0.5,
                head_attention=(1.0,),
                head_cav_deltas=((1.0, 0.0),),
            ),
        ),
        turn=1,
        top_k=2,
    )

    assert decision.selected[0].candidate.destination_id == "continuation"


def test_role_transitions_learn_separate_head_gates() -> None:
    policy = CausalTransitionPolicy(prior_mass=0.1)
    candidates = (
        _candidate("answer", (1.0, 0.0)),
        _candidate("topic", (0.0, 1.0)),
    )
    decision = policy.propose(
        source_id="question",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(0.0,),
        candidates=candidates,
        turn=2,
        top_k=2,
    )
    policy.observe(
        decision,
        actual_destination_id="answer",
        actual_next_role="assistant",
        next_cav=(1.0,),
    )

    assert policy.head_gates("user", "assistant", head_count=2, turn=4)[0] > 1.0
    assert policy.head_gates("assistant", "user", head_count=2, turn=4) == (
        1.0,
        1.0,
    )


def test_snapshot_roundtrip_keeps_only_scalar_statistics() -> None:
    policy = CausalTransitionPolicy(prior_mass=0.1)
    decision = policy.propose(
        source_id="s1",
        from_role="user",
        expected_next_role="assistant",
        source_cav=(12.345, 67.89),
        candidates=(_candidate("s2", (1.0,)),),
        turn=1,
    )
    policy.observe(
        decision,
        actual_destination_id="s2",
        actual_next_role="assistant",
        next_cav=(13.0, 68.0),
    )

    snapshot = policy.snapshot()
    serialized = str(snapshot)
    assert "source_cav" not in serialized
    assert "head_cav_deltas" not in serialized
    assert "12.345" not in serialized

    restored = CausalTransitionPolicy.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot


def test_edge_statistics_are_hard_capped() -> None:
    policy = CausalTransitionPolicy(max_edge_statistics=2)
    for turn in range(3):
        decision = policy.propose(
            source_id=f"s{turn}",
            from_role="user",
            expected_next_role="assistant",
            source_cav=(0.0,),
            candidates=(_candidate(f"d{turn}", (1.0,)),),
            turn=turn * 2,
        )
        policy.observe(
            decision,
            actual_destination_id=f"d{turn}",
            actual_next_role="assistant",
            next_cav=(1.0,),
        )

    assert len(policy.snapshot()["edges"]) == 2
