from tools.matched_eval.typed_action_semantics import (
    canonical_action_concepts,
    canonical_action_proof_terms,
    completed_action_concepts,
    linked_action_concepts,
    matched_action_concepts,
    planned_action_concepts,
)


def test_acquire_cue_links_completed_buy_and_purchase_without_literal_overlap() -> None:
    question = "Which new plants did I recently acquire?"
    bought = "I bought a peace lily and a succulent last weekend."
    purchased = "I purchased two houseplants on 2023-05-20."
    assert matched_action_concepts(question, bought) == ("acquire",)
    assert matched_action_concepts(question, purchased) == ("acquire",)
    assert completed_action_concepts(bought) == ("acquire",)
    assert completed_action_concepts(purchased) == ("acquire", "spend")


def test_completed_acquire_accepts_first_person_got_object_but_not_other_got_senses() -> None:
    assert completed_action_concepts(
        "By the way, I just got a smoker today and I'm excited to use it."
    ) == ("acquire",)
    assert completed_action_concepts("I recently got another bike.") == ("acquire",)
    assert completed_action_concepts("I got to try a smoker recipe.") == ()
    assert completed_action_concepts("I got sick yesterday.") == ()
    assert completed_action_concepts("You can get a smoker someday.") == ()


def test_brought_object_home_is_completed_acquire_but_transport_near_misses_are_not() -> None:
    assert completed_action_concepts(
        "The peace lily was losing leaves since I brought it home."
    ) == ("acquire",)
    assert completed_action_concepts(
        "We recently brought our new succulent back home."
    ) == ("acquire",)
    assert "acquire" not in completed_action_concepts(
        "I brought it to the repair shop."
    )
    assert "acquire" not in completed_action_concepts("She brought it home.")


def test_service_variants_and_first_person_planning_stay_distinct_from_completion() -> None:
    assert linked_action_concepts(
        "The replacement tire needs regular maintenance."
    ) == ("service",)
    assert canonical_action_concepts(
        "The replacement tire needs regular maintenance."
    ) == ()
    assert completed_action_concepts("I replaced the worn tire.") == ("service",)
    assert completed_action_concepts(
        "I might visit a service shop for replacement tire maintenance."
    ) == ()
    assert planned_action_concepts(
        "I'm looking into getting a tire, and I think it is time to replace it."
    ) == ("acquire", "service")
    assert planned_action_concepts("I replaced the tire. I plan a vacation.") == ()
    assert planned_action_concepts("You should plan to replace the tire.") == ()


def test_took_person_to_venue_is_completed_visit_but_transport_near_misses_are_not() -> None:
    assert completed_action_concepts(
        "I took my niece to the Natural History Museum on 2/8."
    ) == ("visit",)
    assert completed_action_concepts(
        "We recently took our friend to the city gallery."
    ) == ("visit",)
    assert "visit" not in completed_action_concepts(
        "I took my medication to the doctor."
    )
    assert "visit" not in completed_action_concepts(
        "You took your niece to the museum."
    )
    assert completed_action_concepts("I might visit one later.") == ()


def test_action_normalization_preserves_distinct_return_and_pickup_roles() -> None:
    assert canonical_action_concepts("return the old boots") == ("return",)
    assert canonical_action_concepts("picked up the larger boots") == ("pickup",)
    assert not matched_action_concepts(
        "What do I need to return?",
        "I picked up the larger boots.",
    )


def test_action_proof_terms_preserve_valid_surfaces_and_split_compounds() -> None:
    assert canonical_action_proof_terms(
        "I purchased the silver camera.", "acquire"
    ) == ("purchased",)
    assert canonical_action_proof_terms(
        "The easy-to-clean lens was cleaned.", "clean"
    ) == ("clean", "cleaned")
