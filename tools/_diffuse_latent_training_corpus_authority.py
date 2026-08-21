"""Public authority boundary for the route-v2 latent-training corpus.

The repository does not yet contain independently audited genuine-output
pins.  Candidate inspection is therefore available under explicitly named
false-only APIs, while every public production verifier is closed before it
can inspect its supplied path.
"""

from __future__ import annotations

from os import PathLike
from typing import NoReturn

from tools._diffuse_latent_training_corpus_authority_filesystem import (
    GenericCorpusBinding,
    inspect_generic_corpus_binding,
    verify_latent_training_corpus_candidate,
    verify_latent_training_fit_candidate,
    verify_latent_training_validation_candidate,
)
from tools._diffuse_latent_training_corpus_authority_models import (
    AUTHORITY_NOT_PINNED_REASON,
    CANDIDATE_EXECUTION_DISABLED_REASON,
    CANDIDATE_RECEIPT_NAME,
    PHASE_CANDIDATE_NAME,
    PRODUCTION_CANDIDATE_NAME,
    DeclaredProductionExecutionCoordinates,
    ProductionAuthorityNotPinned,
    ProductionAuthorityStatus,
    ProductionCandidateExecutionStatus,
    ProductionCandidateExecutionUnavailable,
    ProductionCandidatePublicationReceipt,
    ProductionCorpusCandidateReceipt,
    ProductionExternalLock,
    ProductionLatentTrainingCorpusError,
    ProductionPhaseCandidateReceipt,
    VerifiedLatentTrainingCorpusCandidate,
    VerifiedLatentTrainingPhaseCandidate,
    locked_production_external_lock,
)


def production_authority_status() -> ProductionAuthorityStatus:
    """Report the tracked, fail-closed production-authority state."""

    return ProductionAuthorityStatus()


def _define_closed_verifier(
    public_name: str,
    required_phases: tuple[str, ...],
):
    # These are deliberately closure-owned code literals.  There is no module
    # global, caller parameter, environment variable, or disk value that can
    # substitute for the later audited tracked-code pins.
    pinned_publication_sha256: None = None
    pinned_candidate_sha256: None = None
    pinned_fit_phase_sha256: None = None
    pinned_validation_phase_sha256: None = None

    def verify(path: str | PathLike[str]) -> NoReturn:
        # Do not normalize, stringify, stat, open, or decode ``path`` while the
        # pins are absent.  Hostile path-like objects must remain untouched.
        if (
            pinned_publication_sha256 is None
            or pinned_candidate_sha256 is None
            or (
                "fit" in required_phases
                and pinned_fit_phase_sha256 is None
            )
            or (
                "validation" in required_phases
                and pinned_validation_phase_sha256 is None
            )
        ):
            raise ProductionAuthorityNotPinned(
                f"{public_name} is unavailable: {AUTHORITY_NOT_PINNED_REASON}"
            )
        del path
        raise ProductionAuthorityNotPinned(
            f"{public_name} has no tracked production verifier implementation"
        )

    verify.__name__ = public_name
    verify.__qualname__ = public_name
    verify.__doc__ = (
        "Fail before path access until audited genuine-output identities are "
        "pinned in tracked code."
    )
    return verify


verify_production_latent_training_corpus = _define_closed_verifier(
    "verify_production_latent_training_corpus", ("fit", "validation")
)
verify_production_latent_training_fit_corpus = _define_closed_verifier(
    "verify_production_latent_training_fit_corpus", ("fit",)
)
verify_production_latent_training_validation_corpus = _define_closed_verifier(
    "verify_production_latent_training_validation_corpus", ("validation",)
)


__all__ = [
    "AUTHORITY_NOT_PINNED_REASON",
    "CANDIDATE_EXECUTION_DISABLED_REASON",
    "CANDIDATE_RECEIPT_NAME",
    "DeclaredProductionExecutionCoordinates",
    "GenericCorpusBinding",
    "PHASE_CANDIDATE_NAME",
    "PRODUCTION_CANDIDATE_NAME",
    "ProductionAuthorityNotPinned",
    "ProductionAuthorityStatus",
    "ProductionCandidateExecutionStatus",
    "ProductionCandidateExecutionUnavailable",
    "ProductionCandidatePublicationReceipt",
    "ProductionCorpusCandidateReceipt",
    "ProductionExternalLock",
    "ProductionLatentTrainingCorpusError",
    "ProductionPhaseCandidateReceipt",
    "VerifiedLatentTrainingCorpusCandidate",
    "VerifiedLatentTrainingPhaseCandidate",
    "inspect_generic_corpus_binding",
    "locked_production_external_lock",
    "production_authority_status",
    "verify_latent_training_corpus_candidate",
    "verify_latent_training_fit_candidate",
    "verify_latent_training_validation_candidate",
    "verify_production_latent_training_corpus",
    "verify_production_latent_training_fit_corpus",
    "verify_production_latent_training_validation_corpus",
]
