"""Bounded post-retrieval fusion over exact evidence atoms."""

from memory_condense.search.fusion.latent_router import (
    LatentEvidenceRouter,
    LatentRouterForward,
)
from memory_condense.search.fusion.feature_batch import NodeFeatureBatch
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    DeclaredFeatureExtractorIdentity,
    EvidenceFusionPlan,
    ExtractiveGroup,
    FusionAtomRef,
    FusionCaps,
    FusionMode,
    LatentMembership,
    NodeFeatureReceipt,
    RouterArchitectureReceipt,
    RouterStateReceipt,
    RouterTrainingStatus,
)
from memory_condense.search.fusion.planner import (
    build_evidence_fusion_plan,
    validate_matched_fusion_pair,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomBatchReceipt,
    QwenAtomFeatureCaps,
    QwenAtomFeatureOperationReceipt,
    QwenAtomFeatureProviderReceipt,
    QwenAtomRowReceipt,
)
from memory_condense.search.fusion.qwen_features import QwenAtomFeatureProvider

__all__ = [
    "AuthoritativeHyperedge",
    "DeclaredFeatureExtractorIdentity",
    "EvidenceFusionPlan",
    "ExtractiveGroup",
    "FusionAtomRef",
    "FusionCaps",
    "FusionMode",
    "LatentEvidenceRouter",
    "LatentMembership",
    "LatentRouterForward",
    "NodeFeatureBatch",
    "NodeFeatureReceipt",
    "RouterArchitectureReceipt",
    "RouterStateReceipt",
    "RouterTrainingStatus",
    "QwenAtomBatchReceipt",
    "QwenAtomFeatureCaps",
    "QwenAtomFeatureOperationReceipt",
    "QwenAtomFeatureProvider",
    "QwenAtomFeatureProviderReceipt",
    "QwenAtomRowReceipt",
    "build_evidence_fusion_plan",
    "validate_matched_fusion_pair",
]
