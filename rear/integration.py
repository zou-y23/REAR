"""Backward-compatible re-exports from the cross_view_integration package."""

from cross_view_integration import (
    CrossViewAttentionFusion,
    CrossViewIntegrationModule,
    FinalIntegration,
    SimilarityGuidedAggregation,
    cosine_sim,
)

__all__ = [
    "CrossViewAttentionFusion",
    "CrossViewIntegrationModule",
    "FinalIntegration",
    "SimilarityGuidedAggregation",
    "cosine_sim",
]
