"""Cross-view integration package (Fig. 2): fuse ego and exocentric features."""

from cross_view_integration.attention import CrossViewAttentionFusion
from cross_view_integration.config import CrossViewIntegrationConfig
from cross_view_integration.final_fusion import FinalIntegration
from cross_view_integration.module import CrossViewIntegrationModule
from cross_view_integration.ops import cosine_sim
from cross_view_integration.similarity_agg import SimilarityGuidedAggregation

__all__ = [
    "CrossViewIntegrationModule",
    "CrossViewIntegrationConfig",
    "CrossViewAttentionFusion",
    "FinalIntegration",
    "SimilarityGuidedAggregation",
    "cosine_sim",
]
