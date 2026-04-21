"""Retrieval branch package (Fig. 2): cross-view retrieval + class-adaptive k."""

from retrieval_branch.branch import RetrievalBranch
from retrieval_branch.config import RetrievalBranchConfig
from retrieval_branch.utils import (
    class_adaptive_k,
    frequency_bins_from_counts,
    retrieve_topk_exocentric,
)

__all__ = [
    "RetrievalBranch",
    "RetrievalBranchConfig",
    "retrieve_topk_exocentric",
    "class_adaptive_k",
    "frequency_bins_from_counts",
]
