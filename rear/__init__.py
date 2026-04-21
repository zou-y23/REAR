"""
REAR training / loss utilities. The three Fig. 2 modules live in sibling packages:
  target_branch/, retrieval_branch/, cross_view_integration/
"""

from cross_view_integration import CrossViewIntegrationModule
from rear.model import REAR
from retrieval_branch import RetrievalBranch
from target_branch import TargetBranch

__all__ = [
    "REAR",
    "TargetBranch",
    "RetrievalBranch",
    "CrossViewIntegrationModule",
]
