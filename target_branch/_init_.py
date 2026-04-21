"""Target branch package (Fig. 2): egocentric encoder E."""

from target_branch.branch import TargetBranch
from target_branch.config import TargetBranchConfig
from target_branch.encoder import SharedVideoEncoder

__all__ = ["SharedVideoEncoder", "TargetBranch", "TargetBranchConfig"]
