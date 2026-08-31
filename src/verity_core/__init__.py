"""verity-core: the shared foundation for Verity Labs' RL environment audits."""

from verity_core.config import VerityConfig, load_config
from verity_core.env import (
    DOMAINS,
    REWARD_TYPES,
    Domain,
    Observation,
    RewardResult,
    RewardType,
    StepResult,
    TaskSpec,
    VerityEnv,
)
from verity_core.runner import ExecResult, ResourceLimits, SandboxError, SandboxRunner

__version__ = "0.1.0"

__all__ = [
    "DOMAINS",
    "REWARD_TYPES",
    "Domain",
    "ExecResult",
    "Observation",
    "ResourceLimits",
    "RewardResult",
    "RewardType",
    "SandboxError",
    "SandboxRunner",
    "StepResult",
    "TaskSpec",
    "VerityConfig",
    "VerityEnv",
    "__version__",
    "load_config",
]
