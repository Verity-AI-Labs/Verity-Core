"""verity-core: the shared foundation for Verity Labs' RL environment audits."""

import logging

from verity_core.adapters import (
    DockerTestAdapter,
    ManifestError,
    TerminalAdapter,
    VerifiersAdapter,
    load_env,
)
from verity_core.config import ResolvedConfig, VerityConfig, load_config, resolve_config
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
from verity_core.logs import configure_logging
from verity_core.models import ModelClient, ModelError, ModelResponse, ResponseCache, TokenUsage
from verity_core.runner import ExecResult, ResourceLimits, SandboxError, SandboxRunner
from verity_core.scorecard import AXES, AxisValue, Scorecard

__version__ = "0.1.0"

# A library must not configure logging for its host. The placeholder keeps Python from
# printing "no handlers could be found" while leaving handler and level choices to the
# tool's entry point, which can call configure_logging() if it wants our defaults.
logging.getLogger("verity_core").addHandler(logging.NullHandler())

__all__ = [
    "AXES",
    "DOMAINS",
    "REWARD_TYPES",
    "AxisValue",
    "DockerTestAdapter",
    "Domain",
    "ExecResult",
    "ManifestError",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "Observation",
    "ResolvedConfig",
    "ResourceLimits",
    "ResponseCache",
    "RewardResult",
    "RewardType",
    "SandboxError",
    "SandboxRunner",
    "Scorecard",
    "StepResult",
    "TaskSpec",
    "TerminalAdapter",
    "TokenUsage",
    "VerifiersAdapter",
    "VerityConfig",
    "VerityEnv",
    "__version__",
    "configure_logging",
    "load_config",
    "load_env",
    "resolve_config",
]
