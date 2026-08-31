"""The universal environment interface every Verity tool programs against.

RL environments reach us in many upstream formats. Adapters in
:mod:`verity_core.adapters` normalize each of them into :class:`VerityEnv`, so the
four audit tools (RedTeam, Signal, Clean, Stable) only ever see one shape.

:class:`VerityEnv` is a :class:`~typing.Protocol` rather than an abstract base class:
adapters conform structurally and never need to inherit from anything, which keeps
third-party environments usable without wrapping them in our class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

Domain = Literal["browser", "gui", "tool_use", "code", "math", "other"]
"""Task families we audit. Determines which probes a tool is allowed to run."""

RewardType = Literal["binary", "partial"]
"""``binary`` verifiers return pass/fail; ``partial`` return graded credit."""

DOMAINS: tuple[Domain, ...] = get_args(Domain)
REWARD_TYPES: tuple[RewardType, ...] = get_args(RewardType)

__all__ = [
    "DOMAINS",
    "REWARD_TYPES",
    "Domain",
    "Observation",
    "RewardResult",
    "RewardType",
    "StepResult",
    "TaskSpec",
    "VerityEnv",
]


@dataclass(slots=True)
class TaskSpec:
    """Immutable-by-convention metadata describing one auditable task.

    ``source`` and ``commit`` together pin the task to an exact upstream revision;
    an audit is only reproducible if both are recorded.
    """

    id: str
    domain: Domain
    source: str
    commit: str
    reward_type: RewardType
    instructions: str
    has_gold: bool = False

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}; expected one of {DOMAINS}")
        if self.reward_type not in REWARD_TYPES:
            raise ValueError(
                f"unknown reward_type {self.reward_type!r}; expected one of {REWARD_TYPES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "source": self.source,
            "commit": self.commit,
            "reward_type": self.reward_type,
            "instructions": self.instructions,
            "has_gold": self.has_gold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(
            id=data["id"],
            domain=data["domain"],
            source=data.get("source", ""),
            commit=data.get("commit", ""),
            reward_type=data.get("reward_type", "binary"),
            instructions=data.get("instructions", ""),
            has_gold=bool(data.get("has_gold", False)),
        )


@dataclass(slots=True)
class Observation:
    """What the environment shows the policy after a reset or step."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": dict(self.metadata)}


@dataclass(slots=True)
class StepResult:
    """Outcome of a single action, unpackable as ``(obs, reward, done, info)``."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Any:
        yield from (self.observation, self.reward, self.done, self.info)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": dict(self.info),
        }


@dataclass(slots=True)
class RewardResult:
    """A verifier's verdict on a submission.

    ``reward`` and ``verdict`` are reported separately on purpose: a partial-credit
    verifier can hand back 0.6 while still failing the task, and several audit axes
    depend on being able to see that disagreement.

    ``verifier_logs`` is the raw grader output, kept verbatim because it is the
    evidence a scorecard cites.
    """

    reward: float
    verdict: bool
    verifier_logs: str = ""
    verifier_state: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Any:
        yield from (self.reward, self.verdict, self.verifier_logs, self.verifier_state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "verdict": self.verdict,
            "verifier_logs": self.verifier_logs,
            "verifier_state": dict(self.verifier_state),
        }


@runtime_checkable
class VerityEnv(Protocol):
    """The seven methods every audited environment must expose."""

    def spec(self) -> TaskSpec:
        """Return the task metadata, including the pinned upstream revision."""
        ...

    def reset(self) -> Observation:
        """Return the environment to its initial state and hand back the first observation."""
        ...

    def step(self, action: str) -> StepResult:
        """Apply one action and report the resulting observation, reward, and done flag."""
        ...

    def verify(self, submission: str) -> RewardResult:
        """Run the environment's own verifier against ``submission``."""
        ...

    def gold_solution(self) -> str | None:
        """Return the known-correct solution, or ``None`` when the task ships without one."""
        ...

    def snapshot(self) -> bytes:
        """Freeze current state into an opaque token that :meth:`restore` accepts."""
        ...

    def restore(self, snap: bytes) -> None:
        """Restore state from a token previously produced by :meth:`snapshot`."""
        ...
