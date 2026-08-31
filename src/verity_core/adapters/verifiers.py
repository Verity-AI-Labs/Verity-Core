"""Adapter for PrimeIntellect / ``verifiers``-style environments.

Unlike the container formats, these environments are Python: a task config, a rollout
function, and a reward function. They run in this process, so ``snapshot`` and
``restore`` capture the rollout transcript rather than a filesystem image.

Upstream reward functions have no single signature, so this adapter inspects the
callable and adapts, and normalizes whatever it returns (a float, a bool, a dict, or a
``(reward, info)`` pair) into a :class:`~verity_core.env.RewardResult`.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping
from types import TracebackType
from typing import Any

from verity_core.adapters.base import ManifestError, parse_task_spec, resolve_callable
from verity_core.env import Observation, RewardResult, StepResult, TaskSpec

logger = logging.getLogger(__name__)

REWARD_KEYS = ("reward", "score", "value")
VERDICT_KEYS = ("verdict", "passed", "success", "correct")

__all__ = ["VerifiersAdapter"]


class VerifiersAdapter:
    """Wraps a rollout+reward environment as a :class:`~verity_core.env.VerityEnv`.

    Recognised manifest fields:

    ``reward``
        ``"package.module:function"`` reference to the reward function (required).
    ``rollout``
        Optional ``"package.module:function"`` reference, exposed via :meth:`rollout`.
    ``task``
        Mapping handed to the reward and rollout functions as the task config.
    ``pass_threshold``
        Reward at or above which :attr:`RewardResult.verdict` is true. Defaults to 1.0.
    ``gold_solution``
        Inline reference solution, if the environment ships one.
    """

    default_domain = "other"

    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        reward_fn: Any | None = None,
        rollout_fn: Any | None = None,
    ) -> None:
        self.entry = dict(entry)
        self.context = f"VerifiersAdapter for task {self.entry.get('id', '<missing id>')!r}"
        self._spec = parse_task_spec(self.entry, default_domain=self.default_domain)

        task = self.entry.get("task") or {}
        if not isinstance(task, Mapping):
            raise ManifestError(f"{self.context}: 'task' must be a mapping")
        self.task: dict[str, Any] = dict(task)
        self.pass_threshold = float(self.entry.get("pass_threshold", 1.0))

        # Callables may be injected directly (tests, tools building manifests in code)
        # or named in the manifest and imported on first use, so that listing a corpus
        # does not import every environment's dependencies.
        self._reward_fn = reward_fn
        self._rollout_fn = rollout_fn
        self._transcript: list[str] = []

    def _resolve_reward_fn(self) -> Any:
        if self._reward_fn is None:
            reference = self.entry.get("reward")
            if not reference:
                message = f"{self.context}: manifest entry needs a 'reward' field"
                logger.error("%s", message)
                raise ManifestError(message)
            self._reward_fn = resolve_callable(str(reference), context=self.context)
            logger.debug("resolved reward function env_id=%s ref=%s", self._spec.id, reference)
        return self._reward_fn

    def _resolve_rollout_fn(self) -> Any | None:
        if self._rollout_fn is None:
            reference = self.entry.get("rollout")
            if not reference:
                return None
            self._rollout_fn = resolve_callable(str(reference), context=self.context)
        return self._rollout_fn

    def spec(self) -> TaskSpec:
        return self._spec

    def reset(self) -> Observation:
        logger.info("env reset env_id=%s", self._spec.id)
        self._transcript.clear()
        return Observation(
            text=self._spec.instructions,
            metadata={"task_id": self._spec.id, "task": dict(self.task)},
        )

    def step(self, action: str) -> StepResult:
        """Score ``action`` as a complete submission and finish the episode.

        Rollout+reward environments are single-turn by construction: the reward
        function scores a finished response, not an intermediate move.
        """
        # TODO: support multi-turn verifiers environments by driving the upstream
        # rollout function turn by turn instead of scoring the first action.
        self._transcript.append(action)
        result = self.verify(action)
        return StepResult(
            observation=Observation(text=result.verifier_logs, metadata={"reward": result.reward}),
            reward=result.reward,
            done=True,
            info={"verdict": result.verdict, "verifier_state": result.verifier_state},
        )

    def rollout(self, *args: Any, **kwargs: Any) -> Any:
        """Call the upstream rollout function, passing arguments through untouched.

        Not part of the :class:`~verity_core.env.VerityEnv` interface; tools that want
        the environment's own trajectory generator reach for this directly.
        """
        fn = self._resolve_rollout_fn()
        if fn is None:
            raise ManifestError(f"{self.context}: manifest entry declares no 'rollout' function")
        return fn(*args, **kwargs)

    def verify(self, submission: str) -> RewardResult:
        """Run the environment's reward function against ``submission``."""
        fn = self._resolve_reward_fn()
        raw = self._call_reward_fn(fn, submission)
        result = self._normalize(raw)
        logger.info(
            "verify complete env_id=%s verdict=%s reward=%.3f threshold=%.3f",
            self._spec.id,
            result.verdict,
            result.reward,
            self.pass_threshold,
        )
        return result

    def _call_reward_fn(self, fn: Any, submission: str) -> Any:
        """Invoke the reward function using whichever signature it declares.

        Keyword calling is preferred so that argument order in upstream code cannot
        silently swap the task and the submission.
        """
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            parameters = {}  # type: ignore[assignment]

        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if accepts_kwargs or {"task", "submission"} <= set(parameters):
            return fn(task=self.task, submission=submission)
        if "completion" in parameters:
            return fn(task=self.task, completion=submission)
        if len(parameters) == 1:
            return fn(submission)
        return fn(self.task, submission)

    def _normalize(self, raw: Any) -> RewardResult:
        """Coerce a reward function's return value into a :class:`RewardResult`."""
        if isinstance(raw, RewardResult):
            return raw

        state: dict[str, Any] = {}
        logs = ""
        verdict: bool | None = None

        if isinstance(raw, tuple | list) and len(raw) == 2:
            raw, extra = raw[0], raw[1]
            if isinstance(extra, Mapping):
                state = dict(extra)
            else:
                logs = str(extra)

        if isinstance(raw, Mapping):
            state = {**state, **dict(raw)}
            logs = str(raw.get("logs") or raw.get("verifier_logs") or logs)
            verdict = next(
                (bool(raw[key]) for key in VERDICT_KEYS if key in raw),
                None,
            )
            reward_value = next((raw[key] for key in REWARD_KEYS if key in raw), None)
            if reward_value is None:
                if verdict is None:
                    raise ManifestError(
                        f"{self.context}: reward function returned {raw!r}, which has no "
                        f"reward key ({', '.join(REWARD_KEYS)}) or verdict key "
                        f"({', '.join(VERDICT_KEYS)})"
                    )
                reward_value = 1.0 if verdict else 0.0
            raw = reward_value

        # bool is checked before the numeric branch because it is a subclass of int.
        if isinstance(raw, bool):
            reward = 1.0 if raw else 0.0
            verdict = raw if verdict is None else verdict
        elif isinstance(raw, int | float):
            reward = float(raw)
        else:
            raise ManifestError(
                f"{self.context}: reward function returned unsupported type "
                f"{type(raw).__name__}; expected a number, bool, mapping, or pair"
            )

        return RewardResult(
            reward=reward,
            verdict=reward >= self.pass_threshold if verdict is None else verdict,
            verifier_logs=logs,
            verifier_state=state,
        )

    def gold_solution(self) -> str | None:
        gold = self.entry.get("gold_solution")
        return None if gold is None else str(gold)

    def snapshot(self) -> bytes:
        """Freeze the transcript. There is no container here, so state is the rollout."""
        return json.dumps(
            {
                "version": 1,
                "kind": "verifiers-transcript",
                "task_id": self._spec.id,
                "transcript": list(self._transcript),
            }
        ).encode("utf-8")

    def restore(self, snap: bytes) -> None:
        try:
            token = json.loads(snap.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"{self.context}: snapshot is not valid snapshot JSON") from exc
        if token.get("kind") != "verifiers-transcript":
            raise ManifestError(f"{self.context}: unsupported snapshot kind {token.get('kind')!r}")
        self._transcript = [str(item) for item in token.get("transcript") or []]

    def close(self) -> None:
        """Drop the transcript. There is no container here, so nothing else to release."""
        logger.debug("closing verifiers env id=%s", self._spec.id)
        self._transcript.clear()

    def __enter__(self) -> VerifiersAdapter:
        self.reset()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
