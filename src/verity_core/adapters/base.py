"""Manifest parsing and the shared container-backed adapter implementation.

Two of the three upstream formats we support (Terminal Wrench and generic
container+test setups) differ only in their conventions: where the submission goes,
what applies it, and which command grades it. Those differences are expressed as
class attributes on :class:`ContainerEnv` subclasses rather than as duplicated
lifecycle code, so there is exactly one place where sandbox handling can go wrong.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from types import TracebackType
from typing import Any

from verity_core.env import (
    DOMAINS,
    REWARD_TYPES,
    Observation,
    RewardResult,
    StepResult,
    TaskSpec,
)
from verity_core.runner import ExecResult, ResourceLimits, SandboxRunner

logger = logging.getLogger(__name__)

GOLD_MANIFEST_KEYS = ("gold_solution", "gold_solution_path", "gold_patch")

__all__ = [
    "ContainerEnv",
    "ManifestError",
    "parse_task_spec",
    "require",
    "resolve_callable",
    "resource_limits_from",
]


class ManifestError(ValueError):
    """Raised when a manifest entry is missing a field or describes it incorrectly."""


def require(entry: Mapping[str, Any], key: str, *, context: str) -> Any:
    """Fetch a manifest field, failing loudly when it is absent or blank."""
    value = entry.get(key)
    if value is None or value == "":
        message = f"{context}: manifest entry needs a non-empty {key!r} field"
        logger.error("%s", message)
        raise ManifestError(message)
    return value


def parse_task_spec(entry: Mapping[str, Any], *, default_domain: str = "other") -> TaskSpec:
    """Build a :class:`TaskSpec` from a manifest entry.

    ``source`` and ``commit`` are permitted to be empty so that local, in-development
    manifests still load, but they are what makes an audit reproducible and every
    corpus entry is expected to carry them.
    """
    context = f"task {entry.get('id', '<missing id>')!r}"
    env_id = str(require(entry, "id", context=context))

    domain = str(entry.get("domain") or default_domain)
    if domain not in DOMAINS:
        message = f"{context}: unknown domain {domain!r}; expected one of {DOMAINS}"
        logger.error("%s", message)
        raise ManifestError(message)

    reward_type = str(entry.get("reward_type") or "binary")
    if reward_type not in REWARD_TYPES:
        message = f"{context}: unknown reward_type {reward_type!r}; expected one of {REWARD_TYPES}"
        logger.error("%s", message)
        raise ManifestError(message)

    declared_gold = entry.get("has_gold")
    inferred_gold = any(entry.get(key) for key in GOLD_MANIFEST_KEYS)
    return TaskSpec(
        id=env_id,
        domain=domain,  # type: ignore[arg-type]
        source=str(entry.get("source") or ""),
        commit=str(entry.get("commit") or ""),
        reward_type=reward_type,  # type: ignore[arg-type]
        instructions=str(entry.get("instructions") or entry.get("instruction") or ""),
        has_gold=inferred_gold if declared_gold is None else bool(declared_gold),
    )


def resource_limits_from(entry: Mapping[str, Any]) -> ResourceLimits:
    """Read sandbox limits from a manifest, falling back to the safe defaults."""
    raw = entry.get("limits") or {}
    if not isinstance(raw, Mapping):
        raise ManifestError(f"task {entry.get('id')!r}: 'limits' must be a mapping")
    defaults = ResourceLimits()
    return ResourceLimits(
        cpu_count=float(raw.get("cpu_count", defaults.cpu_count)),
        memory_limit=str(raw.get("memory_limit", defaults.memory_limit)),
        timeout_seconds=int(
            raw.get("timeout_seconds", entry.get("timeout", defaults.timeout_seconds))
        ),
        network_disabled=bool(raw.get("network_disabled", defaults.network_disabled)),
    )


def resolve_callable(reference: str, *, context: str) -> Any:
    """Import a ``"package.module:attribute"`` reference from a manifest.

    Resolution is deliberately deferred to first use by callers: listing a corpus
    should not import every environment's dependencies.
    """
    if ":" not in reference:
        raise ManifestError(
            f"{context}: {reference!r} must be written as 'package.module:attribute'"
        )
    module_name, _, attribute = reference.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ManifestError(f"{context}: cannot import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ManifestError(
            f"{context}: module {module_name!r} has no attribute {attribute!r}"
        ) from exc


class ContainerEnv:
    """A :class:`~verity_core.env.VerityEnv` backed by a container and a test command.

    Subclasses set the format's conventions via the ``default_*`` class attributes.
    Nothing here touches Docker at construction time, so a manifest can be loaded and
    inspected on a machine with no daemon running.
    """

    default_domain: str = "other"
    default_submission_path: str = "/workspace/submission.txt"
    default_test_command: str = ""
    default_apply_command: str = ""
    gold_aliases: tuple[str, ...] = ()
    """Format-specific manifest keys that mean the same thing as ``gold_solution``."""

    reset_before_verify: bool = False
    """Whether each :meth:`verify` starts from a clean container.

    This is the difference between the two container formats. An agentic task is
    graded on the container the agent worked in, so resetting would throw away the
    very thing under test. A patch-and-test task is graded on the submission alone, so
    reusing a container would leave the previous submission applied and let one trial
    contaminate the next. Manifests can override it per task.
    """

    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        runner: SandboxRunner | None = None,
        docker_client: Any | None = None,
    ) -> None:
        self.entry = dict(entry)
        self.context = f"{type(self).__name__} for task {self.entry.get('id', '<missing id>')!r}"
        # Fold format-specific aliases in before the spec is parsed, so `has_gold` is
        # inferred from them too.
        for alias in self.gold_aliases:
            if self.entry.get(alias) and not self.entry.get("gold_solution"):
                self.entry["gold_solution"] = self.entry[alias]
        self._spec = parse_task_spec(self.entry, default_domain=self.default_domain)
        self.limits = resource_limits_from(self.entry)
        self.image = self._resolve_image()
        self.submission_path = str(
            self.entry.get("submission_path") or self.default_submission_path
        )
        self.test_command = str(self.entry.get("test_command") or self.default_test_command)
        self.apply_command = str(self.entry.get("apply_command") or self._default_apply_command())
        self.setup_commands = list(self.entry.get("setup_commands") or [])
        self.reset_before_verify = bool(
            self.entry.get("reset_before_verify", type(self).reset_before_verify)
        )
        self._runner = runner
        self._docker_client = docker_client
        self._started = runner is not None and runner.is_running
        self._history: list[str] = []

    def __enter__(self) -> ContainerEnv:
        self.reset()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _default_apply_command(self) -> str:
        """The command that makes a written submission take effect, if any.

        Overridden by formats whose default depends on ``submission_path``.
        """
        return self.default_apply_command

    def _resolve_image(self) -> str:
        image = self.entry.get("image")
        if image:
            return str(image)
        if self.entry.get("dockerfile") or self.entry.get("build_context"):
            # TODO: implement Docker interaction — build the image from the Dockerfile
            # and cache the resulting tag, keyed by the manifest's pinned commit.
            raise ManifestError(
                f"{self.context}: building from 'dockerfile' is not supported yet; "
                "supply a prebuilt 'image' instead"
            )
        raise ManifestError(f"{self.context}: manifest entry needs an 'image' field")

    @property
    def runner(self) -> SandboxRunner:
        """The sandbox for this task, created on first access."""
        if self._runner is None:
            self._runner = SandboxRunner(
                image=self.image,
                limits=self.limits,
                client=self._docker_client,
            )
        return self._runner

    def _ensure_started(self) -> SandboxRunner:
        """Start the sandbox if a caller reached for it without calling reset first.

        Tools frequently want nothing but a verdict, so ``verify`` on a fresh adapter
        should work rather than raise.
        """
        if not self._started:
            self.reset()
        return self.runner

    def spec(self) -> TaskSpec:
        return self._spec

    def reset(self) -> Observation:
        """Recreate the container and run any manifest setup commands."""
        self._history.clear()
        logger.info("env reset env_id=%s image=%s", self._spec.id, self.image)
        runner = self.runner
        runner.start()
        self._started = True
        for command in self.setup_commands:
            result = runner.exec(str(command))
            if not result.ok:
                logger.warning(
                    "setup command failed env_id=%s command=%r exit_code=%s",
                    self._spec.id,
                    command,
                    result.exit_code,
                )
        return Observation(
            text=self._spec.instructions,
            metadata={"task_id": self._spec.id, "image": self.image, "workdir": runner.workdir},
        )

    def step(self, action: str) -> StepResult:
        """Run ``action`` as a shell command in the sandbox.

        The step reward is always 0.0: for these formats the only reward signal is the
        verifier, and inventing a per-step reward would fabricate a shaping term the
        upstream environment does not define. Callers get ``done`` from the verifier.
        """
        runner = self._ensure_started()
        self._history.append(action)
        result = runner.exec(action)
        return StepResult(
            observation=Observation(
                text=result.stdout or result.stderr,
                metadata={"exit_code": result.exit_code, "timed_out": result.timed_out},
            ),
            reward=0.0,
            done=False,
            info=result.to_dict(),
        )

    def verify(self, submission: str) -> RewardResult:
        """Deliver ``submission`` into the sandbox and run the graded test command.

        Unless :attr:`reset_before_verify` is set, this grades the container *as it
        currently stands*, which is what an agentic task requires. Callers running
        repeated trials against a reused container must therefore call :meth:`reset`
        between them, or a file left behind by one trial will score the next.
        """
        if not self.test_command:
            message = f"{self.context}: manifest entry needs a 'test_command' field"
            logger.error("%s", message)
            raise ManifestError(message)

        if self.reset_before_verify or not self._started:
            self.reset()
        runner = self.runner
        runner.write_file(self.submission_path, submission)

        applied: ExecResult | None = None
        if self.apply_command:
            applied = runner.exec(self.apply_command)

        tested = runner.exec(self.test_command)
        return self._grade(tested, applied=applied)

    def _grade(self, tested: ExecResult, *, applied: ExecResult | None = None) -> RewardResult:
        """Turn the test command's exit status into a reward and a verdict."""
        verdict = tested.ok
        state: dict[str, Any] = {"test": tested.to_dict()}
        if applied is not None:
            state["apply"] = applied.to_dict()
            # A submission that would not even apply cannot be credited by its tests.
            if not applied.ok:
                verdict = False
                state["apply_failed"] = True
                logger.warning(
                    "submission did not apply env_id=%s command=%r exit_code=%s",
                    self._spec.id,
                    applied.command,
                    applied.exit_code,
                )

        # TODO: parse per-test results (pytest report, Terminal-Bench JSON) so
        # partial-credit graders yield fractional reward instead of this binary
        # fallback. Reported in verifier_state so callers can see it happened.
        if self._spec.reward_type == "partial":
            state["grading"] = "binary_fallback"
            logger.warning("partial-credit grading fell back to binary env_id=%s", self._spec.id)

        logs = "\n".join(part for part in (tested.stdout, tested.stderr) if part)
        logger.info(
            "verify complete env_id=%s verdict=%s reward=%.3f exit_code=%s duration=%.2fs",
            self._spec.id,
            verdict,
            1.0 if verdict else 0.0,
            tested.exit_code,
            tested.duration_seconds,
        )
        return RewardResult(
            reward=1.0 if verdict else 0.0,
            verdict=verdict,
            verifier_logs=logs,
            verifier_state=state,
        )

    def gold_solution(self) -> str | None:
        """Return the reference solution, inline from the manifest or read from the container."""
        inline = self.entry.get("gold_solution") or self.entry.get("gold_patch")
        if inline:
            return str(inline)

        path = self.entry.get("gold_solution_path")
        if not path:
            return None
        runner = self._ensure_started()
        return runner.read_file(str(path)).decode("utf-8", errors="replace")

    def snapshot(self) -> bytes:
        return self._ensure_started().snapshot()

    def restore(self, snap: bytes) -> None:
        self.runner.restore(snap)
        self._started = True

    def close(self) -> None:
        """Tear down the sandbox and any images it committed."""
        if self._runner is not None:
            logger.debug("closing env env_id=%s", self._spec.id)
            self._runner.cleanup()
        self._started = False
