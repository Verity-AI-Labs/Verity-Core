"""Run an audit across a corpus, one environment at a time.

Each of the four Verity tools measures different axes but needs the same scaffolding
around the measurement: iterate the corpus, isolate failures, write scorecards, track
progress, and be resumable. That scaffolding lives here so no tool has to rebuild it.

Two properties matter more than anything else in this module.

**One environment cannot take down the run.** A corpus contains untrusted, often broken
environments; that is the point of auditing it. An image that will not pull or a
verifier that throws is data, not a crash, so failures are caught, recorded against the
environment, and the batch moves on.

**A run must be resumable.** Two hundred environments take hours. Losing that work
because environment 180 failed is unacceptable, so with ``resume`` set the runner skips
environments that already have a scorecard on disk.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from verity_core.scorecard import SCORECARD_SUFFIX, Scorecard, scorecard_path

logger = logging.getLogger(__name__)

SUCCESS = "success"
FAILED = "failed"
SKIPPED = "skipped"

__all__ = [
    "FAILED",
    "SKIPPED",
    "SUCCESS",
    "BatchResult",
    "EnvResult",
    "completed_env_ids",
    "run_batch",
]


class _UsageSource(Protocol):
    """Anything exposing a cumulative token count, satisfied by :class:`ModelClient`."""

    @property
    def total_usage(self) -> Any: ...


AuditFn = Callable[[dict[str, Any]], Scorecard | None]
ProgressFn = Callable[["EnvResult", int, int], None]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def completed_env_ids(results_dir: Path | str) -> set[str]:
    """Return the environment ids that already have a scorecard in ``results_dir``.

    Ids are read from inside each file rather than inferred from filenames, so
    resumption still works for scorecards a tool wrote under its own naming scheme.
    Unreadable files are ignored: a scorecard truncated by the crash we are resuming
    from should be re-audited, not trusted.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        return set()

    found: set[str] = set()
    for path in sorted(directory.glob(f"*{SCORECARD_SUFFIX}")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            env_id = payload["env_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("ignoring unreadable scorecard path=%s error=%s", path, exc)
            continue
        found.add(str(env_id))
    logger.debug("found %d completed scorecard(s) in %s", len(found), directory)
    return found


@dataclass(slots=True)
class EnvResult:
    """What happened for one environment in a batch."""

    env_id: str
    outcome: str
    duration_seconds: float = 0.0
    scorecard_path: str | None = None
    error: str | None = None
    error_type: str | None = None
    skip_reason: str | None = None
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome == SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "outcome": self.outcome,
            "duration_seconds": round(self.duration_seconds, 3),
            "scorecard_path": self.scorecard_path,
            "error": self.error,
            "error_type": self.error_type,
            "skip_reason": self.skip_reason,
            "tokens": self.tokens,
        }


@dataclass(slots=True)
class BatchResult:
    """Per-environment outcomes plus the totals a run should be judged on."""

    results: list[EnvResult] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    finished_at: str = ""
    duration_seconds: float = 0.0
    interrupted: bool = False

    def _count(self, outcome: str) -> int:
        return sum(1 for result in self.results if result.outcome == outcome)

    @property
    def succeeded(self) -> int:
        return self._count(SUCCESS)

    @property
    def failed(self) -> int:
        return self._count(FAILED)

    @property
    def skipped(self) -> int:
        return self._count(SKIPPED)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(result.tokens for result in self.results)

    @property
    def failures(self) -> list[EnvResult]:
        """The failed environments, which is what an operator wants to read first."""
        return [result for result in self.results if result.outcome == FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "interrupted": self.interrupted,
            "summary": {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "skipped": self.skipped,
                "total_tokens": self.total_tokens,
            },
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info("batch summary written path=%s", target)


def _tokens_used(source: _UsageSource | None, before: int) -> tuple[int, int]:
    """Return ``(tokens spent since before, new cumulative total)``."""
    if source is None:
        return 0, before
    total = int(getattr(source.total_usage, "total_tokens", 0) or 0)
    return max(total - before, 0), total


def run_batch(
    entries: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    audit_fn: AuditFn,
    *,
    results_dir: Path | str | None = None,
    resume: bool = False,
    write_scorecards: bool = True,
    model_client: _UsageSource | None = None,
    progress: ProgressFn | None = None,
    stop_on_error: bool = False,
) -> BatchResult:
    """Run ``audit_fn`` against every manifest entry and collect the outcomes.

    ``audit_fn`` receives one manifest entry and returns a :class:`Scorecard`, or
    ``None`` to declare the environment not applicable, which is recorded as a skip
    rather than a failure. Anything it raises is caught and recorded against that
    environment; the batch continues unless ``stop_on_error`` is set.

    With ``results_dir`` given, returned scorecards are written there (unless
    ``write_scorecards`` is false, which is useful when the audit function writes them
    itself), and ``resume`` skips environments that already have one.

    Pass ``model_client`` to attribute token spend per environment: the runner reads the
    client's cumulative usage before and after each audit. Without it, per-environment
    token counts fall back to a ``tokens`` value in the scorecard metadata, and are zero
    if that is absent too.

    ``KeyboardInterrupt`` stops the run and returns what completed, with ``interrupted``
    set on the result, so an operator who aborts a long run still keeps the work and can
    resume from it.
    """
    items = list(entries)
    total = len(items)
    batch = BatchResult()
    started = time.monotonic()

    already_done = completed_env_ids(results_dir) if resume and results_dir else set()
    if already_done:
        logger.info("resuming: %d environment(s) already have scorecards", len(already_done))

    tokens_seen = int(getattr(getattr(model_client, "total_usage", None), "total_tokens", 0) or 0)

    logger.info(
        "batch started environments=%d results_dir=%s resume=%s", total, results_dir, resume
    )

    for index, entry in enumerate(items, start=1):
        env_id = str(entry.get("id") or f"<entry {index}>")

        if env_id in already_done:
            logger.info("batch skip env_id=%s (%d/%d) reason=already audited", env_id, index, total)
            result = EnvResult(
                env_id=env_id, outcome=SKIPPED, skip_reason="scorecard already in results directory"
            )
            batch.results.append(result)
            if progress is not None:
                progress(result, index, total)
            continue

        logger.info("batch env start env_id=%s (%d/%d)", env_id, index, total)
        env_started = time.monotonic()
        try:
            scorecard = audit_fn(entry)
        except KeyboardInterrupt:
            # Deliberately not recorded as a failure: the operator stopped the run, the
            # environment did not fail. Returning early keeps the work done so far.
            batch.interrupted = True
            logger.warning("batch interrupted at env_id=%s (%d/%d)", env_id, index, total)
            break
        except Exception as exc:
            duration = time.monotonic() - env_started
            spent, tokens_seen = _tokens_used(model_client, tokens_seen)
            logger.error(
                "batch env failed env_id=%s (%d/%d) error=%s: %s\n%s",
                env_id,
                index,
                total,
                type(exc).__name__,
                exc,
                traceback.format_exc(),
            )
            result = EnvResult(
                env_id=env_id,
                outcome=FAILED,
                duration_seconds=duration,
                error=str(exc),
                error_type=type(exc).__name__,
                tokens=spent,
            )
            batch.results.append(result)
            if progress is not None:
                progress(result, index, total)
            if stop_on_error:
                logger.error("batch stopping after failure because stop_on_error is set")
                break
            continue

        duration = time.monotonic() - env_started
        spent, tokens_seen = _tokens_used(model_client, tokens_seen)

        if scorecard is None:
            logger.info(
                "batch env skipped env_id=%s (%d/%d) reason=audit declined", env_id, index, total
            )
            result = EnvResult(
                env_id=env_id,
                outcome=SKIPPED,
                duration_seconds=duration,
                skip_reason="audit function returned no scorecard",
                tokens=spent,
            )
            batch.results.append(result)
            if progress is not None:
                progress(result, index, total)
            continue

        if spent == 0:
            spent = int(scorecard.metadata.get("tokens") or 0)

        written: Path | None = None
        if results_dir is not None and write_scorecards:
            written = scorecard_path(results_dir, scorecard.env_id)
            scorecard.to_json(written)

        logger.info(
            "batch env done env_id=%s (%d/%d) duration=%.2fs coverage=%.0f%% tokens=%d",
            env_id,
            index,
            total,
            duration,
            scorecard.coverage() * 100,
            spent,
        )
        result = EnvResult(
            env_id=env_id,
            outcome=SUCCESS,
            duration_seconds=duration,
            scorecard_path=None if written is None else str(written),
            tokens=spent,
        )
        batch.results.append(result)
        if progress is not None:
            progress(result, index, total)

    batch.duration_seconds = time.monotonic() - started
    batch.finished_at = _utc_now()
    logger.info(
        "batch finished total=%d succeeded=%d failed=%d skipped=%d duration=%.1fs tokens=%d%s",
        batch.total,
        batch.succeeded,
        batch.failed,
        batch.skipped,
        batch.duration_seconds,
        batch.total_tokens,
        " (interrupted)" if batch.interrupted else "",
    )
    return batch
