"""Aggregate individual scorecards into a corpus-level defect report.

One scorecard says whether one environment is sound. The Phase 0 result is a statement
about a population: what fraction of a corpus is defective, on which axes, in which
domains. This module turns a directory of scorecards into that statement, in JSON for
downstream analysis and Markdown for publication.

Two things are deliberately explicit rather than assumed.

**Unscored is not zero.** An axis nobody measured is excluded from that axis's
statistics and counted under ``missing``, so a thin run reads as low coverage instead of
a clean corpus. Averaging a missing measurement in as zero would manufacture the very
result the audit is supposed to test.

**Flagging direction is a parameter.** Whether a high score means "more defective" is a
property of the rubric, not of this code, so :data:`DEFAULT_DIRECTION` is a documented
default that a caller can invert rather than a fact baked into the aggregation.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from verity_core.scorecard import AXES, SCORECARD_SUFFIX, VALIDITY_AXES, Scorecard

logger = logging.getLogger(__name__)

Direction = Literal["above", "below"]

DEFAULT_DEFECT_THRESHOLD = 0.5
DEFAULT_DIRECTION: Direction = "above"
"""Flag an environment when its axis score is at or above the threshold.

Matches a rubric where a higher score means a more serious defect. Pass
``direction="below"`` for a rubric scored the other way, where a low score is the
problem.
"""

UNKNOWN_DOMAIN = "unknown"

__all__ = [
    "DEFAULT_DEFECT_THRESHOLD",
    "DEFAULT_DIRECTION",
    "AxisStats",
    "CorpusReport",
    "DomainStats",
    "aggregate_scorecards",
    "build_corpus_report",
    "load_scorecards",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_scorecards(results_dir: Path | str) -> list[Scorecard]:
    """Load every scorecard in ``results_dir``, sorted by environment id.

    Files that will not parse are logged and skipped rather than aborting the report: a
    single scorecard truncated by a crash should not make the other 199 unreportable.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")

    scorecards: list[Scorecard] = []
    for path in sorted(directory.glob(f"*{SCORECARD_SUFFIX}")):
        try:
            scorecards.append(Scorecard.from_json(path))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping unreadable scorecard path=%s error=%s", path, exc)
            continue

    scorecards.sort(key=lambda card: card.env_id)
    logger.info("loaded %d scorecard(s) from %s", len(scorecards), directory)
    return scorecards


def _is_flagged(value: float | None, threshold: float, direction: Direction) -> bool:
    if value is None:
        return False
    return value >= threshold if direction == "above" else value <= threshold


def _domain_of(card: Scorecard, overrides: Mapping[str, str] | None) -> str:
    if overrides and card.env_id in overrides:
        return str(overrides[card.env_id])
    return str(card.metadata.get("domain") or UNKNOWN_DOMAIN)


@dataclass(slots=True)
class AxisStats:
    """Distribution of one axis across the corpus, plus how often it was flagged."""

    axis: str
    count: int = 0
    missing: int = 0
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    flagged: int = 0

    @property
    def defect_rate(self) -> float:
        """Fraction of *scored* environments flagged on this axis.

        Denominated in scored environments, not the whole corpus, so an axis measured on
        20 of 200 environments reports the rate among those 20. Reading it needs
        ``count`` alongside it, which is why both travel together.
        """
        return 0.0 if not self.count else self.flagged / self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "count": self.count,
            "missing": self.missing,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "flagged": self.flagged,
            "defect_rate": self.defect_rate,
        }


@dataclass(slots=True)
class DomainStats:
    """How many environments in one domain were flagged on any validity axis."""

    domain: str
    total: int = 0
    flagged: int = 0

    @property
    def defect_rate(self) -> float:
        return 0.0 if not self.total else self.flagged / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "total": self.total,
            "flagged": self.flagged,
            "defect_rate": self.defect_rate,
        }


@dataclass(slots=True)
class CorpusReport:
    """The corpus-level result: per-axis statistics, defect rates, and outliers."""

    generated_at: str = field(default_factory=_utc_now)
    threshold: float = DEFAULT_DEFECT_THRESHOLD
    direction: Direction = DEFAULT_DIRECTION
    total_scorecards: int = 0
    complete_scorecards: int = 0
    mean_coverage: float = 0.0
    flagged_environments: int = 0
    axis_stats: dict[str, AxisStats] = field(default_factory=dict)
    domain_stats: dict[str, DomainStats] = field(default_factory=dict)
    most_flagged: list[dict[str, Any]] = field(default_factory=list)
    defect_axes: tuple[str, ...] = VALIDITY_AXES

    @property
    def defect_rate(self) -> float:
        """Fraction of the corpus flagged on at least one validity axis.

        The headline number: how much of this corpus should not be trusted as-is.
        """
        if not self.total_scorecards:
            return 0.0
        return self.flagged_environments / self.total_scorecards

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "threshold": self.threshold,
            "direction": self.direction,
            "defect_axes": list(self.defect_axes),
            "summary": {
                "total_scorecards": self.total_scorecards,
                "complete_scorecards": self.complete_scorecards,
                "mean_coverage": self.mean_coverage,
                "flagged_environments": self.flagged_environments,
                "defect_rate": self.defect_rate,
            },
            "axes": {axis: stats.to_dict() for axis, stats in self.axis_stats.items()},
            "domains": {name: stats.to_dict() for name, stats in self.domain_stats.items()},
            "most_flagged": list(self.most_flagged),
        }

    def to_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info("corpus report written path=%s", target)

    def to_markdown(self, *, top_n: int = 10) -> str:
        comparator = ">=" if self.direction == "above" else "<="
        lines = [
            "# Verity corpus audit",
            "",
            f"_Generated {self.generated_at}_",
            "",
            "## Summary",
            "",
            f"- **Environments audited:** {self.total_scorecards}",
            f"- **Defect rate:** {_percent(self.defect_rate)} "
            f"({self.flagged_environments} of {self.total_scorecards} flagged on at least one "
            "validity axis)",
            f"- **Fully scored scorecards:** {self.complete_scorecards}",
            f"- **Mean axis coverage:** {_percent(self.mean_coverage)}",
            f"- **Flagging rule:** axis score {comparator} {self.threshold:g} "
            f"on {', '.join(self.defect_axes)}",
            "",
            "## Per-axis statistics",
            "",
            "| Axis | Scored | Missing | Mean | Median | Std | Min | Max | Flagged | Defect rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        for axis in AXES:
            stats = self.axis_stats.get(axis)
            if stats is None:
                continue
            flagged = str(stats.flagged) if axis in self.defect_axes else "n/a"
            rate = _percent(stats.defect_rate) if axis in self.defect_axes else "n/a"
            lines.append(
                f"| {axis} | {stats.count} | {stats.missing} | {_number(stats.mean)} | "
                f"{_number(stats.median)} | {_number(stats.std)} | {_number(stats.minimum)} | "
                f"{_number(stats.maximum)} | {flagged} | {rate} |"
            )

        lines += ["", "## Defect rate by domain", ""]
        if self.domain_stats:
            lines += [
                "| Domain | Environments | Flagged | Defect rate |",
                "| --- | --- | --- | --- |",
            ]
            ordered = sorted(
                self.domain_stats.values(), key=lambda item: (-item.defect_rate, item.domain)
            )
            lines += [
                f"| {stats.domain} | {stats.total} | {stats.flagged} | "
                f"{_percent(stats.defect_rate)} |"
                for stats in ordered
            ]
        else:
            lines.append("_No domain metadata on these scorecards._")

        lines += ["", f"## Most-flagged environments (top {top_n})", ""]
        if self.most_flagged:
            lines += ["| Environment | Domain | Flagged axes |", "| --- | --- | --- |"]
            for item in self.most_flagged[:top_n]:
                axes = ", ".join(item["axes"]) or "—"
                lines.append(f"| `{item['env_id']}` | {item['domain']} | {axes} |")
        else:
            lines.append("_No environment was flagged on any validity axis._")

        return "\n".join(lines) + "\n"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def aggregate_scorecards(
    scorecards: Iterable[Scorecard],
    *,
    threshold: float = DEFAULT_DEFECT_THRESHOLD,
    direction: Direction = DEFAULT_DIRECTION,
    defect_axes: Sequence[str] = VALIDITY_AXES,
    domains: Mapping[str, str] | None = None,
) -> CorpusReport:
    """Compute corpus-level statistics and defect rates from scorecards.

    ``threshold`` and ``direction`` define what counts as a flag; ``defect_axes`` limits
    which axes contribute to the defect rate, defaulting to the validity axes since
    utility scores describe how useful a sound environment is, not whether it is broken.

    ``domains`` maps environment id to domain and takes precedence over each scorecard's
    ``metadata["domain"]``, so a caller holding the corpus can supply domains that the
    audit did not record.
    """
    cards = list(scorecards)
    unknown_axes = [axis for axis in defect_axes if axis not in AXES]
    if unknown_axes:
        raise ValueError(f"unknown defect axes: {', '.join(unknown_axes)}")

    report = CorpusReport(
        threshold=threshold,
        direction=direction,
        total_scorecards=len(cards),
        defect_axes=tuple(defect_axes),
    )
    if not cards:
        logger.warning("aggregating an empty results directory: report will be empty")
        return report

    values: dict[str, list[float]] = {axis: [] for axis in AXES}
    flags: dict[str, int] = dict.fromkeys(AXES, 0)
    missing: dict[str, int] = dict.fromkeys(AXES, 0)
    per_env_flags: list[dict[str, Any]] = []
    domain_stats: dict[str, DomainStats] = {}

    for card in cards:
        domain = _domain_of(card, domains)
        stats = domain_stats.setdefault(domain, DomainStats(domain=domain))
        stats.total += 1

        flagged_axes: list[str] = []
        for axis in AXES:
            value = card.axes[axis].value
            if value is None:
                missing[axis] += 1
                continue
            values[axis].append(float(value))
            if axis in defect_axes and _is_flagged(value, threshold, direction):
                flags[axis] += 1
                flagged_axes.append(axis)

        if flagged_axes:
            report.flagged_environments += 1
            stats.flagged += 1
        per_env_flags.append(
            {
                "env_id": card.env_id,
                "domain": domain,
                "flagged_count": len(flagged_axes),
                "axes": flagged_axes,
                "coverage": card.coverage(),
            }
        )

        if card.is_complete:
            report.complete_scorecards += 1

    report.mean_coverage = statistics.fmean(card.coverage() for card in cards)
    report.axis_stats = {
        axis: _axis_stats(axis, values[axis], missing[axis], flags[axis]) for axis in AXES
    }
    report.domain_stats = domain_stats
    # Ties broken by id so the report is byte-identical across runs on the same data.
    report.most_flagged = sorted(
        (item for item in per_env_flags if item["flagged_count"]),
        key=lambda item: (-item["flagged_count"], item["env_id"]),
    )

    logger.info(
        "aggregated scorecards=%d flagged=%d defect_rate=%.1f%% threshold=%s direction=%s",
        report.total_scorecards,
        report.flagged_environments,
        report.defect_rate * 100,
        threshold,
        direction,
    )
    return report


def _axis_stats(axis: str, values: list[float], missing: int, flagged: int) -> AxisStats:
    if not values:
        return AxisStats(axis=axis, count=0, missing=missing, flagged=flagged)
    return AxisStats(
        axis=axis,
        count=len(values),
        missing=missing,
        mean=statistics.fmean(values),
        median=statistics.median(values),
        # Sample standard deviation, which is undefined for a single observation; zero is
        # the honest answer there rather than an error.
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        flagged=flagged,
    )


def build_corpus_report(
    results_dir: Path | str,
    *,
    threshold: float = DEFAULT_DEFECT_THRESHOLD,
    direction: Direction = DEFAULT_DIRECTION,
    defect_axes: Sequence[str] = VALIDITY_AXES,
    domains: Mapping[str, str] | None = None,
    output_dir: Path | str | None = None,
    basename: str = "corpus-report",
) -> CorpusReport:
    """Load the scorecards in ``results_dir`` and aggregate them into a report.

    With ``output_dir`` set, the report is also written as ``<basename>.json`` and
    ``<basename>.md`` there.
    """
    report = aggregate_scorecards(
        load_scorecards(results_dir),
        threshold=threshold,
        direction=direction,
        defect_axes=defect_axes,
        domains=domains,
    )

    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        report.to_json(target / f"{basename}.json")
        markdown = target / f"{basename}.md"
        markdown.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("corpus report written path=%s", markdown)

    return report
