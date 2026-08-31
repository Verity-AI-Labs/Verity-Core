"""The scorecard: the artifact every Verity audit produces.

A scorecard holds one measurement per rubric axis for a single environment. Four
separate tools write into the same scorecard, so each axis records *which* tool
produced it and carries the raw evidence behind the number. An audit that cannot be
traced back to its evidence is an opinion, so ``evidence`` is a first-class field
rather than an optional annotation.

This module owns the axis *identifiers* and the file format, not the definitions of
what each axis measures; those live in the rubric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALIDITY_AXES: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
UTILITY_AXES: tuple[str, ...] = ("U1", "U2", "U3", "U4", "U6", "U7")

AXES: tuple[str, ...] = VALIDITY_AXES + UTILITY_AXES
"""The 13 scored axes, in report order."""

EXCLUDED_AXES: dict[str, str] = {
    # U5 is the downstream outcome the other axes are meant to predict. Scoring it
    # here would leak the label into the very measurements used to predict it.
    "U5": "transfer value is the predicted outcome, not an audited input",
}

SCHEMA_VERSION = 1

__all__ = [
    "AXES",
    "EXCLUDED_AXES",
    "SCHEMA_VERSION",
    "UTILITY_AXES",
    "VALIDITY_AXES",
    "AxisValue",
    "Scorecard",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_axis(axis: str) -> str:
    if axis in EXCLUDED_AXES:
        raise ValueError(f"axis {axis!r} is deliberately excluded: {EXCLUDED_AXES[axis]}")
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {', '.join(AXES)}")
    return axis


@dataclass(slots=True)
class AxisValue:
    """One axis measurement, with the tool that produced it and its evidence.

    ``value`` stays ``None`` for an axis that has not been measured. That is distinct
    from a measured zero, and conflating the two would turn "we did not look" into
    "we looked and found nothing".
    """

    axis: str
    value: float | None = None
    tool: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_axis(self.axis)
        if self.value is not None:
            self.value = float(self.value)

    @property
    def scored(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "value": self.value,
            "tool": self.tool,
            "evidence": self.evidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisValue:
        value = data.get("value")
        return cls(
            axis=data["axis"],
            value=None if value is None else float(value),
            tool=data.get("tool", ""),
            evidence=dict(data.get("evidence") or {}),
            notes=data.get("notes", ""),
        )


@dataclass(slots=True)
class Scorecard:
    """A complete 13-axis audit record for one environment.

    Every axis is present from construction, unscored ones included, so a reader can
    always tell the difference between an axis that was measured and an axis that was
    skipped. A scorecard with missing keys would make that unanswerable.
    """

    env_id: str
    timestamp: str = field(default_factory=_utc_now)
    axes: dict[str, AxisValue] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for axis in self.axes:
            _validate_axis(axis)
        self.axes = {axis: self.axes.get(axis) or AxisValue(axis=axis) for axis in AXES}

    def set_axis(
        self,
        axis: str,
        value: float | None,
        tool: str,
        evidence: dict[str, Any] | None = None,
        notes: str = "",
    ) -> AxisValue:
        """Record a measurement for ``axis``, replacing any previous one."""
        entry = AxisValue(
            axis=_validate_axis(axis),
            value=value,
            tool=tool,
            evidence=dict(evidence or {}),
            notes=notes,
        )
        self.axes[axis] = entry
        return entry

    def get_axis(self, axis: str) -> AxisValue:
        return self.axes[_validate_axis(axis)]

    @property
    def scored_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis in AXES if self.axes[axis].scored)

    @property
    def unscored_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis in AXES if not self.axes[axis].scored)

    @property
    def is_complete(self) -> bool:
        return not self.unscored_axes

    def coverage(self) -> float:
        """Fraction of the 13 axes that carry a measurement."""
        return len(self.scored_axes) / len(AXES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "env_id": self.env_id,
            "timestamp": self.timestamp,
            "axes": {axis: self.axes[axis].to_dict() for axis in AXES},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scorecard:
        raw_axes = data.get("axes") or {}
        axes = {}
        for axis, entry in raw_axes.items():
            _validate_axis(axis)
            axes[axis] = AxisValue.from_dict({"axis": axis, **entry})
        return cls(
            env_id=data["env_id"],
            timestamp=data.get("timestamp") or _utc_now(),
            axes=axes,
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_json(self, path: Path | str) -> None:
        """Write the scorecard to ``path``, creating parent directories as needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_json(cls, path: Path | str) -> Scorecard:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_markdown(self) -> str:
        """Render a human-readable report, with evidence in collapsible sections."""
        scored = len(self.scored_axes)
        lines = [
            f"# Verity scorecard: `{self.env_id}`",
            "",
            f"- **Generated:** {self.timestamp}",
            f"- **Coverage:** {scored}/{len(AXES)} axes ({self.coverage():.0%})",
        ]
        if self.unscored_axes:
            lines.append(f"- **Unscored:** {', '.join(self.unscored_axes)}")
        if self.metadata:
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- **{key}:** {value}")
        lines.append("")

        for title, group in (("Validity", VALIDITY_AXES), ("Utility", UTILITY_AXES)):
            lines += [
                f"## {title}",
                "",
                "| Axis | Value | Tool | Notes |",
                "| --- | --- | --- | --- |",
            ]
            for axis in group:
                entry = self.axes[axis]
                value = "—" if entry.value is None else f"{entry.value:.3f}"
                lines.append(
                    f"| {axis} | {value} | {entry.tool or '—'} | {_escape_cell(entry.notes)} |"
                )
            lines.append("")

        evidence_axes = [a for a in AXES if self.axes[a].evidence]
        if evidence_axes:
            lines += ["## Evidence", ""]
            for axis in evidence_axes:
                entry = self.axes[axis]
                body = json.dumps(entry.evidence, indent=2, sort_keys=True, default=str)
                lines += [
                    "<details>",
                    f"<summary>{axis}{f' ({entry.tool})' if entry.tool else ''}</summary>",
                    "",
                    "```json",
                    body,
                    "```",
                    "",
                    "</details>",
                    "",
                ]

        excluded = ", ".join(f"{a} ({why})" for a, why in sorted(EXCLUDED_AXES.items()))
        lines += ["---", "", f"Excluded by design: {excluded}.", ""]
        return "\n".join(lines)


def _escape_cell(text: str) -> str:
    """Keep pipes and newlines in free text from breaking the markdown table."""
    if not text:
        return "—"
    return text.replace("|", "\\|").replace("\n", " ")
