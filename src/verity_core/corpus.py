"""Load a corpus of environment manifests from a directory of YAML files.

Verity-Corpus stores one manifest per environment as YAML. This module turns that
directory into a list of validated manifest entry dicts, each ready to hand to
:func:`~verity_core.adapters.load_env`.

It deliberately touches neither Docker nor any environment: loading a 200-entry corpus
to count how many are browser tasks should not start a single container. Validation
here is limited to what can be checked from metadata alone.

Entries come back sorted by ``id`` so that repeated runs visit environments in the same
order. Batch resumption and cross-run comparison both depend on that being stable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from verity_core.adapters import ADAPTERS, FORMAT_ALIASES
from verity_core.env import DOMAINS

logger = logging.getLogger(__name__)

MANIFEST_SUFFIXES = (".yaml", ".yml")
REQUIRED_FIELDS = ("id", "format")
GOLD_KEYS = ("gold_solution", "gold_solution_path", "gold_patch", "solution")
SOURCE_PATH_KEY = "manifest_path"
"""Key under which each entry records the file it came from, for error reporting."""

__all__ = [
    "GOLD_KEYS",
    "MANIFEST_SUFFIXES",
    "REQUIRED_FIELDS",
    "CorpusError",
    "CorpusStats",
    "corpus_stats",
    "find_manifest_files",
    "load_corpus",
    "load_manifest_file",
]


class CorpusError(ValueError):
    """Raised when a manifest file or entry cannot be used."""


def _normalize_format(value: object) -> str:
    """Canonicalize a format name for comparison, without rejecting unknown ones."""
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return FORMAT_ALIASES.get(key, key)


def _as_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def find_manifest_files(directory: Path | str, *, recursive: bool = True) -> list[Path]:
    """Return the YAML files under ``directory``, sorted, ignoring everything else.

    A corpus directory routinely contains a README, a license, and editor detritus, so
    anything without a YAML suffix is skipped rather than treated as a broken manifest.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")

    candidates: Iterator[Path] = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in MANIFEST_SUFFIXES
        # Editor and macOS metadata files can carry a .yaml suffix; they are not manifests.
        and not path.name.startswith(".")
    )
    logger.debug("found %d manifest file(s) under %s", len(files), root)
    return files


def load_manifest_file(path: Path | str) -> list[dict[str, Any]]:
    """Parse one YAML file into a list of raw manifest entries.

    A file may hold a single mapping (the common case, one environment per file) or a
    list of mappings. An empty file yields no entries rather than an error, since a
    placeholder manifest is a normal state for a corpus under construction.
    """
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CorpusError(f"{manifest_path}: not valid YAML: {exc}") from exc

    if loaded is None:
        logger.warning("manifest file is empty, skipping path=%s", manifest_path)
        return []

    if isinstance(loaded, dict):
        raw_entries = [loaded]
    elif isinstance(loaded, list):
        raw_entries = loaded
    else:
        raise CorpusError(
            f"{manifest_path}: expected a mapping or a list of mappings at the top level, "
            f"got {type(loaded).__name__}"
        )

    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise CorpusError(
                f"{manifest_path}: entry {index} is a {type(raw).__name__}, expected a mapping"
            )
        entry = dict(raw)
        entry.setdefault(SOURCE_PATH_KEY, str(manifest_path))
        entries.append(entry)
    return entries


def _validate(entry: dict[str, Any]) -> None:
    """Check the fields that can be verified from metadata alone."""
    where = entry.get(SOURCE_PATH_KEY, "<unknown file>")

    missing = [name for name in REQUIRED_FIELDS if not entry.get(name)]
    if missing:
        raise CorpusError(
            f"{where}: manifest entry is missing required field(s): {', '.join(missing)}"
        )

    resolved = _normalize_format(entry["format"])
    if resolved not in ADAPTERS:
        raise CorpusError(
            f"{where}: entry {entry['id']!r} has unknown format {entry['format']!r}; "
            f"registered formats are {', '.join(sorted(ADAPTERS))}"
        )

    domain = entry.get("domain")
    if domain is not None and str(domain) not in DOMAINS:
        raise CorpusError(
            f"{where}: entry {entry['id']!r} has unknown domain {domain!r}; "
            f"expected one of {', '.join(DOMAINS)}"
        )


def load_corpus(
    directory: Path | str,
    *,
    domain: str | Iterable[str] | None = None,
    # Shadows the builtin, but matches the manifest field name it filters on.
    format: str | Iterable[str] | None = None,
    recursive: bool = True,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate every manifest entry under ``directory``.

    ``domain`` and ``format`` filter the result and accept either a single value or a
    collection. Format matching is alias-aware, so ``format="terminal"`` also selects
    entries written as ``terminal-bench`` or ``tbench``.

    With ``strict`` set (the default) an unusable entry aborts the load. Set it to false
    to log and skip bad entries instead, which is what a corpus under active development
    wants: two malformed manifests should not block auditing the other 198.

    Duplicate ids are always an error, even when ``strict`` is false, because scorecard
    filenames derive from the id and a duplicate would have one environment's results
    silently overwrite another's.
    """
    files = find_manifest_files(directory, recursive=recursive)
    wanted_domains = _as_set(domain)
    wanted_formats = {_normalize_format(f) for f in _as_set(format)} if format else None

    entries: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    skipped = 0

    for path in files:
        try:
            candidates = load_manifest_file(path)
        except CorpusError as exc:
            if strict:
                logger.error("%s", exc)
                raise
            logger.error("skipping unreadable manifest: %s", exc)
            skipped += 1
            continue

        for entry in candidates:
            try:
                _validate(entry)
            except CorpusError as exc:
                if strict:
                    logger.error("%s", exc)
                    raise
                logger.error("skipping invalid entry: %s", exc)
                skipped += 1
                continue

            env_id = str(entry["id"])
            if env_id in seen:
                message = (
                    f"duplicate environment id {env_id!r} in {entry[SOURCE_PATH_KEY]}; "
                    f"already defined in {seen[env_id]}"
                )
                logger.error("%s", message)
                raise CorpusError(message)
            seen[env_id] = str(entry.get(SOURCE_PATH_KEY, path))

            if (
                wanted_domains is not None
                and str(entry.get("domain", "other")) not in wanted_domains
            ):
                continue
            if (
                wanted_formats is not None
                and _normalize_format(entry["format"]) not in wanted_formats
            ):
                continue
            entries.append(entry)

    entries.sort(key=lambda item: str(item["id"]))
    logger.info(
        "corpus loaded path=%s files=%d entries=%d skipped=%d domain_filter=%s format_filter=%s",
        directory,
        len(files),
        len(entries),
        skipped,
        domain or "none",
        format or "none",
    )
    return entries


@dataclass(slots=True)
class CorpusStats:
    """Counts describing a corpus, for coverage checks and report headers."""

    total: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    by_format: dict[str, int] = field(default_factory=dict)
    with_gold: int = 0

    @property
    def gold_coverage(self) -> float:
        """Fraction of entries shipping a reference solution."""
        return 0.0 if not self.total else self.with_gold / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_domain": dict(self.by_domain),
            "by_format": dict(self.by_format),
            "with_gold": self.with_gold,
            "gold_coverage": self.gold_coverage,
        }

    def to_markdown(self) -> str:
        lines = [
            f"**Environments:** {self.total}",
            f"**With gold solutions:** {self.with_gold} ({self.gold_coverage:.0%})",
            "",
            "| Domain | Count |",
            "| --- | --- |",
        ]
        lines += [f"| {name} | {count} |" for name, count in sorted(self.by_domain.items())]
        lines += ["", "| Format | Count |", "| --- | --- |"]
        lines += [f"| {name} | {count} |" for name, count in sorted(self.by_format.items())]
        return "\n".join(lines)


def corpus_stats(entries: Iterable[dict[str, Any]]) -> CorpusStats:
    """Summarize a loaded corpus by domain, format, and gold-solution coverage."""
    stats = CorpusStats()
    for entry in entries:
        stats.total += 1
        domain = str(entry.get("domain", "other"))
        stats.by_domain[domain] = stats.by_domain.get(domain, 0) + 1
        fmt = _normalize_format(entry.get("format", "unknown"))
        stats.by_format[fmt] = stats.by_format.get(fmt, 0) + 1
        if any(entry.get(key) for key in GOLD_KEYS) or entry.get("has_gold"):
            stats.with_gold += 1
    return stats
