"""Adapters that normalize upstream environment formats into :class:`VerityEnv`.

:func:`load_env` is the only entry point a tool needs: hand it a manifest entry and
get back something satisfying the protocol, with no knowledge of which upstream
project the task came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_core.adapters.base import ManifestError
from verity_core.adapters.docker_test import DockerTestAdapter
from verity_core.adapters.terminal import TerminalAdapter
from verity_core.adapters.verifiers import VerifiersAdapter
from verity_core.env import VerityEnv

ADAPTERS: dict[str, type] = {
    "verifiers": VerifiersAdapter,
    "terminal": TerminalAdapter,
    "docker_test": DockerTestAdapter,
}
"""Canonical format name to adapter class."""

FORMAT_ALIASES: dict[str, str] = {
    # Upstream projects and our own corpus notes name these formats inconsistently.
    # Accepting the spellings people actually write keeps typo-hunting out of audits.
    "prime": "verifiers",
    "primeintellect": "verifiers",
    "prime_intellect": "verifiers",
    "prime-intellect": "verifiers",
    "terminal-bench": "terminal",
    "terminal_bench": "terminal",
    "terminalbench": "terminal",
    "tbench": "terminal",
    "terminal-wrench": "terminal",
    "terminal_wrench": "terminal",
    "docker-test": "docker_test",
    "dockertest": "docker_test",
    "docker": "docker_test",
    "r2e": "docker_test",
    "r2e-gym": "docker_test",
    "r2e_gym": "docker_test",
    "swe-gym": "docker_test",
    "swe_gym": "docker_test",
    "swebench": "docker_test",
    "swe-bench": "docker_test",
}

__all__ = [
    "ADAPTERS",
    "FORMAT_ALIASES",
    "DockerTestAdapter",
    "ManifestError",
    "TerminalAdapter",
    "VerifiersAdapter",
    "canonical_format",
    "load_env",
    "register_adapter",
]


def canonical_format(format_name: str) -> str:
    """Map a manifest ``format`` value onto a registered adapter name."""
    key = str(format_name).strip().lower().replace(" ", "_")
    resolved = FORMAT_ALIASES.get(key, key)
    if resolved not in ADAPTERS:
        raise ManifestError(
            f"unknown environment format {format_name!r}; "
            f"registered formats are {', '.join(sorted(ADAPTERS))}"
        )
    return resolved


def register_adapter(format_name: str, adapter: type, *, aliases: tuple[str, ...] = ()) -> None:
    """Register an adapter for a new format.

    Present so a tool can audit a one-off format without a change to verity-core.
    """
    key = str(format_name).strip().lower().replace(" ", "_")
    ADAPTERS[key] = adapter
    for alias in aliases:
        FORMAT_ALIASES[str(alias).strip().lower().replace(" ", "_")] = key


def load_env(manifest_entry: Mapping[str, Any], **kwargs: Any) -> VerityEnv:
    """Build the right adapter for a manifest entry.

    ``kwargs`` pass through to the adapter, which is how callers inject a prepared
    :class:`~verity_core.runner.SandboxRunner`, a Docker client, or reward callables.
    """
    if not isinstance(manifest_entry, Mapping):
        raise ManifestError(
            f"manifest entry must be a mapping, got {type(manifest_entry).__name__}"
        )
    format_name = manifest_entry.get("format")
    if not format_name:
        raise ManifestError(
            f"manifest entry {manifest_entry.get('id', '<missing id>')!r} needs a 'format' field"
        )
    adapter = ADAPTERS[canonical_format(str(format_name))]
    return adapter(manifest_entry, **kwargs)
