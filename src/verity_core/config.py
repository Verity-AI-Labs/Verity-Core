"""Configuration shared by every Verity tool.

Values resolve per field with file settings winning over environment variables,
which in turn win over the built-in defaults. Resolving per field rather than
per source means a ``verity.yaml`` that only pins ``model_name`` still picks up a
``VERITY_CACHE_DIR`` set by CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CACHE_DIR = Path(".verity_cache")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DOCKER_TIMEOUT = 600
DEFAULT_DOCKER_MEMORY_LIMIT = "4g"

CONFIG_FILENAME = "verity.yaml"
CONFIG_PATH_ENV_VAR = "VERITY_CONFIG"
ENV_PREFIX = "VERITY_"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

__all__ = [
    "CONFIG_FILENAME",
    "CONFIG_PATH_ENV_VAR",
    "VerityConfig",
    "load_config",
]


def _parse_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"{source}: cannot interpret {value!r} as a boolean")


def _parse_int(value: Any, *, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: cannot interpret {value!r} as an integer") from exc


def _parse_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


@dataclass(slots=True)
class VerityConfig:
    """Resolved settings for model access, storage locations, and sandbox limits."""

    model_base_url: str = DEFAULT_MODEL_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)
    docker_timeout: int = DEFAULT_DOCKER_TIMEOUT
    docker_memory_limit: str = DEFAULT_DOCKER_MEMORY_LIMIT
    docker_network_disabled: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = _parse_path(self.cache_dir)
        self.results_dir = _parse_path(self.results_dir)
        if self.docker_timeout <= 0:
            raise ValueError(f"docker_timeout must be positive, got {self.docker_timeout}")

    def ensure_dirs(self) -> None:
        """Create the cache and results directories if they do not already exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_base_url": self.model_base_url,
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "results_dir": str(self.results_dir),
            "docker_timeout": self.docker_timeout,
            "docker_memory_limit": self.docker_memory_limit,
            "docker_network_disabled": self.docker_network_disabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerityConfig:
        """Build a config from a mapping, rejecting unknown keys.

        Unknown keys are an error rather than a silent no-op: a typo like
        ``docker_memroy_limit`` would otherwise leave an audit running with limits
        the operator believes they overrode.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown config key(s): {', '.join(unknown)}; "
                f"expected any of {', '.join(sorted(known))}"
            )
        return cls(**_coerce(data, source="config"))


def _coerce(data: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Convert raw string/YAML values into the field types the dataclass expects."""
    coerced: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if key in ("cache_dir", "results_dir"):
            coerced[key] = _parse_path(value)
        elif key == "docker_timeout":
            coerced[key] = _parse_int(value, source=f"{source}.{key}")
        elif key == "docker_network_disabled":
            coerced[key] = _parse_bool(value, source=f"{source}.{key}")
        else:
            coerced[key] = str(value)
    return coerced


def _from_env(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Collect ``VERITY_*`` overrides, ignoring the config-path variable itself."""
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for f in fields(VerityConfig):
        raw = source.get(f"{ENV_PREFIX}{f.name.upper()}")
        if raw is not None and raw != "":
            values[f.name] = raw
    return _coerce(values, source="env")


def _read_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def _discover_config_path(env: dict[str, str] | None = None) -> Path | None:
    source = os.environ if env is None else env
    from_env = source.get(CONFIG_PATH_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()
    candidate = Path.cwd() / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load_config(path: Path | None = None, *, env: dict[str, str] | None = None) -> VerityConfig:
    """Load configuration from ``path``, environment variables, then defaults.

    When ``path`` is omitted the loader checks ``$VERITY_CONFIG`` and then
    ``./verity.yaml``; if neither exists it falls back to environment variables and
    defaults alone. An explicitly requested ``path`` that does not exist is an error,
    since silently ignoring it would hide a misconfigured run.

    ``env`` overrides the process environment and exists so tests can inject
    variables without mutating global state.
    """
    if path is None:
        resolved_path = _discover_config_path(env)
    else:
        resolved_path = Path(path).expanduser()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"config file not found: {resolved_path}")

    values = _from_env(env)
    if resolved_path is not None:
        file_values = _read_file(resolved_path)
        known = {f.name for f in fields(VerityConfig)}
        unknown = sorted(set(file_values) - known)
        if unknown:
            raise ValueError(
                f"{resolved_path}: unknown config key(s): {', '.join(unknown)}; "
                f"expected any of {', '.join(sorted(known))}"
            )
        values.update(_coerce(file_values, source=str(resolved_path)))

    return VerityConfig(**values)
