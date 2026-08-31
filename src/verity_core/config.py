"""Configuration shared by every Verity tool.

Settings resolve in layers. :data:`LAYERS` lists them from lowest to highest
priority, and :func:`resolve_config` folds them in exactly that order, so the
precedence rule is readable as data rather than inferred from which dict gets
updated into which.

Resolution happens per field, not per source: a ``verity.yaml`` that only pins
``model_name`` still picks up a ``VERITY_CACHE_DIR`` set by CI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MODEL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CACHE_DIR = Path(".verity_cache")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DOCKER_TIMEOUT = 600
DEFAULT_DOCKER_MEMORY_LIMIT = "4g"

CONFIG_FILENAME = "verity.yaml"
CONFIG_PATH_ENV_VAR = "VERITY_CONFIG"
ENV_PREFIX = "VERITY_"

LAYERS: tuple[str, ...] = ("defaults", "environment", "file")
"""Configuration layers, lowest priority first.

:func:`resolve_config` applies them in this order and each one overrides the layers
before it, so ``file`` beats ``environment``, which beats ``defaults``. Reordering
this tuple changes the precedence rule; nothing else needs to change with it.
"""

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

__all__ = [
    "CONFIG_FILENAME",
    "CONFIG_PATH_ENV_VAR",
    "LAYERS",
    "ResolvedConfig",
    "VerityConfig",
    "load_config",
    "resolve_config",
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
        """Build a config from a mapping, rejecting unknown keys."""
        _reject_unknown_keys(data, source="config")
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


def _known_fields() -> set[str]:
    return {f.name for f in fields(VerityConfig)}


def _reject_unknown_keys(data: dict[str, Any], *, source: str) -> None:
    """Fail on keys that are not config fields.

    Unknown keys are an error rather than a silent no-op: a typo like
    ``docker_memroy_limit`` would otherwise leave an audit running with limits the
    operator believes they overrode.
    """
    unknown = sorted(set(data) - _known_fields())
    if unknown:
        message = (
            f"{source}: unknown config key(s): {', '.join(unknown)}; "
            f"expected any of {', '.join(sorted(_known_fields()))}"
        )
        logger.error("%s", message)
        raise ValueError(message)


def _defaults_layer() -> dict[str, Any]:
    """The built-in defaults, read off a freshly constructed config."""
    defaults = VerityConfig()
    return {f.name: getattr(defaults, f.name) for f in fields(VerityConfig)}


def _environment_layer(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Collect ``VERITY_*`` overrides, ignoring the config-path variable itself."""
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for f in fields(VerityConfig):
        raw = source.get(f"{ENV_PREFIX}{f.name.upper()}")
        if raw is not None and raw != "":
            values[f.name] = raw
    return _coerce(values, source="env")


def _file_layer(path: Path | None) -> dict[str, Any]:
    """Read and validate a config file, or return nothing when there is none."""
    if path is None:
        return {}
    raw = _read_file(path)
    _reject_unknown_keys(raw, source=str(path))
    return _coerce(raw, source=str(path))


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


@dataclass(slots=True)
class ResolvedConfig:
    """A resolved config plus the layer each field's value came from.

    The provenance map is what lets ``verity-core config`` answer the question an
    operator actually has when a run misbehaves: not "what is the timeout" but "why
    is it that, and which layer set it".
    """

    config: VerityConfig
    sources: dict[str, str]
    path: Path | None = None

    def describe(self) -> list[tuple[str, Any, str]]:
        """Return ``(field, value, layer)`` rows in declaration order, for display."""
        return [
            (f.name, getattr(self.config, f.name), self.sources.get(f.name, "defaults"))
            for f in fields(VerityConfig)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "sources": dict(self.sources),
            "path": None if self.path is None else str(self.path),
        }


def resolve_config(
    path: Path | None = None, *, env: dict[str, str] | None = None
) -> ResolvedConfig:
    """Resolve configuration and record which layer supplied each field.

    When ``path`` is omitted the loader checks ``$VERITY_CONFIG`` and then
    ``./verity.yaml``; if neither exists only the environment and defaults apply. An
    explicitly requested ``path`` that does not exist is an error, since silently
    ignoring it would hide a misconfigured run.

    ``env`` overrides the process environment and exists so tests can inject variables
    without mutating global state.
    """
    if path is None:
        config_path = _discover_config_path(env)
    else:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            message = f"config file not found: {config_path}"
            logger.error("%s", message)
            raise FileNotFoundError(message)

    available = {
        "defaults": _defaults_layer(),
        "environment": _environment_layer(env),
        "file": _file_layer(config_path),
    }

    # Fold the layers in LAYERS order, lowest priority first, so each one overrides
    # the layers before it. The precedence rule lives in LAYERS, not in this loop.
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for layer in LAYERS:
        for name, value in available[layer].items():
            values[name] = value
            sources[name] = layer

    resolved = ResolvedConfig(config=VerityConfig(**values), sources=sources, path=config_path)
    logger.debug(
        "resolved config from %s: %s",
        config_path or "environment and defaults only",
        resolved.sources,
    )
    return resolved


def load_config(path: Path | None = None, *, env: dict[str, str] | None = None) -> VerityConfig:
    """Load configuration, with file values beating environment variables beating defaults.

    Use :func:`resolve_config` instead when you also need to know which layer each
    value came from.
    """
    return resolve_config(path, env=env).config
