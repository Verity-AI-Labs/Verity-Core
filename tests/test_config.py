"""Tests for configuration loading from defaults, files, and environment variables."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from verity_core.config import (
    DEFAULT_DOCKER_MEMORY_LIMIT,
    DEFAULT_MODEL_BASE_URL,
    DEFAULT_MODEL_NAME,
    LAYERS,
    VerityConfig,
    load_config,
    resolve_config,
)


def write_yaml(directory: Path, body: str, name: str = "verity.yaml") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class TestDefaults:
    def test_uses_defaults_when_nothing_is_configured(self) -> None:
        config = load_config(env={})
        assert config.model_base_url == DEFAULT_MODEL_BASE_URL
        assert config.model_name == DEFAULT_MODEL_NAME
        assert config.cache_dir == Path(".verity_cache")
        assert config.results_dir == Path("results")
        assert config.docker_timeout == 600
        assert config.docker_memory_limit == DEFAULT_DOCKER_MEMORY_LIMIT

    def test_network_is_disabled_by_default(self) -> None:
        # Isolation measurements are only meaningful if this default holds.
        assert load_config(env={}).docker_network_disabled is True

    def test_rejects_a_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="docker_timeout must be positive"):
            VerityConfig(docker_timeout=0)


class TestFileLoading:
    def test_loads_values_from_a_yaml_file(self, tmp_path: Path) -> None:
        path = write_yaml(
            tmp_path,
            """
            model_base_url: http://gpu-01:8000/v1
            model_name: Qwen/Qwen3-32B
            cache_dir: /tmp/verity-cache
            results_dir: /tmp/verity-results
            docker_timeout: 120
            docker_memory_limit: 16g
            docker_network_disabled: false
            """,
        )
        config = load_config(path, env={})
        assert config.model_base_url == "http://gpu-01:8000/v1"
        assert config.model_name == "Qwen/Qwen3-32B"
        assert config.cache_dir == Path("/tmp/verity-cache")
        assert config.docker_timeout == 120
        assert config.docker_memory_limit == "16g"
        assert config.docker_network_disabled is False

    def test_a_partial_file_leaves_other_fields_at_their_defaults(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: only-this\n")
        config = load_config(path, env={})
        assert config.model_name == "only-this"
        assert config.model_base_url == DEFAULT_MODEL_BASE_URL

    def test_an_empty_file_is_equivalent_to_no_file(self, tmp_path: Path) -> None:
        assert load_config(write_yaml(tmp_path, ""), env={}) == load_config(env={})

    def test_an_explicit_missing_path_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_config(tmp_path / "absent.yaml", env={})

    def test_rejects_a_file_that_is_not_a_mapping(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "- one\n- two\n")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_config(path, env={})

    def test_rejects_an_unknown_key(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "docker_memroy_limit: 8g\n")
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(path, env={})


class TestEnvironmentLoading:
    def test_reads_every_field_from_environment_variables(self) -> None:
        config = load_config(
            env={
                "VERITY_MODEL_BASE_URL": "http://env:8000/v1",
                "VERITY_MODEL_NAME": "env-model",
                "VERITY_CACHE_DIR": "/tmp/env-cache",
                "VERITY_RESULTS_DIR": "/tmp/env-results",
                "VERITY_DOCKER_TIMEOUT": "42",
                "VERITY_DOCKER_MEMORY_LIMIT": "2g",
                "VERITY_DOCKER_NETWORK_DISABLED": "false",
            }
        )
        assert config.model_base_url == "http://env:8000/v1"
        assert config.model_name == "env-model"
        assert config.cache_dir == Path("/tmp/env-cache")
        assert config.results_dir == Path("/tmp/env-results")
        assert config.docker_timeout == 42
        assert config.docker_memory_limit == "2g"
        assert config.docker_network_disabled is False

    def test_reads_from_the_real_process_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERITY_MODEL_NAME", "from-os-environ")
        assert load_config().model_name == "from-os-environ"

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_parses_truthy_booleans(self, raw: str) -> None:
        assert load_config(env={"VERITY_DOCKER_NETWORK_DISABLED": raw}).docker_network_disabled

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
    def test_parses_falsy_booleans(self, raw: str) -> None:
        assert not load_config(env={"VERITY_DOCKER_NETWORK_DISABLED": raw}).docker_network_disabled

    def test_rejects_an_uninterpretable_boolean(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret 'maybe' as a boolean"):
            load_config(env={"VERITY_DOCKER_NETWORK_DISABLED": "maybe"})

    def test_rejects_an_uninterpretable_integer(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret 'soon' as an integer"):
            load_config(env={"VERITY_DOCKER_TIMEOUT": "soon"})

    def test_an_empty_variable_is_ignored(self) -> None:
        assert load_config(env={"VERITY_MODEL_NAME": ""}).model_name == DEFAULT_MODEL_NAME


class TestPrecedence:
    def test_layers_are_declared_lowest_priority_first(self) -> None:
        # The precedence rule lives in this tuple; the resolver just folds it in order.
        assert LAYERS == ("defaults", "environment", "file")

    def test_the_file_wins_when_the_same_key_is_set_in_both(self, tmp_path: Path) -> None:
        # The documented rule, asserted directly: same key, both sources, file wins.
        path = write_yaml(tmp_path, "model_name: from-file\n")
        config = load_config(path, env={"VERITY_MODEL_NAME": "from-env"})
        assert config.model_name == "from-file"

    @pytest.mark.parametrize(
        ("key", "yaml_body", "env_var", "env_value", "expected"),
        [
            ("model_name", "model_name: from-file", "VERITY_MODEL_NAME", "from-env", "from-file"),
            (
                "model_base_url",
                "model_base_url: http://file:8000/v1",
                "VERITY_MODEL_BASE_URL",
                "http://env:8000/v1",
                "http://file:8000/v1",
            ),
            ("docker_timeout", "docker_timeout: 111", "VERITY_DOCKER_TIMEOUT", "222", 111),
            (
                "docker_memory_limit",
                "docker_memory_limit: 32g",
                "VERITY_DOCKER_MEMORY_LIMIT",
                "1g",
                "32g",
            ),
            (
                "docker_network_disabled",
                "docker_network_disabled: false",
                "VERITY_DOCKER_NETWORK_DISABLED",
                "true",
                False,
            ),
            (
                "cache_dir",
                "cache_dir: /tmp/from-file",
                "VERITY_CACHE_DIR",
                "/tmp/from-env",
                Path("/tmp/from-file"),
            ),
        ],
    )
    def test_the_file_wins_for_every_field(
        self,
        tmp_path: Path,
        key: str,
        yaml_body: str,
        env_var: str,
        env_value: str,
        expected: object,
    ) -> None:
        path = write_yaml(tmp_path, f"{yaml_body}\n")
        config = load_config(path, env={env_var: env_value})
        assert getattr(config, key) == expected

    def test_environment_still_fills_fields_the_file_omits(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: from-file\n")
        config = load_config(path, env={"VERITY_MODEL_BASE_URL": "http://from-env:8000/v1"})
        assert config.model_name == "from-file"
        assert config.model_base_url == "http://from-env:8000/v1"

    def test_environment_wins_over_defaults(self) -> None:
        assert load_config(env={"VERITY_MODEL_NAME": "from-env"}).model_name == "from-env"


class TestProvenance:
    def test_reports_the_layer_that_supplied_each_field(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: from-file\n")
        resolved = resolve_config(path, env={"VERITY_DOCKER_TIMEOUT": "30"})
        assert resolved.sources["model_name"] == "file"
        assert resolved.sources["docker_timeout"] == "environment"
        assert resolved.sources["docker_memory_limit"] == "defaults"

    def test_a_key_set_in_both_layers_is_attributed_to_the_file(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: from-file\n")
        resolved = resolve_config(path, env={"VERITY_MODEL_NAME": "from-env"})
        assert resolved.sources["model_name"] == "file"
        assert resolved.config.model_name == "from-file"

    def test_every_field_is_attributed_to_some_layer(self) -> None:
        resolved = resolve_config(env={})
        assert set(resolved.sources) == {f.name for f in fields(VerityConfig)}
        assert set(resolved.sources.values()) <= set(LAYERS)

    def test_records_the_config_path_it_used(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: m\n")
        assert resolve_config(path, env={}).path == path

    def test_the_path_is_none_when_no_file_was_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_config(env={}).path is None

    def test_describe_returns_rows_in_declaration_order(self) -> None:
        rows = resolve_config(env={"VERITY_MODEL_NAME": "m"}).describe()
        assert [name for name, _, _ in rows] == [f.name for f in fields(VerityConfig)]
        assert ("model_name", "m", "environment") in rows

    def test_serializes_config_sources_and_path(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: from-file\n")
        payload = resolve_config(path, env={}).to_dict()
        assert payload["config"]["model_name"] == "from-file"
        assert payload["sources"]["model_name"] == "file"
        assert payload["path"] == str(path)


class TestDiscovery:
    def test_finds_verity_yaml_in_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_yaml(tmp_path, "model_name: discovered\n")
        monkeypatch.chdir(tmp_path)
        assert load_config(env={}).model_name == "discovered"

    def test_verity_config_variable_points_at_a_file(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path, "model_name: pointed-at\n", name="custom.yaml")
        assert load_config(env={"VERITY_CONFIG": str(path)}).model_name == "pointed-at"

    def test_absent_verity_yaml_falls_back_to_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert load_config(env={}).model_name == DEFAULT_MODEL_NAME


class TestSerializationAndHelpers:
    def test_round_trips_through_a_dict(self, tmp_path: Path) -> None:
        original = VerityConfig(model_name="m", cache_dir=tmp_path / "c")
        assert VerityConfig.from_dict(original.to_dict()) == original

    def test_from_dict_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown config key\\(s\\): typo"):
            VerityConfig.from_dict({"typo": 1})

    def test_paths_are_coerced_and_user_expanded(self) -> None:
        config = VerityConfig(cache_dir="~/verity-cache")
        assert isinstance(config.cache_dir, Path)
        assert "~" not in str(config.cache_dir)

    def test_ensure_dirs_creates_both_directories(self, tmp_path: Path) -> None:
        config = VerityConfig(
            cache_dir=tmp_path / "a" / "cache", results_dir=tmp_path / "b" / "res"
        )
        config.ensure_dirs()
        assert config.cache_dir.is_dir()
        assert config.results_dir.is_dir()

    def test_ensure_dirs_is_idempotent(self, tmp_path: Path) -> None:
        config = VerityConfig(cache_dir=tmp_path / "cache", results_dir=tmp_path / "res")
        config.ensure_dirs()
        config.ensure_dirs()
        assert config.cache_dir.is_dir()
