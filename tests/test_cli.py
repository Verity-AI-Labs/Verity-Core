"""Tests for the verity-core command line interface."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from verity_core.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from verity_core.scorecard import Scorecard, scorecard_path

REWARD_MODULE = "cli_reward_fixture"


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "manifests"
    write(
        root / "browse.yaml",
        """
        id: corpus/browse-01
        format: terminal
        domain: browser
        image: verity/browse:latest
        solution: "touch /workspace/done\\n"
        """,
    )
    write(
        root / "code.yaml",
        """
        id: corpus/code-01
        format: swe-gym
        domain: code
        image: verity/code:latest
        """,
    )
    write(root / "README.md", "# not a manifest\n")
    write(root / "placeholder.yaml", "")
    return root


@pytest.fixture
def reward_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put an importable reward function on the path for verifiers-format manifests."""
    write(
        tmp_path / f"{REWARD_MODULE}.py",
        """
        def grade(submission, **kwargs):
            return 1.0 if "done" in submission else 0.0
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))


def verifiers_manifest(path: Path, *, gold: str | None = "all done") -> Path:
    body = f"""
    id: corpus/gold-check
    format: verifiers
    domain: math
    source: https://github.com/example/envs
    commit: cafe123
    reward: {REWARD_MODULE}:grade
    pass_threshold: 1.0
    """
    if gold is not None:
        body += f'gold_solution: "{gold}"\n'
    return write(path, body)


def scorecards_in(directory: Path, values: dict[str, float | None]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for env_id, value in values.items():
        card = Scorecard(env_id=env_id, metadata={"domain": "browser"})
        if value is not None:
            card.set_axis("V1", value, "verity-core")
        card.to_json(scorecard_path(directory, env_id))
    return directory


class TestParser:
    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args([])
        assert excinfo.value.code == 2

    def test_an_unknown_subcommand_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    @pytest.mark.parametrize("command", ["corpus", "audit", "report", "config"])
    def test_every_documented_subcommand_exists(self, command: str, tmp_path: Path) -> None:
        argv = [command] if command == "config" else [command, str(tmp_path)]
        assert build_parser().parse_args(argv).command == command

    def test_reports_its_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "verity-core" in capsys.readouterr().out

    def test_the_log_level_is_validated(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--log-level", "chatty", "config"])


class TestCorpusCommand:
    def test_summarizes_a_corpus(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["corpus", str(corpus_dir)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "Environments: 2" in out
        assert "With gold solutions: 1 (50%)" in out

    def test_breaks_down_by_domain_and_format(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["corpus", str(corpus_dir)])
        out = capsys.readouterr().out
        assert "By domain:" in out
        assert "browser" in out
        assert "By format:" in out
        assert "docker_test" in out

    def test_emits_json_on_request(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["corpus", str(corpus_dir), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["stats"]["total"] == 2
        assert payload["stats"]["by_domain"] == {"browser": 1, "code": 1}

    def test_lists_environment_ids_on_request(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["corpus", str(corpus_dir), "--list"])
        assert "corpus/browse-01" in capsys.readouterr().out

    def test_lists_environment_ids_in_json_too(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["corpus", str(corpus_dir), "--list", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["environments"] == ["corpus/browse-01", "corpus/code-01"]

    def test_filters_by_domain(self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["corpus", str(corpus_dir), "--domain", "browser", "--json"])
        assert json.loads(capsys.readouterr().out)["stats"]["total"] == 1

    def test_the_domain_filter_is_repeatable(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["corpus", str(corpus_dir), "--domain", "browser", "--domain", "code", "--json"])
        assert json.loads(capsys.readouterr().out)["stats"]["total"] == 2

    def test_filters_by_format(self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["corpus", str(corpus_dir), "--format", "terminal", "--json"])
        assert json.loads(capsys.readouterr().out)["stats"]["by_format"] == {"terminal": 1}

    def test_a_missing_directory_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["corpus", str(tmp_path / "absent")]) == EXIT_ERROR
        assert "corpus directory not found" in capsys.readouterr().err

    def test_an_invalid_manifest_fails_by_default(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(corpus_dir / "broken.yaml", "format: terminal\n")
        assert main(["corpus", str(corpus_dir)]) == EXIT_ERROR
        assert "missing required field" in capsys.readouterr().err

    def test_lenient_mode_skips_invalid_manifests(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(corpus_dir / "broken.yaml", "format: terminal\n")
        assert main(["corpus", str(corpus_dir), "--lenient", "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["stats"]["total"] == 2

    def test_can_stay_at_the_top_level(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(
            corpus_dir / "nested" / "deep.yaml",
            "id: corpus/deep\nformat: terminal\ndomain: gui\nimage: img\n",
        )
        main(["corpus", str(corpus_dir), "--no-recursive", "--json"])
        assert json.loads(capsys.readouterr().out)["stats"]["total"] == 2

    def test_an_empty_corpus_reports_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "empty").mkdir()
        main(["corpus", str(tmp_path / "empty")])
        out = capsys.readouterr().out
        assert "Environments: 0" in out
        assert "(none)" in out


@pytest.mark.usefixtures("reward_module")
class TestAuditCommand:
    def test_scores_the_gold_check_axis(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        assert main(["audit", str(manifest), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["axes"]["V1"]["value"] == 0.0
        assert payload["axes"]["V1"]["tool"] == "verity-core"

    def test_flags_an_environment_whose_verifier_rejects_its_own_gold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The one defect verity-core can find alone, scored 1.0 to match the report's
        # flagging direction.
        manifest = verifiers_manifest(tmp_path / "env.yaml", gold="nothing useful")
        main(["audit", str(manifest), "--json"])
        assert json.loads(capsys.readouterr().out)["axes"]["V1"]["value"] == 1.0

    def test_records_the_evidence_behind_the_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        main(["audit", str(manifest), "--json"])
        evidence = json.loads(capsys.readouterr().out)["axes"]["V1"]["evidence"]
        assert evidence["gold_verdict"] is True
        assert evidence["gold_reward"] == 1.0

    def test_carries_the_provenance_into_metadata(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        main(["audit", str(manifest), "--json"])
        metadata = json.loads(capsys.readouterr().out)["metadata"]
        assert metadata["commit"] == "cafe123"
        assert metadata["domain"] == "math"
        assert metadata["audited_by"] == "verity-core"

    def test_leaves_the_axis_unscored_without_a_gold_solution(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml", gold=None)
        main(["audit", str(manifest), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["axes"]["V1"]["value"] is None
        assert payload["metadata"]["gold_solution"] == "absent"

    def test_skip_gold_records_that_it_measured_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        main(["audit", str(manifest), "--skip-gold", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["metadata"]["gold_check"] == "skipped"
        assert payload["axes"]["V1"]["value"] is None

    def test_leaves_the_other_twelve_axes_for_the_tools(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        main(["audit", str(manifest), "--json"])
        axes = json.loads(capsys.readouterr().out)["axes"]
        scored = [axis for axis, entry in axes.items() if entry["value"] is not None]
        assert scored == ["V1"]

    def test_renders_markdown_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        main(["audit", str(manifest)])
        out = capsys.readouterr().out
        assert "# Verity scorecard: `corpus/gold-check`" in out
        assert "| V1 |" in out

    def test_writes_the_scorecard_when_asked(self, tmp_path: Path) -> None:
        manifest = verifiers_manifest(tmp_path / "env.yaml")
        results = tmp_path / "results"
        main(["audit", str(manifest), "--results-dir", str(results)])
        written = scorecard_path(results, "corpus/gold-check")
        assert Scorecard.from_json(written).env_id == "corpus/gold-check"

    def test_a_missing_manifest_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["audit", str(tmp_path / "absent.yaml")]) == EXIT_ERROR
        assert "verity-core:" in capsys.readouterr().err

    def test_an_empty_manifest_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["audit", str(write(tmp_path / "empty.yaml", ""))]) == EXIT_ERROR
        assert "no manifest entries found" in capsys.readouterr().err

    def test_a_multi_entry_manifest_points_at_the_batch_runner(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = write(
            tmp_path / "many.yaml",
            """
            - id: corpus/a
              format: terminal
              image: img
            - id: corpus/b
              format: terminal
              image: img
            """,
        )
        assert main(["audit", str(manifest)]) == EXIT_ERROR
        assert "use the batch runner" in capsys.readouterr().err

    def test_an_unknown_format_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = write(tmp_path / "env.yaml", "id: corpus/x\nformat: gymnasium\n")
        assert main(["audit", str(manifest)]) == EXIT_ERROR
        assert "unknown environment format" in capsys.readouterr().err


class TestReportCommand:
    def test_renders_a_corpus_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = scorecards_in(tmp_path / "results", {"a": 1.0, "b": 0.0})
        assert main(["report", str(results)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "# Verity corpus audit" in out
        assert "**Defect rate:** 50.0%" in out

    def test_emits_json_on_request(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = scorecards_in(tmp_path / "results", {"a": 1.0, "b": 0.0})
        main(["report", str(results), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["total_scorecards"] == 2
        assert payload["summary"]["defect_rate"] == 0.5

    def test_the_threshold_is_configurable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = scorecards_in(tmp_path / "results", {"a": 0.4})
        main(["report", str(results), "--threshold", "0.3", "--json"])
        assert json.loads(capsys.readouterr().out)["summary"]["flagged_environments"] == 1

    def test_the_direction_is_configurable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = scorecards_in(tmp_path / "results", {"a": 0.1})
        main(["report", str(results), "--direction", "below", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["direction"] == "below"
        assert payload["summary"]["flagged_environments"] == 1

    def test_writes_both_formats_to_an_output_directory(self, tmp_path: Path) -> None:
        results = scorecards_in(tmp_path / "results", {"a": 1.0})
        main(["report", str(results), "--output", str(tmp_path / "out")])
        assert (tmp_path / "out" / "corpus-report.json").is_file()
        assert (tmp_path / "out" / "corpus-report.md").is_file()

    def test_limits_the_most_flagged_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = scorecards_in(tmp_path / "results", {f"env-{i}": 1.0 for i in range(6)})
        main(["report", str(results), "--top", "2"])
        assert capsys.readouterr().out.count("| `env-") == 2

    def test_a_missing_directory_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report", str(tmp_path / "absent")]) == EXIT_ERROR
        assert "results directory not found" in capsys.readouterr().err

    def test_says_so_when_there_are_no_scorecards(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "empty").mkdir()
        assert main(["report", str(tmp_path / "empty")]) == EXIT_OK
        assert "no scorecards found" in capsys.readouterr().err


class TestConfigCommand:
    def test_prints_every_field_with_its_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VERITY_MODEL_NAME", raising=False)
        assert main(["config"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "FIELD" in out
        assert "model_name" in out
        assert "defaults" in out

    def test_shows_which_layer_won(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The question this subcommand exists to answer.
        path = write(tmp_path / "verity.yaml", "model_name: from-file\n")
        monkeypatch.setenv("VERITY_MODEL_NAME", "from-env")
        monkeypatch.setenv("VERITY_DOCKER_TIMEOUT", "45")
        main(["config", "--path", str(path)])
        lines = {
            line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.strip()
        }
        assert lines["model_name"].endswith("file")
        assert "from-file" in lines["model_name"]
        assert lines["docker_timeout"].endswith("environment")

    def test_emits_json_on_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write(tmp_path / "verity.yaml", "model_name: from-file\n")
        main(["config", "--path", str(path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["config"]["model_name"] == "from-file"
        assert payload["sources"]["model_name"] == "file"

    def test_reports_when_no_config_file_was_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VERITY_CONFIG", raising=False)
        main(["config"])
        assert "none found" in capsys.readouterr().out

    def test_a_missing_config_file_fails_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["config", "--path", str(tmp_path / "absent.yaml")]) == EXIT_ERROR
        assert "config file not found" in capsys.readouterr().err

    def test_an_unknown_key_fails_with_a_helpful_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write(tmp_path / "verity.yaml", "docker_memroy_limit: 8g\n")
        assert main(["config", "--path", str(path)]) == EXIT_ERROR
        err = capsys.readouterr().err
        assert "unknown config key(s): docker_memroy_limit" in err
        assert "docker_memory_limit" in err


class TestLogging:
    def test_logs_go_to_stderr_not_stdout(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # JSON on stdout has to stay pipeable whatever the log level.
        main(["--log-level", "info", "corpus", str(corpus_dir), "--json"])
        captured = capsys.readouterr()
        assert json.loads(captured.out)["stats"]["total"] == 2
        assert "corpus loaded" in captured.err

    def test_quiet_by_default(self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["corpus", str(corpus_dir), "--json"])
        assert "corpus loaded" not in capsys.readouterr().err

    def test_debug_level_shows_routine_operations(
        self, corpus_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--log-level", "debug", "corpus", str(corpus_dir), "--json"])
        assert "manifest file(s) under" in capsys.readouterr().err


class TestEndToEnd:
    def test_corpus_then_audit_then_report(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        reward_module: Any,
    ) -> None:
        manifests = tmp_path / "manifests"
        verifiers_manifest(manifests / "healthy.yaml")
        write(
            manifests / "broken.yaml",
            f"""
            id: corpus/broken
            format: verifiers
            domain: math
            reward: {REWARD_MODULE}:grade
            pass_threshold: 1.0
            gold_solution: "nothing useful"
            """,
        )
        results = tmp_path / "results"

        assert main(["corpus", str(manifests), "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["stats"]["total"] == 2

        for name in ("healthy", "broken"):
            assert (
                main(
                    [
                        "audit",
                        str(manifests / f"{name}.yaml"),
                        "--results-dir",
                        str(results),
                        "--json",
                    ]
                )
                == EXIT_OK
            )
        capsys.readouterr()

        assert main(["report", str(results), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["total_scorecards"] == 2
        # Exactly the broken one is flagged: a healthy environment must not register as
        # a defect just because it was audited.
        assert payload["summary"]["flagged_environments"] == 1
        assert payload["most_flagged"][0]["env_id"] == "corpus/broken"
