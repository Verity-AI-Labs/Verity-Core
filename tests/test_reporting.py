"""Tests for corpus-level scorecard aggregation and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_core.reporting import (
    CorpusReport,
    aggregate_scorecards,
    build_corpus_report,
    load_scorecards,
)
from verity_core.scorecard import AXES, VALIDITY_AXES, Scorecard, scorecard_path


def card(
    env_id: str,
    *,
    domain: str | None = None,
    tool: str = "verity-signal",
    **axes: float | None,
) -> Scorecard:
    metadata: dict[str, Any] = {} if domain is None else {"domain": domain}
    scorecard = Scorecard(env_id=env_id, metadata=metadata)
    for axis, value in axes.items():
        scorecard.set_axis(axis, value, tool)
    return scorecard


def write_cards(directory: Path, *cards: Scorecard) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for scorecard in cards:
        scorecard.to_json(scorecard_path(directory, scorecard.env_id))
    return directory


class TestLoadScorecards:
    def test_loads_every_scorecard(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("a", V1=0.1), card("b", V1=0.2))
        assert [item.env_id for item in load_scorecards(tmp_path)] == ["a", "b"]

    def test_sorts_by_env_id(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("z", V1=0.1), card("a", V1=0.2), card("m", V1=0.3))
        assert [item.env_id for item in load_scorecards(tmp_path)] == ["a", "m", "z"]

    def test_preserves_axis_values(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("a", V1=0.25))
        assert load_scorecards(tmp_path)[0].get_axis("V1").value == 0.25

    def test_an_empty_directory_loads_nothing(self, tmp_path: Path) -> None:
        assert load_scorecards(tmp_path) == []

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="results directory not found"):
            load_scorecards(tmp_path / "absent")

    def test_skips_unreadable_scorecards_and_keeps_the_rest(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("a", V1=0.1))
        (tmp_path / "truncated.json").write_text('{"env_id": "b"', encoding="utf-8")
        assert [item.env_id for item in load_scorecards(tmp_path)] == ["a"]

    def test_warns_about_skipped_scorecards(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "broken.json").write_text("{}", encoding="utf-8")
        with caplog.at_level("WARNING", logger="verity_core.reporting"):
            load_scorecards(tmp_path)
        assert any("unreadable scorecard" in record.message for record in caplog.records)

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("a", V1=0.1))
        (tmp_path / "report.md").write_text("# not a scorecard", encoding="utf-8")
        assert len(load_scorecards(tmp_path)) == 1


class TestAxisStatistics:
    def test_computes_the_distribution_of_an_axis(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.0), card("b", V1=0.5), card("c", V1=1.0)])
        stats = report.axis_stats["V1"]
        assert stats.count == 3
        assert stats.mean == pytest.approx(0.5)
        assert stats.median == pytest.approx(0.5)
        assert stats.minimum == pytest.approx(0.0)
        assert stats.maximum == pytest.approx(1.0)
        assert stats.std == pytest.approx(0.5)

    def test_a_single_observation_has_zero_spread(self) -> None:
        stats = aggregate_scorecards([card("a", V1=0.4)]).axis_stats["V1"]
        assert stats.count == 1
        assert stats.std == 0.0
        assert stats.mean == pytest.approx(0.4)

    def test_unscored_axes_are_counted_as_missing_not_zero(self) -> None:
        # The distinction the whole scorecard design rests on.
        report = aggregate_scorecards([card("a", V1=1.0), card("b"), card("c")])
        stats = report.axis_stats["V1"]
        assert stats.count == 1
        assert stats.missing == 2
        assert stats.mean == pytest.approx(1.0)

    def test_an_entirely_unmeasured_axis_has_no_statistics(self) -> None:
        stats = aggregate_scorecards([card("a", V1=0.5)]).axis_stats["V2"]
        assert stats.count == 0
        assert stats.missing == 1
        assert stats.mean is None
        assert stats.median is None
        assert stats.std is None

    def test_reports_every_axis(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.5)])
        assert set(report.axis_stats) == set(AXES)

    def test_a_measured_zero_is_included(self) -> None:
        stats = aggregate_scorecards([card("a", V1=0.0), card("b", V1=0.0)]).axis_stats["V1"]
        assert stats.count == 2
        assert stats.mean == 0.0


class TestDefectRate:
    def test_flags_environments_at_or_above_the_threshold(self) -> None:
        report = aggregate_scorecards(
            [card("a", V1=0.8), card("b", V1=0.5), card("c", V1=0.2)], threshold=0.5
        )
        assert report.axis_stats["V1"].flagged == 2
        assert report.flagged_environments == 2
        assert report.defect_rate == pytest.approx(2 / 3)

    def test_the_threshold_is_configurable(self) -> None:
        cards = [card("a", V1=0.8), card("b", V1=0.5), card("c", V1=0.2)]
        assert aggregate_scorecards(cards, threshold=0.9).flagged_environments == 0
        assert aggregate_scorecards(cards, threshold=0.1).flagged_environments == 3

    def test_the_direction_can_be_inverted(self) -> None:
        # For a rubric where a low score is the defect.
        cards = [card("a", V1=0.1), card("b", V1=0.9)]
        report = aggregate_scorecards(cards, threshold=0.5, direction="below")
        assert report.flagged_environments == 1
        assert report.most_flagged[0]["env_id"] == "a"

    def test_an_environment_flagged_on_several_axes_counts_once(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.9, V2=0.9, V3=0.9)], threshold=0.5)
        assert report.flagged_environments == 1
        assert report.defect_rate == 1.0

    def test_utility_axes_do_not_contribute_by_default(self) -> None:
        report = aggregate_scorecards([card("a", U1=1.0, U2=1.0)], threshold=0.5)
        assert report.flagged_environments == 0
        assert report.axis_stats["U1"].flagged == 0

    def test_the_defect_axes_are_configurable(self) -> None:
        report = aggregate_scorecards([card("a", U1=1.0)], threshold=0.5, defect_axes=["U1"])
        assert report.flagged_environments == 1

    def test_unknown_defect_axes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown defect axes: Z9"):
            aggregate_scorecards([card("a", V1=0.5)], defect_axes=["V1", "Z9"])

    def test_an_unscored_axis_is_never_flagged(self) -> None:
        report = aggregate_scorecards([card("a")], threshold=0.0)
        assert report.flagged_environments == 0

    def test_the_per_axis_rate_is_over_scored_environments(self) -> None:
        # V1 measured on two of four; one of those two flagged, so the rate is 50%.
        report = aggregate_scorecards(
            [card("a", V1=0.9), card("b", V1=0.1), card("c"), card("d")], threshold=0.5
        )
        stats = report.axis_stats["V1"]
        assert (stats.count, stats.flagged) == (2, 1)
        assert stats.defect_rate == pytest.approx(0.5)

    def test_an_empty_corpus_has_a_zero_defect_rate(self) -> None:
        report = aggregate_scorecards([])
        assert report.total_scorecards == 0
        assert report.defect_rate == 0.0

    def test_warns_when_there_is_nothing_to_aggregate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="verity_core.reporting"):
            aggregate_scorecards([])
        assert any("empty results directory" in record.message for record in caplog.records)


class TestCoverage:
    def test_counts_complete_scorecards(self) -> None:
        full = card("a", **dict.fromkeys(AXES, 0.1))
        report = aggregate_scorecards([full, card("b", V1=0.1)])
        assert report.complete_scorecards == 1

    def test_reports_mean_axis_coverage(self) -> None:
        full = card("a", **dict.fromkeys(AXES, 0.1))
        report = aggregate_scorecards([full, card("b")])
        assert report.mean_coverage == pytest.approx(0.5)


class TestDomainBreakdown:
    def test_groups_defect_rates_by_domain(self) -> None:
        report = aggregate_scorecards(
            [
                card("a", domain="browser", V1=0.9),
                card("b", domain="browser", V1=0.1),
                card("c", domain="code", V1=0.9),
            ],
            threshold=0.5,
        )
        assert report.domain_stats["browser"].total == 2
        assert report.domain_stats["browser"].flagged == 1
        assert report.domain_stats["browser"].defect_rate == pytest.approx(0.5)
        assert report.domain_stats["code"].defect_rate == 1.0

    def test_scorecards_without_a_domain_group_under_unknown(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.9)])
        assert report.domain_stats["unknown"].total == 1

    def test_a_domain_map_overrides_scorecard_metadata(self) -> None:
        # A caller holding the corpus can supply domains the audit never recorded.
        report = aggregate_scorecards([card("a", V1=0.9)], domains={"a": "gui"})
        assert set(report.domain_stats) == {"gui"}

    def test_the_domain_map_only_affects_ids_it_names(self) -> None:
        report = aggregate_scorecards(
            [card("a", domain="code", V1=0.9), card("b", domain="code", V1=0.9)],
            domains={"a": "gui"},
        )
        assert set(report.domain_stats) == {"gui", "code"}


class TestMostFlagged:
    def test_ranks_environments_by_flag_count(self) -> None:
        report = aggregate_scorecards(
            [
                card("one-axis", V1=0.9),
                card("three-axes", V1=0.9, V2=0.9, V3=0.9),
                card("two-axes", V1=0.9, V2=0.9),
            ],
            threshold=0.5,
        )
        assert [item["env_id"] for item in report.most_flagged] == [
            "three-axes",
            "two-axes",
            "one-axis",
        ]

    def test_lists_which_axes_were_flagged(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.9, V3=0.9, V2=0.1)], threshold=0.5)
        assert report.most_flagged[0]["axes"] == ["V1", "V3"]

    def test_excludes_unflagged_environments(self) -> None:
        report = aggregate_scorecards([card("a", V1=0.9), card("clean", V1=0.0)], threshold=0.5)
        assert [item["env_id"] for item in report.most_flagged] == ["a"]

    def test_ties_are_broken_by_id_for_reproducibility(self) -> None:
        report = aggregate_scorecards([card("zeta", V1=0.9), card("alpha", V1=0.9)], threshold=0.5)
        assert [item["env_id"] for item in report.most_flagged] == ["alpha", "zeta"]


class TestJsonReport:
    def test_serializes_the_summary(self) -> None:
        payload = aggregate_scorecards([card("a", V1=0.9), card("b", V1=0.1)]).to_dict()
        assert payload["summary"]["total_scorecards"] == 2
        assert payload["summary"]["flagged_environments"] == 1
        assert payload["summary"]["defect_rate"] == pytest.approx(0.5)

    def test_records_the_flagging_rule_it_used(self) -> None:
        payload = aggregate_scorecards([card("a", V1=0.9)], threshold=0.75).to_dict()
        assert payload["threshold"] == 0.75
        assert payload["direction"] == "above"
        assert payload["defect_axes"] == list(VALIDITY_AXES)

    def test_includes_per_axis_and_per_domain_sections(self) -> None:
        payload = aggregate_scorecards([card("a", domain="gui", V1=0.9)]).to_dict()
        assert payload["axes"]["V1"]["flagged"] == 1
        assert payload["domains"]["gui"]["total"] == 1

    def test_writes_a_json_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "report.json"
        aggregate_scorecards([card("a", V1=0.9)]).to_json(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["summary"]["total_scorecards"] == 1

    def test_an_empty_report_serializes(self) -> None:
        assert CorpusReport().to_dict()["summary"]["total_scorecards"] == 0


class TestMarkdownReport:
    @pytest.fixture
    def report(self) -> CorpusReport:
        return aggregate_scorecards(
            [
                card("corpus/a", domain="browser", V1=0.9, V2=0.8, U1=0.4),
                card("corpus/b", domain="browser", V1=0.1),
                card("corpus/c", domain="code", V1=0.9),
            ],
            threshold=0.5,
        )

    def test_leads_with_the_defect_rate(self, report: CorpusReport) -> None:
        assert "**Defect rate:** 66.7%" in report.to_markdown()

    def test_states_the_flagging_rule(self, report: CorpusReport) -> None:
        assert "**Flagging rule:** axis score >= 0.5" in report.to_markdown()

    def test_shows_the_comparator_for_inverted_flagging(self) -> None:
        inverted = aggregate_scorecards([card("a", V1=0.1)], direction="below")
        assert "axis score <= 0.5" in inverted.to_markdown()

    def test_includes_a_per_axis_table(self, report: CorpusReport) -> None:
        markdown = report.to_markdown()
        assert "| Axis | Scored | Missing | Mean | Median | Std | Min | Max |" in markdown
        assert "| V1 | 3 | 0 |" in markdown

    def test_marks_utility_axes_as_not_flagged(self, report: CorpusReport) -> None:
        row = next(line for line in report.to_markdown().splitlines() if line.startswith("| U1 |"))
        assert row.endswith("| n/a | n/a |")

    def test_renders_unmeasured_statistics_as_a_dash(self, report: CorpusReport) -> None:
        row = next(line for line in report.to_markdown().splitlines() if line.startswith("| U6 |"))
        assert "—" in row

    def test_includes_the_domain_breakdown(self, report: CorpusReport) -> None:
        markdown = report.to_markdown()
        assert "## Defect rate by domain" in markdown
        assert "| code | 1 | 1 | 100.0% |" in markdown
        assert "| browser | 2 | 1 | 50.0% |" in markdown

    def test_orders_domains_by_defect_rate(self, report: CorpusReport) -> None:
        lines = report.to_markdown().splitlines()
        domains = [line for line in lines if line.startswith(("| code |", "| browser |"))]
        assert domains[0].startswith("| code |")

    def test_lists_the_most_flagged_environments(self, report: CorpusReport) -> None:
        markdown = report.to_markdown()
        assert "## Most-flagged environments" in markdown
        assert "| `corpus/a` | browser | V1, V2 |" in markdown

    def test_limits_the_most_flagged_list(self) -> None:
        cards = [card(f"env-{i:02d}", V1=0.9) for i in range(20)]
        markdown = aggregate_scorecards(cards, threshold=0.5).to_markdown(top_n=3)
        assert markdown.count("| `env-") == 3

    def test_says_so_when_nothing_was_flagged(self) -> None:
        markdown = aggregate_scorecards([card("a", V1=0.0)], threshold=0.5).to_markdown()
        assert "_No environment was flagged on any validity axis._" in markdown

    def test_says_so_when_there_is_no_domain_metadata(self) -> None:
        markdown = aggregate_scorecards([card("a", V1=0.9)]).to_markdown()
        assert "| unknown | 1 | 1 | 100.0% |" in markdown

    def test_an_empty_report_still_renders(self) -> None:
        markdown = aggregate_scorecards([]).to_markdown()
        assert "**Environments audited:** 0" in markdown
        assert "_No domain metadata on these scorecards._" in markdown

    def test_ends_with_a_newline(self, report: CorpusReport) -> None:
        assert report.to_markdown().endswith("\n")


class TestBuildCorpusReport:
    def test_aggregates_a_results_directory(self, tmp_path: Path) -> None:
        write_cards(tmp_path, card("a", V1=0.9), card("b", V1=0.1))
        report = build_corpus_report(tmp_path, threshold=0.5)
        assert report.total_scorecards == 2
        assert report.flagged_environments == 1

    def test_writes_both_output_formats(self, tmp_path: Path) -> None:
        results = write_cards(tmp_path / "results", card("a", V1=0.9))
        out = tmp_path / "report"
        build_corpus_report(results, output_dir=out)
        assert (out / "corpus-report.json").is_file()
        assert (out / "corpus-report.md").is_file()

    def test_the_basename_is_configurable(self, tmp_path: Path) -> None:
        results = write_cards(tmp_path / "results", card("a", V1=0.9))
        build_corpus_report(results, output_dir=tmp_path / "out", basename="phase0")
        assert (tmp_path / "out" / "phase0.md").is_file()

    def test_writes_nothing_without_an_output_directory(self, tmp_path: Path) -> None:
        results = write_cards(tmp_path / "results", card("a", V1=0.9))
        build_corpus_report(results)
        assert sorted(path.name for path in results.iterdir()) == ["a.json"]

    def test_the_written_json_matches_the_returned_report(self, tmp_path: Path) -> None:
        results = write_cards(tmp_path / "results", card("a", V1=0.9), card("b", V1=0.2))
        report = build_corpus_report(results, output_dir=tmp_path / "out", threshold=0.5)
        payload = json.loads((tmp_path / "out" / "corpus-report.json").read_text(encoding="utf-8"))
        assert payload["summary"]["defect_rate"] == pytest.approx(report.defect_rate)

    def test_a_missing_results_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_corpus_report(tmp_path / "absent")

    def test_round_trips_from_batch_written_scorecards(self, tmp_path: Path) -> None:
        # The real pipeline: run_batch writes scorecards, reporting aggregates them.
        from verity_core.batch import run_batch

        def audit(entry: dict[str, Any]) -> Scorecard:
            scorecard = Scorecard(env_id=str(entry["id"]), metadata={"domain": entry["domain"]})
            scorecard.set_axis("V1", 0.9, "verity-core")
            return scorecard

        corpus = [
            {"id": "corpus/x", "format": "terminal", "domain": "browser"},
            {"id": "corpus/y", "format": "terminal", "domain": "code"},
        ]
        run_batch(corpus, audit, results_dir=tmp_path)
        report = build_corpus_report(tmp_path, threshold=0.5)
        assert report.total_scorecards == 2
        assert report.defect_rate == 1.0
        assert set(report.domain_stats) == {"browser", "code"}
