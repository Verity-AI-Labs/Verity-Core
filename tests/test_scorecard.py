"""Tests for the scorecard, focused on serialization round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_core.scorecard import (
    AXES,
    EXCLUDED_AXES,
    SCHEMA_VERSION,
    UTILITY_AXES,
    VALIDITY_AXES,
    AxisValue,
    Scorecard,
)


@pytest.fixture
def filled() -> Scorecard:
    card = Scorecard(env_id="terminal-bench/hello-world", timestamp="2026-08-31T00:00:00+00:00")
    card.set_axis("V1", 0.75, "verity-redteam", {"exploits": 3, "trials": 50}, "reward hackable")
    card.set_axis("V2", 1.0, "verity-clean", {"contaminated": False})
    card.set_axis("U2", 0.0, "verity-signal", {"pass_rate": 0.0}, "no measurable signal")
    card.metadata["commit"] = "9f8e7d6"
    return card


class TestAxisSet:
    def test_there_are_thirteen_axes(self) -> None:
        assert len(AXES) == 13
        assert len(set(AXES)) == 13

    def test_axes_are_the_validity_and_utility_groups(self) -> None:
        assert AXES == VALIDITY_AXES + UTILITY_AXES
        assert VALIDITY_AXES == ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
        assert UTILITY_AXES == ("U1", "U2", "U3", "U4", "U6", "U7")

    def test_u5_is_excluded_with_a_stated_reason(self) -> None:
        assert "U5" not in AXES
        assert "U5" in EXCLUDED_AXES
        assert EXCLUDED_AXES["U5"]


class TestAxisValue:
    def test_defaults_to_unscored(self) -> None:
        entry = AxisValue(axis="V1")
        assert entry.value is None
        assert entry.scored is False

    def test_coerces_an_integer_value_to_float(self) -> None:
        assert isinstance(AxisValue(axis="V1", value=1).value, float)

    def test_round_trips_through_a_dict(self) -> None:
        entry = AxisValue(axis="U3", value=0.5, tool="verity-stable", evidence={"n": 2}, notes="x")
        assert AxisValue.from_dict(entry.to_dict()) == entry

    def test_rejects_the_excluded_axis(self) -> None:
        with pytest.raises(ValueError, match="deliberately excluded"):
            AxisValue(axis="U5")

    def test_rejects_an_unknown_axis(self) -> None:
        with pytest.raises(ValueError, match="unknown axis 'V9'"):
            AxisValue(axis="V9")


class TestScorecardConstruction:
    def test_every_axis_is_present_from_the_start(self) -> None:
        card = Scorecard(env_id="e")
        assert set(card.axes) == set(AXES)
        assert all(not entry.scored for entry in card.axes.values())

    def test_timestamp_defaults_to_an_iso_utc_string(self) -> None:
        assert Scorecard(env_id="e").timestamp.endswith("+00:00")

    def test_partially_supplied_axes_are_completed(self) -> None:
        card = Scorecard(env_id="e", axes={"V1": AxisValue(axis="V1", value=1.0)})
        assert len(card.axes) == 13
        assert card.axes["V1"].value == 1.0
        assert card.axes["V2"].value is None

    def test_rejects_an_excluded_axis_key(self) -> None:
        with pytest.raises(ValueError, match="deliberately excluded"):
            Scorecard(env_id="e", axes={"U5": AxisValue(axis="V1")})


class TestSetAxis:
    def test_records_value_tool_evidence_and_notes(self, filled: Scorecard) -> None:
        entry = filled.get_axis("V1")
        assert entry.value == 0.75
        assert entry.tool == "verity-redteam"
        assert entry.evidence == {"exploits": 3, "trials": 50}
        assert entry.notes == "reward hackable"

    def test_setting_an_axis_twice_replaces_it(self, filled: Scorecard) -> None:
        filled.set_axis("V1", 0.1, "verity-signal", {})
        assert filled.get_axis("V1").value == 0.1
        assert filled.get_axis("V1").tool == "verity-signal"

    def test_a_none_value_stays_unscored(self, filled: Scorecard) -> None:
        filled.set_axis("V3", None, "verity-clean", {}, "not measurable")
        assert filled.get_axis("V3").scored is False
        assert filled.get_axis("V3").tool == "verity-clean"

    def test_zero_is_a_measurement_not_an_absence(self, filled: Scorecard) -> None:
        assert filled.get_axis("U2").value == 0.0
        assert filled.get_axis("U2").scored is True
        assert "U2" in filled.scored_axes

    def test_evidence_is_copied_rather_than_referenced(self, filled: Scorecard) -> None:
        evidence = {"k": 1}
        filled.set_axis("U4", 1.0, "t", evidence)
        evidence["k"] = 2
        assert filled.get_axis("U4").evidence == {"k": 1}

    @pytest.mark.parametrize("axis", ["U5", "V8", "z1", ""])
    def test_rejects_invalid_axes(self, filled: Scorecard, axis: str) -> None:
        with pytest.raises(ValueError):
            filled.set_axis(axis, 1.0, "t", {})


class TestCoverage:
    def test_reports_scored_and_unscored_axes(self, filled: Scorecard) -> None:
        assert filled.scored_axes == ("V1", "V2", "U2")
        assert "V3" in filled.unscored_axes
        assert len(filled.scored_axes) + len(filled.unscored_axes) == 13

    def test_coverage_is_the_scored_fraction(self, filled: Scorecard) -> None:
        assert filled.coverage() == pytest.approx(3 / 13)

    def test_an_empty_card_has_zero_coverage(self) -> None:
        assert Scorecard(env_id="e").coverage() == 0.0

    def test_completeness_requires_every_axis(self, filled: Scorecard) -> None:
        assert filled.is_complete is False
        for axis in AXES:
            filled.set_axis(axis, 1.0, "t", {})
        assert filled.is_complete is True
        assert filled.coverage() == 1.0


class TestSerialization:
    def test_json_file_round_trip_preserves_the_scorecard(
        self, filled: Scorecard, tmp_path: Path
    ) -> None:
        path = tmp_path / "scorecard.json"
        filled.to_json(path)
        assert Scorecard.from_json(path) == filled

    def test_dict_round_trip_preserves_the_scorecard(self, filled: Scorecard) -> None:
        assert Scorecard.from_dict(filled.to_dict()) == filled

    def test_round_trip_preserves_none_distinctly_from_zero(self, tmp_path: Path) -> None:
        card = Scorecard(env_id="e")
        card.set_axis("V1", None, "t", {})
        card.set_axis("V2", 0.0, "t", {})
        path = tmp_path / "sc.json"
        card.to_json(path)
        loaded = Scorecard.from_json(path)
        assert loaded.get_axis("V1").value is None
        assert loaded.get_axis("V2").value == 0.0

    def test_round_trip_preserves_evidence_and_metadata(
        self, filled: Scorecard, tmp_path: Path
    ) -> None:
        path = tmp_path / "sc.json"
        filled.to_json(path)
        loaded = Scorecard.from_json(path)
        assert loaded.get_axis("V1").evidence == {"exploits": 3, "trials": 50}
        assert loaded.metadata == {"commit": "9f8e7d6"}

    def test_to_json_creates_missing_parent_directories(
        self, filled: Scorecard, tmp_path: Path
    ) -> None:
        path = tmp_path / "deep" / "nested" / "sc.json"
        filled.to_json(path)
        assert path.is_file()

    def test_to_json_accepts_a_string_path(self, filled: Scorecard, tmp_path: Path) -> None:
        path = tmp_path / "sc.json"
        filled.to_json(str(path))
        assert Scorecard.from_json(str(path)) == filled

    def test_written_file_is_valid_indented_json(self, filled: Scorecard, tmp_path: Path) -> None:
        path = tmp_path / "sc.json"
        filled.to_json(path)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text)["env_id"] == filled.env_id

    def test_dict_contains_all_axes_and_the_schema_version(self, filled: Scorecard) -> None:
        payload = filled.to_dict()
        assert payload["schema_version"] == SCHEMA_VERSION
        assert list(payload["axes"]) == list(AXES)

    def test_from_dict_tolerates_a_sparse_payload(self) -> None:
        card = Scorecard.from_dict({"env_id": "e", "axes": {"V1": {"value": 0.5, "tool": "t"}}})
        assert card.get_axis("V1").value == 0.5
        assert len(card.axes) == 13

    def test_from_dict_rejects_an_excluded_axis(self) -> None:
        with pytest.raises(ValueError, match="deliberately excluded"):
            Scorecard.from_dict({"env_id": "e", "axes": {"U5": {"value": 1.0}}})


class TestMarkdown:
    def test_includes_the_env_id_and_coverage(self, filled: Scorecard) -> None:
        report = filled.to_markdown()
        assert "terminal-bench/hello-world" in report
        assert "3/13 axes" in report

    def test_lists_every_axis_in_a_grouped_table(self, filled: Scorecard) -> None:
        report = filled.to_markdown()
        assert "## Validity" in report
        assert "## Utility" in report
        for axis in AXES:
            assert f"| {axis} |" in report

    def test_shows_values_tools_and_unscored_placeholders(self, filled: Scorecard) -> None:
        report = filled.to_markdown()
        assert "| V1 | 0.750 | verity-redteam | reward hackable |" in report
        assert "| V3 | — | — | — |" in report

    def test_renders_evidence_for_scored_axes(self, filled: Scorecard) -> None:
        report = filled.to_markdown()
        assert "## Evidence" in report
        assert '"exploits": 3' in report

    def test_notes_the_excluded_axis(self, filled: Scorecard) -> None:
        assert "U5" in filled.to_markdown().split("---")[-1]

    def test_escapes_pipes_and_newlines_in_notes(self) -> None:
        card = Scorecard(env_id="e")
        card.set_axis("V1", 1.0, "t", {}, notes="a | b\nc")
        row = next(line for line in card.to_markdown().splitlines() if line.startswith("| V1 "))
        assert "a \\| b c" in row
        # The note's pipe is escaped and its newline flattened, so the row still has
        # exactly the four cells the table header declares.
        assert row.replace("\\|", "").count("|") == 5

    def test_reports_metadata(self, filled: Scorecard) -> None:
        assert "**commit:** 9f8e7d6" in filled.to_markdown()

    def test_omits_the_evidence_section_when_there_is_none(self) -> None:
        assert "## Evidence" not in Scorecard(env_id="e").to_markdown()
