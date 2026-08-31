"""Tests for loading a corpus of YAML manifests from a directory."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from verity_core.corpus import (
    CorpusError,
    CorpusStats,
    corpus_stats,
    find_manifest_files,
    load_corpus,
    load_manifest_file,
)


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def manifest(env_id: str, *, fmt: str = "terminal", domain: str = "tool_use", **extra: Any) -> str:
    lines = [f"id: {env_id}", f"format: {fmt}", f"domain: {domain}", "image: verity/task:latest"]
    lines += [f"{key}: {value}" for key, value in extra.items()]
    return "\n".join(lines) + "\n"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "manifests"
    write(root / "browse-01.yaml", manifest("corpus/browse-01", domain="browser"))
    write(root / "browse-02.yml", manifest("corpus/browse-02", domain="browser", solution="x"))
    write(root / "code-01.yaml", manifest("corpus/code-01", fmt="swe-gym", domain="code"))
    write(
        root / "math-01.yaml",
        "id: corpus/math-01\nformat: verifiers\ndomain: math\nreward: mod:fn\n",
    )
    return root


class TestFindManifestFiles:
    def test_finds_both_yaml_suffixes(self, corpus_dir: Path) -> None:
        names = {path.name for path in find_manifest_files(corpus_dir)}
        assert names == {"browse-01.yaml", "browse-02.yml", "code-01.yaml", "math-01.yaml"}

    def test_skips_non_yaml_files(self, corpus_dir: Path) -> None:
        write(corpus_dir / "README.md", "# corpus\n")
        write(corpus_dir / "notes.txt", "not a manifest\n")
        (corpus_dir / "schema.json").write_text("{}", encoding="utf-8")
        assert len(find_manifest_files(corpus_dir)) == 4

    def test_skips_dotfiles_that_look_like_manifests(self, corpus_dir: Path) -> None:
        write(corpus_dir / ".DS_Store.yaml", manifest("corpus/hidden"))
        assert all(not path.name.startswith(".") for path in find_manifest_files(corpus_dir))

    def test_descends_into_subdirectories(self, corpus_dir: Path) -> None:
        write(corpus_dir / "browser" / "nested.yaml", manifest("corpus/nested", domain="browser"))
        assert len(find_manifest_files(corpus_dir)) == 5

    def test_can_stay_at_the_top_level(self, corpus_dir: Path) -> None:
        write(corpus_dir / "browser" / "nested.yaml", manifest("corpus/nested"))
        assert len(find_manifest_files(corpus_dir, recursive=False)) == 4

    def test_returns_files_in_sorted_order(self, corpus_dir: Path) -> None:
        found = find_manifest_files(corpus_dir)
        assert found == sorted(found)

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="corpus directory not found"):
            find_manifest_files(tmp_path / "absent")

    def test_a_file_path_is_not_a_corpus_directory(self, corpus_dir: Path) -> None:
        with pytest.raises(CorpusError, match="corpus directory not found"):
            find_manifest_files(corpus_dir / "browse-01.yaml")


class TestLoadManifestFile:
    def test_reads_a_single_mapping(self, tmp_path: Path) -> None:
        path = write(tmp_path / "one.yaml", manifest("corpus/one"))
        entries = load_manifest_file(path)
        assert len(entries) == 1
        assert entries[0]["id"] == "corpus/one"

    def test_reads_a_list_of_mappings(self, tmp_path: Path) -> None:
        path = write(
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
        assert [entry["id"] for entry in load_manifest_file(path)] == ["corpus/a", "corpus/b"]

    def test_an_empty_file_yields_no_entries(self, tmp_path: Path) -> None:
        # A placeholder manifest is normal in a corpus under construction.
        assert load_manifest_file(write(tmp_path / "empty.yaml", "")) == []

    def test_a_comment_only_file_yields_no_entries(self, tmp_path: Path) -> None:
        path = write(tmp_path / "todo.yaml", "# still to be written\n")
        assert load_manifest_file(path) == []

    def test_records_the_file_each_entry_came_from(self, tmp_path: Path) -> None:
        path = write(tmp_path / "one.yaml", manifest("corpus/one"))
        assert load_manifest_file(path)[0]["manifest_path"] == str(path)

    def test_malformed_yaml_is_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path / "bad.yaml", "id: [unclosed\n")
        with pytest.raises(CorpusError, match="not valid YAML"):
            load_manifest_file(path)

    def test_a_scalar_top_level_is_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path / "scalar.yaml", "just a string\n")
        with pytest.raises(CorpusError, match="expected a mapping or a list of mappings"):
            load_manifest_file(path)

    def test_a_non_mapping_list_item_is_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path / "list.yaml", "- one\n- two\n")
        with pytest.raises(CorpusError, match="entry 0 is a str"):
            load_manifest_file(path)


class TestLoadCorpus:
    def test_loads_every_valid_entry(self, corpus_dir: Path) -> None:
        assert len(load_corpus(corpus_dir)) == 4

    def test_entries_are_sorted_by_id(self, corpus_dir: Path) -> None:
        ids = [entry["id"] for entry in load_corpus(corpus_dir)]
        assert ids == sorted(ids)
        assert ids[0] == "corpus/browse-01"

    def test_entries_are_usable_by_load_env(self, corpus_dir: Path) -> None:
        from verity_core import load_env

        entry = next(e for e in load_corpus(corpus_dir) if e["format"] == "terminal")
        assert load_env(entry).spec().id == entry["id"]

    def test_ignores_empty_and_non_yaml_files(self, corpus_dir: Path) -> None:
        write(corpus_dir / "placeholder.yaml", "")
        write(corpus_dir / "README.md", "# not a manifest\n")
        assert len(load_corpus(corpus_dir)) == 4

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="corpus directory not found"):
            load_corpus(tmp_path / "absent")

    def test_an_empty_directory_yields_an_empty_corpus(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert load_corpus(tmp_path / "empty") == []


class TestValidation:
    @pytest.mark.parametrize("missing", ["id", "format"])
    def test_a_missing_required_field_names_the_field(self, tmp_path: Path, missing: str) -> None:
        body = {"id": "corpus/x", "format": "terminal", "image": "img"}
        del body[missing]
        write(tmp_path / "bad.yaml", "\n".join(f"{k}: {v}" for k, v in body.items()) + "\n")
        with pytest.raises(CorpusError, match=f"missing required field\\(s\\): {missing}"):
            load_corpus(tmp_path)

    def test_an_empty_required_field_counts_as_missing(self, tmp_path: Path) -> None:
        write(tmp_path / "bad.yaml", "id: ''\nformat: terminal\n")
        with pytest.raises(CorpusError, match="missing required field"):
            load_corpus(tmp_path)

    def test_the_error_names_the_offending_file(self, tmp_path: Path) -> None:
        path = write(tmp_path / "broken.yaml", "format: terminal\n")
        with pytest.raises(CorpusError, match=r"broken\.yaml"):
            load_corpus(tmp_path)
        assert path.exists()

    def test_an_unknown_format_lists_the_registered_ones(self, tmp_path: Path) -> None:
        write(tmp_path / "bad.yaml", "id: corpus/x\nformat: gymnasium\n")
        with pytest.raises(CorpusError, match="registered formats are docker_test, terminal"):
            load_corpus(tmp_path)

    def test_an_unknown_domain_is_rejected(self, tmp_path: Path) -> None:
        write(tmp_path / "bad.yaml", manifest("corpus/x", domain="olfactory"))
        with pytest.raises(CorpusError, match="unknown domain 'olfactory'"):
            load_corpus(tmp_path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        # Scorecard filenames derive from the id, so a duplicate would overwrite results.
        write(tmp_path / "a.yaml", manifest("corpus/same"))
        write(tmp_path / "b.yaml", manifest("corpus/same"))
        with pytest.raises(CorpusError, match="duplicate environment id 'corpus/same'"):
            load_corpus(tmp_path)

    def test_duplicate_ids_are_rejected_even_when_not_strict(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", manifest("corpus/same"))
        write(tmp_path / "b.yaml", manifest("corpus/same"))
        with pytest.raises(CorpusError, match="duplicate environment id"):
            load_corpus(tmp_path, strict=False)


class TestNonStrictMode:
    def test_skips_invalid_entries_and_keeps_the_rest(self, corpus_dir: Path) -> None:
        write(corpus_dir / "broken.yaml", "format: terminal\n")
        write(corpus_dir / "also-broken.yaml", "id: corpus/nope\nformat: gymnasium\n")
        assert len(load_corpus(corpus_dir, strict=False)) == 4

    def test_skips_unparseable_files(self, corpus_dir: Path) -> None:
        write(corpus_dir / "broken.yaml", "id: [unclosed\n")
        assert len(load_corpus(corpus_dir, strict=False)) == 4

    def test_logs_an_error_for_each_skipped_entry(
        self, corpus_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write(corpus_dir / "broken.yaml", "format: terminal\n")
        with caplog.at_level("ERROR", logger="verity_core.corpus"):
            load_corpus(corpus_dir, strict=False)
        assert any("skipping invalid entry" in record.message for record in caplog.records)


class TestFiltering:
    def test_filters_by_a_single_domain(self, corpus_dir: Path) -> None:
        entries = load_corpus(corpus_dir, domain="browser")
        assert {entry["id"] for entry in entries} == {"corpus/browse-01", "corpus/browse-02"}

    def test_filters_by_several_domains(self, corpus_dir: Path) -> None:
        entries = load_corpus(corpus_dir, domain=["browser", "math"])
        assert len(entries) == 3

    def test_filters_by_format(self, corpus_dir: Path) -> None:
        entries = load_corpus(corpus_dir, format="verifiers")
        assert [entry["id"] for entry in entries] == ["corpus/math-01"]

    def test_format_filtering_is_alias_aware(self, corpus_dir: Path) -> None:
        # The corpus entry says swe-gym; asking for docker_test must still match.
        entries = load_corpus(corpus_dir, format="docker_test")
        assert [entry["id"] for entry in entries] == ["corpus/code-01"]

    def test_combines_domain_and_format_filters(self, corpus_dir: Path) -> None:
        assert load_corpus(corpus_dir, domain="browser", format="verifiers") == []

    def test_an_unmatched_filter_yields_nothing(self, corpus_dir: Path) -> None:
        assert load_corpus(corpus_dir, domain="gui") == []

    def test_entries_without_a_domain_default_to_other(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", "id: corpus/a\nformat: terminal\nimage: img\n")
        assert len(load_corpus(tmp_path, domain="other")) == 1


class TestCorpusStats:
    def test_counts_by_domain_and_format(self, corpus_dir: Path) -> None:
        stats = corpus_stats(load_corpus(corpus_dir))
        assert stats.total == 4
        assert stats.by_domain == {"browser": 2, "code": 1, "math": 1}
        assert stats.by_format == {"terminal": 2, "docker_test": 1, "verifiers": 1}

    def test_counts_entries_with_gold_solutions(self, corpus_dir: Path) -> None:
        stats = corpus_stats(load_corpus(corpus_dir))
        assert stats.with_gold == 1
        assert stats.gold_coverage == pytest.approx(0.25)

    @pytest.mark.parametrize(
        "key", ["gold_solution", "gold_patch", "gold_solution_path", "solution"]
    )
    def test_recognizes_every_gold_field(self, tmp_path: Path, key: str) -> None:
        write(tmp_path / "a.yaml", manifest("corpus/a", **{key: "value"}))
        assert corpus_stats(load_corpus(tmp_path)).with_gold == 1

    def test_recognizes_an_explicit_has_gold_flag(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", manifest("corpus/a", has_gold="true"))
        assert corpus_stats(load_corpus(tmp_path)).with_gold == 1

    def test_an_empty_corpus_has_zero_coverage(self) -> None:
        stats = corpus_stats([])
        assert stats.total == 0
        assert stats.gold_coverage == 0.0

    def test_format_counts_are_canonical_not_as_written(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", manifest("corpus/a", fmt="terminal-bench"))
        write(tmp_path / "b.yaml", manifest("corpus/b", fmt="tbench"))
        assert corpus_stats(load_corpus(tmp_path)).by_format == {"terminal": 2}

    def test_serializes_to_a_dict(self, corpus_dir: Path) -> None:
        payload = corpus_stats(load_corpus(corpus_dir)).to_dict()
        assert payload["total"] == 4
        assert payload["by_domain"]["browser"] == 2
        assert payload["gold_coverage"] == pytest.approx(0.25)

    def test_renders_markdown_tables(self, corpus_dir: Path) -> None:
        report = corpus_stats(load_corpus(corpus_dir)).to_markdown()
        assert "**Environments:** 4" in report
        assert "| Domain | Count |" in report
        assert "| browser | 2 |" in report
        assert "| verifiers | 1 |" in report

    def test_stats_can_be_built_directly(self) -> None:
        stats = CorpusStats(total=2, with_gold=1)
        assert stats.gold_coverage == 0.5
