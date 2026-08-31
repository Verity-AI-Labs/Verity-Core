"""Tests for the batch runner: error isolation, resumption, and accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from verity_core.batch import FAILED, SKIPPED, SUCCESS, BatchResult, completed_env_ids, run_batch
from verity_core.models import ModelClient, TokenUsage
from verity_core.scorecard import Scorecard, scorecard_path


def entries(*env_ids: str) -> list[dict[str, Any]]:
    return [{"id": env_id, "format": "terminal", "image": "img"} for env_id in env_ids]


def audit_ok(entry: dict[str, Any]) -> Scorecard:
    card = Scorecard(env_id=str(entry["id"]))
    card.set_axis("V1", 1.0, "test-tool")
    return card


class FakeModelClient:
    """Stands in for ModelClient, exposing only the cumulative usage batch reads."""

    def __init__(self) -> None:
        self.total_usage = TokenUsage()

    def spend(self, tokens: int) -> None:
        self.total_usage = self.total_usage + TokenUsage(prompt_tokens=tokens, total_tokens=tokens)


class TestHappyPath:
    def test_runs_the_audit_for_every_entry(self) -> None:
        seen: list[str] = []

        def audit(entry: dict[str, Any]) -> Scorecard:
            seen.append(str(entry["id"]))
            return audit_ok(entry)

        run_batch(entries("a", "b", "c"), audit)
        assert seen == ["a", "b", "c"]

    def test_counts_successes(self) -> None:
        batch = run_batch(entries("a", "b"), audit_ok)
        assert (batch.total, batch.succeeded, batch.failed, batch.skipped) == (2, 2, 0, 0)

    def test_records_the_env_id_for_each_outcome(self) -> None:
        batch = run_batch(entries("a", "b"), audit_ok)
        assert [result.env_id for result in batch.results] == ["a", "b"]

    def test_an_empty_corpus_is_not_an_error(self) -> None:
        batch = run_batch([], audit_ok)
        assert batch.total == 0
        assert batch.total_tokens == 0

    def test_records_timing(self) -> None:
        batch = run_batch(entries("a"), audit_ok)
        assert batch.duration_seconds >= 0.0
        assert batch.finished_at != ""
        assert batch.results[0].duration_seconds >= 0.0

    def test_the_audit_receives_the_whole_manifest_entry(self) -> None:
        captured: list[dict[str, Any]] = []

        def audit(entry: dict[str, Any]) -> Scorecard:
            captured.append(entry)
            return audit_ok(entry)

        run_batch([{"id": "a", "format": "terminal", "image": "custom:1"}], audit)
        assert captured[0]["image"] == "custom:1"


class TestErrorIsolation:
    def test_continues_after_one_environment_raises(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            if entry["id"] == "b":
                raise RuntimeError("container would not start")
            return audit_ok(entry)

        batch = run_batch(entries("a", "b", "c"), audit)
        assert [result.outcome for result in batch.results] == [SUCCESS, FAILED, SUCCESS]
        assert batch.succeeded == 2
        assert batch.failed == 1

    def test_records_the_error_message_and_type(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            raise ValueError("manifest is nonsense")

        failure = run_batch(entries("a"), audit).results[0]
        assert failure.outcome == FAILED
        assert failure.error == "manifest is nonsense"
        assert failure.error_type == "ValueError"

    def test_exposes_the_failures_for_reporting(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            if entry["id"] in {"b", "d"}:
                raise RuntimeError("boom")
            return audit_ok(entry)

        batch = run_batch(entries("a", "b", "c", "d"), audit)
        assert [result.env_id for result in batch.failures] == ["b", "d"]

    def test_logs_the_failure_with_a_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            raise RuntimeError("deep failure")

        with caplog.at_level("ERROR", logger="verity_core.batch"):
            run_batch(entries("a"), audit)
        combined = "\n".join(record.message for record in caplog.records)
        assert "batch env failed env_id=a" in combined
        assert "Traceback" in combined

    def test_every_environment_failing_still_returns_a_result(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            raise RuntimeError("boom")

        batch = run_batch(entries("a", "b"), audit)
        assert batch.failed == 2
        assert batch.succeeded == 0

    def test_stop_on_error_halts_the_run(self) -> None:
        seen: list[str] = []

        def audit(entry: dict[str, Any]) -> Scorecard:
            seen.append(str(entry["id"]))
            raise RuntimeError("boom")

        batch = run_batch(entries("a", "b", "c"), audit, stop_on_error=True)
        assert seen == ["a"]
        assert batch.total == 1

    def test_a_failure_does_not_write_a_scorecard(self, tmp_path: Path) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            raise RuntimeError("boom")

        run_batch(entries("a"), audit, results_dir=tmp_path)
        assert list(tmp_path.glob("*.json")) == []

    def test_a_keyboard_interrupt_stops_the_run_and_keeps_the_work(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            if entry["id"] == "c":
                raise KeyboardInterrupt
            return audit_ok(entry)

        batch = run_batch(entries("a", "b", "c", "d"), audit)
        assert batch.interrupted is True
        assert batch.succeeded == 2
        assert batch.failed == 0

    def test_an_interrupt_is_not_recorded_as_a_failure(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            raise KeyboardInterrupt

        batch = run_batch(entries("a"), audit)
        assert batch.results == []
        assert batch.interrupted is True


class TestScorecardWriting:
    def test_writes_a_scorecard_per_environment(self, tmp_path: Path) -> None:
        run_batch(entries("suite/a", "suite/b"), audit_ok, results_dir=tmp_path)
        written = sorted(path.name for path in tmp_path.glob("*.json"))
        assert written == ["suite__a.json", "suite__b.json"]

    def test_the_written_scorecard_round_trips(self, tmp_path: Path) -> None:
        run_batch(entries("suite/a"), audit_ok, results_dir=tmp_path)
        reloaded = Scorecard.from_json(scorecard_path(tmp_path, "suite/a"))
        assert reloaded.env_id == "suite/a"
        assert reloaded.get_axis("V1").value == 1.0

    def test_records_where_each_scorecard_landed(self, tmp_path: Path) -> None:
        batch = run_batch(entries("a"), audit_ok, results_dir=tmp_path)
        assert batch.results[0].scorecard_path == str(scorecard_path(tmp_path, "a"))

    def test_writes_nothing_without_a_results_directory(self, tmp_path: Path) -> None:
        batch = run_batch(entries("a"), audit_ok)
        assert batch.results[0].scorecard_path is None
        assert list(tmp_path.glob("*.json")) == []

    def test_write_scorecards_can_be_disabled(self, tmp_path: Path) -> None:
        # For tools that write their own scorecards and only want the batch accounting.
        run_batch(entries("a"), audit_ok, results_dir=tmp_path, write_scorecards=False)
        assert list(tmp_path.glob("*.json")) == []

    def test_creates_the_results_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "results"
        run_batch(entries("a"), audit_ok, results_dir=target)
        assert (target / "a.json").is_file()


class TestSkipping:
    def test_a_none_return_is_a_skip_not_a_failure(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard | None:
            return None if entry["id"] == "b" else audit_ok(entry)

        batch = run_batch(entries("a", "b"), audit)
        assert batch.skipped == 1
        assert batch.failed == 0
        assert batch.results[1].skip_reason == "audit function returned no scorecard"

    def test_a_skip_writes_no_scorecard(self, tmp_path: Path) -> None:
        run_batch(entries("a"), lambda entry: None, results_dir=tmp_path)
        assert list(tmp_path.glob("*.json")) == []


class TestResumption:
    def test_skips_environments_that_already_have_a_scorecard(self, tmp_path: Path) -> None:
        run_batch(entries("a", "b"), audit_ok, results_dir=tmp_path)

        audited: list[str] = []

        def audit(entry: dict[str, Any]) -> Scorecard:
            audited.append(str(entry["id"]))
            return audit_ok(entry)

        batch = run_batch(entries("a", "b", "c"), audit, results_dir=tmp_path, resume=True)
        assert audited == ["c"]
        assert batch.skipped == 2
        assert batch.succeeded == 1

    def test_records_why_each_environment_was_skipped(self, tmp_path: Path) -> None:
        run_batch(entries("a"), audit_ok, results_dir=tmp_path)
        batch = run_batch(entries("a"), audit_ok, results_dir=tmp_path, resume=True)
        assert batch.results[0].skip_reason == "scorecard already in results directory"

    def test_re_audits_everything_without_resume(self, tmp_path: Path) -> None:
        run_batch(entries("a"), audit_ok, results_dir=tmp_path)
        batch = run_batch(entries("a"), audit_ok, results_dir=tmp_path, resume=False)
        assert batch.succeeded == 1
        assert batch.skipped == 0

    def test_resuming_an_interrupted_run_finishes_the_corpus(self, tmp_path: Path) -> None:
        corpus = entries("a", "b", "c", "d")

        def flaky(entry: dict[str, Any]) -> Scorecard:
            if entry["id"] == "c":
                raise KeyboardInterrupt
            return audit_ok(entry)

        first = run_batch(corpus, flaky, results_dir=tmp_path, resume=True)
        assert first.interrupted is True

        second = run_batch(corpus, audit_ok, results_dir=tmp_path, resume=True)
        assert second.skipped == 2
        assert second.succeeded == 2
        assert len(completed_env_ids(tmp_path)) == 4

    def test_resume_is_harmless_when_the_directory_is_missing(self, tmp_path: Path) -> None:
        batch = run_batch(entries("a"), audit_ok, results_dir=tmp_path / "absent", resume=True)
        assert batch.succeeded == 1


class TestCompletedEnvIds:
    def test_reads_ids_from_scorecard_contents(self, tmp_path: Path) -> None:
        run_batch(entries("suite/a", "suite/b"), audit_ok, results_dir=tmp_path)
        assert completed_env_ids(tmp_path) == {"suite/a", "suite/b"}

    def test_a_missing_directory_has_nothing_completed(self, tmp_path: Path) -> None:
        assert completed_env_ids(tmp_path / "absent") == set()

    def test_finds_scorecards_written_under_other_filenames(self, tmp_path: Path) -> None:
        # A tool may name files its own way; the id inside the file is what counts.
        audit_ok({"id": "suite/custom"}).to_json(tmp_path / "whatever-name.json")
        assert completed_env_ids(tmp_path) == {"suite/custom"}

    def test_ignores_truncated_scorecards(self, tmp_path: Path) -> None:
        # Exactly the state a crash mid-write leaves behind; such an env must be re-run.
        (tmp_path / "broken.json").write_text('{"env_id": "suite/a"', encoding="utf-8")
        assert completed_env_ids(tmp_path) == set()

    def test_ignores_json_without_an_env_id(self, tmp_path: Path) -> None:
        (tmp_path / "other.json").write_text(json.dumps({"unrelated": True}), encoding="utf-8")
        assert completed_env_ids(tmp_path) == set()

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not a scorecard", encoding="utf-8")
        assert completed_env_ids(tmp_path) == set()

    def test_warns_about_files_it_could_not_read(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "broken.json").write_text("{", encoding="utf-8")
        with caplog.at_level("WARNING", logger="verity_core.batch"):
            completed_env_ids(tmp_path)
        assert any("unreadable scorecard" in record.message for record in caplog.records)


class TestTokenAccounting:
    def test_attributes_tokens_per_environment_from_the_model_client(self) -> None:
        client = FakeModelClient()
        spend = {"a": 100, "b": 250}

        def audit(entry: dict[str, Any]) -> Scorecard:
            client.spend(spend[str(entry["id"])])
            return audit_ok(entry)

        batch = run_batch(entries("a", "b"), audit, model_client=client)
        assert [result.tokens for result in batch.results] == [100, 250]
        assert batch.total_tokens == 350

    def test_counts_tokens_spent_before_a_failure(self) -> None:
        client = FakeModelClient()

        def audit(entry: dict[str, Any]) -> Scorecard:
            client.spend(40)
            raise RuntimeError("failed after spending")

        batch = run_batch(entries("a"), audit, model_client=client)
        assert batch.results[0].tokens == 40
        assert batch.total_tokens == 40

    def test_ignores_spend_that_happened_before_the_batch(self) -> None:
        client = FakeModelClient()
        client.spend(1000)
        batch = run_batch(entries("a"), audit_ok, model_client=client)
        assert batch.total_tokens == 0

    def test_falls_back_to_scorecard_metadata(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard:
            card = audit_ok(entry)
            card.metadata["tokens"] = 77
            return card

        assert run_batch(entries("a"), audit).total_tokens == 77

    def test_tokens_are_zero_without_any_source(self) -> None:
        assert run_batch(entries("a"), audit_ok).total_tokens == 0

    def test_accounts_tokens_from_a_real_model_client(self, tmp_path: Path) -> None:
        # FakeModelClient could drift from ModelClient's usage surface, so this pins the
        # accounting against the real client over a mocked transport.
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "stub",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                },
            )

        client = ModelClient(
            cache_dir=None, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        )

        def audit(entry: dict[str, Any]) -> Scorecard:
            client.complete("stub", [{"role": "user", "content": str(entry["id"])}])
            return audit_ok(entry)

        batch = run_batch(entries("a", "b"), audit, model_client=client)
        assert [result.tokens for result in batch.results] == [20, 20]
        assert batch.total_tokens == 40


class TestProgress:
    def test_reports_progress_after_each_environment(self) -> None:
        seen: list[tuple[str, str, int, int]] = []

        def progress(result: Any, index: int, total: int) -> None:
            seen.append((result.env_id, result.outcome, index, total))

        run_batch(entries("a", "b"), audit_ok, progress=progress)
        assert seen == [("a", SUCCESS, 1, 2), ("b", SUCCESS, 2, 2)]

    def test_reports_failures_and_skips_too(self) -> None:
        outcomes: list[str] = []

        def audit(entry: dict[str, Any]) -> Scorecard | None:
            if entry["id"] == "a":
                raise RuntimeError("boom")
            return None if entry["id"] == "b" else audit_ok(entry)

        run_batch(
            entries("a", "b", "c"),
            audit,
            progress=lambda result, index, total: outcomes.append(result.outcome),
        )
        assert outcomes == [FAILED, SKIPPED, SUCCESS]

    def test_logs_a_line_per_environment(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="verity_core.batch"):
            run_batch(entries("a", "b"), audit_ok)
        combined = "\n".join(record.message for record in caplog.records)
        assert "batch env start env_id=a (1/2)" in combined
        assert "batch env done env_id=b (2/2)" in combined
        assert "batch finished total=2 succeeded=2 failed=0 skipped=0" in combined


class TestBatchResultSerialization:
    def test_summarizes_counts_and_totals(self) -> None:
        def audit(entry: dict[str, Any]) -> Scorecard | None:
            if entry["id"] == "b":
                raise RuntimeError("boom")
            return None if entry["id"] == "c" else audit_ok(entry)

        payload = run_batch(entries("a", "b", "c"), audit).to_dict()
        assert payload["summary"] == {
            "total": 3,
            "succeeded": 1,
            "failed": 1,
            "skipped": 1,
            "total_tokens": 0,
        }
        assert len(payload["results"]) == 3

    def test_writes_a_summary_file(self, tmp_path: Path) -> None:
        batch = run_batch(entries("a"), audit_ok)
        target = tmp_path / "nested" / "batch.json"
        batch.to_json(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["summary"]["succeeded"] == 1
        assert payload["results"][0]["env_id"] == "a"

    def test_an_empty_batch_serializes(self) -> None:
        payload = BatchResult().to_dict()
        assert payload["summary"]["total"] == 0
