"""Tests for the sandbox runner, driven by a fake Docker client."""

from __future__ import annotations

import json

import pytest
from docker.errors import DockerException

from conftest import FakeDockerClient
from verity_core.runner import (
    KILL_GRACE_SECONDS,
    SIGKILL_EXIT_CODE,
    TIMEOUT_EXIT_CODE,
    ExecResult,
    ResourceLimits,
    SandboxError,
    SandboxRunner,
    _is_timeout,
)

IMAGE = "verity/test-image:latest"


@pytest.fixture
def runner(docker_client: FakeDockerClient) -> SandboxRunner:
    return SandboxRunner(
        image=IMAGE,
        limits=ResourceLimits(cpu_count=1.5, memory_limit="512m", timeout_seconds=30),
        client=docker_client,
    )


class TestResourceLimits:
    def test_network_is_disabled_by_default(self) -> None:
        assert ResourceLimits().network_disabled is True

    def test_converts_cpu_count_to_nano_cpus(self) -> None:
        assert ResourceLimits(cpu_count=2.5).nano_cpus == 2_500_000_000

    @pytest.mark.parametrize("cpu_count", [0, -1.0])
    def test_rejects_a_non_positive_cpu_count(self, cpu_count: float) -> None:
        with pytest.raises(ValueError, match="cpu_count must be positive"):
            ResourceLimits(cpu_count=cpu_count)

    def test_rejects_a_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ResourceLimits(timeout_seconds=0)


class TestExecResult:
    def test_ok_requires_a_zero_exit_and_no_timeout(self) -> None:
        assert ExecResult(0, "", "", 0.1).ok is True
        assert ExecResult(1, "", "", 0.1).ok is False
        assert ExecResult(0, "", "", 0.1, timed_out=True).ok is False

    def test_serializes_every_field(self) -> None:
        payload = ExecResult(2, "out", "err", 0.5, timed_out=True, command="ls").to_dict()
        assert payload == {
            "command": "ls",
            "exit_code": 2,
            "stdout": "out",
            "stderr": "err",
            "duration_seconds": 0.5,
            "timed_out": True,
        }


class TestTimeoutDetection:
    def test_124_is_conclusive(self) -> None:
        assert _is_timeout(TIMEOUT_EXIT_CODE, 0.1, 30) is True

    def test_137_at_the_deadline_counts_as_a_timeout(self) -> None:
        assert _is_timeout(SIGKILL_EXIT_CODE, 30.0, 30) is True

    def test_137_well_before_the_deadline_is_not_a_timeout(self) -> None:
        # An OOM kill shares this status but happens early; calling it a hang would
        # misreport the finding.
        assert _is_timeout(SIGKILL_EXIT_CODE, 0.2, 30) is False

    def test_an_ordinary_failure_is_not_a_timeout(self) -> None:
        assert _is_timeout(1, 0.1, 30) is False


class TestStart:
    def test_applies_every_resource_limit(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        kwargs = docker_client.last.create_kwargs
        assert kwargs["mem_limit"] == "512m"
        assert kwargs["nano_cpus"] == 1_500_000_000
        assert kwargs["network_disabled"] is True
        assert kwargs["detach"] is True
        assert kwargs["privileged"] is False

    def test_uses_the_requested_image_and_workdir(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        assert docker_client.last.image == IMAGE
        assert docker_client.last.create_kwargs["working_dir"] == "/workspace"

    def test_labels_the_container_for_cleanup(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        assert docker_client.last.create_kwargs["labels"] == {"verity.sandbox": "1"}

    def test_network_can_be_enabled_deliberately(self, docker_client: FakeDockerClient) -> None:
        runner = SandboxRunner(
            image=IMAGE,
            limits=ResourceLimits(network_disabled=False),
            client=docker_client,
        )
        runner.start()
        assert docker_client.last.create_kwargs["network_disabled"] is False

    def test_restarting_replaces_the_previous_container(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        first = docker_client.last
        runner.start()
        assert first.removed is True
        assert len(docker_client.created) == 2

    def test_a_docker_failure_becomes_a_sandbox_error(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        docker_client.run_error = DockerException("no such image")
        with pytest.raises(SandboxError, match="failed to start container"):
            runner.start()

    def test_using_the_runner_before_starting_is_an_error(self, runner: SandboxRunner) -> None:
        assert runner.is_running is False
        with pytest.raises(SandboxError, match="call start\\(\\) first"):
            _ = runner.container


class TestExec:
    def test_wraps_the_command_in_a_timeout(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        runner.exec("echo hi")
        assert docker_client.last.raw_commands[0] == [
            "timeout",
            f"--kill-after={KILL_GRACE_SECONDS}s",
            "30s",
            "/bin/sh",
            "-c",
            "echo hi",
        ]

    def test_a_per_call_timeout_overrides_the_default(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        runner.exec("sleep 1", timeout=5)
        assert docker_client.last.raw_commands[0][2] == "5s"

    def test_a_command_given_as_a_list_is_quoted(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        result = runner.exec(["echo", "two words"])
        assert result.command == "echo 'two words'"

    def test_captures_stdout_and_stderr_separately(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (0, "to stdout", "to stderr"))
        runner.start()
        result = runner.exec("noisy")
        assert result.stdout == "to stdout"
        assert result.stderr == "to stderr"
        assert result.ok is True

    def test_reports_a_nonzero_exit_code(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (3, "", "failed"))
        runner.start()
        result = runner.exec("false")
        assert result.exit_code == 3
        assert result.ok is False

    def test_flags_a_timed_out_command(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (TIMEOUT_EXIT_CODE, "", ""))
        runner.start()
        result = runner.exec("sleep 999")
        assert result.timed_out is True
        assert result.ok is False

    def test_records_the_duration(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        assert runner.exec("echo hi").duration_seconds >= 0

    def test_exec_auto_starts_nothing_and_requires_a_container(self, runner: SandboxRunner) -> None:
        with pytest.raises(SandboxError, match="call start\\(\\) first"):
            runner.exec("echo hi")


class TestFileTransfer:
    def test_writes_then_reads_a_file(self, runner: SandboxRunner) -> None:
        runner.start()
        runner.write_file("/workspace/sub.py", "print('hi')\n")
        assert runner.read_file("/workspace/sub.py") == b"print('hi')\n"

    def test_preserves_quotes_and_newlines(self, runner: SandboxRunner) -> None:
        # Written through an archive rather than echoed through a shell, so content
        # that would need escaping survives untouched.
        payload = "line one\n'single' \"double\" $VAR `cmd`\n"
        runner.start()
        runner.write_file("/workspace/tricky.sh", payload)
        assert runner.read_file("/workspace/tricky.sh").decode() == payload

    def test_accepts_bytes(self, runner: SandboxRunner) -> None:
        runner.start()
        runner.write_file("/workspace/blob", b"\x00\x01\x02")
        assert runner.read_file("/workspace/blob") == b"\x00\x01\x02"

    def test_reading_an_absent_file_is_an_error(self, runner: SandboxRunner) -> None:
        runner.start()
        with pytest.raises(SandboxError, match="no such file in container"):
            runner.read_file("/workspace/absent")


class TestSnapshotAndRestore:
    def test_the_token_is_versioned_json(self, runner: SandboxRunner) -> None:
        runner.start()
        token = json.loads(runner.snapshot().decode())
        assert token["version"] == 1
        assert token["kind"] == "docker-commit"
        assert token["image"].startswith("sha256:")
        assert token["source_image"] == IMAGE

    def test_restore_starts_a_container_from_the_snapshot_image(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        snap = runner.snapshot()
        image_id = json.loads(snap.decode())["image"]
        runner.restore(snap)
        assert docker_client.last.image == image_id
        assert runner.is_running is True

    def test_restore_replaces_the_running_container(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        original = docker_client.last
        runner.restore(runner.snapshot())
        assert original.removed is True

    def test_restore_rejects_a_non_json_token(self, runner: SandboxRunner) -> None:
        runner.start()
        with pytest.raises(SandboxError, match="not valid verity snapshot JSON"):
            runner.restore(b"\xff\xfe not json")

    def test_restore_rejects_an_unknown_snapshot_kind(self, runner: SandboxRunner) -> None:
        runner.start()
        token = json.dumps({"kind": "overlayfs", "image": "x"}).encode()
        with pytest.raises(SandboxError, match="unsupported snapshot kind"):
            runner.restore(token)

    def test_restore_rejects_a_token_without_an_image(self, runner: SandboxRunner) -> None:
        runner.start()
        token = json.dumps({"kind": "docker-commit"}).encode()
        with pytest.raises(SandboxError, match="missing an image reference"):
            runner.restore(token)


class TestCleanup:
    def test_removes_the_container_and_snapshot_images(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        image_id = json.loads(runner.snapshot().decode())["image"]
        runner.cleanup()
        assert docker_client.last.removed is True
        assert docker_client.images.removed == [image_id]
        assert runner.is_running is False

    def test_cleanup_is_safe_before_start(self, runner: SandboxRunner) -> None:
        runner.cleanup()
        assert runner.is_running is False

    def test_cleanup_is_idempotent(self, runner: SandboxRunner) -> None:
        runner.start()
        runner.snapshot()
        runner.cleanup()
        runner.cleanup()
        assert runner.is_running is False

    def test_stop_leaves_snapshot_images_in_place(
        self, runner: SandboxRunner, docker_client: FakeDockerClient
    ) -> None:
        runner.start()
        runner.snapshot()
        runner.stop()
        assert docker_client.images.removed == []

    def test_the_context_manager_starts_and_cleans_up(
        self, docker_client: FakeDockerClient
    ) -> None:
        with SandboxRunner(image=IMAGE, client=docker_client) as runner:
            assert runner.is_running is True
            container = docker_client.last
        assert container.removed is True
        assert runner.is_running is False

    def test_cleanup_runs_even_when_the_body_raises(self, docker_client: FakeDockerClient) -> None:
        with pytest.raises(RuntimeError), SandboxRunner(image=IMAGE, client=docker_client):
            raise RuntimeError("audit blew up")
        assert docker_client.last.removed is True
