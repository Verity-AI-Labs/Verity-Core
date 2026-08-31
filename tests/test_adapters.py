"""Tests for the adapters and the load_env factory.

The central guarantee is that whatever upstream format a manifest names, a tool gets
back an object satisfying the VerityEnv protocol and can call every method on it.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from conftest import FakeDockerClient
from verity_core.adapters import (
    ADAPTERS,
    DockerTestAdapter,
    ManifestError,
    TerminalAdapter,
    VerifiersAdapter,
    canonical_format,
    load_env,
    register_adapter,
)
from verity_core.adapters.base import ContainerEnv, resolve_callable
from verity_core.env import RewardResult, VerityEnv

IMAGE = "verity/task:latest"

TERMINAL_MANIFEST: dict[str, Any] = {
    "id": "terminal-bench/hello-world",
    "format": "terminal",
    "image": IMAGE,
    "source": "https://github.com/laude-institute/terminal-bench",
    "commit": "abc1234",
    "instructions": "Create /workspace/done.",
    "solution": "touch /workspace/done\n",
}

DOCKER_TEST_MANIFEST: dict[str, Any] = {
    "id": "swe-gym/task-42",
    "format": "docker_test",
    "image": IMAGE,
    "test_command": "pytest -q",
    "gold_patch": "--- a\n+++ b\n",
}

VERIFIERS_MANIFEST: dict[str, Any] = {
    "id": "prime/gsm8k-0",
    "format": "verifiers",
    "domain": "math",
    "instructions": "Answer the question.",
    "task": {"answer": "42"},
}


def exact_answer(task: dict[str, Any], submission: str) -> float:
    return 1.0 if submission.strip() == task["answer"] else 0.0


class TestFactory:
    @pytest.mark.parametrize(
        ("manifest", "kwargs", "expected"),
        [
            (VERIFIERS_MANIFEST, {"reward_fn": exact_answer}, VerifiersAdapter),
            (TERMINAL_MANIFEST, {}, TerminalAdapter),
            (DOCKER_TEST_MANIFEST, {}, DockerTestAdapter),
        ],
        ids=["verifiers", "terminal", "docker_test"],
    )
    def test_selects_the_adapter_named_by_the_format(
        self, manifest: dict[str, Any], kwargs: dict[str, Any], expected: type
    ) -> None:
        env = load_env(manifest, **kwargs)
        assert isinstance(env, expected)
        # Whatever the upstream format, a tool gets something it can call.
        assert isinstance(env, VerityEnv)

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("verifiers", "verifiers"),
            ("prime", "verifiers"),
            ("Prime-Intellect", "verifiers"),
            ("terminal", "terminal"),
            ("terminal-bench", "terminal"),
            ("Terminal Wrench", "terminal"),
            ("tbench", "terminal"),
            ("docker_test", "docker_test"),
            ("r2e-gym", "docker_test"),
            ("SWE-Gym", "docker_test"),
        ],
    )
    def test_resolves_format_aliases(self, alias: str, expected: str) -> None:
        assert canonical_format(alias) == expected

    def test_rejects_an_unknown_format(self) -> None:
        with pytest.raises(ManifestError, match="unknown environment format"):
            load_env({"id": "x", "format": "gymnasium"})

    def test_rejects_a_manifest_without_a_format(self) -> None:
        with pytest.raises(ManifestError, match="needs a 'format' field"):
            load_env({"id": "x", "image": IMAGE})

    def test_rejects_a_manifest_that_is_not_a_mapping(self) -> None:
        with pytest.raises(ManifestError, match="must be a mapping"):
            load_env(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_a_new_format_can_be_registered(self) -> None:
        class CustomAdapter(ContainerEnv):
            default_test_command = "true"

        try:
            register_adapter("custom_fmt", CustomAdapter, aliases=("custom",))
            env = load_env({"id": "c/1", "format": "custom", "image": IMAGE})
            assert isinstance(env, CustomAdapter)
        finally:
            ADAPTERS.pop("custom_fmt", None)


class TestTaskSpecParsing:
    def test_reads_identity_and_provenance(self) -> None:
        spec = load_env(TERMINAL_MANIFEST).spec()
        assert spec.id == "terminal-bench/hello-world"
        assert spec.source == "https://github.com/laude-institute/terminal-bench"
        assert spec.commit == "abc1234"
        assert spec.instructions == "Create /workspace/done."

    def test_each_format_has_a_sensible_default_domain(self) -> None:
        assert load_env(TERMINAL_MANIFEST).spec().domain == "tool_use"
        assert load_env(DOCKER_TEST_MANIFEST).spec().domain == "code"

    def test_a_manifest_domain_overrides_the_default(self) -> None:
        spec = load_env({**TERMINAL_MANIFEST, "domain": "browser"}).spec()
        assert spec.domain == "browser"

    def test_accepts_instruction_in_the_singular(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "instructions"}
        spec = load_env({**manifest, "instruction": "singular key"}).spec()
        assert spec.instructions == "singular key"

    def test_infers_has_gold_from_a_solution_field(self) -> None:
        assert load_env(TERMINAL_MANIFEST).spec().has_gold is True
        assert load_env(DOCKER_TEST_MANIFEST).spec().has_gold is True

    def test_has_gold_is_false_without_a_reference_solution(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "solution"}
        assert load_env(manifest).spec().has_gold is False

    def test_an_explicit_has_gold_is_respected(self) -> None:
        assert load_env({**TERMINAL_MANIFEST, "has_gold": False}).spec().has_gold is False

    def test_rejects_a_missing_id(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "id"}
        with pytest.raises(ManifestError, match="non-empty 'id' field"):
            load_env(manifest)

    def test_rejects_an_unknown_domain(self) -> None:
        with pytest.raises(ManifestError, match="unknown domain"):
            load_env({**TERMINAL_MANIFEST, "domain": "olfactory"})

    def test_rejects_an_unknown_reward_type(self) -> None:
        with pytest.raises(ManifestError, match="unknown reward_type"):
            load_env({**TERMINAL_MANIFEST, "reward_type": "vibes"})


class TestResourceLimitsFromManifest:
    def test_defaults_disable_the_network(self, docker_client: FakeDockerClient) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        assert docker_client.last.create_kwargs["network_disabled"] is True

    def test_reads_limits_from_the_manifest(self, docker_client: FakeDockerClient) -> None:
        manifest = {
            **TERMINAL_MANIFEST,
            "limits": {"cpu_count": 4, "memory_limit": "8g", "timeout_seconds": 45},
        }
        env = load_env(manifest, docker_client=docker_client)
        env.reset()
        kwargs = docker_client.last.create_kwargs
        assert kwargs["nano_cpus"] == 4_000_000_000
        assert kwargs["mem_limit"] == "8g"

    def test_a_bare_timeout_field_sets_the_deadline(self, docker_client: FakeDockerClient) -> None:
        env = load_env({**TERMINAL_MANIFEST, "timeout": 15}, docker_client=docker_client)
        env.reset()
        env.step("echo hi")
        assert docker_client.last.raw_commands[0][2] == "15s"

    def test_the_network_can_be_enabled_deliberately(self, docker_client: FakeDockerClient) -> None:
        manifest = {**TERMINAL_MANIFEST, "limits": {"network_disabled": False}}
        env = load_env(manifest, docker_client=docker_client)
        env.reset()
        assert docker_client.last.create_kwargs["network_disabled"] is False

    def test_rejects_limits_that_are_not_a_mapping(self) -> None:
        with pytest.raises(ManifestError, match="'limits' must be a mapping"):
            load_env({**TERMINAL_MANIFEST, "limits": ["4g"]})


class TestContainerLifecycle:
    def test_reset_starts_a_container_and_returns_the_instructions(
        self, docker_client: FakeDockerClient
    ) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        observation = env.reset()
        assert observation.text == "Create /workspace/done."
        assert observation.metadata["task_id"] == "terminal-bench/hello-world"
        assert len(docker_client.created) == 1

    def test_reset_runs_manifest_setup_commands(self, docker_client: FakeDockerClient) -> None:
        manifest = {**TERMINAL_MANIFEST, "setup_commands": ["git init", "pip install -e ."]}
        env = load_env(manifest, docker_client=docker_client)
        env.reset()
        assert docker_client.last.commands == ["git init", "pip install -e ."]

    def test_reset_recreates_the_container(self, docker_client: FakeDockerClient) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        env.reset()
        assert len(docker_client.created) == 2
        assert docker_client.created[0].removed is True

    def test_step_runs_the_action_and_returns_its_output(
        self, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (0, f"ran {command}", ""))
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        result = env.step("ls -la")
        assert result.observation.text == "ran ls -la"
        assert result.info["exit_code"] == 0

    def test_step_carries_no_reward_of_its_own(self, docker_client: FakeDockerClient) -> None:
        # The verifier is the only source of reward for these formats.
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        result = env.step("echo hi")
        assert result.reward == 0.0
        assert result.done is False

    def test_step_reports_stderr_when_stdout_is_empty(
        self, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (1, "", "command not found"))
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        assert env.step("nope").observation.text == "command not found"

    def test_close_cleans_up_the_container(self, docker_client: FakeDockerClient) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        env.close()
        assert docker_client.last.removed is True

    def test_the_context_manager_resets_and_closes(self, docker_client: FakeDockerClient) -> None:
        with load_env(TERMINAL_MANIFEST, docker_client=docker_client) as env:
            assert env.spec().id == "terminal-bench/hello-world"
        assert docker_client.last.removed is True

    def test_no_container_is_created_until_it_is_needed(
        self, docker_client: FakeDockerClient
    ) -> None:
        load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        assert docker_client.created == []


class TestTerminalAdapter:
    def test_defaults_follow_terminal_bench_conventions(self) -> None:
        env = load_env(TERMINAL_MANIFEST)
        assert env.test_command == "bash /tests/run-tests.sh"
        assert env.submission_path == "/workspace/solution.sh"
        assert env.apply_command == "bash /workspace/solution.sh"

    def test_the_apply_command_follows_an_overridden_submission_path(self) -> None:
        env = load_env({**TERMINAL_MANIFEST, "submission_path": "/tmp/candidate.sh"})
        assert env.apply_command == "bash /tmp/candidate.sh"

    def test_solution_is_accepted_as_the_gold_solution(self) -> None:
        assert load_env(TERMINAL_MANIFEST).gold_solution() == "touch /workspace/done\n"

    def test_writes_the_submission_then_runs_it_before_the_tests(
        self, docker_client: FakeDockerClient
    ) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.verify("touch /workspace/done\n")
        assert docker_client.last.files["/workspace/solution.sh"] == b"touch /workspace/done\n"
        assert docker_client.last.commands == [
            "bash /workspace/solution.sh",
            "bash /tests/run-tests.sh",
        ]

    def test_passing_tests_yield_a_full_reward(self, docker_client: FakeDockerClient) -> None:
        docker_client.set_handler(lambda command, files: (0, "2 passed", ""))
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        result = env.verify("touch /workspace/done\n")
        assert result.reward == 1.0
        assert result.verdict is True
        assert "2 passed" in result.verifier_logs

    def test_failing_tests_yield_no_reward(self, docker_client: FakeDockerClient) -> None:
        docker_client.set_handler(lambda command, files: (1, "", "1 failed"))
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        result = env.verify("echo nothing\n")
        assert result.reward == 0.0
        assert result.verdict is False
        assert "1 failed" in result.verifier_logs

    def test_the_test_exit_status_is_recorded_as_evidence(
        self, docker_client: FakeDockerClient
    ) -> None:
        docker_client.set_handler(lambda command, files: (7, "out", "err"))
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        state = env.verify("x").verifier_state
        assert state["test"]["exit_code"] == 7
        assert state["test"]["command"] == "bash /tests/run-tests.sh"

    def test_verify_auto_starts_the_sandbox(self, docker_client: FakeDockerClient) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.verify("x")
        assert len(docker_client.created) == 1

    def test_an_agentic_task_grades_the_container_it_was_given(
        self, docker_client: FakeDockerClient
    ) -> None:
        # Resetting here would discard the agent's work, which is the thing under test.
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        env.step("touch /workspace/done")
        env.verify("x")
        assert len(docker_client.created) == 1

    def test_repeated_trials_can_be_isolated_on_request(
        self, docker_client: FakeDockerClient
    ) -> None:
        env = load_env(
            {**TERMINAL_MANIFEST, "reset_before_verify": True}, docker_client=docker_client
        )
        env.verify("first")
        env.verify("second")
        assert len(docker_client.created) == 2


class TestDockerTestAdapter:
    def test_defaults_target_a_patch_submission(self) -> None:
        env = load_env(DOCKER_TEST_MANIFEST)
        assert env.submission_path == "/workspace/submission.patch"
        assert env.test_command == "pytest -q"
        assert env.apply_command == ""

    def test_each_verification_starts_from_a_clean_container(
        self, docker_client: FakeDockerClient
    ) -> None:
        # Reusing a container would leave the previous patch applied.
        env = load_env(DOCKER_TEST_MANIFEST, docker_client=docker_client)
        env.verify("patch one")
        env.verify("patch two")
        assert len(docker_client.created) == 2

    def test_gold_patch_is_accepted_as_the_gold_solution(self) -> None:
        assert load_env(DOCKER_TEST_MANIFEST).gold_solution() == "--- a\n+++ b\n"

    def test_runs_the_apply_command_before_the_tests(self, docker_client: FakeDockerClient) -> None:
        manifest = {
            **DOCKER_TEST_MANIFEST,
            "apply_command": "git apply /workspace/submission.patch",
        }
        env = load_env(manifest, docker_client=docker_client)
        env.verify("--- a\n")
        assert docker_client.last.commands == [
            "git apply /workspace/submission.patch",
            "pytest -q",
        ]

    def test_a_submission_that_will_not_apply_fails_regardless_of_the_tests(
        self, docker_client: FakeDockerClient
    ) -> None:
        def handler(command: str, files: dict[str, bytes]) -> tuple[int, str, str]:
            if command.startswith("git apply"):
                return 1, "", "patch does not apply"
            return 0, "all tests passed", ""

        docker_client.set_handler(handler)
        manifest = {
            **DOCKER_TEST_MANIFEST,
            "apply_command": "git apply /workspace/submission.patch",
        }
        env = load_env(manifest, docker_client=docker_client)
        result = env.verify("garbage")
        assert result.verdict is False
        assert result.reward == 0.0
        assert result.verifier_state["apply_failed"] is True

    def test_a_partial_reward_type_records_the_binary_fallback(
        self, docker_client: FakeDockerClient
    ) -> None:
        manifest = {**DOCKER_TEST_MANIFEST, "reward_type": "partial"}
        env = load_env(manifest, docker_client=docker_client)
        result = env.verify("x")
        assert result.verifier_state["grading"] == "binary_fallback"

    def test_requires_a_test_command(self, docker_client: FakeDockerClient) -> None:
        manifest = {k: v for k, v in DOCKER_TEST_MANIFEST.items() if k != "test_command"}
        env = load_env(manifest, docker_client=docker_client)
        with pytest.raises(ManifestError, match="needs a 'test_command' field"):
            env.verify("x")


class TestContainerImageResolution:
    def test_requires_an_image(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "image"}
        with pytest.raises(ManifestError, match="needs an 'image' field"):
            load_env(manifest)

    def test_building_from_a_dockerfile_is_reported_as_unsupported(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "image"}
        with pytest.raises(ManifestError, match="not supported yet"):
            load_env({**manifest, "dockerfile": "Dockerfile"})


class TestContainerGoldAndSnapshots:
    def test_gold_can_be_read_from_a_path_inside_the_container(
        self, docker_client: FakeDockerClient
    ) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "solution"}
        env = load_env(
            {**manifest, "gold_solution_path": "/opt/solution.sh"},
            docker_client=docker_client,
        )
        env.reset()
        docker_client.last.files["/opt/solution.sh"] = b"echo gold\n"
        assert env.gold_solution() == "echo gold\n"

    def test_gold_is_none_when_the_task_ships_without_one(self) -> None:
        manifest = {k: v for k, v in TERMINAL_MANIFEST.items() if k != "solution"}
        assert load_env(manifest).gold_solution() is None

    def test_snapshot_and_restore_go_through_the_sandbox(
        self, docker_client: FakeDockerClient
    ) -> None:
        env = load_env(TERMINAL_MANIFEST, docker_client=docker_client)
        env.reset()
        snap = env.snapshot()
        assert json.loads(snap.decode())["kind"] == "docker-commit"
        env.restore(snap)
        assert docker_client.last.image == json.loads(snap.decode())["image"]


class TestVerifiersAdapter:
    def test_reset_exposes_the_task_configuration(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        observation = env.reset()
        assert observation.text == "Answer the question."
        assert observation.metadata["task"] == {"answer": "42"}

    def test_scores_a_correct_submission(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        result = env.verify("42")
        assert result.reward == 1.0
        assert result.verdict is True

    def test_scores_an_incorrect_submission(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        result = env.verify("41")
        assert result.reward == 0.0
        assert result.verdict is False

    def test_step_scores_the_action_and_ends_the_episode(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        env.reset()
        result = env.step("42")
        assert result.reward == 1.0
        assert result.done is True
        assert result.info["verdict"] is True

    def test_the_transcript_round_trips_through_a_snapshot(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        env.reset()
        env.step("42")
        snap = env.snapshot()
        env.reset()
        env.restore(snap)
        assert json.loads(snap.decode())["transcript"] == ["42"]

    def test_restore_rejects_a_container_snapshot(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        token = json.dumps({"kind": "docker-commit", "image": "x"}).encode()
        with pytest.raises(ManifestError, match="unsupported snapshot kind"):
            env.restore(token)

    def test_gold_solution_comes_from_the_manifest(self) -> None:
        env = load_env({**VERIFIERS_MANIFEST, "gold_solution": "42"}, reward_fn=exact_answer)
        assert env.gold_solution() == "42"
        assert env.spec().has_gold is True

    def test_requires_a_reward_reference(self) -> None:
        with pytest.raises(ManifestError, match="needs a 'reward' field"):
            load_env(VERIFIERS_MANIFEST).verify("42")

    def test_rejects_a_task_that_is_not_a_mapping(self) -> None:
        with pytest.raises(ManifestError, match="'task' must be a mapping"):
            load_env({**VERIFIERS_MANIFEST, "task": ["answer"]})

    def test_rollout_without_a_declared_function_is_an_error(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        with pytest.raises(ManifestError, match="declares no 'rollout' function"):
            env.rollout()

    def test_works_as_a_context_manager(self) -> None:
        with load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer) as env:
            assert env.verify("42").verdict is True

    def test_close_clears_the_transcript_and_is_repeatable(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=exact_answer)
        env.reset()
        env.step("42")
        env.close()
        env.close()
        assert json.loads(env.snapshot().decode())["transcript"] == []

    def test_rollout_passes_arguments_through(self) -> None:
        env = load_env(
            VERIFIERS_MANIFEST,
            reward_fn=exact_answer,
            rollout_fn=lambda *args, **kwargs: (args, kwargs),
        )
        assert env.rollout(1, key="v") == ((1,), {"key": "v"})


class TestVerifiersRewardNormalization:
    def test_accepts_a_float(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: 1.0)
        assert env.verify("x").reward == 1.0

    def test_accepts_a_bool(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: True)
        result = env.verify("x")
        assert (result.reward, result.verdict) == (1.0, True)

    def test_accepts_a_mapping_with_reward_and_verdict(self) -> None:
        env = load_env(
            VERIFIERS_MANIFEST,
            reward_fn=lambda task, submission: {
                "reward": 0.6,
                "passed": False,
                "logs": "3 of 5",
                "tests": 5,
            },
        )
        result = env.verify("x")
        assert result.reward == 0.6
        assert result.verdict is False
        assert result.verifier_logs == "3 of 5"
        assert result.verifier_state["tests"] == 5

    def test_accepts_a_mapping_keyed_on_score(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: {"score": 0.25})
        assert env.verify("x").reward == 0.25

    def test_derives_a_reward_from_a_verdict_only_mapping(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: {"success": True})
        result = env.verify("x")
        assert (result.reward, result.verdict) == (1.0, True)

    def test_accepts_a_reward_and_info_pair(self) -> None:
        env = load_env(
            VERIFIERS_MANIFEST, reward_fn=lambda task, submission: (0.5, {"tests_passed": 2})
        )
        result = env.verify("x")
        assert result.reward == 0.5
        assert result.verifier_state == {"tests_passed": 2}

    def test_accepts_a_reward_and_log_pair(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: (0.5, "some logs"))
        assert env.verify("x").verifier_logs == "some logs"

    def test_passes_a_reward_result_straight_through(self) -> None:
        expected = RewardResult(0.3, True, "logs", {"k": "v"})
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: expected)
        assert env.verify("x") is expected

    def test_the_verdict_threshold_defaults_to_a_full_reward(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: 0.99)
        assert env.verify("x").verdict is False

    def test_the_verdict_threshold_is_configurable(self) -> None:
        env = load_env(
            {**VERIFIERS_MANIFEST, "pass_threshold": 0.5},
            reward_fn=lambda task, submission: 0.6,
        )
        assert env.verify("x").verdict is True

    def test_rejects_an_unsupported_return_type(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: object())
        with pytest.raises(ManifestError, match="unsupported type"):
            env.verify("x")

    def test_rejects_a_mapping_with_neither_reward_nor_verdict(self) -> None:
        env = load_env(VERIFIERS_MANIFEST, reward_fn=lambda task, submission: {"note": "hi"})
        with pytest.raises(ManifestError, match="no reward key"):
            env.verify("x")


class TestVerifiersRewardSignatures:
    def test_calls_a_task_and_submission_signature_by_keyword(self) -> None:
        seen: dict[str, Any] = {}

        def reward(task: dict[str, Any], submission: str) -> float:
            seen.update(task=task, submission=submission)
            return 1.0

        load_env(VERIFIERS_MANIFEST, reward_fn=reward).verify("answer")
        assert seen == {"task": {"answer": "42"}, "submission": "answer"}

    def test_supports_a_kwargs_only_signature(self) -> None:
        def reward(**kwargs: Any) -> float:
            return 1.0 if kwargs["submission"] == "42" else 0.0

        assert load_env(VERIFIERS_MANIFEST, reward_fn=reward).verify("42").reward == 1.0

    def test_supports_a_completion_parameter_name(self) -> None:
        def reward(task: dict[str, Any], completion: str) -> float:
            return 1.0 if completion == task["answer"] else 0.0

        assert load_env(VERIFIERS_MANIFEST, reward_fn=reward).verify("42").reward == 1.0

    def test_supports_a_single_argument_signature(self) -> None:
        assert load_env(VERIFIERS_MANIFEST, reward_fn=len).verify("abc").reward == 3.0


class TestResolveCallable:
    def test_imports_a_module_attribute(self) -> None:
        assert resolve_callable("math:sqrt", context="test") is not None

    def test_rejects_a_reference_without_a_colon(self) -> None:
        with pytest.raises(ManifestError, match=re.escape("package.module:attribute")):
            resolve_callable("math.sqrt", context="test")

    def test_reports_a_missing_module(self) -> None:
        with pytest.raises(ManifestError, match="cannot import module"):
            resolve_callable("no_such_module_xyz:fn", context="test")

    def test_reports_a_missing_attribute(self) -> None:
        with pytest.raises(ManifestError, match="has no attribute"):
            resolve_callable("math:not_a_real_function", context="test")

    def test_a_manifest_reward_reference_is_imported_lazily(self) -> None:
        env = load_env({**VERIFIERS_MANIFEST, "reward": "no_such_module_xyz:fn"})
        # Construction succeeds; the import only happens when the reward is needed.
        assert env.spec().id == "prime/gsm8k-0"
        with pytest.raises(ManifestError, match="cannot import module"):
            env.verify("x")
