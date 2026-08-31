"""Tests for the VerityEnv protocol and its type definitions."""

from __future__ import annotations

from typing import Any, get_args

import pytest

from verity_core.env import (
    DOMAINS,
    REWARD_TYPES,
    Domain,
    Observation,
    RewardResult,
    RewardType,
    StepResult,
    TaskSpec,
    VerityEnv,
)


def make_spec(**overrides: Any) -> TaskSpec:
    fields: dict[str, Any] = {
        "id": "corpus/task-1",
        "domain": "code",
        "source": "https://github.com/example/tasks",
        "commit": "0f1e2d3",
        "reward_type": "binary",
        "instructions": "Fix the failing test.",
        "has_gold": True,
    }
    fields.update(overrides)
    return TaskSpec(**fields)


class ConformingEnv:
    """A minimal structural implementation, standing in for a real adapter."""

    def spec(self) -> TaskSpec:
        return make_spec()

    def reset(self) -> Observation:
        return Observation(text="start")

    def step(self, action: str) -> StepResult:
        return StepResult(Observation(text=action), 0.0, False)

    def verify(self, submission: str) -> RewardResult:
        return RewardResult(1.0, True)

    def gold_solution(self) -> str | None:
        return None

    def snapshot(self) -> bytes:
        return b"{}"

    def restore(self, snap: bytes) -> None:
        return None


class PartialEnv:
    def spec(self) -> TaskSpec:
        return make_spec()

    def reset(self) -> Observation:
        return Observation(text="start")


class TestDomainConstants:
    def test_constants_match_the_literal_types(self) -> None:
        assert get_args(Domain) == DOMAINS
        assert get_args(RewardType) == REWARD_TYPES

    def test_expected_domains_are_present(self) -> None:
        assert set(DOMAINS) == {"browser", "gui", "tool_use", "code", "math", "other"}
        assert set(REWARD_TYPES) == {"binary", "partial"}


class TestTaskSpec:
    def test_round_trips_through_a_dict(self) -> None:
        spec = make_spec()
        assert TaskSpec.from_dict(spec.to_dict()) == spec

    def test_from_dict_defaults_optional_fields(self) -> None:
        spec = TaskSpec.from_dict({"id": "t", "domain": "math"})
        assert (spec.source, spec.commit, spec.instructions) == ("", "", "")
        assert spec.reward_type == "binary"
        assert spec.has_gold is False

    def test_rejects_an_unknown_domain(self) -> None:
        with pytest.raises(ValueError, match="unknown domain"):
            make_spec(domain="quantum")

    def test_rejects_an_unknown_reward_type(self) -> None:
        with pytest.raises(ValueError, match="unknown reward_type"):
            make_spec(reward_type="graded")

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_accepts_every_declared_domain(self, domain: str) -> None:
        assert make_spec(domain=domain).domain == domain


class TestObservation:
    def test_metadata_defaults_to_an_independent_dict(self) -> None:
        first, second = Observation(text="a"), Observation(text="b")
        first.metadata["k"] = 1
        assert second.metadata == {}

    def test_serializes_metadata_as_a_copy(self) -> None:
        obs = Observation(text="a", metadata={"k": 1})
        payload = obs.to_dict()
        payload["metadata"]["k"] = 2
        assert obs.metadata["k"] == 1


class TestStepResult:
    def test_unpacks_as_a_gym_style_tuple(self) -> None:
        result = StepResult(Observation(text="obs"), 0.5, True, {"note": "n"})
        observation, reward, done, info = result
        assert observation.text == "obs"
        assert reward == 0.5
        assert done is True
        assert info == {"note": "n"}

    def test_serializes_nested_observation(self) -> None:
        payload = StepResult(Observation(text="obs"), 1.0, False).to_dict()
        assert payload["observation"] == {"text": "obs", "metadata": {}}
        assert payload["done"] is False


class TestRewardResult:
    def test_unpacks_in_declared_order(self) -> None:
        reward, verdict, logs, state = RewardResult(0.6, False, "3/5 passed", {"n": 5})
        assert (reward, verdict, logs) == (0.6, False, "3/5 passed")
        assert state == {"n": 5}

    def test_partial_reward_can_disagree_with_the_verdict(self) -> None:
        result = RewardResult(reward=0.6, verdict=False)
        assert result.reward > 0
        assert result.verdict is False

    def test_serializes_all_fields(self) -> None:
        payload = RewardResult(1.0, True, "ok", {"k": "v"}).to_dict()
        assert payload == {
            "reward": 1.0,
            "verdict": True,
            "verifier_logs": "ok",
            "verifier_state": {"k": "v"},
        }


class TestProtocolConformance:
    def test_a_full_implementation_conforms(self) -> None:
        assert isinstance(ConformingEnv(), VerityEnv)

    def test_a_partial_implementation_does_not_conform(self) -> None:
        assert not isinstance(PartialEnv(), VerityEnv)

    def test_conforming_env_is_usable_through_the_protocol(self) -> None:
        env: VerityEnv = ConformingEnv()
        assert env.spec().id == "corpus/task-1"
        assert env.reset().text == "start"
        assert env.step("ls").observation.text == "ls"
        assert env.verify("answer").verdict is True
        assert env.gold_solution() is None
        assert env.restore(env.snapshot()) is None
