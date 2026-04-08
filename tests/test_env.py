import pytest
import numpy as np
from env.incident_env import IncidentResponseEnv
from graders.correctness_grader import grade_episode, run_grader as run_correctness
from graders.efficiency_grader import grade_efficiency, run_grader as run_efficiency


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------

def test_reset_returns_valid_obs():
    env = IncidentResponseEnv()
    obs, info = env.reset(seed=0)
    assert "metrics" in obs
    assert "step_count" in obs
    assert obs["metrics"].shape == (3,)
    assert obs["step_count"] == 0


def test_step_returns_five_tuple():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    result = env.step(0)  # check_logs
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_observation_within_bounds():
    env = IncidentResponseEnv()
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        metrics = obs["metrics"]
        assert metrics[0] <= 100.0, "CPU must be <= 100"
        assert metrics[1] <= 100.0, "Memory must be <= 100"
        assert metrics[2] <= 2000.0, "Latency must be <= 2000"
        assert all(metrics >= 0.0)


def test_truncation_at_max_steps():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    truncated = False
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(4)  # ignore
        if terminated:
            break
    assert truncated


def test_resolve_terminates_episode():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    env.step(0)  # check_logs
    # apply correct fix
    fix_map = {"database_failure": 2, "service_crash": 1, "high_load": 3}
    env.step(fix_map[env.root_cause])
    _, _, terminated, _, _ = env.step(5)  # resolve
    assert terminated


def test_correct_resolution_gives_positive_reward():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    fix_map = {"database_failure": 2, "service_crash": 1, "high_load": 3}
    env.step(0)  # check_logs: +2 -1 = +1
    env.step(fix_map[env.root_cause])  # correct fix: +3 -1 = +2
    _, reward, _, _, _ = env.step(5)  # resolve: +10 -1 = +9
    assert reward == 9


def test_premature_resolve_gives_negative_reward():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(5)  # resolve immediately
    assert reward < 0
    assert terminated


def test_wrong_fix_gives_negative_reward():
    env = IncidentResponseEnv()
    env.reset(seed=0)
    wrong_fix = {"database_failure": 1, "service_crash": 2, "high_load": 1}
    _, reward, _, _, _ = env.step(wrong_fix[env.root_cause])
    assert reward < 0


def test_seed_reproducibility():
    env = IncidentResponseEnv()
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert info1["root_cause"] == info2["root_cause"]
    np.testing.assert_array_equal(obs1["metrics"], obs2["metrics"])


def test_difficulty_levels():
    for difficulty in ["easy", "medium", "hard"]:
        env = IncidentResponseEnv(config={"difficulty": difficulty})
        obs, info = env.reset(seed=0)
        assert len(info["logs"]) == 2


# ---------------------------------------------------------------------------
# Correctness Grader
# ---------------------------------------------------------------------------

def test_correctness_grader_perfect_score():
    # check_logs → correct fix → resolve
    fix_map = {"database_failure": 2, "service_crash": 1, "high_load": 3}
    for root_cause, fix_action in fix_map.items():
        result = grade_episode([0, fix_action, 5], root_cause)
        assert result["score"] == 1.0
        assert result["passed"]


def test_correctness_grader_no_logs():
    result = grade_episode([2, 5], "database_failure")
    assert result["score"] < 1.0
    assert "log" in result["feedback"].lower()


def test_correctness_grader_wrong_fix():
    result = grade_episode([0, 1, 5], "database_failure")  # wrong fix
    assert result["score"] < 0.5
    assert not result["passed"]


# ---------------------------------------------------------------------------
# Efficiency Grader
# ---------------------------------------------------------------------------

def test_efficiency_grader_optimal():
    result = grade_efficiency([0, 2, 5], "database_failure", max_steps=20)
    assert result["efficiency"] == 1.0
    assert result["resolved"]


def test_efficiency_grader_extra_steps():
    # 7 steps instead of 3
    actions = [0, 4, 4, 4, 2, 4, 5]
    result = grade_efficiency(actions, "database_failure", max_steps=20)
    assert result["efficiency"] < 1.0
    assert result["steps_taken"] == 7
