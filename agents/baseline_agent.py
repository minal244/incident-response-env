"""
Rule-Based Baseline Agent
--------------------------
A deterministic agent that solves IncidentResponseEnv optimally.

Strategy (3 steps):
  1. check_logs        — always gather information first
  2. apply correct fix — infer root cause from alerts/metrics
  3. resolve           — call resolve once the fix is applied

This agent demonstrates the environment is solvable and provides an
upper-bound reference for RL agent evaluation.
"""

from env.incident_env import IncidentResponseEnv
from graders import correctness_grader, efficiency_grader

# Action indices
CHECK_LOGS = 0
RESTART_SERVICE = 1
RESTART_DATABASE = 2
SCALE_RESOURCES = 3
IGNORE = 4
RESOLVE = 5


class RuleBasedAgent:
    """
    Deterministic agent that resolves incidents in exactly 3 steps
    by reading the alert signal from the info dict.
    """

    def __init__(self):
        self._step = 0
        self._fix_action = None

    def reset(self):
        self._step = 0
        self._fix_action = None

    def act(self, obs: dict, info: dict) -> int:
        """
        Choose an action given the current observation and info.

        Step 0: check_logs
        Step 1: apply fix based on alert
        Step 2+: resolve
        """
        if self._step == 0:
            self._step += 1
            return CHECK_LOGS

        if self._step == 1:
            self._step += 1
            self._fix_action = self._infer_fix(info)
            return self._fix_action

        return RESOLVE

    def _infer_fix(self, info: dict) -> int:
        """Map alert signal to the correct remediation action."""
        alerts = info.get("alerts", [])
        if "database_down" in alerts:
            return RESTART_DATABASE
        if "service_unavailable" in alerts:
            return RESTART_SERVICE
        if "high_cpu_usage" in alerts:
            return SCALE_RESOURCES
        # Fallback: infer from metrics (high latency → DB, high memory → service)
        metrics = info.get("metrics", [50, 50, 300])
        cpu, memory, latency = metrics[0], metrics[1], metrics[2]
        if latency > 800:
            return RESTART_DATABASE
        if memory > 65:
            return RESTART_SERVICE
        return SCALE_RESOURCES


def run_baseline(difficulty: str = "medium", seed: int = 42) -> dict:
    """
    Run one episode with the rule-based agent and return graded results.
    """
    env = IncidentResponseEnv(config={"difficulty": difficulty})
    agent = RuleBasedAgent()

    obs, info = env.reset(seed=seed)
    agent.reset()

    actions_taken = []
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs, info)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    root_cause = env.root_cause
    correctness = correctness_grader.grade_episode(actions_taken, root_cause)
    efficiency = efficiency_grader.grade_efficiency(
        actions_taken, root_cause, max_steps=env.max_steps
    )

    return {
        "difficulty": difficulty,
        "root_cause": root_cause,
        "actions_taken": actions_taken,
        "total_reward": total_reward,
        "correctness": correctness,
        "efficiency": efficiency,
    }


if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        result = run_baseline(difficulty=difficulty)
        print(f"\n--- {difficulty.upper()} ---")
        print(f"Root cause  : {result['root_cause']}")
        print(f"Total reward: {result['total_reward']}")
        print(f"Correctness : {result['correctness']['score']} — {result['correctness']['feedback']}")
        print(f"Efficiency  : {result['efficiency']['score']} — {result['efficiency']['feedback']}")
