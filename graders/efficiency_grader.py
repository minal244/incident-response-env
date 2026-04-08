"""
Efficiency Grader
-----------------
Evaluates how efficiently an agent resolves an incident.
An agent that resolves the incident in fewer steps scores higher.

Scoring formula:
    efficiency = 1 - (steps_taken - min_steps) / (max_steps - min_steps)
    final_score = correctness_weight * resolved + efficiency_weight * efficiency

Where:
    min_steps = 3  (check_logs → correct_fix → resolve)
    max_steps = env.max_steps

Score range: 0.0 – 1.0
"""

from env.incident_env import IncidentResponseEnv
from graders.correctness_grader import grade_episode

MIN_STEPS = 3  # theoretical optimum: check_logs + fix + resolve


def grade_efficiency(
    actions_taken: list[int],
    root_cause: str,
    max_steps: int = 20,
) -> dict:
    """
    Grade efficiency of an episode.

    Parameters
    ----------
    actions_taken : list[int]
    root_cause : str
    max_steps : int

    Returns
    -------
    dict with keys:
        score        : float in [0.0, 1.0]
        steps_taken  : int
        efficiency   : float — how close to optimal step count
        resolved     : bool — did the agent actually resolve correctly?
        feedback     : str
    """
    correctness = grade_episode(actions_taken, root_cause)
    resolved = correctness["passed"]

    steps_taken = len(actions_taken)
    efficiency = 1.0 - (steps_taken - MIN_STEPS) / max(max_steps - MIN_STEPS, 1)
    efficiency = max(0.0, min(1.0, efficiency))

    # Only reward efficiency if the incident was actually resolved
    score = 0.6 * float(resolved) + 0.4 * efficiency if resolved else efficiency * 0.2

    if resolved:
        if steps_taken == MIN_STEPS:
            feedback = f"Optimal resolution in {steps_taken} steps."
        else:
            feedback = f"Resolved in {steps_taken} steps ({steps_taken - MIN_STEPS} extra vs. optimal)."
    else:
        feedback = f"Incident not resolved after {steps_taken} steps."

    return {
        "score": round(score, 3),
        "steps_taken": steps_taken,
        "efficiency": round(efficiency, 3),
        "resolved": resolved,
        "feedback": feedback,
    }


def run_grader(env: IncidentResponseEnv, policy_fn, seed: int = 42) -> dict:
    """
    Run a full episode with the given policy and grade efficiency.

    Parameters
    ----------
    env : IncidentResponseEnv
    policy_fn : callable(obs, info) -> int
    seed : int

    Returns
    -------
    Grade dict from grade_efficiency().
    """
    obs, info = env.reset(seed=seed)
    root_cause = info["root_cause"]
    actions_taken = []
    done = False

    while not done:
        action = policy_fn(obs, info)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return grade_efficiency(actions_taken, root_cause, max_steps=env.max_steps)
