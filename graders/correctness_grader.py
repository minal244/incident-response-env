"""
Correctness Grader
------------------
Programmatic grader that evaluates whether an agent correctly resolved
the incident. Scores an episode rollout on three criteria:

  1. Information gathering  — did the agent check logs before acting?
  2. Correct remediation    — did the agent apply the right fix action?
  3. Clean resolution       — did the agent call `resolve` after fixing?

Score range: 0.0 – 1.0
"""

from env.incident_env import IncidentResponseEnv


def grade_episode(actions_taken: list[int], root_cause: str) -> dict:
    """
    Grade a single episode given the sequence of action indices taken
    and the true root cause.

    Parameters
    ----------
    actions_taken : list[int]
        Ordered list of action indices from the episode.
    root_cause : str
        The ground-truth root cause ("database_failure", "service_crash",
        or "high_load").

    Returns
    -------
    dict with keys:
        score        : float in [0.0, 1.0]
        breakdown    : dict of sub-scores
        passed       : bool (score >= 0.5)
        feedback     : str
    """
    action_map = {
        0: "check_logs",
        1: "restart_service",
        2: "restart_database",
        3: "scale_resources",
        4: "ignore",
        5: "resolve",
    }

    correct_fix = {
        "database_failure": "restart_database",
        "service_crash": "restart_service",
        "high_load": "scale_resources",
    }

    action_names = [action_map[a] for a in actions_taken]

    # --- Sub-score 1: Did agent check logs? ---
    checked_logs = "check_logs" in action_names
    logs_score = 1.0 if checked_logs else 0.0

    # --- Sub-score 2: Did agent apply the correct fix? ---
    fix = correct_fix[root_cause]
    applied_fix = fix in action_names
    fix_score = 1.0 if applied_fix else 0.0

    # --- Sub-score 3: Did agent call resolve AFTER the fix? ---
    resolved_correctly = False
    if applied_fix and "resolve" in action_names:
        fix_idx = next(i for i, a in enumerate(action_names) if a == fix)
        resolve_idx = next(i for i, a in enumerate(action_names) if a == "resolve")
        resolved_correctly = resolve_idx > fix_idx
    resolve_score = 1.0 if resolved_correctly else 0.0

    # Weighted aggregate: fix is most important
    score = 0.25 * logs_score + 0.45 * fix_score + 0.30 * resolve_score

    feedback_parts = []
    if not checked_logs:
        feedback_parts.append("Agent skipped log inspection before acting.")
    if not applied_fix:
        feedback_parts.append(f"Agent never applied the correct fix ({fix}).")
    if not resolved_correctly:
        feedback_parts.append("Agent did not call resolve after remediation.")
    feedback = " ".join(feedback_parts) if feedback_parts else "Incident resolved correctly."

    return {
        "score": round(score, 3),
        "breakdown": {
            "logs_checked": logs_score,
            "correct_fix_applied": fix_score,
            "clean_resolution": resolve_score,
        },
        "passed": score >= 0.5,
        "feedback": feedback,
    }


def run_grader(env: IncidentResponseEnv, policy_fn, seed: int = 42) -> dict:
    """
    Run a full episode with the given policy and grade the result.

    Parameters
    ----------
    env : IncidentResponseEnv
    policy_fn : callable(obs, info) -> int
        A function that maps observation + info to an action index.
    seed : int

    Returns
    -------
    Grade dict from grade_episode().
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

    return grade_episode(actions_taken, root_cause)
