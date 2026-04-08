"""
Inference script for IncidentResponseEnv — OpenEnv Round 1 submission.

An LLM agent (via OpenAI-compatible client) diagnoses a simulated system
outage by observing metrics, logs, and alerts, then selecting the correct
remediation action.
"""

import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env.incident_env import IncidentResponseEnv
from graders.correctness_grader import grade_episode

# ---------------------------------------------------------------------------
# Environment configuration (mandatory)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # unused; kept for spec compliance

TASK_NAME = "incident-response"
BENCHMARK = "incident-response-env"
DIFFICULTY = os.getenv("DIFFICULTY", "medium")
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------
ACTIONS = {
    0: "check_logs",
    1: "restart_service",
    2: "restart_database",
    3: "scale_resources",
    4: "ignore",
    5: "resolve",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an on-call Site Reliability Engineer responding to a system incident.
    At each step you receive:
      - metrics: [cpu_usage (0-100%), memory_usage (0-100%), latency (0-2000ms)]
      - logs: recent log snippets (partial — not all logs are visible)
      - alerts: active alert tags
      - step: current step number

    You must choose exactly one action by responding with its number only. No explanation.

    Available actions:
      0 — check_logs        (gather more information; always do this first)
      1 — restart_service   (fixes service_crash root cause)
      2 — restart_database  (fixes database_failure root cause)
      3 — scale_resources   (fixes high_load root cause)
      4 — ignore            (do nothing; penalized)
      5 — resolve           (close the incident; only call after applying the correct fix)

    Strategy:
      1. Start with check_logs (action 0).
      2. Read the alerts and metrics to identify the root cause.
      3. Apply the matching fix (1, 2, or 3).
      4. Call resolve (action 5).

    Respond with a single digit: 0, 1, 2, 3, 4, or 5.
""").strip()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------
def build_user_prompt(step: int, obs: dict, info: dict) -> str:
    metrics = obs["metrics"]
    return textwrap.dedent(f"""
        Step: {step}
        Metrics: cpu={metrics[0]:.1f}% memory={metrics[1]:.1f}% latency={metrics[2]:.0f}ms
        Logs: {info['logs']}
        Alerts: {info['alerts']}

        Choose your action (0-5):
    """).strip()


def get_action(client: OpenAI, step: int, obs: dict, info: dict) -> tuple[int, str]:
    user_prompt = build_user_prompt(step, obs, info)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        action_idx = int(text[0])
        if action_idx not in ACTIONS:
            raise ValueError(f"out of range: {action_idx}")
        return action_idx, ACTIONS[action_idx]
    except Exception as exc:
        # Fallback: check_logs first, then resolve
        fallback = 0 if step == 1 else 5
        return fallback, ACTIONS[fallback]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = IncidentResponseEnv(config={"difficulty": DIFFICULTY})

    actions_taken: List[int] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset(seed=42)

        for step in range(1, MAX_STEPS + 1):
            action_idx, action_name = get_action(client, step, obs, info)

            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            actions_taken.append(action_idx)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_name, reward=reward, done=done, error=None)

            if done:
                break

        # Score via correctness grader — already in [0, 1]
        grade = grade_episode(actions_taken, env.root_cause)
        score = grade["score"]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
