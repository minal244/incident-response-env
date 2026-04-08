# IncidentResponseEnv

A Gymnasium-compatible reinforcement learning environment in which an agent acts as an on-call engineer diagnosing and resolving simulated system outages.

---

## Motivation

Incident response is a high-stakes, multi-step reasoning task: an engineer must gather evidence, correlate signals, identify the root cause, and apply the correct remediation — all under time pressure. Random or greedy policies fail because the correct action depends on hidden state (the root cause) that must be inferred from partial observations. This makes it a strong testbed for agents that need to reason under uncertainty, perform sequential diagnosis, and avoid costly irreversible actions.

This environment is inspired by real-world Site Reliability Engineering (SRE) workflows and is related to benchmarks such as [WebArena](https://webarena.dev/) and [AgentBench](https://github.com/THUDM/AgentBench) that evaluate agent decision-making in interactive systems.

---

## Task Description

The agent is presented with a live system alert. Somewhere in the infrastructure, one of three root causes has triggered:

| Root Cause         | Signal                          | Correct Fix        |
|--------------------|---------------------------------|--------------------|
| `database_failure` | High latency (1000–2000 ms)     | `restart_database` |
| `service_crash`    | High memory usage (70–95%)      | `restart_service`  |
| `high_load`        | High CPU usage (85–100%)        | `scale_resources`  |

The agent must:
1. Inspect logs to gather information
2. Identify and apply the correct remediation
3. Call `resolve` to close the incident

The root cause is **not** directly observable — the agent must infer it from noisy metrics and partial log snippets.

---

## Observation Space

```python
spaces.Dict({
    "metrics": spaces.Box(
        low=[0.0, 0.0, 0.0],
        high=[100.0, 100.0, 2000.0],
        dtype=np.float32
    ),  # [cpu_usage (%), memory_usage (%), latency (ms)]
    "step_count": spaces.Discrete(21)  # current step in episode
})
```

Additionally, `info` returned from `step()` and `reset()` contains:
- `logs`: list of 2 log strings (partial observability — only 2 of up to 5 are shown)
- `alerts`: list of alert tags (e.g. `["database_down"]`)
- `root_cause`: ground-truth root cause (available for debugging; hide during training)

---

## Action Space

```python
spaces.Discrete(6)
```

| Index | Action             | Effect                                                  |
|-------|--------------------|---------------------------------------------------------|
| 0     | `check_logs`       | Sets `logs_checked = True`; reward +2                  |
| 1     | `restart_service`  | Correct fix for `service_crash`; else penalty           |
| 2     | `restart_database` | Correct fix for `database_failure`; else penalty        |
| 3     | `scale_resources`  | Correct fix for `high_load`; else penalty               |
| 4     | `ignore`           | No-op; reward -2                                        |
| 5     | `resolve`          | Ends episode; +10 if resolved correctly, -10 otherwise  |

---

## Reward Function

The reward is shaped across multiple components to guide learning:

$$r_t = r_{\text{action}} + r_{\text{step}}$$

Where:

- **Action reward** $r_{\text{action}}$:
  - `check_logs`: $+2$
  - Correct fix action: $+3$
  - Wrong fix action: $-3$
  - `ignore`: $-2$
  - `resolve` (when `logs_checked` and `correct_action_taken`): $+10$
  - `resolve` (premature): $-10$, episode terminates

- **Step penalty** $r_{\text{step}} = -1$ per step (efficiency incentive)

**Optimal episode reward:** $r = 2 + 3 + 10 - 3 = +12$ (3 steps: check_logs → fix → resolve)

**Random policy expected reward:** Negative (wrong fixes penalized at -3, step penalty accumulates)

---

## Difficulty Levels

Configured via `config={"difficulty": "easy"|"medium"|"hard"}`.

| Level    | Effect                                                                 |
|----------|------------------------------------------------------------------------|
| `easy`   | Clean logs, clear signals                                              |
| `medium` | One noise log entry appended (`"Minor warning: cache miss"`)           |
| `hard`   | Two noise entries added, logs shuffled — partial observability hardest |

---

## Example Episode

```python
from env.incident_env import IncidentResponseEnv

env = IncidentResponseEnv(config={"difficulty": "medium"})
obs, info = env.reset(seed=0)

# info["root_cause"] == "database_failure" (hidden during training)
# info["alerts"]     == ["database_down"]
# obs["metrics"]     == [55.0, 40.0, 1342.0]  (high latency signal)

# Step 1: check_logs → reward +2 -1 = +1
obs, r, term, trunc, info = env.step(0)

# Step 2: restart_database → reward +3 -1 = +2
obs, r, term, trunc, info = env.step(2)

# Step 3: resolve → reward +10 -1 = +9, episode ends
obs, r, term, trunc, info = env.step(5)
# Total reward: +12
```

---

## Graders

Two graders are provided in `graders/`:

**1. Correctness Grader** (`graders/correctness_grader.py`)

Programmatic grader. Evaluates:
- Was `check_logs` called before acting? (weight 0.25)
- Was the correct fix applied? (weight 0.45)
- Was `resolve` called after the fix? (weight 0.30)

Score: 0.0 – 1.0. Pass threshold: ≥ 0.5.

**2. Efficiency Grader** (`graders/efficiency_grader.py`)

Evaluates step efficiency relative to the theoretical optimum (3 steps):

$$\text{efficiency} = 1 - \frac{\text{steps\_taken} - 3}{\text{max\_steps} - 3}$$

$$\text{score} = 0.6 \cdot \mathbb{1}[\text{resolved}] + 0.4 \cdot \text{efficiency}$$

---

## Baseline Agent

A deterministic rule-based agent is provided in `agents/baseline_agent.py`.

**Strategy:** check_logs → infer fix from alert → resolve (always 3 steps, near-perfect score)

```bash
python agents/baseline_agent.py
```

Expected output:
```
--- EASY ---
Root cause  : database_failure
Total reward: 12
Correctness : 1.0 — Incident resolved correctly.
Efficiency  : 1.0 — Optimal resolution in 3 steps.
```

The baseline scores 1.0 on both graders. A random policy typically scores 0.0–0.2 on correctness and terminates early with negative total reward.

---

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
incident-response-env/
├── env/
│   ├── __init__.py
│   └── incident_env.py       # Core Gymnasium environment
├── graders/
│   ├── __init__.py
│   ├── correctness_grader.py # Programmatic correctness grader
│   └── efficiency_grader.py  # Step-efficiency grader
├── agents/
│   ├── __init__.py
│   └── baseline_agent.py     # Rule-based optimal baseline
├── tests/
│   └── test_env.py
└── requirements.txt
```
