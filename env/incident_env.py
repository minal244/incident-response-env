import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class IncidentResponseEnv(gym.Env):
    """
    Incident Response RL Environment

    Agent acts as an on-call engineer diagnosing system failures.
    """

    def __init__(self, config=None):
        super(IncidentResponseEnv, self).__init__()

        # Config
        self.config = config or {"difficulty": "medium"}
        self.difficulty = self.config.get("difficulty", "medium")

        self.max_steps = 20

        # Actions
        self.actions = {
            0: "check_logs",
            1: "restart_service",
            2: "restart_database",
            3: "scale_resources",
            4: "ignore",
            5: "resolve"
        }

        self.action_space = spaces.Discrete(len(self.actions))

        # Observation Space
        self.observation_space = spaces.Dict({
            "metrics": spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([100.0, 100.0, 2000.0], dtype=np.float32),
                dtype=np.float32
            ),
            "step_count": spaces.Discrete(self.max_steps + 1)
        })

        # Environment state
        self.current_step = 0
        self.root_cause = None

        # Internal flags (for resolution logic)
        self.logs_checked = False
        self.correct_action_taken = False

    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.current_step = 0

        self.root_cause = random.choice([
            "database_failure",
            "service_crash",
            "high_load"
        ])

        self.logs_checked = False
        self.correct_action_taken = False

        observation = self._get_observation()

        return observation, self._get_info()

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self.current_step += 1

        reward = 0
        terminated = False
        truncated = False

        action_name = self.actions[action]

        # -------- LOGIC --------

        if action_name == "check_logs":
            self.logs_checked = True
            reward += 2

        elif action_name == "restart_database":
            if self.root_cause == "database_failure":
                self.correct_action_taken = True
                reward += 3
            else:
                reward -= 3

        elif action_name == "restart_service":
            if self.root_cause == "service_crash":
                self.correct_action_taken = True
                reward += 3
            else:
                reward -= 3

        elif action_name == "scale_resources":
            if self.root_cause == "high_load":
                self.correct_action_taken = True
                reward += 3
            else:
                reward -= 3

        elif action_name == "ignore":
            reward -= 2

        elif action_name == "resolve":
            if self._is_resolved():
                reward += 10
            else:
                reward -= 10
            terminated = True

        # Step penalty (efficiency)
        reward -= 1

        # Max step truncation
        if self.current_step >= self.max_steps:
            truncated = True

        observation = self._get_observation()

        return observation, reward, terminated, truncated, self._get_info()

    # =========================
    # RESOLUTION LOGIC
    # =========================
    def _is_resolved(self):
        """
        Proper resolution requires:
        1. Logs checked (information gathering)
        2. Correct action taken
        """
        return self.logs_checked and self.correct_action_taken

    # =========================
    # OBSERVATION
    # =========================
    def _get_observation(self):
        metrics = np.array([
            self._cpu_usage(),
            self._memory_usage(),
            self._latency()
        ], dtype=np.float32)

        return {
            "metrics": metrics,
            "step_count": self.current_step
        }

    # =========================
    # INFO (IMPORTANT FOR DEBUG)
    # =========================
    def _get_info(self):
        return {
            "logs": self._generate_logs(),
            "alerts": self._generate_alerts(),
            "root_cause": self.root_cause  # hidden in real training, useful for debug
        }

    # =========================
    # METRICS GENERATION
    # =========================
    def _cpu_usage(self):
        if self.root_cause == "high_load":
            return random.randint(85, 100)
        return random.randint(40, 70)

    def _memory_usage(self):
        if self.root_cause == "service_crash":
            return random.randint(70, 95)
        return random.randint(30, 60)

    def _latency(self):
        if self.root_cause == "database_failure":
            return random.randint(1000, 2000)
        return random.randint(100, 500)

    # =========================
    # LOG GENERATION (PARTIAL OBSERVABILITY)
    # =========================
    def _generate_logs(self):
        base_logs = {
            "database_failure": [
                "DB connection timeout",
                "Retrying connection...",
                "Database unreachable"
            ],
            "service_crash": [
                "Service A crashed",
                "Segmentation fault detected",
                "Restart required"
            ],
            "high_load": [
                "CPU usage above 95%",
                "Request queue increasing",
                "Latency spike detected"
            ]
        }

        logs = base_logs[self.root_cause].copy()

        # Difficulty-based noise
        if self.difficulty == "medium":
            logs.append("Minor warning: cache miss")

        elif self.difficulty == "hard":
            logs.extend([
                "Cache layer timeout",
                "Background job delayed"
            ])
            random.shuffle(logs)

        return logs[:2]  # Partial observability (only 2 logs shown)

    # =========================
    # ALERT GENERATION
    # =========================
    def _generate_alerts(self):
        alerts = {
            "database_failure": ["database_down"],
            "service_crash": ["service_unavailable"],
            "high_load": ["high_cpu_usage"]
        }
        return alerts[self.root_cause]

    # =========================
    # RENDER (BONUS FEATURE)
    # =========================
    def render(self):
        print("\n--- INCIDENT STATE ---")
        print("Step:", self.current_step)
        print("Metrics:", self._get_observation()["metrics"])
        print("Logs:", self._generate_logs())
        print("Alerts:", self._generate_alerts())
        print("----------------------")