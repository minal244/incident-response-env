"""
OpenEnv HTTP server for IncidentResponseEnv.

Exposes the Gymnasium environment as a REST API so the OpenEnv platform
can ping the Space and call reset()/step()/state() over HTTP.

Endpoints:
    GET  /              health check
    GET  /health        health check
    POST /reset         start a new episode
    POST /step          take one action
    GET  /state         current observation + info
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.incident_env import IncidentResponseEnv

app = FastAPI(title="IncidentResponseEnv", version="1.0.0")

# ---------------------------------------------------------------------------
# Global env state (single-session; sufficient for OpenEnv evaluation)
# ---------------------------------------------------------------------------
_env: Optional[IncidentResponseEnv] = None
_current_obs: Optional[Dict] = None
_current_info: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    difficulty: Optional[str] = "medium"


class StepRequest(BaseModel):
    action: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _serialize_obs(obs: Dict) -> Dict:
    """Convert numpy arrays to plain Python types for JSON serialisation."""
    return {
        "metrics": obs["metrics"].tolist(),
        "step_count": int(obs["step_count"]),
    }


def _serialize_info(info: Dict) -> Dict:
    return {
        "logs": info.get("logs", []),
        "alerts": info.get("alerts", []),
        # root_cause intentionally omitted (hidden from evaluator)
    }


@app.get("/")
def root():
    return HTMLResponse("""
    <html>
    <head><title>IncidentResponseEnv</title></head>
    <body>
    <h1>Incident Response Environment</h1>
    <p>This is an OpenEnv-compatible API for incident response simulation.</p>
    <p>Endpoints:</p>
    <ul>
    <li>GET /health - Health check</li>
    <li>POST /reset - Reset environment</li>
    <li>POST /step - Take action</li>
    <li>GET /state - Get current state</li>
    </ul>
    </body>
    </html>
    """)

@app.get("/health")
def health():
    return {"status": "ok", "env": "incident-response-env"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env, _current_obs, _current_info

    _env = IncidentResponseEnv(config={"difficulty": req.difficulty})
    obs, info = _env.reset(seed=req.seed)

    _current_obs = obs
    _current_info = info

    return {
        "observation": _serialize_obs(obs),
        "info": _serialize_info(info),
    }


@app.post("/step")
def step(req: StepRequest):
    global _current_obs, _current_info

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")

    if req.action not in range(_env.action_space.n):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action {req.action}. Must be 0–{_env.action_space.n - 1}.",
        )

    obs, reward, terminated, truncated, info = _env.step(req.action)
    _current_obs = obs
    _current_info = info

    return {
        "observation": _serialize_obs(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": _serialize_info(info),
    }


@app.get("/state")
def state():
    if _current_obs is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {
        "observation": _serialize_obs(_current_obs),
        "info": _serialize_info(_current_info),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
