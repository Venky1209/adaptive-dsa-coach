---
title: Adaptive DSA Coach
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
short_description: Adaptive DSA tutoring environment for the OpenEnv hackathon.
---

# adaptive-dsa-coach

Adaptive DSA Coach is an OpenEnv-compatible tutoring environment for the OpenEnv x Scaler x Meta PyTorch hackathon. It simulates a student progressing through DSA practice while handling burnout, distraction, motivation, and career-context pressure.

## Overview

The environment models a coaching loop around six tracked topics:

- `arrays`
- `strings`
- `dp`
- `graphs`
- `trees`
- `system_design`

Agents act through typed actions and receive dense rewards for useful coaching behavior. The environment is deterministic, JSON-serializable, and designed to work with OpenEnv-style HTTP deployment.

## Tasks

Three tasks are available:

- `EASY` - shorter episode, lower stress, reward shaping around basic coaching support
- `MEDIUM` - placement-season pressure with burnout handling and topic targeting
- `HARD` - longer episode with stronger emphasis on recovery, session management, and sustained progress

Each task has fixed initial state values, max steps, and a success threshold.

## Action Space

Supported action types:

- `build_plan`
- `recommend_exercise`
- `evaluate_solution`
- `give_hint`
- `handle_distraction`
- `handle_burnout`
- `give_motivation`
- `advance_session`
- `end_day`

Action parameters are sent as a `dict` and are interpreted deterministically by the environment.

## Observation Space

All observations include the following required fields:

- `topic_mastery`
- `motivation`
- `burnout_risk`
- `streak`
- `daily_time_left`
- `current_topic`
- `current_problem_id`
- `time_on_problem`
- `recent_accuracy`
- `recent_speed`
- `event_flags`
- `career_context`

## Reward Design

The environment uses shaped rewards to encourage useful tutoring behavior:

- Weak-topic targeting is rewarded for plan and exercise actions.
- High-quality feedback and optimization hints are rewarded during solution review.
- Distraction handling is rewarded when the redirect is productive.
- Burnout handling is rewarded when the agent reduces difficulty or changes topic appropriately.
- Motivation support is rewarded more strongly when motivation is already low.
- `end_day` can grant a large bonus when the episode finishes sustainably.

Rewards are always clamped to the range `-1.0` to `1.0`.

## Determinism

The environment is deterministic:

- `random.seed(42)` is used for event injection and reproducibility.
- Graders are deterministic for the same input.
- State is JSON serializable.

## Files

The project is organized as:

```text
adaptive-dsa-coach/
├── env/
│   ├── __init__.py
│   ├── models.py
│   ├── environment.py
│   ├── tasks.py
│   └── graders.py
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server locally on port `7860`:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Reset the environment by posting a task name to `/reset`, then step through actions with `/step`, and inspect the current state with `/state`.

Available server routes:

- `GET /`
- `GET /health`
- `GET /tasks`
- `GET /state`
- `POST /state`
- `POST /reset`
- `POST /step`

## Inference

The inference entrypoint lives at the repo root:

```bash
python inference.py EASY
python inference.py MEDIUM
python inference.py HARD
```

Environment variables supported by the inference script:

- `API_BASE_URL` - defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME` - defaults to `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` - Hugging Face token used for OpenAI-compatible requests

The script prints the exact required transcript format:

```text
[START] task=X env=adaptive_dsa_coach model=Y
[STEP] step=N action=X reward=0.00 done=false error=null
[END] success=true steps=N score=0.00 rewards=r1,r2,...
```

## Docker

Build and run the container:

```bash
docker build -t adaptive-dsa-coach .
docker run --rm -p 7860:7860 adaptive-dsa-coach
```

The container listens on `7860` and exposes a health check at `/health`.

## OpenEnv Manifest

The OpenEnv manifest is configured for Hugging Face Spaces with:

- `spec_version: 1`
- `type: space`
- `runtime: fastapi`
- `app: app:app`
- `port: 7860`

## Notes

- The project is designed for deterministic grading and reproducible outputs.
- The environment state and all payloads are JSON serializable.
- The current implementation keeps the coaching loop practical and focused on DSA tutoring, burnout handling, and placement/hackathon support.