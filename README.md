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

Adaptive DSA Coach is a deterministic OpenEnv tutoring environment built for the OpenEnv x Scaler x Meta hackathon. It simulates a student working through DSA practice while dealing with distraction, burnout, motivation swings, and placement/hackathon pressure.

## At a Glance

| Item | Value |
| --- | --- |
| Environment name | `adaptive_dsa_coach` |
| Tasks | `EASY`, `MEDIUM`, `HARD` |
| Server | FastAPI on port `7860` |
| Deployment | Docker / Hugging Face Spaces |
| Validation | `./validate.sh <space-url>` |
| Inference | `python inference.py <TASK>` |

## What It Models

The environment tracks six DSA topics:

- `arrays`
- `strings`
- `dp`
- `graphs`
- `trees`
- `system_design`

Each episode keeps the learner state JSON-serializable and deterministic. The runtime injects recurring events such as distraction, motivation drops, and burnout signals so agents must respond instead of following a static script.

## Tasks

Three fixed tasks are available:

- `EASY` - shorter episodes, lower stress, and simpler recovery behavior
- `MEDIUM` - placement-season pressure with stronger burnout handling
- `HARD` - longer episodes with more aggressive session management and recovery

Every task has its own initial state, episode length, and success threshold.

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

Actions accept a parameter dictionary and are applied deterministically.

## Observation Space

Each observation includes:

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

Rewards are shaped to encourage coaching behaviors that actually improve the learner state:

- Targeting weak topics is rewarded.
- Good feedback and optimization hints are rewarded during solution review.
- Distraction handling is rewarded when it redirects attention productively.
- Burnout handling is rewarded when the agent lowers pressure or shifts topics appropriately.
- Motivation support is rewarded more strongly when motivation is already low.
- `end_day` can produce a large bonus if the episode closes sustainably.

All rewards are clamped to `[-1.0, 1.0]`.

## Determinism

The project is designed for reproducibility:

- `random.seed(42)` is used for event injection.
- Graders are deterministic for the same state and trajectory.
- State and API payloads are JSON serializable.

## Repository Layout

```text
adaptive-dsa-coach/
├── app.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── validate.sh
├── Dockerfile
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── graders.py
│   ├── models.py
│   └── tasks.py
└── server/
    ├── __init__.py
    └── app.py
```

The root `app.py` is the main FastAPI app. `server/app.py` re-exports the same app for OpenEnv packaging and deployment compatibility.

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then interact with the environment through the HTTP endpoints below.

## API Endpoints

- `GET /` - environment summary and active task list
- `GET /health` - basic health check
- `GET /tasks` - available task specs
- `GET /state` - current environment state
- `POST /state` - current environment state
- `POST /reset` - reset to a task
- `POST /step` - submit one action and receive the next state

## Inference

Run the inference loop with one of the supported tasks:

```bash
python inference.py EASY
python inference.py MEDIUM
python inference.py HARD
```

Supported environment variables:

- `API_BASE_URL` - defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME` - defaults to `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` - Hugging Face token used for OpenAI-compatible requests

The script prints the expected transcript format for each episode:

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

The container listens on `7860` and exposes `/health` for readiness checks.

## OpenEnv Manifest

The Hugging Face Spaces manifest is configured as:

- `spec_version: 1`
- `type: space`
- `runtime: fastapi`
- `app: server.app:app`
- `port: 7860`

## Validation

Use the bundled script to verify the repo end to end:

```bash
./validate.sh https://your-space.hf.space
```

The validator checks three things:

1. The live Hugging Face Space responds to `/reset`.
2. The Docker image builds successfully.
3. `openenv validate` passes for the local repository.

## Notes

- The environment is intentionally practical rather than overly abstract.
- The focus is on useful coaching behavior, not model personalization.
- The repo is ready for GitHub, Hugging Face Spaces, and local OpenEnv validation.