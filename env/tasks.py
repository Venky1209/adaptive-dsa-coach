from __future__ import annotations

from typing import Any, Final

from .models import (
    ACTION_TYPES,
    AdaptiveDSAObservation,
    AdaptiveDSATaskSpec,
    TASK_NAMES,
    TOPIC_NAMES,
)

DEFAULT_CURRENT_PROBLEM_SUFFIX: Final[str] = "bootstrap-001"

TASK_MAX_STEPS: Final[dict[str, int]] = {
    "EASY": 6,
    "MEDIUM": 10,
    "HARD": 20,
}

TASK_SUCCESS_THRESHOLDS: Final[dict[str, float]] = {
    "EASY": 0.5,
    "MEDIUM": 0.6,
    "HARD": 0.7,
}

TASK_INITIAL_STATES: Final[dict[str, dict[str, object]]] = {
    "EASY": {
        "motivation": 0.6,
        "burnout_risk": 0.2,
        "streak": 3,
        "event_flags": ["distraction"],
        "career_context": "normal",
        "daily_time_left": 120,
        "topic_mastery": {
            "arrays": 0.3,
            "strings": 0.4,
            "dp": 0.7,
            "graphs": 0.25,
            "trees": 0.35,
            "system_design": 0.5,
        },
    },
    "MEDIUM": {
        "motivation": 0.45,
        "burnout_risk": 0.55,
        "streak": 7,
        "event_flags": ["motivation_drop", "burnout_signal"],
        "career_context": "placement_season",
        "daily_time_left": 90,
        "topic_mastery": {
            "arrays": 0.6,
            "strings": 0.5,
            "dp": 0.3,
            "graphs": 0.2,
            "trees": 0.45,
            "system_design": 0.4,
        },
    },
    "HARD": {
        "motivation": 0.35,
        "burnout_risk": 0.65,
        "streak": 1,
        "event_flags": ["distraction", "burnout_signal"],
        "career_context": "hackathon_deadline",
        "daily_time_left": 180,
        "topic_mastery": {
            "arrays": 0.25,
            "strings": 0.3,
            "dp": 0.2,
            "graphs": 0.15,
            "trees": 0.2,
            "system_design": 0.3,
        },
    },
}


def _weakest_topic(topic_mastery: dict[str, float]) -> str:
    return min(topic_mastery, key=topic_mastery.get)


def _build_problem_id(task_name: str) -> str:
    return f"{task_name.lower()}-{DEFAULT_CURRENT_PROBLEM_SUFFIX}"


def build_initial_observation(task_name: str) -> AdaptiveDSAObservation:
    if task_name not in TASK_INITIAL_STATES:
        raise KeyError(f"unknown task: {task_name}")

    initial_state = TASK_INITIAL_STATES[task_name]
    topic_mastery = {topic: float(initial_state["topic_mastery"][topic]) for topic in TOPIC_NAMES}

    return AdaptiveDSAObservation(
        topic_mastery=topic_mastery,
        motivation=float(initial_state["motivation"]),
        burnout_risk=float(initial_state["burnout_risk"]),
        streak=int(initial_state["streak"]),
        daily_time_left=int(initial_state["daily_time_left"]),
        current_topic=_weakest_topic(topic_mastery),
        current_problem_id=_build_problem_id(task_name),
        time_on_problem=0,
        recent_accuracy=0.0,
        recent_speed=0.0,
        event_flags=list(initial_state["event_flags"]),
        career_context=str(initial_state["career_context"]),
    )


def build_task_spec(task_name: str) -> AdaptiveDSATaskSpec:
    if task_name not in TASK_NAMES:
        raise KeyError(f"unknown task: {task_name}")

    return AdaptiveDSATaskSpec(
        task_name=task_name,  # type: ignore[arg-type]
        max_steps=TASK_MAX_STEPS[task_name],
        success_threshold=TASK_SUCCESS_THRESHOLDS[task_name],
        initial_state=build_initial_observation(task_name),
        grader=f"grade_{task_name.lower()}",
        description=(
            f"{task_name} AdaptiveDSA Coach task with exact initial state, "
            f"deterministic transitions, and dense coaching rewards."
        ),
    )


TASK_SPECS: Final[dict[str, AdaptiveDSATaskSpec]] = {
    task_name: build_task_spec(task_name) for task_name in TASK_NAMES
}


def grade_easy_task(*args: Any, **kwargs: Any):
    # Lazy import avoids a circular dependency: graders imports get_task_spec from this module.
    from .graders import grade_easy

    return grade_easy(*args, **kwargs)


def grade_medium_task(*args: Any, **kwargs: Any):
    # Lazy import avoids a circular dependency: graders imports get_task_spec from this module.
    from .graders import grade_medium

    return grade_medium(*args, **kwargs)


def grade_hard_task(*args: Any, **kwargs: Any):
    # Lazy import avoids a circular dependency: graders imports get_task_spec from this module.
    from .graders import grade_hard

    return grade_hard(*args, **kwargs)


# Explicit task manifest for static validators that enumerate tasks and graders directly.
TASKS: list[dict[str, Any]] = [
    {
        "task_id": "EASY",
        "task_name": "EASY",
        "difficulty": "easy",
        "spec": TASK_SPECS["EASY"],
        "grader": grade_easy_task,
        "grader_name": "grade_easy",
    },
    {
        "task_id": "MEDIUM",
        "task_name": "MEDIUM",
        "difficulty": "medium",
        "spec": TASK_SPECS["MEDIUM"],
        "grader": grade_medium_task,
        "grader_name": "grade_medium",
    },
    {
        "task_id": "HARD",
        "task_name": "HARD",
        "difficulty": "hard",
        "spec": TASK_SPECS["HARD"],
        "grader": grade_hard_task,
        "grader_name": "grade_hard",
    },
]

TASKS_WITH_GRADERS: dict[str, dict[str, Any]] = {
    entry["task_name"]: entry for entry in TASKS
}


def list_task_names() -> tuple[str, ...]:
    return TASK_NAMES


def list_task_specs() -> tuple[AdaptiveDSATaskSpec, ...]:
    return tuple(TASK_SPECS[task_name] for task_name in TASK_NAMES)


def get_task_spec(task_name: str) -> AdaptiveDSATaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise KeyError(f"unknown task: {task_name}") from exc


def get_default_task_name() -> str:
    return TASK_NAMES[0]


def get_max_steps(task_name: str) -> int:
    return TASK_MAX_STEPS[task_name]


def get_success_threshold(task_name: str) -> float:
    return TASK_SUCCESS_THRESHOLDS[task_name]


def get_topic_names() -> tuple[str, ...]:
    return TOPIC_NAMES


def get_action_types() -> tuple[str, ...]:
    return ACTION_TYPES


__all__ = [
    "DEFAULT_CURRENT_PROBLEM_SUFFIX",
    "TASKS",
    "TASKS_WITH_GRADERS",
    "TASK_INITIAL_STATES",
    "TASK_MAX_STEPS",
    "TASK_SPECS",
    "TASK_SUCCESS_THRESHOLDS",
    "build_initial_observation",
    "build_task_spec",
    "grade_easy_task",
    "grade_hard_task",
    "grade_medium_task",
    "get_action_types",
    "get_default_task_name",
    "get_max_steps",
    "get_success_threshold",
    "get_task_spec",
    "get_topic_names",
    "list_task_names",
    "list_task_specs",
]