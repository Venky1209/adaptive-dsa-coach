"""OpenEnv-style task registry with three tasks and three graders."""

from __future__ import annotations

from typing import Any

from env.graders import TASK_GRADERS, grade_easy, grade_hard, grade_medium, get_task_grader
from env.tasks import (
	TASK_INITIAL_STATES,
	TASK_MAX_STEPS,
	TASK_SPECS,
	TASK_SUCCESS_THRESHOLDS,
	build_initial_observation,
	build_task_spec,
	get_action_types,
	get_default_task_name,
	get_max_steps,
	get_success_threshold,
	get_task_spec,
	get_topic_names,
	list_task_names,
	list_task_specs,
)

# Keep the task/grader pairing explicit for scanner-friendly discovery.
TASKS_WITH_GRADERS: dict[str, dict[str, Any]] = {
	"EASY": {"spec": get_task_spec("EASY"), "grader": grade_easy},
	"MEDIUM": {"spec": get_task_spec("MEDIUM"), "grader": grade_medium},
	"HARD": {"spec": get_task_spec("HARD"), "grader": grade_hard},
}

TASKS = tuple(TASKS_WITH_GRADERS.keys())


def list_tasks_with_graders() -> dict[str, dict[str, Any]]:
	return TASKS_WITH_GRADERS


__all__ = [
	"TASKS",
	"TASKS_WITH_GRADERS",
	"TASK_GRADERS",
	"TASK_INITIAL_STATES",
	"TASK_MAX_STEPS",
	"TASK_SPECS",
	"TASK_SUCCESS_THRESHOLDS",
	"build_initial_observation",
	"build_task_spec",
	"get_action_types",
	"get_default_task_name",
	"get_max_steps",
	"get_success_threshold",
	"get_task_grader",
	"get_task_spec",
	"get_topic_names",
	"grade_easy",
	"grade_hard",
	"grade_medium",
	"list_task_names",
	"list_task_specs",
	"list_tasks_with_graders",
]