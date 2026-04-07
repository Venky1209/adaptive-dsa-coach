from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .models import (
    ACTION_TYPES,
    Action,
    AdaptiveDSAAction,
    AdaptiveDSAEnvironmentState,
    AdaptiveDSAObservation,
    Observation,
    State,
    TOPIC_NAMES,
    VALID_DISTRACTION_REDIRECTS,
    VALID_EVENT_FLAGS,
    VALID_MOTIVATION_STYLES,
)
from .tasks import get_default_task_name, get_task_spec, list_task_names


EVENT_INJECTION_POOL = ("distraction", "motivation_drop", "burnout_signal")
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _normalize_task_name(task_name: str | None) -> str:
    candidate = str(task_name or get_default_task_name()).strip().upper()
    return candidate if candidate in list_task_names() else get_default_task_name()


def _bottom_topics(observation: AdaptiveDSAObservation, count: int = 2) -> tuple[str, ...]:
    ranked = sorted(TOPIC_NAMES, key=lambda topic: (observation.topic_mastery[topic], topic))
    return tuple(ranked[:count])


def _average_topic_mastery(observation: AdaptiveDSAObservation) -> float:
    return sum(observation.topic_mastery[topic] for topic in TOPIC_NAMES) / len(TOPIC_NAMES)


def _difficulty_from_motivation(motivation: float) -> str:
    if motivation < 0.4:
        return "easy"
    if motivation < 0.7:
        return "medium"
    return "hard"


def _difficulty_rank(label: str | None) -> int | None:
    if label is None:
        return None
    return DIFFICULTY_ORDER.get(str(label).strip().lower())


def _coerce_action(value: Any) -> AdaptiveDSAAction:
    if isinstance(value, AdaptiveDSAAction):
        return value
    if isinstance(value, Action):
        return value
    if isinstance(value, dict):
        return AdaptiveDSAAction.model_validate(value)
    if hasattr(value, "model_dump"):
        return AdaptiveDSAAction.model_validate(value.model_dump())
    raise TypeError(f"Unsupported action type: {type(value)!r}")


@dataclass(slots=True)
class _StepOutcome:
    observation: AdaptiveDSAObservation
    reward: float
    done: bool
    info: dict[str, Any]


class AdaptiveDSACoachEnv:
    def __init__(self, task_name: str = "easy", seed: int = 42) -> None:
        self._rng = random.Random(seed)
        random.seed(seed)
        self._task_name = _normalize_task_name(task_name)
        self._spec = get_task_spec(self._task_name)
        self._initial_observation = self._spec.initial_state.model_copy(deep=True)
        self._state = self._build_state()

    def _build_state(self) -> AdaptiveDSAEnvironmentState:
        observation = self._spec.initial_state.model_copy(deep=True)
        observation.current_topic = observation.current_topic or _bottom_topics(observation, 1)[0]
        observation.current_problem_id = f"{self._task_name.lower()}-{observation.current_topic}-000"
        observation.time_on_problem = 0
        observation.daily_time_left = max(0, observation.daily_time_left)
        observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy)
        observation.recent_speed = max(0.0, float(observation.recent_speed))
        observation.event_flags = [flag for flag in observation.event_flags if flag in VALID_EVENT_FLAGS]
        return AdaptiveDSAEnvironmentState(
            task_name=self._task_name,
            step_count=0,
            max_steps=self._spec.max_steps,
            success_threshold=self._spec.success_threshold,
            observation=observation,
            reward_history=[],
            action_history=[],
            done=False,
            success=False,
        )

    def reset(self, task_name: str | None = None, seed: int | None = None, **_: Any) -> AdaptiveDSAObservation:
        if seed is not None:
            random.seed(seed)
            self._rng.seed(seed)

        if task_name is not None:
            self._task_name = _normalize_task_name(task_name)

        self._spec = get_task_spec(self._task_name)
        self._initial_observation = self._spec.initial_state.model_copy(deep=True)
        self._state = self._build_state()
        return self._state.observation.model_copy(deep=True)

    def state(self) -> AdaptiveDSAEnvironmentState:
        return self._state.model_copy(deep=True)

    def _inject_event_if_needed(self, step_count: int, observation: AdaptiveDSAObservation) -> None:
        if step_count % 3 != 0:
            return

        available = [event for event in EVENT_INJECTION_POOL if event not in observation.event_flags]
        if available:
            observation.event_flags.append(self._rng.choice(available))

    def _apply_action(self, observation: AdaptiveDSAObservation, action: AdaptiveDSAAction, step_count: int) -> tuple[AdaptiveDSAObservation, float, dict[str, Any]]:
        next_observation = observation.model_copy(deep=True)
        reward = 0.0
        info: dict[str, Any] = {"action_type": action.action_type}

        params = dict(action.params)
        topic = str(params.get("topic") or params.get("topic_name") or params.get("focus_topic") or params.get("exercise_topic") or "").strip()
        difficulty = str(params.get("difficulty") or params.get("level") or params.get("target_difficulty") or "").strip().lower() or None
        feedback = str(params.get("feedback") or params.get("notes") or params.get("message") or params.get("hint") or params.get("solution_feedback") or "").strip()
        optimization_hint = str(params.get("optimization_hint") or params.get("optimization") or params.get("improvement_hint") or "").strip()
        redirect = str(params.get("redirect") or params.get("redirect_to") or params.get("destination") or params.get("fallback") or "").strip().lower() or None
        style = str(params.get("style") or params.get("motivation_style") or "").strip().lower() or None

        weak_topics = _bottom_topics(next_observation, 2)
        target_difficulty = _difficulty_from_motivation(next_observation.motivation)

        if action.action_type in {"build_plan", "recommend_exercise"}:
            if topic in weak_topics:
                reward += 0.15
            if difficulty == target_difficulty:
                reward += 0.20
            if difficulty == "hard" and next_observation.burnout_risk > 0.7:
                reward -= 0.15
            if topic in TOPIC_NAMES:
                next_observation.current_topic = topic
            if action.action_type == "recommend_exercise" and topic in TOPIC_NAMES:
                next_observation.topic_mastery[topic] = _clamp_unit_interval(next_observation.topic_mastery[topic] + 0.05)
                next_observation.recent_accuracy = _clamp_unit_interval(next_observation.recent_accuracy + 0.05)
                next_observation.recent_speed = max(0.0, next_observation.recent_speed + 0.15)
            next_observation.current_problem_id = f"{self._task_name.lower()}-{next_observation.current_topic}-{step_count:03d}"

        elif action.action_type == "evaluate_solution":
            if feedback:
                if len(feedback) > 50:
                    reward += 0.25
                if optimization_hint:
                    reward += 0.10
                next_observation.recent_accuracy = _clamp_unit_interval(next_observation.recent_accuracy + min(0.2, len(feedback) / 400.0))
                next_observation.recent_speed = max(0.0, next_observation.recent_speed + 0.25)
            else:
                reward -= 0.10

        elif action.action_type == "give_hint":
            if params:
                reward += 0.05
                next_observation.recent_speed = max(0.0, next_observation.recent_speed + 0.10)
                next_observation.recent_accuracy = _clamp_unit_interval(next_observation.recent_accuracy + 0.02)
            if topic in TOPIC_NAMES:
                next_observation.current_topic = topic
                next_observation.current_problem_id = f"{self._task_name.lower()}-{topic}-{step_count:03d}"

        elif action.action_type == "handle_distraction":
            if "distraction" in next_observation.event_flags and redirect in VALID_DISTRACTION_REDIRECTS:
                reward += 0.20
                next_observation.event_flags = [flag for flag in next_observation.event_flags if flag != "distraction"]
                if redirect in TOPIC_NAMES:
                    next_observation.current_topic = redirect
            else:
                reward -= 0.20

        elif action.action_type == "handle_burnout":
            lowers_difficulty = difficulty is not None and _difficulty_rank(difficulty) is not None and _difficulty_rank(difficulty) < _difficulty_rank(target_difficulty)
            topic_shift = topic in TOPIC_NAMES and topic != next_observation.current_topic
            if "burnout_signal" in next_observation.event_flags and (lowers_difficulty or topic_shift):
                reward += 0.25
                next_observation.burnout_risk = _clamp_unit_interval(next_observation.burnout_risk - 0.2)
                next_observation.motivation = _clamp_unit_interval(next_observation.motivation + 0.1)
                next_observation.event_flags = [flag for flag in next_observation.event_flags if flag != "burnout_signal"]
                if topic in TOPIC_NAMES:
                    next_observation.current_topic = topic
            else:
                reward -= 0.20

        elif action.action_type == "give_motivation":
            reward += 0.15 if next_observation.motivation < 0.4 else 0.05
            if style in VALID_MOTIVATION_STYLES:
                next_observation.motivation = _clamp_unit_interval(next_observation.motivation + (0.08 if style == "encouraging" else 0.05 if style == "honest_recovery" else 0.06))

        elif action.action_type == "advance_session":
            if topic in TOPIC_NAMES:
                next_observation.current_topic = topic
            next_observation.recent_speed = max(0.0, next_observation.recent_speed + 0.10)

        elif action.action_type == "end_day":
            weak_topics_initial = _bottom_topics(self._initial_observation, 2)
            improved = sum(1 for topic_name in weak_topics_initial if next_observation.topic_mastery[topic_name] > self._initial_observation.topic_mastery[topic_name])
            if next_observation.motivation > 0.5 and next_observation.burnout_risk < 0.6 and improved >= 2:
                reward += 0.30
                next_observation.streak += 1
            next_observation.recent_speed = 0.0

        next_observation.time_on_problem = observation.time_on_problem + 120
        next_observation.daily_time_left = max(0, next_observation.daily_time_left - 2)
        self._inject_event_if_needed(step_count, next_observation)

        return next_observation, _clamp_reward(reward), info

    def step(self, action: Action | AdaptiveDSAAction | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        action_model = _coerce_action(action)
        next_step_count = self._state.step_count + 1
        next_observation, reward, info = self._apply_action(self._state.observation, action_model, next_step_count)

        done = next_step_count >= self._state.max_steps
        score = _clamp_unit_interval(
            0.30 * next_observation.motivation
            + 0.25 * (1.0 - next_observation.burnout_risk)
            + 0.25 * _average_topic_mastery(next_observation)
            + 0.10 * min(1.0, next_observation.streak / 10.0)
            + 0.10 * max(0.0, 1.0 - len(next_observation.event_flags) / 3.0)
        )
        success = done and score >= self._spec.success_threshold

        self._state = AdaptiveDSAEnvironmentState(
            task_name=self._task_name,
            step_count=next_step_count,
            max_steps=self._spec.max_steps,
            success_threshold=self._spec.success_threshold,
            observation=next_observation.model_copy(deep=True),
            reward_history=[*self._state.reward_history, reward],
            action_history=[*self._state.action_history, action_model.model_copy(deep=True)],
            done=done,
            success=success,
        )

        info.update({"episode_score": score, "done": done, "success": success})
        return next_observation.model_copy(deep=True), reward, done, info


State = AdaptiveDSAEnvironmentState


__all__ = ["AdaptiveDSACoachEnv", "State"]

