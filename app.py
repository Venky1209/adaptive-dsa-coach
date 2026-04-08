from __future__ import annotations

import random
import threading
from typing import Any, Final

from fastapi import FastAPI, HTTPException

from env.models import (
	ACTION_TYPES,
	AdaptiveDSAAction,
	AdaptiveDSAEnvironmentState,
	AdaptiveDSAObservation,
	AdaptiveDSAResetPayload,
	AdaptiveDSAResetResponse,
	AdaptiveDSAStepPayload,
	AdaptiveDSAStepResponse,
	AdaptiveDSAStepResult,
	TOPIC_NAMES,
	VALID_DISTRACTION_REDIRECTS,
	VALID_EVENT_FLAGS,
	VALID_MOTIVATION_STYLES,
)
from env.tasks import build_initial_observation, get_default_task_name, get_task_spec, list_task_names, list_task_specs
from env.graders import TASK_GRADERS, grade_task


EVENT_INJECTION_POOL: Final[tuple[str, ...]] = ("distraction", "motivation_drop", "burnout_signal")
DIFFICULTY_ORDER: Final[dict[str, int]] = {"easy": 0, "medium": 1, "hard": 2}


def _clamp_unit_interval(value: float) -> float:
	return max(0.0, min(1.0, float(value)))


def _clamp_reward(value: float) -> float:
	return max(-1.0, min(1.0, float(value)))


def _safe_lower(value: Any) -> str:
	return str(value).strip().lower()


def _safe_str(value: Any, default: str = "") -> str:
	if value is None:
		return default
	text = str(value).strip()
	return text if text else default


def _difficulty_from_motivation(motivation: float) -> str:
	if motivation < 0.4:
		return "easy"
	if motivation < 0.7:
		return "medium"
	return "hard"


def _difficulty_rank(label: str | None) -> int | None:
	if label is None:
		return None
	return DIFFICULTY_ORDER.get(_safe_lower(label))


def _bottom_topics(observation: AdaptiveDSAObservation, count: int = 2) -> tuple[str, ...]:
	ranked = sorted(TOPIC_NAMES, key=lambda topic: (observation.topic_mastery[topic], topic))
	return tuple(ranked[:count])


def _average_topic_mastery(observation: AdaptiveDSAObservation) -> float:
	return sum(observation.topic_mastery[topic] for topic in TOPIC_NAMES) / len(TOPIC_NAMES)


def _count_improved_topics(
	initial_state: AdaptiveDSAObservation,
	current_state: AdaptiveDSAObservation,
	topics: tuple[str, ...],
) -> int:
	improved = 0
	for topic in topics:
		if current_state.topic_mastery[topic] > initial_state.topic_mastery[topic]:
			improved += 1
	return improved


def _clean_event_flags(flags: list[str]) -> list[str]:
	deduped = []
	for flag in flags:
		if flag in VALID_EVENT_FLAGS and flag not in deduped:
			deduped.append(flag)
	return deduped


def _problem_id(task_name: str, topic: str, step_count: int) -> str:
	return f"{task_name.lower()}-{topic}-{step_count:03d}"


def _normalized_state(state: AdaptiveDSAEnvironmentState) -> AdaptiveDSAEnvironmentState:
	return AdaptiveDSAEnvironmentState.model_validate(state.model_dump())


class AdaptiveDSARuntime:
	def __init__(self, task_name: str | None = None) -> None:
		self._lock = threading.Lock()
		self._rng = random.Random(42)
		self._task_name = str(task_name or get_default_task_name()).strip().upper()
		self._spec = get_task_spec(self._task_name)
		self._initial_observation = self._spec.initial_state.model_copy(deep=True)
		self._state = self._build_state(self._spec)

	def _build_state(self, spec) -> AdaptiveDSAEnvironmentState:
		observation = spec.initial_state.model_copy(deep=True)
		observation.event_flags = _clean_event_flags(observation.event_flags)
		observation.current_topic = observation.current_topic or _bottom_topics(observation, 1)[0]
		observation.current_problem_id = _problem_id(spec.task_name, observation.current_topic, 0)
		observation.time_on_problem = 0
		observation.daily_time_left = max(0, observation.daily_time_left)
		observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy)
		observation.recent_speed = max(0.0, float(observation.recent_speed))

		return AdaptiveDSAEnvironmentState(
			task_name=spec.task_name,
			step_count=0,
			max_steps=spec.max_steps,
			success_threshold=spec.success_threshold,
			observation=observation,
			reward_history=[],
			action_history=[],
			done=False,
			success=False,
		)

	def reset(self, task_name: str) -> AdaptiveDSAEnvironmentState:
		with self._lock:
			random.seed(42)
			self._rng.seed(42)
			self._task_name = str(task_name).strip().upper()
			self._spec = get_task_spec(self._task_name)
			self._initial_observation = self._spec.initial_state.model_copy(deep=True)
			self._state = self._build_state(self._spec)
			return self._state.model_copy(deep=True)

	def get_state(self) -> AdaptiveDSAEnvironmentState:
		with self._lock:
			return self._state.model_copy(deep=True)

	def _inject_event_if_needed(self, step_count: int, observation: AdaptiveDSAObservation) -> tuple[AdaptiveDSAObservation, str | None]:
		if step_count % 3 != 0:
			return observation, None

		injected_event = None
		available_events = list(EVENT_INJECTION_POOL)
		attempts = 0
		while available_events and attempts < len(EVENT_INJECTION_POOL):
			candidate = self._rng.choice(available_events)
			if candidate not in observation.event_flags:
				injected_event = candidate
				observation.event_flags.append(candidate)
				break
			available_events.remove(candidate)
			attempts += 1

		observation.event_flags = _clean_event_flags(observation.event_flags)
		return observation, injected_event

	def _apply_action(self, state: AdaptiveDSAEnvironmentState, action: AdaptiveDSAAction) -> tuple[AdaptiveDSAObservation, float, dict[str, Any]]:
		observation = state.observation.model_copy(deep=True)
		reward = 0.0
		info: dict[str, Any] = {
			"action_type": action.action_type,
			"reward_components": {},
		}

		action_type = action.action_type
		params = dict(action.params)
		current_bottom_topics = _bottom_topics(observation, 2)

		topic = _safe_str(params.get("topic") or params.get("topic_name") or params.get("focus_topic") or params.get("exercise_topic"))
		difficulty = _safe_lower(params.get("difficulty") or params.get("level") or params.get("target_difficulty")) or None
		feedback = _safe_str(params.get("feedback") or params.get("notes") or params.get("message") or params.get("hint") or params.get("solution_feedback"))
		optimization_hint = _safe_str(params.get("optimization_hint") or params.get("optimization") or params.get("improvement_hint"))
		redirect = _safe_lower(params.get("redirect") or params.get("redirect_to") or params.get("destination") or params.get("fallback")) or None
		style = _safe_lower(params.get("style") or params.get("motivation_style")) or None

		if action_type in {"build_plan", "recommend_exercise"}:
			if topic in current_bottom_topics:
				reward += 0.15
				info["reward_components"]["targeted_weak_topic"] = 0.15

			target_difficulty = _difficulty_from_motivation(observation.motivation)
			if difficulty and difficulty == target_difficulty:
				reward += 0.20
				info["reward_components"]["difficulty_match"] = 0.20

			if difficulty == "hard" and observation.burnout_risk > 0.7:
				reward -= 0.15
				info["reward_components"]["hard_topic_burnout_penalty"] = -0.15

			if topic in TOPIC_NAMES:
				observation.current_topic = topic
			elif not observation.current_topic:
				observation.current_topic = current_bottom_topics[0]

			if action_type == "recommend_exercise" and topic in TOPIC_NAMES:
				observation.topic_mastery[topic] = _clamp_unit_interval(observation.topic_mastery[topic] + 0.05)
				info["topic_mastery_gain"] = {topic: 0.05}
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.05)
				observation.recent_speed = max(0.0, observation.recent_speed + 0.15)

			observation.current_problem_id = _problem_id(state.task_name, observation.current_topic, state.step_count + 1)

		elif action_type == "evaluate_solution":
			if feedback:
				if len(feedback) > 50:
					reward += 0.25
					info["reward_components"]["feedback_quality"] = 0.25
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + min(0.2, len(feedback) / 400.0))
				observation.recent_speed = max(0.0, observation.recent_speed + 0.25)
			else:
				reward -= 0.10
				info["reward_components"]["missing_feedback_penalty"] = -0.10

			if optimization_hint:
				reward += 0.10
				info["reward_components"]["optimization_hint"] = 0.10
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.03)

		elif action_type == "give_hint":
			hint_text = _safe_str(params.get("hint") or params.get("hint_text") or params.get("content"))
			if hint_text:
				reward += 0.05
				info["reward_components"]["hint_provided"] = 0.05
				observation.recent_speed = max(0.0, observation.recent_speed + 0.10)
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.02)

			if topic in TOPIC_NAMES:
				observation.current_topic = topic
				observation.current_problem_id = _problem_id(state.task_name, observation.current_topic, state.step_count + 1)

		elif action_type == "handle_distraction":
			if "distraction" in observation.event_flags and redirect in VALID_DISTRACTION_REDIRECTS:
				reward += 0.20
				info["reward_components"]["productive_redirect"] = 0.20
				observation.event_flags = [flag for flag in observation.event_flags if flag != "distraction"]
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.04)
				if redirect in TOPIC_NAMES:
					observation.current_topic = redirect
				elif redirect == "system_design":
					observation.current_topic = "system_design"
			else:
				reward -= 0.20
				info["reward_components"]["ignored_or_entertainment_penalty"] = -0.20

		elif action_type == "handle_burnout":
			lowers_difficulty = difficulty is not None and _difficulty_rank(difficulty) is not None and _difficulty_rank(difficulty) < _difficulty_rank(_difficulty_from_motivation(observation.motivation))
			topic_shift = topic in TOPIC_NAMES and topic != observation.current_topic
			if "burnout_signal" in observation.event_flags and (lowers_difficulty or topic_shift):
				reward += 0.25
				info["reward_components"]["burnout_handled"] = 0.25
				observation.burnout_risk = _clamp_unit_interval(observation.burnout_risk - 0.2)
				observation.motivation = _clamp_unit_interval(observation.motivation + 0.1)
				observation.event_flags = [flag for flag in observation.event_flags if flag != "burnout_signal"]
				if topic in TOPIC_NAMES:
					observation.current_topic = topic
					observation.current_problem_id = _problem_id(state.task_name, observation.current_topic, state.step_count + 1)
				observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.03)
			else:
				reward -= 0.20
				info["reward_components"]["burnout_ignored_penalty"] = -0.20

		elif action_type == "give_motivation":
			if observation.motivation < 0.4:
				reward += 0.15
				info["reward_components"]["low_motivation_support"] = 0.15
			else:
				reward += 0.05
				info["reward_components"]["steady_motivation_support"] = 0.05

			style_boost = {
				"encouraging": 0.08,
				"honest_recovery": 0.05,
				"career_linked": 0.06,
			}.get(style or "", 0.03)
			observation.motivation = _clamp_unit_interval(observation.motivation + style_boost)
			observation.recent_accuracy = _clamp_unit_interval(observation.recent_accuracy + 0.02)

		elif action_type == "advance_session":
			next_topic = _safe_str(params.get("next_topic") or params.get("topic") or params.get("focus_topic"))
			if next_topic in TOPIC_NAMES:
				observation.current_topic = next_topic
				reward += 0.05
				info["reward_components"]["session_advanced"] = 0.05
			else:
				observation.current_topic = _bottom_topics(observation, 1)[0]
			observation.current_problem_id = _problem_id(state.task_name, observation.current_topic, state.step_count + 1)
			observation.recent_speed = max(0.0, observation.recent_speed + 0.10)

		elif action_type == "end_day":
			initial_bottom_topics = _bottom_topics(self._initial_observation, 2)
			improved_weak_topics = _count_improved_topics(self._initial_observation, observation, initial_bottom_topics)

			preconditions_met = sum(
				1
				for condition in (
					observation.motivation > 0.5,
					observation.burnout_risk < 0.6,
					improved_weak_topics >= 2,
				)
				if condition
			)
			if preconditions_met >= 3:
				observation.streak += 1
				info["streak_incremented"] = True

			conditions = [
				observation.streak > state.observation.streak,
				observation.motivation > 0.5,
				observation.burnout_risk < 0.6,
				improved_weak_topics >= 2,
			]
			conditions_met = sum(1 for condition in conditions if condition)
			info["end_day_conditions_met"] = conditions_met
			info["improved_weak_topics"] = improved_weak_topics

			if conditions_met >= 4:
				reward += 1.0
				info["reward_components"]["end_day_bonus"] = 1.0
			elif conditions_met == 3:
				reward += 0.65
				info["reward_components"]["end_day_bonus"] = 0.65
			elif conditions_met == 2:
				reward += 0.30
				info["reward_components"]["end_day_bonus"] = 0.30

			observation.daily_time_left = self._spec.initial_state.daily_time_left
			observation.current_topic = _bottom_topics(observation, 1)[0]
			observation.current_problem_id = _problem_id(state.task_name, observation.current_topic, state.step_count + 1)
			observation.recent_speed = 0.0

		else:
			info["reward_components"]["neutral_action"] = 0.0

		return observation, _clamp_reward(reward), info

	def step(self, action: AdaptiveDSAAction) -> tuple[AdaptiveDSAEnvironmentState, AdaptiveDSAStepResult]:
		with self._lock:
			if self._state.done:
				raise HTTPException(status_code=409, detail="episode already finished; call /reset to start a new task")

			previous_state = self._state.model_copy(deep=True)
			next_observation, reward, info = self._apply_action(previous_state, action)

			next_observation.time_on_problem = previous_state.observation.time_on_problem + 120
			next_observation.daily_time_left = max(0, next_observation.daily_time_left - 2)

			next_step_count = previous_state.step_count + 1
			next_observation, injected_event = self._inject_event_if_needed(next_step_count, next_observation)
			if injected_event is not None:
				info["injected_event"] = injected_event

			episode_score = self._episode_score(next_observation)
			done = next_step_count >= previous_state.max_steps
			success = done and episode_score >= previous_state.success_threshold

			next_state = AdaptiveDSAEnvironmentState(
				task_name=previous_state.task_name,
				step_count=next_step_count,
				max_steps=previous_state.max_steps,
				success_threshold=previous_state.success_threshold,
				observation=next_observation,
				reward_history=[*previous_state.reward_history, reward],
				action_history=[*previous_state.action_history, action.model_copy(deep=True)],
				done=done,
				success=success,
			)
			next_state = _normalized_state(next_state)
			self._state = next_state

			info.update(
				{
					"episode_score": episode_score,
					"step_count": next_step_count,
					"done": done,
					"success": success,
					"current_topic": next_observation.current_topic,
					"current_problem_id": next_observation.current_problem_id,
				}
			)

			result = AdaptiveDSAStepResult(
				observation=next_observation.model_copy(deep=True),
				reward=reward,
				done=done,
				success=success,
				info=info,
			)
			return next_state.model_copy(deep=True), result

	def _episode_score(self, observation: AdaptiveDSAObservation) -> float:
		motivation_component = observation.motivation
		burnout_component = 1.0 - observation.burnout_risk
		mastery_component = _average_topic_mastery(observation)
		streak_component = min(1.0, observation.streak / 10.0)
		event_component = max(0.0, 1.0 - len(observation.event_flags) / 3.0)
		return _clamp_unit_interval(
			0.30 * motivation_component
			+ 0.25 * burnout_component
			+ 0.25 * mastery_component
			+ 0.10 * streak_component
			+ 0.10 * event_component
		)


def create_app() -> FastAPI:
	application = FastAPI(title="AdaptiveDSA Coach", version="1.0.0")
	application.state.runtime = AdaptiveDSARuntime()

	@application.get("/")
	def root() -> dict[str, Any]:
		runtime: AdaptiveDSARuntime = application.state.runtime
		state = runtime.get_state()
		tasks_with_graders = [task_name for task_name in list_task_names() if task_name in TASK_GRADERS]
		return {
			"name": "adaptive-dsa-coach",
			"env": "adaptive_dsa_coach",
			"tasks": list(list_task_names()),
			"tasks_with_graders": tasks_with_graders,
			"tasks_with_graders_count": len(tasks_with_graders),
			"current_task": state.task_name,
		}

	@application.get("/metadata")
	def metadata() -> dict[str, Any]:
		tasks_with_graders = [task_name for task_name in list_task_names() if task_name in TASK_GRADERS]
		return {
			"name": "adaptive-dsa-coach",
			"description": "Adaptive DSA tutoring environment with deterministic multi-task grading.",
			"tasks": list(list_task_names()),
			"tasks_with_graders": tasks_with_graders,
			"tasks_with_graders_count": len(tasks_with_graders),
		}

	@application.get("/health")
	def health() -> dict[str, Any]:
		return {"status": "ok", "env": "adaptive_dsa_coach"}

	@application.get("/tasks")
	def tasks() -> dict[str, Any]:
		tasks_payload: list[dict[str, Any]] = []
		difficulty_map = {"EASY": "easy", "MEDIUM": "medium", "HARD": "hard"}
		for spec in list_task_specs():
			task_name = spec.task_name
			tasks_payload.append(
				{
					**spec.model_dump(),
					"task_id": task_name,
					"difficulty": difficulty_map.get(task_name, "unknown"),
					"grader": spec.grader,
					"grader_name": f"grade_{task_name.lower()}",
					"has_grader": task_name in TASK_GRADERS,
				}
			)
		tasks_with_graders = [task for task in tasks_payload if task.get("has_grader")]
		return {
			"tasks": tasks_payload,
			"tasks_with_graders": tasks_with_graders,
			"tasks_with_graders_count": len(tasks_with_graders),
			"default_task": get_default_task_name(),
		}

	@application.get("/state", response_model=AdaptiveDSAResetResponse)
	def get_state() -> AdaptiveDSAResetResponse:
		runtime: AdaptiveDSARuntime = application.state.runtime
		return AdaptiveDSAResetResponse(state=runtime.get_state())

	@application.post("/state", response_model=AdaptiveDSAResetResponse)
	def post_state() -> AdaptiveDSAResetResponse:
		runtime: AdaptiveDSARuntime = application.state.runtime
		return AdaptiveDSAResetResponse(state=runtime.get_state())

	@application.post("/reset", response_model=AdaptiveDSAResetResponse)
	def reset(payload: dict[str, Any] | None = None) -> AdaptiveDSAResetResponse:
		runtime: AdaptiveDSARuntime = application.state.runtime
		request_body = payload or {}
		task_name = request_body.get("task_name") or request_body.get("task_id") or request_body.get("task") or get_default_task_name()
		state = runtime.reset(str(task_name).strip().upper())
		return AdaptiveDSAResetResponse(state=state)

	@application.post("/step", response_model=AdaptiveDSAStepResponse)
	def step(payload: dict[str, Any]) -> AdaptiveDSAStepResponse:
		runtime: AdaptiveDSARuntime = application.state.runtime
		action_payload = payload.get("action", payload)
		action = AdaptiveDSAAction.model_validate(action_payload)
		state, result = runtime.step(action)
		return AdaptiveDSAStepResponse(state=state, result=result)

	@application.post("/grader")
	def grader(payload: dict[str, Any] | None = None) -> dict[str, Any]:
		request_body = payload or {}
		task_name = str(request_body.get("task_name") or request_body.get("task_id") or request_body.get("task") or get_default_task_name()).strip().upper()
		if task_name not in TASK_GRADERS:
			raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")

		initial_state = request_body.get("initial_state")
		final_state = request_body.get("final_state")
		actions = request_body.get("actions")
		trajectory = request_body.get("trajectory")

		# Support minimal validator payloads that only provide task identifiers.
		if initial_state is None:
			initial_state = build_initial_observation(task_name).model_dump()
		if final_state is None:
			final_state = build_initial_observation(task_name).model_dump()
		if actions is None:
			actions = []
		if trajectory is None:
			trajectory = []

		result = grade_task(
			task_name=task_name,
			initial_state=initial_state,
			final_state=final_state,
			actions=actions,
			trajectory=trajectory,
		)
		return {
			"task_name": result.task_name,
			"task_id": result.task_name,
			"grader": f"grade_{result.task_name.lower()}",
			"score": result.score,
			"breakdown": result.breakdown,
			"has_grader": True,
		}

	return application


app = create_app()


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
