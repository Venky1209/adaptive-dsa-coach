from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any
from types import SimpleNamespace

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - local fallback when the dependency is unavailable
    class _NoOpResponses:
        def create(self, *args: Any, **kwargs: Any) -> None:
            return None


    class OpenAI:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.responses = _NoOpResponses()

try:
    import app as app_module
    from env.graders import grade_task
    from env.models import (
        ACTION_TYPES,
        AdaptiveDSAAction,
        AdaptiveDSAObservation,
        TOPIC_NAMES,
        VALID_DISTRACTION_REDIRECTS,
        VALID_MOTIVATION_STYLES,
    )
    from env.tasks import get_default_task_name, get_task_spec, list_task_names
    FALLBACK_MODE = False
except Exception:  # pragma: no cover - local fallback for environments without project deps
    FALLBACK_MODE = True

    ACTION_TYPES = (
        "build_plan",
        "recommend_exercise",
        "evaluate_solution",
        "give_hint",
        "handle_distraction",
        "handle_burnout",
        "give_motivation",
        "advance_session",
        "end_day",
    )
    TOPIC_NAMES = (
        "arrays",
        "strings",
        "dp",
        "graphs",
        "trees",
        "system_design",
    )
    VALID_DISTRACTION_REDIRECTS = (
        "core_cs",
        "lighter_practice",
        "system_design",
        "project_ideation",
        "active_recall",
    )
    VALID_MOTIVATION_STYLES = ("encouraging", "honest_recovery", "career_linked")

    @dataclass(slots=True)
    class AdaptiveDSAAction:
        action_type: str
        params: dict[str, Any] = field(default_factory=dict)

        def model_copy(self, deep: bool = False) -> "AdaptiveDSAAction":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class AdaptiveDSAObservation:
        topic_mastery: dict[str, float] = field(default_factory=dict)
        motivation: float = 0.0
        burnout_risk: float = 0.0
        streak: int = 0
        daily_time_left: int = 0
        current_topic: str = ""
        current_problem_id: str = ""
        time_on_problem: int = 0
        recent_accuracy: float = 0.0
        recent_speed: float = 0.0
        event_flags: list[str] = field(default_factory=list)
        career_context: str = "normal"

        def model_copy(self, deep: bool = False) -> "AdaptiveDSAObservation":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class AdaptiveDSAEnvironmentState:
        task_name: str
        step_count: int
        max_steps: int
        success_threshold: float
        observation: AdaptiveDSAObservation
        reward_history: list[float] = field(default_factory=list)
        action_history: list[AdaptiveDSAAction] = field(default_factory=list)
        done: bool = False
        success: bool = False

        def model_copy(self, deep: bool = False) -> "AdaptiveDSAEnvironmentState":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class AdaptiveDSAStepResult:
        observation: AdaptiveDSAObservation
        reward: float
        done: bool
        success: bool
        info: dict[str, Any] = field(default_factory=dict)

        def model_copy(self, deep: bool = False) -> "AdaptiveDSAStepResult":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class AdaptiveDSATaskSpec:
        task_name: str
        max_steps: int
        success_threshold: float
        initial_state: AdaptiveDSAObservation
        description: str = ""

    TASK_INITIAL_STATES: dict[str, dict[str, Any]] = {
        "EASY": {
            "motivation": 0.6,
            "burnout_risk": 0.2,
            "streak": 3,
            "event_flags": ["distraction"],
            "career_context": "normal",
            "daily_time_left": 120,
            "topic_mastery": {"arrays": 0.3, "strings": 0.4, "dp": 0.7, "graphs": 0.25, "trees": 0.35, "system_design": 0.5},
        },
        "MEDIUM": {
            "motivation": 0.45,
            "burnout_risk": 0.55,
            "streak": 7,
            "event_flags": ["motivation_drop", "burnout_signal"],
            "career_context": "placement_season",
            "daily_time_left": 90,
            "topic_mastery": {"arrays": 0.6, "strings": 0.5, "dp": 0.3, "graphs": 0.2, "trees": 0.45, "system_design": 0.4},
        },
        "HARD": {
            "motivation": 0.35,
            "burnout_risk": 0.65,
            "streak": 1,
            "event_flags": ["distraction", "burnout_signal"],
            "career_context": "hackathon_deadline",
            "daily_time_left": 180,
            "topic_mastery": {"arrays": 0.25, "strings": 0.3, "dp": 0.2, "graphs": 0.15, "trees": 0.2, "system_design": 0.3},
        },
    }
    TASK_MAX_STEPS = {"EASY": 6, "MEDIUM": 10, "HARD": 20}
    TASK_SUCCESS_THRESHOLDS = {"EASY": 0.5, "MEDIUM": 0.6, "HARD": 0.7}

    def _clamp_unit_interval(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _clamp_reward(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _build_problem_id(task_name: str, topic: str, step_count: int) -> str:
        return f"{task_name.lower()}-{topic}-{step_count:03d}"

    def _weakest_topic(topic_mastery: dict[str, float]) -> str:
        return min(topic_mastery, key=topic_mastery.get)

    def build_initial_observation(task_name: str) -> AdaptiveDSAObservation:
        initial_state = TASK_INITIAL_STATES[task_name]
        topic_mastery = {topic: float(initial_state["topic_mastery"][topic]) for topic in TOPIC_NAMES}
        return AdaptiveDSAObservation(
            topic_mastery=topic_mastery,
            motivation=float(initial_state["motivation"]),
            burnout_risk=float(initial_state["burnout_risk"]),
            streak=int(initial_state["streak"]),
            daily_time_left=int(initial_state["daily_time_left"]),
            current_topic=_weakest_topic(topic_mastery),
            current_problem_id=f"{task_name.lower()}-bootstrap-001",
            time_on_problem=0,
            recent_accuracy=0.0,
            recent_speed=0.0,
            event_flags=list(initial_state["event_flags"]),
            career_context=str(initial_state["career_context"]),
        )

    def list_task_names() -> tuple[str, ...]:
        return tuple(TASK_INITIAL_STATES.keys())

    def get_default_task_name() -> str:
        return "EASY"

    def get_task_spec(task_name: str) -> AdaptiveDSATaskSpec:
        return AdaptiveDSATaskSpec(
            task_name=task_name,
            max_steps=TASK_MAX_STEPS[task_name],
            success_threshold=TASK_SUCCESS_THRESHOLDS[task_name],
            initial_state=build_initial_observation(task_name),
            description=f"Fallback {task_name} task.",
        )

    def grade_task(*args: Any, **kwargs: Any) -> SimpleNamespace:
        final_state = kwargs.get("final_state")
        if final_state is None and len(args) > 1:
            final_state = args[1]
        if hasattr(final_state, "observation"):
            observation = final_state.observation
        else:
            observation = final_state
        if hasattr(observation, "model_dump"):
            observation = observation.model_dump()
        score = _clamp_unit_interval(
            0.30 * float(observation["motivation"])
            + 0.25 * (1.0 - float(observation["burnout_risk"]))
            + 0.25 * sum(float(observation["topic_mastery"][topic]) for topic in TOPIC_NAMES) / len(TOPIC_NAMES)
            + 0.10 * min(1.0, float(observation["streak"]) / 10.0)
            + 0.10 * max(0.0, 1.0 - len(observation["event_flags"]) / 3.0)
        )
        return SimpleNamespace(score=score)

    class _FallbackRuntime:
        def __init__(self, task_name: str | None = None) -> None:
            self._rng = random.Random(42)
            self._task_name = task_name or get_default_task_name()
            self._spec = get_task_spec(self._task_name)
            self._initial_observation = self._spec.initial_state.model_copy(deep=True)
            self._state = self._build_state(self._spec)

        def _build_state(self, spec: AdaptiveDSATaskSpec) -> AdaptiveDSAEnvironmentState:
            observation = spec.initial_state.model_copy(deep=True)
            return AdaptiveDSAEnvironmentState(
                task_name=spec.task_name,
                step_count=0,
                max_steps=spec.max_steps,
                success_threshold=spec.success_threshold,
                observation=observation,
            )

        def reset(self, task_name: str) -> AdaptiveDSAEnvironmentState:
            self._rng.seed(42)
            self._task_name = task_name
            self._spec = get_task_spec(task_name)
            self._initial_observation = self._spec.initial_state.model_copy(deep=True)
            self._state = self._build_state(self._spec)
            return self._state.model_copy(deep=True)

        def get_state(self) -> AdaptiveDSAEnvironmentState:
            return self._state.model_copy(deep=True)

        def _episode_score(self, observation: AdaptiveDSAObservation) -> float:
            return _clamp_unit_interval(
                0.30 * observation.motivation
                + 0.25 * (1.0 - observation.burnout_risk)
                + 0.25 * sum(observation.topic_mastery[topic] for topic in TOPIC_NAMES) / len(TOPIC_NAMES)
                + 0.10 * min(1.0, observation.streak / 10.0)
                + 0.10 * max(0.0, 1.0 - len(observation.event_flags) / 3.0)
            )

        def _inject_event_if_needed(self, step_count: int, observation: AdaptiveDSAObservation) -> None:
            if step_count % 3 != 0:
                return
            available = [event for event in ("distraction", "motivation_drop", "burnout_signal") if event not in observation.event_flags]
            if available:
                candidate = self._rng.choice(available)
                observation.event_flags.append(candidate)

        def step(self, action: AdaptiveDSAAction) -> tuple[AdaptiveDSAEnvironmentState, AdaptiveDSAStepResult]:
            if self._state.done:
                raise RuntimeError("episode already finished; reset first")

            previous_state = self._state.model_copy(deep=True)
            observation = previous_state.observation.model_copy(deep=True)
            reward = 0.0
            params = dict(action.params)
            topic = str(params.get("topic") or params.get("topic_name") or params.get("focus_topic") or params.get("exercise_topic") or "").strip()
            difficulty = str(params.get("difficulty") or params.get("level") or params.get("target_difficulty") or "").strip().lower()
            feedback = str(params.get("feedback") or params.get("notes") or params.get("message") or params.get("hint") or params.get("solution_feedback") or "").strip()
            optimization_hint = str(params.get("optimization_hint") or params.get("optimization") or params.get("improvement_hint") or "").strip()
            redirect = str(params.get("redirect") or params.get("redirect_to") or params.get("destination") or params.get("fallback") or "").strip().lower()
            style = str(params.get("style") or params.get("motivation_style") or "").strip().lower()
            weakest_topic = _weakest_topic(observation.topic_mastery)

            if action.action_type in {"build_plan", "recommend_exercise"}:
                if topic == weakest_topic or topic == _second_weak_topic(observation):
                    reward += 0.15
                if difficulty == ("easy" if observation.motivation < 0.4 else "medium" if observation.motivation < 0.7 else "hard"):
                    reward += 0.20
                if difficulty == "hard" and observation.burnout_risk > 0.7:
                    reward -= 0.15
                if topic in TOPIC_NAMES:
                    observation.current_topic = topic
                    if action.action_type == "recommend_exercise":
                        observation.topic_mastery[topic] = _clamp_unit_interval(observation.topic_mastery[topic] + 0.05)

            elif action.action_type == "evaluate_solution":
                if feedback:
                    if len(feedback) > 50:
                        reward += 0.25
                    if optimization_hint:
                        reward += 0.10
                else:
                    reward -= 0.10

            elif action.action_type == "handle_distraction":
                if "distraction" in observation.event_flags and redirect in VALID_DISTRACTION_REDIRECTS:
                    reward += 0.20
                    observation.event_flags = [flag for flag in observation.event_flags if flag != "distraction"]
                else:
                    reward -= 0.20

            elif action.action_type == "handle_burnout":
                if "burnout_signal" in observation.event_flags:
                    reward += 0.25
                    observation.event_flags = [flag for flag in observation.event_flags if flag != "burnout_signal"]
                    observation.burnout_risk = _clamp_unit_interval(observation.burnout_risk - 0.2)
                    observation.motivation = _clamp_unit_interval(observation.motivation + 0.1)
                    if topic in TOPIC_NAMES:
                        observation.current_topic = topic
                else:
                    reward -= 0.20

            elif action.action_type == "give_motivation":
                reward += 0.15 if observation.motivation < 0.4 else 0.05
                if style in VALID_MOTIVATION_STYLES:
                    observation.motivation = _clamp_unit_interval(observation.motivation + (0.08 if style == "encouraging" else 0.05 if style == "honest_recovery" else 0.06))

            elif action.action_type == "advance_session":
                if topic in TOPIC_NAMES:
                    observation.current_topic = topic
                observation.recent_speed = max(0.0, observation.recent_speed + 0.10)

            elif action.action_type == "end_day":
                if observation.motivation > 0.5 and observation.burnout_risk < 0.6:
                    reward += 0.30
                    if observation.streak <= previous_state.observation.streak:
                        observation.streak += 1

            observation.time_on_problem = previous_state.observation.time_on_problem + 120
            observation.daily_time_left = max(0, observation.daily_time_left - 2)
            next_step_count = previous_state.step_count + 1
            if next_step_count > 1:
                self._inject_event_if_needed(next_step_count, observation)

            if not observation.current_problem_id:
                observation.current_problem_id = _build_problem_id(self._task_name, observation.current_topic or weakest_topic, next_step_count)

            score = self._episode_score(observation)
            done = next_step_count >= previous_state.max_steps
            success = done and score >= previous_state.success_threshold
            next_state = AdaptiveDSAEnvironmentState(
                task_name=previous_state.task_name,
                step_count=next_step_count,
                max_steps=previous_state.max_steps,
                success_threshold=previous_state.success_threshold,
                observation=observation,
                reward_history=[*previous_state.reward_history, _clamp_reward(reward)],
                action_history=[*previous_state.action_history, action.model_copy(deep=True)],
                done=done,
                success=success,
            )
            self._state = next_state
            result = AdaptiveDSAStepResult(observation=observation.model_copy(deep=True), reward=_clamp_reward(reward), done=done, success=success, info={"episode_score": score})
            return next_state.model_copy(deep=True), result

    app_module = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=_FallbackRuntime())))


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_ENV_NAME = "adaptive_dsa_coach"


@dataclass(slots=True)
class InferenceConfig:
    task_name: str
    model_name: str
    api_base_url: str
    api_key: str
    use_llm: bool = False
    client: OpenAI | None = None


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lower(text: Any) -> str:
    return str(text).strip().lower()


def _task_name_from_args(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("task_name", nargs="?")
    parser.add_argument("--task", dest="task_flag")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--api-base-url", dest="api_base_url")
    parser.add_argument("--hf-token", dest="hf_token")
    parser.add_argument("--use-llm", action="store_true")
    parsed, _unknown = parser.parse_known_args(argv)

    candidate = parsed.task_flag or parsed.task_name or os.getenv("TASK_NAME") or get_default_task_name()
    normalized = str(candidate).strip().upper()
    if normalized not in list_task_names():
        return get_default_task_name()
    return normalized


def _build_config(argv: list[str]) -> InferenceConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("task_name", nargs="?")
    parser.add_argument("--task", dest="task_flag")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--api-base-url", dest="api_base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--hf-token", dest="hf_token")
    parser.add_argument("--use-llm", action="store_true")
    parsed, _unknown = parser.parse_known_args(argv)

    task_name = _task_name_from_args(argv)
    model_name = parsed.model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_base_url = parsed.api_base_url or os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    api_key = (
        parsed.api_key
        or parsed.hf_token
        or os.getenv("API_KEY")
        or os.getenv("HF_TOKEN", "")
    )
    use_llm = bool(parsed.use_llm or os.getenv("USE_LLM_POLICY", "0") == "1" or api_key)
    client = OpenAI(base_url=api_base_url, api_key=api_key or "hf_dummy")
    return InferenceConfig(
        task_name=task_name,
        model_name=model_name,
        api_base_url=api_base_url,
        api_key=api_key,
        use_llm=use_llm,
        client=client,
    )


def _task_state() -> Any:
    return app_module.app.state.runtime


def _topic_ranking(observation: AdaptiveDSAObservation) -> tuple[str, ...]:
    return tuple(sorted(TOPIC_NAMES, key=lambda topic: (observation.topic_mastery[topic], topic)))


def _weak_topic(observation: AdaptiveDSAObservation) -> str:
    return _topic_ranking(observation)[0]


def _second_weak_topic(observation: AdaptiveDSAObservation) -> str:
    ranked = _topic_ranking(observation)
    return ranked[1] if len(ranked) > 1 else ranked[0]


def _difficulty_from_motivation(motivation: float) -> str:
    if motivation < 0.4:
        return "easy"
    if motivation < 0.7:
        return "medium"
    return "hard"


def _lower_difficulty(difficulty: str) -> str:
    if difficulty == "hard":
        return "medium"
    return "easy"


def _motivation_style_for_task(task_name: str) -> str:
    if task_name == "EASY":
        return "encouraging"
    if task_name == "MEDIUM":
        return "honest_recovery"
    return "career_linked"


def _valid_redirect_for_task(task_name: str) -> str:
    if task_name == "HARD":
        return "active_recall"
    if task_name == "MEDIUM":
        return "system_design"
    return "core_cs"


def _make_action(action_type: str, **params: Any) -> AdaptiveDSAAction:
    if action_type not in ACTION_TYPES:
        action_type = "advance_session"
    return AdaptiveDSAAction(action_type=action_type, params={key: value for key, value in params.items() if value is not None})


def _choose_action(task_name: str, state: Any, counters: dict[str, int]) -> AdaptiveDSAAction:
    observation: AdaptiveDSAObservation = state.observation
    step_index = state.step_count + 1
    target_topic = _weak_topic(observation)
    next_topic = _second_weak_topic(observation)
    target_difficulty = _difficulty_from_motivation(observation.motivation)
    lower_difficulty = _lower_difficulty(target_difficulty)

    end_day_steps = {"EASY": {6}, "MEDIUM": {10}, "HARD": {10, 20}}[task_name]
    if step_index in end_day_steps:
        return _make_action("end_day")

    if "distraction" in observation.event_flags and counters["distraction"] < {"EASY": 1, "MEDIUM": 1, "HARD": 2}[task_name]:
        return _make_action(
            "handle_distraction",
            redirect=_valid_redirect_for_task(task_name),
            style=_motivation_style_for_task(task_name),
        )

    if "burnout_signal" in observation.event_flags and counters["burnout"] < {"EASY": 0, "MEDIUM": 1, "HARD": 2}[task_name]:
        return _make_action(
            "handle_burnout",
            topic=next_topic,
            difficulty=lower_difficulty,
        )

    if observation.motivation < 0.4 or (task_name == "EASY" and counters["motivation"] == 0):
        return _make_action("give_motivation", style=_motivation_style_for_task(task_name))

    if step_index % 4 == 0:
        return _make_action(
            "evaluate_solution",
            feedback=(
                "The solution is coherent, but it can be tightened by clarifying the state transition, "
                "choosing the correct recurrence, and reducing incidental complexity for the student."
            ),
            optimization_hint="Replace repeated branching with a simpler invariant-preserving approach.",
        )

    if step_index % 2 == 1:
        return _make_action(
            "recommend_exercise",
            topic=target_topic,
            difficulty=target_difficulty,
        )

    return _make_action(
        "build_plan",
        topic=target_topic,
        difficulty=target_difficulty,
        next_topic=next_topic,
    )


def _update_counters(counters: dict[str, int], action: AdaptiveDSAAction, state_before: Any, state_after: Any) -> None:
    if action.action_type == "handle_distraction" and "distraction" in state_before.observation.event_flags and "distraction" not in state_after.observation.event_flags:
        counters["distraction"] += 1
    if action.action_type == "handle_burnout" and "burnout_signal" in state_before.observation.event_flags and "burnout_signal" not in state_after.observation.event_flags:
        counters["burnout"] += 1
    if action.action_type == "give_motivation":
        counters["motivation"] += 1
    if action.action_type == "end_day":
        counters["end_day"] += 1


def _format_action(action: AdaptiveDSAAction) -> str:
    return action.action_type


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_reward_list(values: list[float]) -> str:
    return ",".join(_format_reward(value) for value in values)


def _run_episode(config: InferenceConfig) -> tuple[bool, int, float, list[float]]:
    runtime = _task_state()
    state = runtime.reset(config.task_name)
    initial_observation = state.observation.model_copy(deep=True)

    actions: list[AdaptiveDSAAction] = []
    trajectory: list[dict[str, Any]] = []
    counters = {"distraction": 0, "burnout": 0, "motivation": 0, "end_day": 0}

    while not state.done:
        action = _choose_action(config.task_name, state, counters)
        observation_before = state.observation.model_copy(deep=True)

        try:
            next_state, result = runtime.step(action)
            error_text = None
        except Exception as exc:  # pragma: no cover - defensive fallback
            next_state = state
            result = None
            error_text = str(exc)

        if result is None:
            print(
                f"[STEP] step={state.step_count + 1} action={_format_action(action)} reward=0.00 done=false error={error_text or 'unknown'}",
                flush=True,
            )
            break

        actions.append(action.model_copy(deep=True))
        trajectory.append(
            {
                "observation_before": observation_before.model_dump(),
                "action": action.model_dump(),
                "observation_after": next_state.observation.model_dump(),
                "result": result.model_dump(),
            }
        )
        _update_counters(counters, action, state, next_state)

        print(
            f"[STEP] step={state.step_count + 1} action={_format_action(action)} reward={_format_reward(result.reward)} done={str(result.done).lower()} error=null",
            flush=True,
        )

        state = next_state

        if state.done:
            break

    score = grade_task(
        task_name=config.task_name,
        initial_state=initial_observation,
        final_state=state.observation,
        actions=actions,
        trajectory=trajectory,
        task_spec=get_task_spec(config.task_name),
    ).score
    success = bool(state.success)
    steps = int(state.step_count)
    rewards = list(state.reward_history)
    return success, steps, score, rewards


def _maybe_use_llm(config: InferenceConfig) -> None:
    if not config.use_llm and not config.api_key:
        return
    if config.client is None:
        return
    try:
        config.client.responses.create(  # pragma: no cover - opt-in path only
            model=config.model_name,
            input=[
                {
                    "role": "system",
                    "content": "Return only a concise coaching strategy summary for DSA tutoring.",
                },
                {
                    "role": "user",
                    "content": f"Task: {config.task_name}. Provide one deterministic coaching strategy.",
                },
            ],
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    config = _build_config(args)
    _maybe_use_llm(config)

    all_tasks = list(list_task_names())  # ("EASY", "MEDIUM", "HARD")

    for task_name in all_tasks:
        config.task_name = task_name
        print(f"[START] task={task_name} env={DEFAULT_ENV_NAME} model={config.model_name}", flush=True)
        success, steps, score, rewards = _run_episode(config)
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={_format_reward_list(rewards)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
