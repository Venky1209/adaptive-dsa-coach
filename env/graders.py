from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    ACTION_TYPES,
    AdaptiveDSAAction,
    AdaptiveDSAEnvironmentState,
    AdaptiveDSAObservation,
    AdaptiveDSATaskSpec,
    TOPIC_NAMES,
    VALID_DISTRACTION_REDIRECTS,
)
from .tasks import get_task_spec


@dataclass(slots=True)
class GradingResult:
    task_name: str
    score: float
    breakdown: dict[str, float]


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _coerce_observation(value: Any) -> AdaptiveDSAObservation:
    if isinstance(value, AdaptiveDSAObservation):
        return value
    data = _as_mapping(value)
    if "observation" in data and not isinstance(data.get("topic_mastery"), dict):
        return _coerce_observation(data["observation"])
    return AdaptiveDSAObservation.model_validate(data)


def _coerce_action(value: Any) -> AdaptiveDSAAction:
    if isinstance(value, AdaptiveDSAAction):
        return value
    data = _as_mapping(value)
    if "action" in data and not isinstance(data.get("action_type"), str):
        return _coerce_action(data["action"])
    if "result" in data and not isinstance(data.get("action_type"), str):
        return _coerce_action(data["result"])
    return AdaptiveDSAAction.model_validate(data)


def _coerce_state(value: Any) -> AdaptiveDSAEnvironmentState:
    if isinstance(value, AdaptiveDSAEnvironmentState):
        return value
    data = _as_mapping(value)
    if "state" in data and not isinstance(data.get("observation"), dict):
        return _coerce_state(data["state"])
    if "result" in data and not isinstance(data.get("observation"), dict):
        return _coerce_state(data["result"])
    return AdaptiveDSAEnvironmentState.model_validate(data)


def _get_task_name(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    data = _as_mapping(value)
    task_name = data.get("task_name") or data.get("task")
    return str(task_name) if task_name is not None else None


def _record_before_observation(record: Any) -> AdaptiveDSAObservation | None:
    data = _as_mapping(record)
    for key in ("observation_before", "state_before", "before", "prev_observation", "pre_state"):
        if key in data and data[key] is not None:
            try:
                return _coerce_observation(data[key])
            except Exception:
                continue
    return None


def _record_after_observation(record: Any) -> AdaptiveDSAObservation | None:
    data = _as_mapping(record)
    for key in ("observation_after", "state_after", "after", "next_observation", "post_state"):
        if key in data and data[key] is not None:
            try:
                return _coerce_observation(data[key])
            except Exception:
                continue
    return None


def _record_action(record: Any) -> AdaptiveDSAAction | None:
    data = _as_mapping(record)
    for key in ("action", "step_action", "move"):
        if key in data and data[key] is not None:
            try:
                return _coerce_action(data[key])
            except Exception:
                continue
    if "action_type" in data:
        try:
            return _coerce_action(data)
        except Exception:
            return None
    return None


def _iter_records(*sources: Any) -> list[Any]:
    records: list[Any] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            records.extend(list(source))
        else:
            records.append(source)
    return records


def _extract_actions(*sources: Any) -> list[AdaptiveDSAAction]:
    actions: list[AdaptiveDSAAction] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            for item in source:
                try:
                    actions.append(_coerce_action(item))
                except Exception:
                    continue
        else:
            try:
                actions.append(_coerce_action(source))
            except Exception:
                continue
    return actions


def _extract_task_spec(task_name: str | None, task_spec: Any | None = None) -> AdaptiveDSATaskSpec:
    if task_spec is not None:
        if isinstance(task_spec, AdaptiveDSATaskSpec):
            return task_spec
        return AdaptiveDSATaskSpec.model_validate(_as_mapping(task_spec))
    if task_name is None:
        raise ValueError("task_name is required for grading")
    return get_task_spec(task_name)


def _bottom_topics(initial_state: AdaptiveDSAObservation, count: int = 2) -> tuple[str, ...]:
    ranked_topics = sorted(TOPIC_NAMES, key=lambda topic: (initial_state.topic_mastery[topic], topic))
    return tuple(ranked_topics[:count])


def _topic_from_action(action: AdaptiveDSAAction) -> str | None:
    for key in ("topic", "topic_name", "exercise_topic", "target_topic", "focus_topic"):
        value = action.params.get(key)
        if isinstance(value, str):
            return value
    focus_topics = action.params.get("focus_topics")
    if isinstance(focus_topics, Sequence) and not isinstance(focus_topics, (str, bytes, bytearray)):
        for item in focus_topics:
            if isinstance(item, str):
                return item
    return None


def _redirect_from_action(action: AdaptiveDSAAction) -> str | None:
    for key in ("redirect", "redirect_to", "destination", "fallback"):
        value = action.params.get(key)
        if isinstance(value, str):
            return value
    return None


def _motivation_style_from_action(action: AdaptiveDSAAction) -> str | None:
    value = action.params.get("style") or action.params.get("motivation_style")
    return value if isinstance(value, str) else None


def _feedback_text_from_action(action: AdaptiveDSAAction) -> str:
    for key in ("feedback", "notes", "message", "hint", "solution_feedback"):
        value = action.params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _trajectory_before_states(trajectory: Sequence[Any]) -> list[AdaptiveDSAObservation]:
    before_states: list[AdaptiveDSAObservation] = []
    for record in trajectory:
        observation = _record_before_observation(record)
        if observation is not None:
            before_states.append(observation)
    return before_states


def _trajectory_after_states(trajectory: Sequence[Any]) -> list[AdaptiveDSAObservation]:
    after_states: list[AdaptiveDSAObservation] = []
    for record in trajectory:
        observation = _record_after_observation(record)
        if observation is not None:
            after_states.append(observation)
    return after_states


def _criterion_targeted_weak_topic(initial_state: AdaptiveDSAObservation, actions: Sequence[AdaptiveDSAAction]) -> bool:
    weak_topics = set(_bottom_topics(initial_state, count=2))
    for action in actions:
        if action.action_type not in {"build_plan", "recommend_exercise"}:
            continue
        topic = _topic_from_action(action)
        if topic in weak_topics:
            return True
    return False


def _criterion_distraction_handled_productively(
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any],
) -> bool:
    valid_redirect_used = False
    for action in actions:
        if action.action_type != "handle_distraction":
            continue
        redirect = _redirect_from_action(action)
        if redirect in VALID_DISTRACTION_REDIRECTS:
            valid_redirect_used = True
            break

    distraction_removed = "distraction" not in final_state.event_flags
    if not valid_redirect_used and trajectory:
        for record in trajectory:
            action = _record_action(record)
            if action is None or action.action_type != "handle_distraction":
                continue
            redirect = _redirect_from_action(action)
            after_observation = _record_after_observation(record)
            if redirect in VALID_DISTRACTION_REDIRECTS and after_observation is not None:
                if "distraction" not in after_observation.event_flags:
                    valid_redirect_used = True
                    distraction_removed = True
                    break

    return valid_redirect_used and distraction_removed


def _criterion_evaluate_or_hint_used(actions: Sequence[AdaptiveDSAAction]) -> bool:
    return any(action.action_type in {"evaluate_solution", "give_hint"} for action in actions)


def _criterion_final_motivation_at_least(final_state: AdaptiveDSAObservation, minimum: float) -> bool:
    return final_state.motivation >= minimum


def _criterion_handle_burnout_called(actions: Sequence[AdaptiveDSAAction]) -> bool:
    return any(action.action_type == "handle_burnout" for action in actions)


def _criterion_give_motivation_called_when_low(
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    before_states: Sequence[AdaptiveDSAObservation],
) -> bool:
    if before_states:
        for action, before_state in zip(actions, before_states, strict=False):
            if action.action_type == "give_motivation" and before_state.motivation < 0.4:
                return True
        return False

    return any(action.action_type == "give_motivation" and final_state.motivation < 0.4 for action in actions)


def _criterion_final_burnout_under(final_state: AdaptiveDSAObservation, threshold: float) -> bool:
    return final_state.burnout_risk < threshold


def _criterion_final_motivation_over(final_state: AdaptiveDSAObservation, threshold: float) -> bool:
    return final_state.motivation > threshold


def _criterion_end_day_count(actions: Sequence[AdaptiveDSAAction]) -> int:
    return sum(1 for action in actions if action.action_type == "end_day")


def _criterion_productive_distraction_count(actions: Sequence[AdaptiveDSAAction], trajectory: Sequence[Any]) -> int:
    count = 0
    for action in actions:
        if action.action_type != "handle_distraction":
            continue
        if _redirect_from_action(action) in VALID_DISTRACTION_REDIRECTS:
            count += 1

    if count == 0 and trajectory:
        for record in trajectory:
            action = _record_action(record)
            if action is None or action.action_type != "handle_distraction":
                continue
            if _redirect_from_action(action) in VALID_DISTRACTION_REDIRECTS:
                after_observation = _record_after_observation(record)
                if after_observation is None or "distraction" not in after_observation.event_flags:
                    count += 1
    return count


def _criterion_handle_burnout_count(actions: Sequence[AdaptiveDSAAction]) -> int:
    return sum(1 for action in actions if action.action_type == "handle_burnout")


def _criterion_topics_improved(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    threshold: float = 0.0,
) -> int:
    improved = 0
    for topic in TOPIC_NAMES:
        if final_state.topic_mastery[topic] - initial_state.topic_mastery[topic] > threshold:
            improved += 1
    return improved


def _criterion_final_state_combo(final_state: AdaptiveDSAObservation) -> bool:
    return final_state.motivation > 0.5 and final_state.burnout_risk < 0.5


def _weighted(binary_value: bool, weight: float) -> float:
    return weight if binary_value else 0.0


def _weighted_ratio(count: int, threshold: int, weight: float) -> float:
    if threshold <= 0:
        return weight
    return weight * min(1.0, max(0.0, count / threshold))


def _score_easy(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any],
) -> GradingResult:
    components = {
        "weak_topic_targeted": _weighted(_criterion_targeted_weak_topic(initial_state, actions), 0.25),
        "distraction_handled_productively": _weighted(
            _criterion_distraction_handled_productively(final_state, actions, trajectory),
            0.25,
        ),
        "evaluate_or_hint_used": _weighted(_criterion_evaluate_or_hint_used(actions), 0.25),
        "final_motivation": _weighted(_criterion_final_motivation_at_least(final_state, 0.5), 0.25),
    }
    score = _clamp_score(sum(components.values()))
    return GradingResult(task_name="EASY", score=score, breakdown=components)


def _score_medium(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any],
    before_states: Sequence[AdaptiveDSAObservation],
) -> GradingResult:
    components = {
        "handle_burnout_called": _weighted(_criterion_handle_burnout_called(actions), 0.2),
        "motivation_given_when_low": _weighted(
            _criterion_give_motivation_called_when_low(final_state, actions, before_states),
            0.2,
        ),
        "weak_topic_targeted": _weighted(_criterion_targeted_weak_topic(initial_state, actions), 0.2),
        "final_burnout_under_threshold": _weighted(_criterion_final_burnout_under(final_state, 0.5), 0.2),
        "final_motivation_over_threshold": _weighted(_criterion_final_motivation_over(final_state, 0.45), 0.2),
    }
    score = _clamp_score(sum(components.values()))
    return GradingResult(task_name="MEDIUM", score=score, breakdown=components)


def _score_hard(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any],
) -> GradingResult:
    end_day_count = _criterion_end_day_count(actions)
    productive_distraction_count = _criterion_productive_distraction_count(actions, trajectory)
    burnout_count = _criterion_handle_burnout_count(actions)
    topics_improved = _criterion_topics_improved(initial_state, final_state)

    components = {
        "end_day_twice": _weighted_ratio(end_day_count, 2, 0.15),
        "productive_distraction_handled_twice": _weighted_ratio(productive_distraction_count, 2, 0.15),
        "handle_burnout_twice": _weighted_ratio(burnout_count, 2, 0.15),
        "three_topics_improved": _weighted_ratio(topics_improved, 3, 0.15),
        "streak_improved": _weighted(final_state.streak > initial_state.streak, 0.20),
        "strong_finish": _weighted(_criterion_final_state_combo(final_state), 0.20),
    }
    score = _clamp_score(sum(components.values()))
    return GradingResult(task_name="HARD", score=score, breakdown=components)


def _normalize_grading_inputs(*args: Any, **kwargs: Any) -> tuple[str, AdaptiveDSAObservation, AdaptiveDSAObservation, list[AdaptiveDSAAction], list[Any], AdaptiveDSATaskSpec]:
    task_name = _get_task_name(kwargs.pop("task_name", None))
    task_name = task_name or _get_task_name(kwargs.pop("task", None))

    task_spec = kwargs.pop("task_spec", None)
    initial_state = kwargs.pop("initial_state", None)
    final_state = kwargs.pop("final_state", None)
    state = kwargs.pop("state", None)
    environment_state = kwargs.pop("environment_state", None)
    actions = kwargs.pop("actions", None)
    trajectory = kwargs.pop("trajectory", None)
    if trajectory is None:
        trajectory = kwargs.pop("history", None) or kwargs.pop("transitions", None) or kwargs.pop("step_results", None)

    if args:
        if task_name is None:
            task_name = _get_task_name(args[0])
        if len(args) > 1 and final_state is None and state is None and environment_state is None:
            final_state = args[1]
        if len(args) > 2 and initial_state is None:
            initial_state = args[2]
        if len(args) > 3 and actions is None:
            actions = args[3]
        if len(args) > 4 and trajectory is None:
            trajectory = args[4]

    if final_state is None:
        final_state = state if state is not None else environment_state

    if final_state is not None and not _as_mapping(final_state):
        final_state = None

    spec = _extract_task_spec(task_name, task_spec)

    final_observation = _coerce_observation(final_state if final_state is not None else spec.initial_state)
    initial_observation = _coerce_observation(initial_state if initial_state is not None else spec.initial_state)

    action_list = _extract_actions(actions, getattr(final_state, "action_history", None), _as_mapping(final_state).get("action_history"))
    trajectory_list = _iter_records(trajectory)

    if not action_list and trajectory_list:
        for record in trajectory_list:
            action = _record_action(record)
            if action is not None:
                action_list.append(action)

    return spec.task_name, initial_observation, final_observation, action_list, trajectory_list, spec


def grade_easy(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any] | None = None,
) -> GradingResult:
    return _score_easy(initial_state, final_state, actions, trajectory or [])


def grade_medium(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any] | None = None,
    before_states: Sequence[AdaptiveDSAObservation] | None = None,
) -> GradingResult:
    return _score_medium(initial_state, final_state, actions, trajectory or [], before_states or [])


def grade_hard(
    initial_state: AdaptiveDSAObservation,
    final_state: AdaptiveDSAObservation,
    actions: Sequence[AdaptiveDSAAction],
    trajectory: Sequence[Any] | None = None,
) -> GradingResult:
    return _score_hard(initial_state, final_state, actions, trajectory or [])


def grade_task(*args: Any, **kwargs: Any) -> GradingResult:
    task_name, initial_state, final_state, actions, trajectory, _task_spec = _normalize_grading_inputs(*args, **kwargs)
    before_states = _trajectory_before_states(trajectory)

    if task_name == "EASY":
        return grade_easy(initial_state, final_state, actions, trajectory)
    if task_name == "MEDIUM":
        return grade_medium(initial_state, final_state, actions, trajectory, before_states)
    if task_name == "HARD":
        return grade_hard(initial_state, final_state, actions, trajectory)
    raise ValueError(f"unsupported task: {task_name}")


def build_grader(task_name: str):
    def _grader(*args: Any, **kwargs: Any) -> float:
        result = grade_task(task_name=task_name, *args, **kwargs)
        return result.score

    return _grader


def grade(*args: Any, **kwargs: Any) -> float:
    return grade_task(*args, **kwargs).score


def grade_episode(task_name: str, trajectory: Sequence[Any] | None = None, final_state: Any | None = None, **kwargs: Any) -> float:
    normalized_task_name = str(task_name).strip().upper()
    if normalized_task_name not in {"EASY", "MEDIUM", "HARD"}:
        normalized_task_name = get_task_spec("EASY").task_name

    trajectory_list = list(trajectory or [])
    if final_state is None and trajectory_list:
        final_state = trajectory_list[-1]

    if final_state is None:
        final_state = get_task_spec(normalized_task_name).initial_state

    return grade_task(
        task_name=normalized_task_name,
        final_state=final_state,
        trajectory=trajectory_list,
        **kwargs,
    ).score


TASK_GRADERS = {
    "EASY": build_grader("EASY"),
    "MEDIUM": build_grader("MEDIUM"),
    "HARD": build_grader("HARD"),
}


def get_task_grader(task_name: str):
    normalized_task_name = str(task_name).strip().upper()
    if normalized_task_name not in TASK_GRADERS:
        raise KeyError(f"unknown task: {task_name}")
    return TASK_GRADERS[normalized_task_name]


def score_breakdown(*args: Any, **kwargs: Any) -> dict[str, float]:
    return grade_task(*args, **kwargs).breakdown


__all__ = [
    "GradingResult",
    "build_grader",
    "grade",
    "grade_episode",
    "grade_easy",
    "grade_hard",
    "grade_medium",
    "grade_task",
    "get_task_grader",
    "TASK_GRADERS",
    "score_breakdown",
]
