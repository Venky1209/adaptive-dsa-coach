from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


TOPIC_NAMES: tuple[str, ...] = (
    "arrays",
    "strings",
    "dp",
    "graphs",
    "trees",
    "system_design",
)

VALID_DISTRACTION_REDIRECTS: tuple[str, ...] = (
    "core_cs",
    "lighter_practice",
    "system_design",
    "project_ideation",
    "active_recall",
)

VALID_MOTIVATION_STYLES: tuple[str, ...] = (
    "encouraging",
    "honest_recovery",
    "career_linked",
)

VALID_CAREER_CONTEXTS: tuple[str, ...] = (
    "normal",
    "placement_season",
    "hackathon_deadline",
)

VALID_EVENT_FLAGS: tuple[str, ...] = (
    "distraction",
    "motivation_drop",
    "burnout_signal",
)

ACTION_TYPES: tuple[str, ...] = (
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

TASK_NAMES: tuple[str, ...] = ("EASY", "MEDIUM", "HARD")


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


class AdaptiveDSAObservation(BaseSchema):
    topic_mastery: dict[str, float] = Field(default_factory=dict)
    motivation: float = Field(ge=0.0, le=1.0)
    burnout_risk: float = Field(ge=0.0, le=1.0)
    streak: int
    daily_time_left: int
    current_topic: str
    current_problem_id: str
    time_on_problem: int
    recent_accuracy: float = Field(ge=0.0, le=1.0)
    recent_speed: float
    event_flags: list[str] = Field(default_factory=list)
    career_context: Literal["normal", "placement_season", "hackathon_deadline"]

    @field_validator("topic_mastery")
    @classmethod
    def validate_topic_mastery(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for topic in TOPIC_NAMES:
            if topic not in value:
                raise ValueError(f"missing topic mastery for {topic}")
            normalized[topic] = _clamp_unit_interval(value[topic])

        extra_topics = sorted(set(value) - set(TOPIC_NAMES))
        if extra_topics:
            raise ValueError(f"unexpected topic mastery keys: {extra_topics}")
        return normalized

    @field_validator("event_flags")
    @classmethod
    def validate_event_flags(cls, value: list[str]) -> list[str]:
        invalid_flags = sorted(set(value) - set(VALID_EVENT_FLAGS))
        if invalid_flags:
            raise ValueError(f"unexpected event flags: {invalid_flags}")
        return list(dict.fromkeys(value))

    @field_validator("motivation", "burnout_risk", "recent_accuracy")
    @classmethod
    def validate_unit_interval_fields(cls, value: float) -> float:
        return _clamp_unit_interval(value)


class AdaptiveDSAAction(BaseSchema):
    action_type: Literal[
        "build_plan",
        "recommend_exercise",
        "evaluate_solution",
        "give_hint",
        "handle_distraction",
        "handle_burnout",
        "give_motivation",
        "advance_session",
        "end_day",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class AdaptiveDSATaskSpec(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]
    max_steps: int = Field(ge=1)
    success_threshold: float = Field(ge=0.0, le=1.0)
    initial_state: AdaptiveDSAObservation
    grader: str = ""
    description: str = ""

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()


class AdaptiveDSAEnvironmentState(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    success_threshold: float = Field(ge=0.0, le=1.0)
    observation: AdaptiveDSAObservation
    reward_history: list[float] = Field(default_factory=list)
    action_history: list[AdaptiveDSAAction] = Field(default_factory=list)
    done: bool = False
    success: bool = False

    @model_validator(mode="after")
    def validate_reward_history(self) -> "AdaptiveDSAEnvironmentState":
        self.reward_history = [_clamp_reward(reward) for reward in self.reward_history]
        return self


class AdaptiveDSAStepResult(BaseSchema):
    observation: AdaptiveDSAObservation
    reward: float = Field(ge=-1.0, le=1.0)
    done: bool
    success: bool
    info: dict[str, Any] = Field(default_factory=dict)


class AdaptiveDSAResetPayload(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]


class AdaptiveDSAResetResponse(BaseSchema):
    state: AdaptiveDSAEnvironmentState


class AdaptiveDSAStepPayload(BaseSchema):
    action: AdaptiveDSAAction


class AdaptiveDSAStepResponse(BaseSchema):
    state: AdaptiveDSAEnvironmentState
    result: AdaptiveDSAStepResult


class AdaptiveDSAInferRequest(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]
    model_name: str


class AdaptiveDSAInferResponse(BaseSchema):
    success: bool
    steps: int
    score: float = Field(ge=0.0, le=1.0)
    rewards: list[float] = Field(default_factory=list)


Action = AdaptiveDSAAction
Observation = AdaptiveDSAObservation
State = AdaptiveDSAEnvironmentState


__all__ = [
    "ACTION_TYPES",
    "Action",
    "AdaptiveDSAAction",
    "AdaptiveDSAEnvironmentState",
    "AdaptiveDSAInferRequest",
    "AdaptiveDSAInferResponse",
    "AdaptiveDSAObservation",
    "AdaptiveDSAResetPayload",
    "AdaptiveDSAResetResponse",
    "AdaptiveDSAStepPayload",
    "AdaptiveDSAStepResponse",
    "AdaptiveDSAStepResult",
    "AdaptiveDSATaskSpec",
    "BaseSchema",
    "Observation",
    "State",
    "TASK_NAMES",
    "TOPIC_NAMES",
    "VALID_CAREER_CONTEXTS",
    "VALID_DISTRACTION_REDIRECTS",
    "VALID_EVENT_FLAGS",
    "VALID_MOTIVATION_STYLES",
]
