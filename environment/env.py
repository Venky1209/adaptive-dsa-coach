"""Compatibility wrapper for the OpenEnv-style package layout."""

from env.environment import AdaptiveDSACoachEnv
from env.models import Action, Observation, State

Environment = AdaptiveDSACoachEnv

__all__ = ["Action", "AdaptiveDSACoachEnv", "Environment", "Observation", "State"]