"""Reward functions for RLGym."""

from .ball_rewards import SpeedTowardBall, TouchVelocity, VelocityBallToGoal
from .base import BaseReward
from .combined import CombinedReward
from .game_rewards import DemoReward, GoalReward, SaveReward
from .player_rewards import AerialHeight, BallProximity, InAir, OnGround, SaveBoost
from .team_rewards import PassingReward, TeamSpacing

__all__ = [
    "BaseReward",
    "CombinedReward",
    "TouchVelocity",
    "VelocityBallToGoal",
    "SpeedTowardBall",
    "SaveBoost",
    "InAir",
    "OnGround",
    "BallProximity",
    "AerialHeight",
    "GoalReward",
    "SaveReward",
    "DemoReward",
    "TeamSpacing",
    "PassingReward",
]
