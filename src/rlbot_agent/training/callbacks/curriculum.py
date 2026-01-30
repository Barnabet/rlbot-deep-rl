"""Curriculum learning callback for phase-based training."""

from typing import Any, Callable, Dict, List, Optional

from ...core.config import CurriculumConfig, CurriculumPhase


class CurriculumCallback:
    """Callback for curriculum learning with phase transitions.

    Manages training phases that control:
    - Which rewards are active
    - Team spirit coefficient
    - Team size (logged but not enforced - deferred feature)
    - Self-play flag for historical checkpoint usage
    """

    # Mapping from reward names in config to reward class names
    REWARD_NAME_MAP = {
        'touch_velocity': 'TouchVelocity',
        'velocity_ball_to_goal': 'VelocityBallToGoal',
        'speed_toward_ball': 'SpeedTowardBall',
        'goal': 'GoalReward',
        'save_boost': 'SaveBoost',
        'demo': 'DemoReward',
        'aerial_height': 'AerialHeight',
        'team_spacing_penalty': 'TeamSpacing',
    }

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
        reward_fn: Any,
        initial_weights: Optional[Dict[str, float]] = None,
        on_self_play_start: Optional[Callable[[int], None]] = None,
    ):
        """Initialize curriculum callback.

        Args:
            curriculum_config: Curriculum configuration with phases
            reward_fn: CombinedReward function to modify
            initial_weights: Initial reward weights from config
            on_self_play_start: Callback when self-play phase starts
        """
        self.config = curriculum_config
        self.reward_fn = reward_fn
        self.initial_weights = initial_weights or {}
        self.on_self_play_start = on_self_play_start

        self.current_phase_idx = -1  # -1 means not started
        self.current_phase: Optional[CurriculumPhase] = None
        self._self_play_started = False
        self._prev_team_size = None

    def get_phase_for_step(self, step: int) -> tuple:
        """Get the curriculum phase for a given step.

        Args:
            step: Current training step

        Returns:
            Tuple of (phase_index, phase) or (len(phases)-1, last_phase)
        """
        if not self.config.enabled or not self.config.phases:
            return -1, None

        for i, phase in enumerate(self.config.phases):
            if step < phase.end_step:
                return i, phase

        # Past all phases, stay on last one
        return len(self.config.phases) - 1, self.config.phases[-1]

    def on_step(
        self,
        step: int,
        metrics_dict: Dict[str, float],
    ) -> None:
        """Called after each training step.

        Args:
            step: Current global step
            metrics_dict: Dictionary to add curriculum metrics to
        """
        if not self.config.enabled or not self.config.phases:
            return

        new_idx, new_phase = self.get_phase_for_step(step)

        # Check for phase transition
        if new_idx != self.current_phase_idx and new_phase is not None:
            self._transition_to_phase(new_phase, step)
            self.current_phase_idx = new_idx
            self.current_phase = new_phase

        # Add curriculum metrics
        if self.current_phase is not None:
            metrics_dict['curriculum/phase_idx'] = float(self.current_phase_idx)
            metrics_dict['curriculum/phase_name'] = self.current_phase.name
            metrics_dict['curriculum/team_spirit'] = self.current_phase.team_spirit
            metrics_dict['curriculum/team_size'] = float(self.current_phase.team_size)
            metrics_dict['curriculum/use_historical_checkpoints'] = float(
                self.current_phase.use_historical_checkpoints
            )

            # Calculate progress within current phase
            if self.current_phase_idx > 0:
                prev_end = self.config.phases[self.current_phase_idx - 1].end_step
            else:
                prev_end = 0
            phase_start = prev_end
            phase_end = self.current_phase.end_step
            phase_progress = (step - phase_start) / (phase_end - phase_start) if phase_end > phase_start else 1.0
            metrics_dict['curriculum/phase_progress'] = min(1.0, phase_progress)

    def _transition_to_phase(self, phase: CurriculumPhase, step: int) -> None:
        """Handle transition to a new curriculum phase.

        Note: Reward weights are controlled by smooth interpolation in RewardConfig,
        not by curriculum phases. Phases are mainly for tracking, team spirit, and
        triggering self-play.

        Args:
            phase: New phase to transition to
            step: Current training step
        """
        print(f"\n{'='*60}")
        print(f"CURRICULUM: Entering phase '{phase.name}' at step {step:,}")
        print(f"{'='*60}")

        # NOTE: We no longer override reward weights here.
        # All rewards use smooth weight interpolation defined in RewardConfig.
        # The 'active rewards' in curriculum phases are informational only.

        # Update team spirit
        if hasattr(self.reward_fn, 'set_team_spirit'):
            self.reward_fn.set_team_spirit(phase.team_spirit)
            print(f"  Team spirit: {phase.team_spirit}")

        # Check for team size change (deferred - just warn)
        if self._prev_team_size is not None and phase.team_size != self._prev_team_size:
            print(f"  WARNING: Team size changed from {self._prev_team_size} to {phase.team_size}")
            print(f"           Dynamic team size changes are not yet implemented.")
            print(f"           Restart training with updated config to use new team size.")
        self._prev_team_size = phase.team_size

        # Check for self-play start
        if phase.use_historical_checkpoints and not self._self_play_started:
            print(f"  Self-play enabled with historical checkpoints")
            self._self_play_started = True
            if self.on_self_play_start is not None:
                self.on_self_play_start(step)

        print(f"  Phase focus: {phase.rewards} (all rewards active with smooth weights)")
        print(f"{'='*60}\n")

    def is_self_play_active(self) -> bool:
        """Check if self-play is currently active.

        Returns:
            True if current phase uses historical checkpoints
        """
        if self.current_phase is None:
            return False
        return self.current_phase.use_historical_checkpoints

    def get_current_team_size(self) -> int:
        """Get the team size for the current phase.

        Returns:
            Team size (1, 2, or 3)
        """
        if self.current_phase is None:
            return 1
        return self.current_phase.team_size

    def get_current_phase_name(self) -> str:
        """Get the name of the current phase.

        Returns:
            Phase name or 'none' if curriculum is disabled
        """
        if self.current_phase is None:
            return 'none'
        return self.current_phase.name
