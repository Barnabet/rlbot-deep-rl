"""Environment statistics tracker for monitoring agent capabilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    # Scoring
    goals_scored: int = 0
    goals_conceded: int = 0

    # Ball interaction
    ball_touches: int = 0
    touch_velocities: List[float] = field(default_factory=list)

    # Movement
    total_speed: float = 0.0
    speed_samples: int = 0
    supersonic_steps: int = 0

    # Aerial
    air_steps: int = 0
    max_height: float = 0.0

    # Boost
    boost_collected: float = 0.0
    boost_used: float = 0.0
    avg_boost: float = 0.0
    boost_samples: int = 0

    # Positioning
    ball_distance_sum: float = 0.0
    facing_ball_sum: float = 0.0  # dot product of forward and ball direction
    position_samples: int = 0

    # Combat
    demos_inflicted: int = 0
    demos_received: int = 0

    # Episode info
    episode_length: int = 0
    episode_reward: float = 0.0

    # Per-reward component sums (for breakdown logging)
    reward_components: Dict[str, float] = field(default_factory=dict)

    # Previous state for delta calculations
    _prev_ball_touches: int = 0
    _prev_boost: float = 33.0
    _prev_ball_vel: Optional[np.ndarray] = None


class EnvStatsTracker:
    """Tracks environment statistics across episodes and rollouts."""

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.episode_stats = [EpisodeStats() for _ in range(n_envs)]
        self.completed_episodes: List[EpisodeStats] = []
        self.rollout_step = 0

        # Action tracking for the rollout
        self.action_counts = np.zeros(1944, dtype=np.int64)
        self.total_actions = 0

    def reset(self):
        """Reset for new rollout."""
        self.completed_episodes = []
        self.rollout_step = 0
        self.action_counts = np.zeros(1944, dtype=np.int64)
        self.total_actions = 0
        self.multi_discrete_counts = None  # Reset multi-discrete tracking

    def track_actions(self, actions: np.ndarray):
        """Track action distribution.

        Args:
            actions: Array of action indices - either flat [n_envs] or multi-discrete [n_envs, 8]
        """
        actions = np.asarray(actions)
        if actions.ndim == 2 and actions.shape[-1] == 8:
            # Multi-discrete: [n_envs, 8] - track per-head stats
            if self.multi_discrete_counts is None:
                # Initialize per-head counts: [8 heads, max_options]
                self.multi_discrete_counts = [np.zeros(3 if i < 5 else 2, dtype=np.int64) for i in range(8)]
            for env_actions in actions:
                for head_idx, action in enumerate(env_actions):
                    self.multi_discrete_counts[head_idx][int(action)] += 1
            self.total_actions += len(actions)
        else:
            # Flat: [n_envs] - use action_counts array
            for a in actions:
                self.action_counts[int(a)] += 1
            self.total_actions += len(actions)

    def reset_env(self, env_idx: int):
        """Reset stats for a single environment (on episode end)."""
        # Save completed episode
        if self.episode_stats[env_idx].episode_length > 0:
            self.completed_episodes.append(self.episode_stats[env_idx])
        # Start fresh episode
        self.episode_stats[env_idx] = EpisodeStats()

    def update(self, env_idx: int, state: Any, player: Any, reward: float,
               reward_breakdown: Optional[Dict[str, float]] = None,
               info: Optional[Dict[str, Any]] = None):
        """Update statistics from a step.

        Only tracks detailed stats for blue team (team_num=0) to avoid
        metrics canceling out in self-play (e.g., goals scored = goals conceded).
        All players still contribute to training data.

        Args:
            env_idx: Environment index
            state: Game state (StateAdapter)
            player: Player data (PlayerAdapter)
            reward: Step reward
            reward_breakdown: Optional per-reward component values
            info: Optional info dict from environment (for goal tracking)
        """
        # Only track detailed metrics for blue team (team_num=0)
        # Orange team data is still used for training, just not for metrics
        is_blue_team = hasattr(player, 'team_num') and player.team_num == 0
        if not is_blue_team:
            return

        stats = self.episode_stats[env_idx]
        stats.episode_length += 1
        stats.episode_reward += reward

        # Track per-reward components (only sum raw/weighted, store weights once)
        if reward_breakdown:
            for key, value in reward_breakdown.items():
                if '/weight' in key:
                    # Store weight once, don't sum
                    stats.reward_components[key] = value
                else:
                    # Sum raw and weighted rewards
                    if key not in stats.reward_components:
                        stats.reward_components[key] = 0.0
                    stats.reward_components[key] += value

        # === Scoring ===
        # Method 1: From FixedEpisodeLengthWrapper info
        # Method 2: From reward breakdown (GoalReward/raw)
        # Method 3: From state.goal_scored attribute
        goal_detected = False

        if info and info.get('goal_scored_this_step'):
            goal_detected = True
            # Use reward breakdown to determine scorer
            goal_reward = reward_breakdown.get('GoalReward/raw', 0) if reward_breakdown else 0
            if goal_reward > 0:
                stats.goals_scored += 1
            elif goal_reward < 0:
                stats.goals_conceded += 1

        if not goal_detected and reward_breakdown:
            # Check reward breakdown directly
            goal_reward = reward_breakdown.get('GoalReward/raw', 0)
            if goal_reward > 0:
                stats.goals_scored += 1
                goal_detected = True
            elif goal_reward < 0:
                stats.goals_conceded += 1
                goal_detected = True

        try:
            car = player.car_data
            ball = state.ball

            # Fallback: check state directly
            if not goal_detected and hasattr(state, 'goal_scored') and state.goal_scored:
                scoring_team = getattr(state, 'scoring_team', None)
                if scoring_team == player.team_num:
                    stats.goals_scored += 1
                else:
                    stats.goals_conceded += 1

            # === Ball touches ===
            current_touches = player.ball_touches
            if current_touches > stats._prev_ball_touches:
                new_touches = current_touches - stats._prev_ball_touches
                stats.ball_touches += new_touches

                # Track touch velocity
                if stats._prev_ball_vel is not None:
                    ball_vel = np.array(ball.linear_velocity)
                    delta_vel = np.linalg.norm(ball_vel - stats._prev_ball_vel)
                    stats.touch_velocities.append(delta_vel)

            stats._prev_ball_touches = current_touches
            stats._prev_ball_vel = np.array(ball.linear_velocity).copy()

            # === Movement ===
            car_vel = np.array(car.linear_velocity)
            speed = np.linalg.norm(car_vel)
            stats.total_speed += speed
            stats.speed_samples += 1

            # Supersonic (speed > 2200)
            if speed > 2200:
                stats.supersonic_steps += 1

            # === Aerial ===
            if not player.on_ground:
                stats.air_steps += 1

            car_height = car.position[2]
            if car_height > stats.max_height:
                stats.max_height = car_height

            # === Boost ===
            current_boost = player.boost_amount
            stats.avg_boost += current_boost
            stats.boost_samples += 1

            # Track boost usage
            if current_boost < stats._prev_boost:
                stats.boost_used += stats._prev_boost - current_boost
            elif current_boost > stats._prev_boost:
                stats.boost_collected += current_boost - stats._prev_boost
            stats._prev_boost = current_boost

            # === Positioning ===
            car_pos = np.array(car.position)
            ball_pos = np.array(ball.position)

            # Distance to ball
            ball_dist = np.linalg.norm(ball_pos - car_pos)
            stats.ball_distance_sum += ball_dist

            # Facing ball (dot product)
            to_ball = ball_pos - car_pos
            to_ball_norm = np.linalg.norm(to_ball)
            if to_ball_norm > 1e-6:
                to_ball_dir = to_ball / to_ball_norm
                forward = car.forward if hasattr(car, 'forward') and not callable(car.forward) else car.forward()
                facing_dot = np.dot(forward, to_ball_dir)
                stats.facing_ball_sum += facing_dot

            stats.position_samples += 1

            # === Combat ===
            if player.is_demoed:
                # Check if this is a new demo (respawn timer just started)
                if hasattr(stats, '_was_demoed') and not stats._was_demoed:
                    stats.demos_received += 1
                stats._was_demoed = True
            else:
                stats._was_demoed = False

        except Exception as e:
            # Don't crash training if stats collection fails
            pass

        self.rollout_step += 1

    def get_rollout_stats(self) -> Dict[str, float]:
        """Get aggregated statistics for the rollout."""
        # Use completed episodes for episode-level stats (length, reward)
        # Use all episodes (including ongoing) for per-step stats (air%, boost, etc.)
        all_episodes = self.completed_episodes + [s for s in self.episode_stats if s.episode_length > 0]

        if not all_episodes:
            return {}

        stats = {}

        # Episode-level stats - ONLY from COMPLETED episodes
        # Don't fall back to partial episodes - that causes metric spikes
        stats['env/episodes_completed'] = len(self.completed_episodes)
        if self.completed_episodes:
            stats['env/avg_episode_length'] = np.mean([e.episode_length for e in self.completed_episodes])
            stats['env/avg_episode_reward'] = np.mean([e.episode_reward for e in self.completed_episodes])
        # If no completed episodes, don't report these metrics (they'll be missing from this log)

        # Scoring
        total_goals = sum(e.goals_scored for e in all_episodes)
        total_conceded = sum(e.goals_conceded for e in all_episodes)
        stats['env/goals_scored'] = total_goals
        stats['env/goals_conceded'] = total_conceded
        stats['env/goal_diff'] = total_goals - total_conceded
        if len(self.completed_episodes) > 0:
            stats['env/goals_per_episode'] = total_goals / len(self.completed_episodes)

        # Ball touches
        total_touches = sum(e.ball_touches for e in all_episodes)
        stats['env/ball_touches'] = total_touches
        stats['env/touches_per_episode'] = total_touches / len(all_episodes) if all_episodes else 0

        # Touch velocity
        all_touch_vels = []
        for e in all_episodes:
            all_touch_vels.extend(e.touch_velocities)
        if all_touch_vels:
            stats['env/avg_touch_velocity'] = np.mean(all_touch_vels)
            stats['env/max_touch_velocity'] = np.max(all_touch_vels)

        # Movement
        total_speed = sum(e.total_speed for e in all_episodes)
        total_samples = sum(e.speed_samples for e in all_episodes)
        if total_samples > 0:
            stats['env/avg_speed'] = total_speed / total_samples

        supersonic_steps = sum(e.supersonic_steps for e in all_episodes)
        if total_samples > 0:
            stats['env/supersonic_pct'] = 100 * supersonic_steps / total_samples

        # Aerial
        air_steps = sum(e.air_steps for e in all_episodes)
        if total_samples > 0:
            stats['env/air_pct'] = 100 * air_steps / total_samples
        stats['env/max_height'] = max(e.max_height for e in all_episodes) if all_episodes else 0

        # Boost
        total_boost = sum(e.avg_boost for e in all_episodes)
        boost_samples = sum(e.boost_samples for e in all_episodes)
        if boost_samples > 0:
            stats['env/avg_boost'] = total_boost / boost_samples
        stats['env/boost_used'] = sum(e.boost_used for e in all_episodes)
        stats['env/boost_collected'] = sum(e.boost_collected for e in all_episodes)

        # Positioning
        total_ball_dist = sum(e.ball_distance_sum for e in all_episodes)
        total_facing = sum(e.facing_ball_sum for e in all_episodes)
        pos_samples = sum(e.position_samples for e in all_episodes)
        if pos_samples > 0:
            stats['env/avg_ball_distance'] = total_ball_dist / pos_samples
            stats['env/avg_facing_ball'] = total_facing / pos_samples  # -1 to 1
            stats['env/facing_ball_pct'] = 100 * (total_facing / pos_samples + 1) / 2  # 0-100%

        # Combat
        stats['env/demos_inflicted'] = sum(e.demos_inflicted for e in all_episodes)
        stats['env/demos_received'] = sum(e.demos_received for e in all_episodes)

        # Per-reward component breakdown with percentage of total
        # Use same episode set as avg_episode_reward for consistency
        # Only compute reward breakdown if we have completed episodes
        # (to avoid spikes from partial episode data)
        if not self.completed_episodes:
            return stats

        episodes_for_rewards = self.completed_episodes

        # Collect unique reward names (without /raw, /weighted, /weight suffix)
        reward_names = set()
        for e in episodes_for_rewards:
            for key in e.reward_components.keys():
                # Extract base name (e.g., "TouchVelocity" from "TouchVelocity/raw")
                base_name = key.split('/')[0]
                reward_names.add(base_name)

        # Calculate total weighted reward for percentage calculation
        total_weighted_reward = stats.get('env/avg_episode_reward', 0.0)

        # Total steps for per-step metrics (more representative of learning signal)
        total_steps = sum(e.episode_length for e in episodes_for_rewards)

        # First pass: compute per-step weighted contributions for each reward
        per_step_contributions = {}

        for name in reward_names:
            # Average raw reward per episode (unweighted)
            raw_key = f'{name}/raw'
            raw_total = sum(e.reward_components.get(raw_key, 0.0) for e in episodes_for_rewards)
            raw_avg = raw_total / len(episodes_for_rewards) if episodes_for_rewards else 0
            stats[f'rewards/{name}/raw'] = raw_avg

            # Weight (should be same across episodes, take from first)
            weight_key = f'{name}/weight'
            weight = 0.0
            for e in episodes_for_rewards:
                if weight_key in e.reward_components:
                    weight = e.reward_components[weight_key]
                    break
            stats[f'rewards/{name}/weight'] = weight

            # Compute weighted contribution: raw * weight
            weighted_contribution = raw_avg * weight

            # Percentage of total episode reward (sum-based, can be misleading)
            if abs(total_weighted_reward) > 1e-6:
                pct = 100 * weighted_contribution / total_weighted_reward
            else:
                pct = 0.0
            stats[f'rewards/{name}/pct'] = pct

            # Per-step weighted reward (better represents learning signal magnitude)
            # This shows the actual per-step signal the model experiences
            weighted_total = sum(e.reward_components.get(raw_key, 0.0) for e in episodes_for_rewards) * weight
            per_step = weighted_total / total_steps if total_steps > 0 else 0
            stats[f'rewards/{name}/per_step'] = per_step
            per_step_contributions[name] = abs(per_step)  # Use abs for magnitude comparison

        # Per-step percentage (better metric for comparing reward importance)
        # This shows what % of the per-step learning signal comes from each reward
        total_per_step_magnitude = sum(per_step_contributions.values())
        for name in reward_names:
            if total_per_step_magnitude > 1e-8:
                step_pct = 100 * per_step_contributions[name] / total_per_step_magnitude
            else:
                step_pct = 0.0
            stats[f'rewards/{name}/step_pct'] = step_pct

        # Action distribution stats
        if self.total_actions > 0:
            if hasattr(self, 'multi_discrete_counts') and self.multi_discrete_counts is not None:
                # Multi-discrete: direct per-head stats
                head_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
                total_entropy = 0
                max_entropy = 0

                for head_idx, (name, counts) in enumerate(zip(head_names, self.multi_discrete_counts)):
                    total = counts.sum()
                    if total > 0:
                        probs = counts / total
                        # For binary heads (jump, boost, handbrake), log prob of action=1
                        if len(counts) == 2:
                            stats[f'inputs/{name}_pct'] = 100 * probs[1]
                        else:
                            # For 3-way heads, log prob of positive action (index 2 = +1)
                            stats[f'inputs/{name}_pos_pct'] = 100 * probs[2]

                        # Per-head entropy
                        nonzero = probs[probs > 0]
                        head_entropy = -np.sum(nonzero * np.log(nonzero))
                        total_entropy += head_entropy
                        max_entropy += np.log(len(counts))

                stats['inputs/total_entropy'] = total_entropy
                stats['inputs/entropy_ratio'] = total_entropy / max_entropy if max_entropy > 0 else 0
            else:
                # Flat actions: decode to get stats
                action_probs = self.action_counts / self.total_actions

                jump_actions = 0
                boost_actions = 0
                handbrake_actions = 0

                for action_idx in range(1944):
                    count = self.action_counts[action_idx]
                    j = (action_idx // 243) % 2
                    b = (action_idx // 486) % 2
                    h = (action_idx // 972) % 2

                    if j == 1:
                        jump_actions += count
                    if b == 1:
                        boost_actions += count
                    if h == 1:
                        handbrake_actions += count

                stats['inputs/jump_pct'] = 100 * jump_actions / self.total_actions
                stats['inputs/boost_pct'] = 100 * boost_actions / self.total_actions
                stats['inputs/handbrake_pct'] = 100 * handbrake_actions / self.total_actions

                nonzero_probs = action_probs[action_probs > 0]
                entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
                max_entropy = np.log(1944)
                stats['inputs/action_entropy'] = entropy
                stats['inputs/entropy_ratio'] = entropy / max_entropy

        return stats
