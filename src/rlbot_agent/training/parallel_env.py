"""Parallel environment manager using multiprocessing."""

import multiprocessing as mp
from multiprocessing import connection
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import traceback
import sys
import os


def _worker_process(
    worker_id: int,
    n_envs: int,
    env_factory: Callable[[], Any],
    pipe: connection.Connection,
):
    """Worker process that manages multiple environments.

    Supports multi-agent environments where each env returns [n_agents, ...] arrays.

    Args:
        worker_id: Unique identifier for this worker
        n_envs: Number of environments to manage
        env_factory: Factory function to create environments
        pipe: Connection for communication with main process
    """
    try:
        # Create environments
        envs = [env_factory() for _ in range(n_envs)]

        # Track agents per env (for multi-agent support)
        agents_per_env = [getattr(env, 'n_agents', 1) for env in envs]

        while True:
            cmd, data = pipe.recv()

            if cmd == "step":
                # Step all environments with given actions
                # actions shape: [total_agents_this_worker, action_dim] or [total_agents, ]
                actions = data
                results = []
                action_idx = 0

                for i, env in enumerate(envs):
                    n_agents = agents_per_env[i]
                    try:
                        # Extract actions for this env's agents
                        env_actions = actions[action_idx:action_idx + n_agents]
                        action_idx += n_agents

                        obs, reward, dones, infos = env.step(env_actions)

                        # Check if episode ended (all agents done together)
                        if dones[0] > 0.5:  # Episode ended
                            obs = env.reset()
                            agents_per_env[i] = getattr(env, 'n_agents', 1)

                        # obs: [n_agents, obs_dim], reward: [n_agents], dones: [n_agents]
                        results.append((obs, reward, dones, infos, n_agents))

                    except Exception as e:
                        print(f"Worker {worker_id} env {i} step error: {e}")
                        traceback.print_exc()
                        obs = env.reset()
                        n_agents = getattr(env, 'n_agents', 1)
                        agents_per_env[i] = n_agents
                        results.append((
                            obs,
                            np.zeros(n_agents),
                            np.ones(n_agents),
                            [{} for _ in range(n_agents)],
                            n_agents
                        ))

                pipe.send(("step_result", results))

            elif cmd == "reset":
                # Reset all environments
                results = []
                for i, env in enumerate(envs):
                    try:
                        obs = env.reset()  # [n_agents, obs_dim]
                        n_agents = getattr(env, 'n_agents', 1)
                        agents_per_env[i] = n_agents
                        results.append((obs, n_agents))
                    except Exception as e:
                        print(f"Worker {worker_id} reset error: {e}")
                        traceback.print_exc()
                        results.append((np.zeros((1, 113), dtype=np.float32), 1))

                pipe.send(("reset_result", results))

            elif cmd == "get_agents_per_env":
                pipe.send(("agents_per_env", agents_per_env))

            elif cmd == "close":
                for env in envs:
                    try:
                        env.close()
                    except:
                        pass
                pipe.send(("closed", None))
                break

            elif cmd == "ping":
                pipe.send(("pong", worker_id))

    except Exception as e:
        print(f"Worker {worker_id} fatal error: {e}")
        traceback.print_exc()
        try:
            pipe.send(("error", str(e)))
        except:
            pass


class ParallelEnvManager:
    """Manages multiple environments across multiple worker processes.

    Supports multi-agent environments where each game has multiple players.
    Each player's experience is collected separately for training.

    Architecture:
        Main Process <-> Worker 1 (n_envs_per_worker games, each with n_agents players)
                    <-> Worker 2 (...)
                    <-> Worker N (...)

    Total "virtual envs" = n_workers * n_envs_per_worker * n_agents_per_game
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        n_workers: int,
        n_envs_per_worker: int,
    ):
        """Initialize parallel environment manager.

        Args:
            env_factory: Factory function that creates a single environment
            n_workers: Number of worker processes
            n_envs_per_worker: Number of game environments per worker
        """
        self.n_workers = n_workers
        self.n_envs_per_worker = n_envs_per_worker
        self.n_games = n_workers * n_envs_per_worker

        # Create worker processes
        self.workers: List[mp.Process] = []
        self.pipes: List[connection.Connection] = []

        # Use fork on Linux (faster, no serialization needed)
        # Use spawn on other platforms (safer with CUDA)
        if sys.platform == "linux":
            ctx = mp.get_context("fork")
        else:
            ctx = mp.get_context("spawn")

        for worker_id in range(n_workers):
            parent_pipe, child_pipe = ctx.Pipe()
            worker = ctx.Process(
                target=_worker_process,
                args=(worker_id, n_envs_per_worker, env_factory, child_pipe),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)
            self.pipes.append(parent_pipe)

        # Verify all workers started
        for i, pipe in enumerate(self.pipes):
            pipe.send(("ping", None))

        for i, pipe in enumerate(self.pipes):
            cmd, data = pipe.recv()
            if cmd != "pong":
                raise RuntimeError(f"Worker {i} failed to start: {cmd} {data}")

        # Get total agents count after first reset
        self.n_agents_per_game = 2  # Default for 1v1, updated on reset
        self.n_envs = self.n_games * self.n_agents_per_game  # Total "virtual envs"

        print(f"ParallelEnvManager: {n_workers} workers x {n_envs_per_worker} games")

    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            Observations array of shape [n_total_agents, obs_dim]
        """
        # Send reset command to all workers
        for pipe in self.pipes:
            pipe.send(("reset", None))

        # Collect results - each env returns [n_agents, obs_dim]
        all_obs = []
        total_agents = 0
        for pipe in self.pipes:
            cmd, results = pipe.recv()
            if cmd != "reset_result":
                raise RuntimeError(f"Unexpected response: {cmd}")
            for obs, n_agents in results:
                all_obs.append(obs)  # [n_agents, obs_dim]
                total_agents += n_agents

        # Update total env count
        self.n_agents_per_game = total_agents // self.n_games if self.n_games > 0 else 2
        self.n_envs = total_agents

        print(f"  {self.n_agents_per_game} agents per game = {self.n_envs} total agents")

        # Concatenate all observations
        return np.concatenate(all_obs, axis=0)  # [n_total_agents, obs_dim]

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments in parallel.

        Args:
            actions: Actions array of shape [n_total_agents, action_dim] or [n_total_agents]

        Returns:
            Tuple of (observations, rewards, dones, infos)
            - observations: [n_total_agents, obs_dim]
            - rewards: [n_total_agents]
            - dones: [n_total_agents]
            - infos: List of info dicts (one per agent)
        """
        # Distribute actions to workers
        # Each worker gets actions for all its games' agents
        agents_per_worker = self.n_envs_per_worker * self.n_agents_per_game
        for i, pipe in enumerate(self.pipes):
            start_idx = i * agents_per_worker
            end_idx = start_idx + agents_per_worker
            worker_actions = actions[start_idx:end_idx]
            pipe.send(("step", worker_actions))

        # Collect results from all workers
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []

        for pipe in self.pipes:
            cmd, results = pipe.recv()
            if cmd != "step_result":
                raise RuntimeError(f"Unexpected response: {cmd}")

            for obs, rewards, dones, infos, n_agents in results:
                # obs: [n_agents, obs_dim], rewards: [n_agents], dones: [n_agents]
                all_obs.append(obs)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_infos.extend(infos)

        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_rewards, axis=0),
            np.concatenate(all_dones, axis=0),
            all_infos,
        )

    def close(self):
        """Close all environments and terminate workers."""
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except:
                pass

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.workers = []
        self.pipes = []

    def __del__(self):
        self.close()

    @property
    def num_envs(self) -> int:
        """Total number of agents (virtual environments)."""
        return self.n_envs
