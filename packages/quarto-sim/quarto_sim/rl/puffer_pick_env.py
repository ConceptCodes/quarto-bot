from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import gymnasium as gym

import mujoco
import pufferlib.emulation

from quarto_sim.rl.puffer_config import PickAndPlaceEnvConfig
from quarto_sim.core.pick.env import PickEnv


def _yaw_to_quaternion(yaw: float) -> np.ndarray:
    """Return quaternion for a yaw rotation (roll=pitch=0)."""
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


class RandomizedPickAndPlaceEnv(gym.Env):
    """Single-agent pick-and-place task with domain randomization."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: PickAndPlaceEnvConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode
        self.core = PickEnv(xml_path=config.xml_path)

        base_obs = self.core._get_obs().astype(np.float32)
        self._base_obs_dim = base_obs.shape[0]
        # Extra features: target xyz + piece-to-target delta xyz
        self._extra_obs_dim = 6
        obs_dim = self._base_obs_dim + self._extra_obs_dim
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._piece_joint_addr = int(self.core.model.joint("quarto_joint").qposadr)
        self._gripper_site_id = self.core.model.site("gripperframe").id
        self._board_coords = self._precompute_board_coords()
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._target_position = np.zeros(3, dtype=np.float32)
        self._target_index = (0, 0)
        self._prev_piece_target_dist = None

    def _precompute_board_coords(self) -> List[Tuple[float, float]]:
        """Generate grid coordinates for Quarto board squares."""
        rand = self.cfg.randomization
        rows, cols = rand.board_grid_size
        x_bounds, y_bounds = rand.placement_xy_bounds
        spacing = rand.board_square_spacing
        if spacing <= 0:
            x_span = x_bounds[1] - x_bounds[0]
            y_span = y_bounds[1] - y_bounds[0]
            spacing_x = x_span / max(cols - 1, 1)
            spacing_y = y_span / max(rows - 1, 1)
        else:
            spacing_x = spacing_y = spacing

        coords: List[Tuple[float, float]] = []
        for row in range(rows):
            for col in range(cols):
                x = x_bounds[0] + col * spacing_x
                y = y_bounds[0] + row * spacing_y
                coords.append((x, y))
        return coords

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _sample_piece_position(self) -> np.ndarray:
        rand = self.cfg.randomization
        x = self._rng.uniform(*rand.piece_xy_bounds[0])
        y = self._rng.uniform(*rand.piece_xy_bounds[1])
        z = self._rng.uniform(*rand.piece_z_bounds)
        return np.array([x, y, z], dtype=np.float32)

    def _sample_target_position(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        rand = self.cfg.randomization
        rows, cols = rand.board_grid_size
        idx = self._rng.integers(0, rows * cols)
        row, col = divmod(idx, cols)
        x, y = self._board_coords[idx]
        jitter = self._rng.uniform(-0.005, 0.005, size=2)
        target = np.array(
            [x + jitter[0], y + jitter[1], rand.placement_z], dtype=np.float32
        )
        return target, (row, col)

    def _write_piece_pose(self, pos: np.ndarray) -> None:
        addr = self._piece_joint_addr
        self.core.data.qpos[addr : addr + 3] = pos
        if self.cfg.randomization.randomize_rotation:
            yaw = float(self._rng.uniform(-np.pi, np.pi))
            quat = _yaw_to_quaternion(yaw)
            self.core.data.qpos[addr + 3 : addr + 7] = quat
        mujoco.mj_forward(self.core.model, self.core.data)
        self.core._initial_piece_pos = np.copy(self.core.data.qpos[addr : addr + 3])

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        piece_pos = self._get_piece_position()
        delta = self._target_position - piece_pos
        return np.concatenate(
            [base_obs.astype(np.float32), self._target_position, delta], axis=0
        )

    def _get_piece_position(self) -> np.ndarray:
        addr = self._piece_joint_addr
        return np.array(self.core.data.qpos[addr : addr + 3], dtype=np.float32)

    def _get_gripper_position(self) -> np.ndarray:
        return np.array(self.core.data.site_xpos[self._gripper_site_id], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        base_obs = self.core.reset()
        piece_pos = self._sample_piece_position()
        self._write_piece_pose(piece_pos)
        self._target_position, self._target_index = self._sample_target_position()
        self._prev_piece_target_dist = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        obs = self._augment_obs(self.core._get_obs())
        info = {
            "target_position": self._target_position.copy(),
            "target_square": self._target_index,
        }
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        self._step_count += 1
        scaled_action = np.clip(action, -1.0, 1.0) * self.cfg.action_scale
        base_obs, _, _, info = self.core.step(scaled_action)
        obs = self._augment_obs(base_obs)

        dist_to_piece = float(info.get("distance", 0.0))
        piece_pos = np.array(info.get("piece_pos", self._get_piece_position()))
        piece_target_delta = self._target_position - piece_pos
        piece_to_target_xy = float(np.linalg.norm(piece_target_delta[:2]))
        z_error = abs(piece_target_delta[2])
        gripper_closed = bool(info.get("gripper_closed", False))
        piece_lifted = bool(info.get("piece_lifted", False))

        reward = self.cfg.reward_distance_scale * dist_to_piece
        reward += -1.5 * piece_to_target_xy
        reward += self.cfg.reward_energy_scale * float(np.linalg.norm(scaled_action))
        if piece_lifted:
            reward += 4.0
        if gripper_closed and dist_to_piece < 0.02:
            reward += 2.0
        if self._prev_piece_target_dist is not None:
            reward += 1.0 * (self._prev_piece_target_dist - piece_to_target_xy)
        self._prev_piece_target_dist = piece_to_target_xy

        success = (
            piece_to_target_xy < 0.02
            and z_error < 0.02
            and not gripper_closed
            and piece_lifted
        )
        if success:
            reward += self.cfg.reward_success_bonus

        terminated = success
        truncated = self._step_count >= self.cfg.max_episode_steps

        self._episode_reward += reward
        self._episode_length += 1

        step_info = {
            "target_position": self._target_position.copy(),
            "target_square": self._target_index,
            "piece_position": piece_pos,
            "piece_to_target_xy": piece_to_target_xy,
            "distance_to_piece": dist_to_piece,
            "success": success,
        }
        if terminated or truncated:
            step_info["episode_return"] = self._episode_reward
            step_info["episode_length"] = self._episode_length
            step_info["episode_success"] = success

        return (
            obs.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            step_info,
        )

    def render(self):
        mode = self.render_mode or "human"
        return self.core.render(mode=mode)

    def close(self):
        self.core.close()


def make_puffer_pick_env(
    config: PickAndPlaceEnvConfig, render_mode: Optional[str] = None, buf=None, seed: Optional[int] = None
):
    """Instantiate a GymnasiumPufferEnv wrapping the randomized pick task."""
    env = RandomizedPickAndPlaceEnv(config=config, render_mode=render_mode)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed or 0)


def env_creator(config: PickAndPlaceEnvConfig, render_mode: Optional[str] = None) -> Callable:
    """Return a callable suitable for pufferlib.vector.make."""

    def _make(buf=None, seed: Optional[int] = None):
        return make_puffer_pick_env(config=config, render_mode=render_mode, buf=buf, seed=seed)

    return _make
