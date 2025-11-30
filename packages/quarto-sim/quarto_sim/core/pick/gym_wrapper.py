import gymnasium
from gymnasium import spaces
import numpy as np
from env import PickEnv


class PickEnvGym(gymnasium.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.env = PickEnv()
        obs = self.env._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        info = {}  # Optionally, add info dict
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.clip(action, -1, 1) * 0.05
        obs, reward, done, info = self.env.step(action)
        # For Gymnasium: split 'done' into 'terminated' and 'truncated'
        terminated = (
            done
            and info.get("gripper_closed", False)
            and info.get("piece_lifted", False)
        )
        truncated = done and not terminated
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode or "human"
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
