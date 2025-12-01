from typing import Callable, Optional
import gymnasium as gym
import pufferlib.emulation
import numpy as np
import functools

from quarto_brain.env import QuartoEnv


def make_quarto_env(
    render_mode: Optional[str] = None,
    opponent_type: Optional[str] = None,
    buf=None,
    seed: Optional[int] = None,
):
    """Instantiate a GymnasiumPufferEnv wrapping the Quarto logic."""
    env = QuartoEnv(render_mode=render_mode, opponent_type=opponent_type)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed or 0)


def env_creator(
    render_mode: Optional[str] = None, opponent_type: str = "random"
) -> Callable:
    """Return a callable suitable for pufferlib.vector.make."""
    return functools.partial(
        make_quarto_env, render_mode=render_mode, opponent_type=opponent_type
    )
