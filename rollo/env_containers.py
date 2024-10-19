from dataclasses import dataclass

import jax
from jax._src.stages import Wrapped

from brax import State, envs
from brax.envs import Env


class EnvContainer:
    """
    Dataclass for containing Brax environment related things.
    Very useful since Python doesn't like mutating arguments in callbacks.
    """

    batch_size: int
    env: Env
    env_state: State
    jit_env_step: Wrapped
    jit_env_reset: Wrapped

    def __init__(
        self,
        env_name: str,
        backend: str,
        batch_size: int,
        auto_reset: bool = True,
        episode_length: int = 256,
    ):
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.env = envs.create(
            env_name,
            auto_reset=auto_reset,
            backend=backend,
            episode_length=episode_length,
            batch_size=batch_size,
        )
        self.jit_env_reset = jax.jit(self.env.reset)
        self.jit_env_step = jax.jit(self.env.step)
        self.env_state = self.jit_env_reset(jax.random.PRNGKey(0))
