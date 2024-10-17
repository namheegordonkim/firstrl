import jax.random
import torch
from tqdm import tqdm

from brax import State
from rollo.env_containers import EnvContainer
from rollo.torch_nets import ProbMLP
from utils.tree_utils import tree_stack
from torch import nn

import jax.numpy as jp
import numpy as np


class BilliardsRollouter:

    def __init__(self, env_container: EnvContainer):
        # def __init__(self, logger, env_container: EnvContainer):
        self.env_container = env_container

    def single_action_rollout(self, state: State, action: np.array, num_steps: int):
        # self.logger.info(f"Running action sequence rollout...")
        s0 = state
        trajectories = [s0]
        for t in tqdm(range(num_steps)):
            if t > 0:
                action = np.zeros_like(action)
            s0 = self.env_container.jit_env_step(s0, jp.array(action))
            trajectories.append(s0)
        trajectory_stacked = tree_stack(trajectories, axis=1)
        return trajectory_stacked


class Rollouter:

    def __init__(self, logger, env_container: EnvContainer):
        self.logger = logger
        self.env_container = env_container

    def action_sequence_rollout(self, state, action_sequences):
        self.logger.info(f"Running action sequence rollout...")
        max_horizon = action_sequences.shape[1]
        s0 = state
        trajectories = [s0]
        for t in tqdm(range(action_sequences.shape[1])):
            actions = action_sequences[:, t]
            s0 = self.env_container.jit_env_step(s0, jp.array(actions))
            trajectories.append(s0)
        trajectory_stacked = tree_stack(trajectories, axis=1)
        return trajectory_stacked


class PolicyRollouter:
    def __init__(self, env_container: EnvContainer):
        self.env_container = env_container

    def rollout(self, state, policy: ProbMLP, max_horizon: int, deterministic: bool, terminate: bool = True):
        trajectories = [state]
        actions = []
        s0 = self.env_container.jit_env_reset(rng=jax.random.PRNGKey(0))
        for t in tqdm(range(max_horizon)):

            # def handle_terminate(a, b):
            #     return jax.tree.map(lambda x, y: x.at[state.done.astype(bool)].set(y[state.done.astype(bool)]), a, b)
            #
            # if terminate:
            #     state = jax.jit(handle_terminate)(state, s0)
            a = (
                policy.sample(
                    torch.as_tensor(np.array(state.obs), dtype=torch.float),
                    deterministic,
                )
                .detach()
                .cpu()
                .numpy()
            )
            actions.append(a)
            state = self.env_container.jit_env_step(state, jp.array(a))
            trajectories.append(state)
        trajectory_stacked = tree_stack(trajectories, axis=1)
        # Abusing duck-typing to be compact
        actions_stacked = np.stack(actions, axis=1)
        self.env_container.env_state = state
        return trajectory_stacked, actions_stacked
