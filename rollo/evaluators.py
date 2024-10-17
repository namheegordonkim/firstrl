import jax
import jax.numpy as jp
import numpy as np

import brax
from brax import kinematics, State


class Evaluator:
    """
    Will prepare Evaluator that evaluates trajectory against a specific goal
    And another that evaluates trajectory against all possible goals
    """

    @staticmethod
    def evaluate(
        xpos: np.ndarray,
        goal_xpos: np.ndarray,
    ):
        """
        `xpos` is shape (B, T, n_balls, 2) where T is the number of timesteps
        Index 0 is assumed to be the cue ball
        `goal_xpos` is (B, G, 2) where G is the number of goals.
        When evaluating against one specifi goal, G=1.
        When evaluating against multiple, G>1.
        """
        deltas = (
            xpos[:, :, :, None] - goal_xpos[:, None, None]
        )  # shape is (B, T, n_balls, G, 2)
        distances = jp.linalg.norm(deltas, axis=-1)  # shape is (B, T, n_balls, G)
        min_distances = jp.min(distances, axis=1)  # shape is (B, n_balls, G)
        min_min_distances = jp.min(min_distances, axis=-1)  # shape is (B, n_balls)
        x_intercept = 2
        rewards = 1 - np.mean(min_min_distances[:, 1:], axis=-1) / x_intercept
        rewards = 1e0 * np.clip(rewards, 0, 1)
        failure_yes = min_min_distances[:, 0] <= 5e-2
        success_yes = np.all(min_min_distances[:, 1:] <= 5e-2, axis=-1)
        return rewards, success_yes, failure_yes
