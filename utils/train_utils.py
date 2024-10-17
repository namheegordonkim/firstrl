import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(
            (arr - self.mean) / np.sqrt(self.var + self.epsilon), -1000, 1000
        )

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * np.sqrt(self.var + self.epsilon) + self.mean


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]


def unit_triangle_wave_np(x, low=-1, high=1):
    # x is given in [low, high]. Should scale to [-1, 1].
    # (high - low) / 2 should map to 0
    low = np.array(low)
    high = np.array(high)
    x -= (high + low) / 2
    # original scale is high - low. need to scale to 2
    x *= 2 / (high - low)

    x = np.arcsin(np.sin(x * np.pi / 2)) * 2 / np.pi

    x *= (high - low) / 2
    x += (high + low) / 2
    return x
