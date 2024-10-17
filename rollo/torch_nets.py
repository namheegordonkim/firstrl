import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import ThroughDataset


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip((arr - self.mean) / torch.sqrt(self.var + self.epsilon), -1000, 1000)


class ScaleShift(nn.Module):
    def __init__(self, size, scale, shift):
        super().__init__()
        self.size = size
        self.scale = nn.Parameter(torch.ones(size, dtype=torch.float) * scale)
        self.shift = nn.Parameter(torch.ones(size, dtype=torch.float) * shift)

    def forward(self, x):
        return x * self.scale + self.shift


class MLP(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.input_rms = RunningMeanStd(shape=(input_size,))

    def forward(self, x: torch.Tensor):
        x = self.input_rms.normalize(x)
        y = self.layers.forward(x)
        return y


class ProbMLP(MLP):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__(input_size, output_size, hidden_size)
        del self.layers
        self.mu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            # ScaleShift(self.output_size, scale=1e-3, shift=0),
        )
        # self.logstd = nn.Sequential(
        #     nn.Linear(self.input_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.output_size),
        #     ScaleShift(self.output_size, scale=1, shift=-4),
        # )
        self.logstd = nn.Parameter(torch.ones(self.output_size, dtype=torch.float) * 0, requires_grad=False)

        self.input_rms = RunningMeanStd(shape=(input_size,))

    def forward(self, x: torch.Tensor):
        x = self.input_rms.normalize(x)

        mu = self.mu.forward(x)
        # mu = torch.clip(mu, min=-0.999, max=0.999)
        # logstd = torch.clip(self.logstd.forward(x), min=-10, max=1)
        logstd = self.logstd[None].repeat_interleave(x.shape[0], dim=0)
        dist = Normal(mu, logstd.exp())
        dist = TransformedDistribution(dist, [TanhTransform()])
        return dist

    def sample(self, x: torch.Tensor, deterministic=False):
        dist = self.forward(x)
        if deterministic:
            y = dist.transforms[0](dist.base_dist.mean)
        else:
            y = dist.sample()
        y = torch.clip(y, min=-1, max=1)
        return y
