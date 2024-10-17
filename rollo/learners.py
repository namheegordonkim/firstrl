from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from rollo.env_containers import EnvContainer
from rollo.rollouters import PolicyRollouter
import jax

from rollo.torch_nets import ProbMLP, RunningMeanStd
from utils.train_utils import ThroughDataset


class Learner:
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def weigh(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError


class ReinforceLearner(Learner):

    def __init__(
        self,
        args,
        env_container: EnvContainer,
        rollouter: PolicyRollouter,
        policy: ProbMLP,
        value: nn.Module,
    ):
        self.args = args

        self.env_container = env_container
        self.rollouter = rollouter
        self.policy = policy
        self.value = value

        self.obs_size = self.env_container.env.observation_size
        self.action_size = self.env_container.env.action_size

        self.s0 = None
        self.tau = None
        self.a = None
        self.R = None
        self.V = None
        self.A = None
        self.obs_hist = None
        self.a_hist = None
        self.R_hist = None
        self.xp_max_length = int(1e5)

        self.jax_rng = jax.random.PRNGKey(0)
        self.global_simsteps_elapsed = 0
        self.gamma = 0.999

        self.value_optimizer = Adam(params=self.value.parameters(), lr=5e-5)
        self.policy_optimizer = Adam(params=self.policy.parameters(), lr=5e-5)

    def reset(self):
        self.s0 = self.env_container.jit_env_reset(rng=self.jax_rng)
        self.env_container.env_state = self.s0
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)

        self.tau = None
        self.a = None

        self.R = None
        self.V = None
        self.A = None

    def sample(self, rollout_length, deterministic):
        self.tau, self.a = self.rollouter.rollout(self.env_container.env_state, self.policy, rollout_length, deterministic)
        self.R = None
        self.V = None
        self.A = None

    def evaluate(self):
        # Compute cumulative reward from trajectory tau
        trajectory_rewards = np.array(self.tau.reward)
        discount_exponents = np.arange(trajectory_rewards.shape[-1])
        discount_exponents = discount_exponents[None] - discount_exponents[:, None]
        discount_exponents = discount_exponents[None].repeat(trajectory_rewards.shape[0], axis=0)

        idxs = np.arange(trajectory_rewards.shape[-1])[None].repeat(trajectory_rewards.shape[0], axis=0)[..., None].repeat(trajectory_rewards.shape[-1], axis=-1).transpose(0, 2, 1)

        tmp = np.zeros_like(self.tau.done, dtype=int)
        e, f = np.where(self.tau.done[..., ::-1])
        tmp[e, f] = f
        tmp = np.roll(tmp, 1, axis=-1)
        tmp2 = self.tau.done.shape[1] - 1 - np.maximum.accumulate(tmp, axis=-1)[..., ::-1]
        discount_exponents[idxs >= tmp2[..., None]] = -1

        gamma_seq = self.gamma**discount_exponents
        gamma_seq[gamma_seq > 1] = 0
        discounted_trajectory_rewards = trajectory_rewards[..., None, :] * gamma_seq
        discounted_cumulative_rewards = discounted_trajectory_rewards.sum(-1)
        self.R = discounted_cumulative_rewards

    def weigh(self):
        s_t = np.array(self.tau.obs)
        s_t = torch.as_tensor(s_t)
        with torch.no_grad():
            self.V = self.value.forward(s_t)[..., -1].detach().cpu().numpy()

        # Off-policy: memorize into history
        self.obs_hist = np.array(self.tau.obs[:, :-1].reshape(-1, self.tau.obs.shape[-1]))
        self.a_hist = self.a.reshape(-1, self.a.shape[-1])
        self.R_hist = self.R[:, :-1].reshape(-1)

        # if self.obs_hist is None:
        #     self.obs_hist = np.array(self.tau.obs[:, :-1].reshape(-1, self.tau.obs.shape[-1]))
        #     self.a_hist = self.a.reshape(-1, self.a.shape[-1])
        #     self.R_hist = self.R[:, :-1].reshape(-1)
        # else:
        #     self.obs_hist = np.concatenate([self.obs_hist, self.tau.obs[:, :-1].reshape(-1, self.tau.obs.shape[-1])], axis=0)
        #     self.a_hist = np.concatenate([self.a_hist, self.a.reshape(-1, self.a.shape[-1])], axis=0)
        #     self.R_hist = np.concatenate([self.R_hist, self.R[:, :-1].reshape(-1)], axis=0)

        # FIFO pruning of history
        # while self.obs_hist.shape[0] > self.xp_max_length:
        #     self.obs_hist = self.obs_hist[self.env_container.batch_size :]
        #     self.a_hist = self.a_hist[self.env_container.batch_size :]
        #     self.R_hist = self.R_hist[self.env_container.batch_size :]

        # Advantage estimation: keeping it simple; can apply bootstrapping if necessary
        # self.A = (self.R - self.R_hist.mean()) / (self.R_hist.std() + 1e-8) - self.V
        self.A = self.R - self.V

    def learn(self, n_epochs: int):
        device = torch.device("cuda")
        # Prepare common data
        s_t = torch.as_tensor(self.obs_hist)
        a_t = torch.as_tensor(self.a_hist)
        R_t = torch.as_tensor(self.R_hist)

        # First, train the value network: predict cumulative reward from state
        self.value.input_rms.update(s_t)
        self.policy.input_rms.update(s_t)
        self.value = self.value.to(device)
        self.policy = self.policy.to(device)

        # value_mean = R_t.mean()
        # value_std = R_t.std()
        # R_t = (R_t - value_mean) / (value_std + 1e-8)

        dataset = ThroughDataset(s_t, a_t, R_t)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        eval_s, eval_a, eval_R = next(iter(dataloader))
        eval_s = eval_s.to(device=device, dtype=torch.float)
        eval_a = eval_a.to(device=device, dtype=torch.float)
        eval_R = eval_R.to(device=device, dtype=torch.float)

        eval_every = 1
        pbar = tqdm(total=n_epochs)
        for epoch in range(n_epochs):
            if epoch % eval_every == 0:
                with torch.no_grad():
                    RR_hat = self.value.forward(eval_s)[..., -1]
                    eval_value_loss = torch.nn.functional.mse_loss(RR_hat, eval_R)

            for i, (ss, aa, RR) in enumerate(dataloader):
                ss = ss.to(device=device, dtype=torch.float)
                aa = aa.to(device=device, dtype=torch.float)
                RR = RR.to(device=device, dtype=torch.float)

                self.value_optimizer.zero_grad()
                RR_hat = self.value.forward(ss)[..., -1]
                value_loss = torch.nn.functional.mse_loss(RR_hat, RR)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.value_optimizer.step()
            pbar.update(1)
            pbar.set_postfix({"value_loss": f"{eval_value_loss.item():.2e}"})
        pbar.close()

        self.value = self.value.cpu()
        with torch.no_grad():
            A_t = R_t - self.value.forward(s_t)[..., -1]
            # A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-8)
            # A_t = torch.clip(A_t, min=-3)
        dataset = ThroughDataset(s_t, a_t, R_t, A_t)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        eval_s, eval_a, eval_R, eval_A = next(iter(dataloader))
        eval_s = eval_s.to(device=device, dtype=torch.float)
        eval_a = eval_a.to(device=device, dtype=torch.float)
        eval_R = eval_R.to(device=device, dtype=torch.float)
        eval_A = eval_A.to(device=device, dtype=torch.float)

        pbar = tqdm(total=n_epochs)
        for epoch in range(n_epochs):
            if epoch % eval_every == 0:
                with torch.no_grad():
                    dist = self.policy.forward(eval_s)
                    log_probs = dist.log_prob(eval_a)
                    eval_policy_loss = -torch.mean(log_probs * eval_A)

            for i, (ss, aa, RR, AA) in enumerate(dataloader):
                ss = ss.to(device=device, dtype=torch.float)
                aa = aa.to(device=device, dtype=torch.float)
                RR = RR.to(device=device, dtype=torch.float)
                AA = AA.to(device=device, dtype=torch.float)
                # with torch.no_grad():
                #     AA = RR - self.value.forward(ss)[..., -1]
                #     AA = (AA - AA.mean()) / (AA.std() + 1e-8)
                #     AA = torch.clip(AA, min=0)

                self.policy_optimizer.zero_grad()
                dist = self.policy.forward(ss)
                log_probs = dist.log_prob(aa)
                policy_loss = -torch.mean(log_probs * AA)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"policy_loss": f"{eval_policy_loss.item():.2e}"})
        pbar.close()
        self.policy = self.policy.cpu()
        self.policy.logstd.data -= 0.1
        self.policy.logstd.data = torch.clip(self.policy.logstd.data, min=-10, max=1)
        print(self.policy.logstd)
