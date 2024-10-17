import jax
import numpy as np
import torch
from jax import tree_map
from torch import nn
from tqdm import tqdm

from rollo.action_populators import ActionPopulator
from rollo.env_containers import EnvContainer
from rollo.evaluators import Evaluator
from rollo.models import MyKNeighbors
from rollo.rollouters import BilliardsRollouter
from utils.torch_nets import GCFitPredict
from utils.train_utils import unit_triangle_wave_np


class GCSLOptimizer:
    def __init__(
        self,
        args,
        # logger,
        # writer,
        env_container: EnvContainer,
        action_populator: ActionPopulator,
        rollouter: BilliardsRollouter,
        evaluator: Evaluator,
        proposal_model: nn.Module,
    ):
        self.action_size = 2
        self.g_hist = None
        self.o0_hist = None
        self.args = args
        # self.logger = logger
        # self.writer = writer

        self.env_container = env_container
        self.action_populator = action_populator
        self.rollouter = rollouter
        self.evaluator = evaluator
        self.proposal_model = proposal_model

        self.obs_size = self.env_container.env.observation_size
        self.deployment_model = MyKNeighbors(1, self.obs_size, 2)

        self.n_parallels = 4
        self.parallel_size = self.env_container.batch_size // self.n_parallels

        self.s0 = None
        self.s0_dots_T = None
        self.g = None
        self.g_all = None
        self.g_idxs = None
        self.a = None
        self.a_noised = None
        self.r = None
        self.climb_i = 0

        self.parallel_s0 = None
        self.parallel_g_idxs = None
        self.parallel_g = None
        self.parallel_best_a = None
        self.parallel_best_r = None
        self.parallel_best_s0_dots_T = None
        self.n_elites = 1
        # self.n_elites = 8

        self.x0_dots_T_hist = None
        self.a_hist = None

        self.augmented_o0 = None
        self.augmented_g = None
        self.augmented_a = None

        self.outer_i = 0
        self.jax_rng = jax.random.PRNGKey(0)

        self.g_all = np.array(
            [
                [-0.725, -1.30],
                [-0.725, 0],
                [-0.725, 1.30],
                [0.725, -1.30],
                [0.725, 0],
                [0.725, 1.30],
            ]
        )
        self.s0_test = self.env_container.jit_env_reset(rng=jax.random.PRNGKey(0))
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)
        np.random.seed(0)
        self.g_test_idxs = np.random.choice(
            self.g_all.shape[0], size=self.env_container.batch_size
        )
        np.random.seed(0)
        self.g_test = self.g_all[self.g_test_idxs]
        self.global_simsteps_elapsed = 0

    def reset(self):
        self.parallel_best_a = np.zeros(
            (self.n_parallels, self.n_elites, self.action_size)
        )
        self.parallel_best_r = np.ones((self.n_parallels, self.n_elites)) * -np.inf

        self.s0 = self.env_container.jit_env_reset(rng=self.jax_rng)
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)
        self.g_idxs = np.random.choice(
            self.g_all.shape[0], size=self.env_container.batch_size
        )
        self.g = self.g_all[self.g_idxs]

        # Just select the first n_parallels; should preserve uniformness
        self.parallel_s0 = tree_map(lambda x: x[: self.n_parallels], self.s0)
        self.parallel_g_idxs = self.g_idxs[: self.n_parallels]
        self.parallel_g = self.g_all[self.parallel_g_idxs]

        # reassign
        self.s0 = tree_map(lambda x: x.repeat(self.parallel_size, 0), self.parallel_s0)
        self.g_idxs = self.parallel_g_idxs.repeat(self.parallel_size, 0)
        self.g = self.g_all[self.g_idxs]
        self.climb_i = 0

        self.parallel_best_s0_dots_T = tree_map(
            lambda x: x[:, None, None].repeat(self.n_elites, 1).repeat(65, 2),
            self.parallel_s0,
        )

        self.s0_dots_T = None

    def sample(self, deterministic=False):
        # o0 = np.concatenate(
        #     [
        #         self.s0.pipeline_state.x.pos[..., :-1],
        #         self.s0.pipeline_state.xd.vel[..., :-1],
        #     ],
        #     axis=-1,
        # )
        o0 = np.array(
            self.s0.pipeline_state.x.pos[..., :-1],
        )
        o0 = o0.reshape(o0.shape[0], -1)
        g = self.g
        o0g = torch.as_tensor(np.concatenate([o0, g], axis=-1), dtype=torch.float)
        a = self.proposal_model.forward(o0g).detach().cpu().numpy()
        if deterministic:
            noise = a * 0
        else:
            noise = np.random.normal(0, 1, size=a.shape) * 0.1
        self.a_noised = a + noise

        # self.logger.info(f"Sampling and adding noise")
        # noise = np.random.normal(0, noise_std, size=self.a.shape)
        # self.a_noised = self.a + noise
        self.a_noised = unit_triangle_wave_np(self.a_noised)

        # self.logger.info(f"Rolling out")
        aa = self.action_populator.populate(self.a_noised)
        self.s0_dots_T = self.rollouter.single_action_rollout(self.s0, aa, 64)
        self.global_simsteps_elapsed += self.env_container.batch_size * 64

        # x0_dots_T = np.concatenate(
        #     [
        #         self.s0_dots_T.pipeline_state.x.pos[..., :-1],
        #         self.s0_dots_T.pipeline_state.xd.vel[..., :-1],
        #     ],
        #     axis=-1,
        # )

    def remember(self):
        x0_dots_T = np.array(
            self.s0_dots_T.pipeline_state.x.pos[..., :-1],
        )
        x0_dots_T = x0_dots_T.reshape((x0_dots_T.shape[0], x0_dots_T.shape[1], -1))
        if True or self.x0_dots_T_hist is None:
            self.x0_dots_T_hist = x0_dots_T
            self.a_hist = self.a_noised * 1
        else:
            self.x0_dots_T_hist = np.concatenate(
                [self.x0_dots_T_hist, x0_dots_T], axis=0
            )
            self.a_hist = np.concatenate([self.a_hist, self.a_noised * 1], axis=0)

    def evaluate(self):
        # self.logger.info(f"Evaluating...")
        noise = 0.1 * np.random.normal(
            0, 1, self.s0_dots_T.pipeline_state.x.pos[:, :, :, :-1].shape
        )
        r, rr, rrr = self.evaluator.evaluate(
            self.s0_dots_T.pipeline_state.x.pos[:, :, :, :-1],
            # self.s0_dots_T.pipeline_state.x.pos[:, :, :, :-1] + noise,
            self.g[:, None],
        )
        self.r = r * 1
        self.r += rr
        self.r *= 1 - rrr

    def filter(self):
        # self.logger.info(f"Filtering...")
        parallel_r = self.r.reshape(self.n_parallels, self.parallel_size)
        argsort_parallel_r = np.argsort(parallel_r, axis=1)[..., ::-1]  # descending
        elite_idxs = argsort_parallel_r[:, : self.n_elites]
        parallel_a = self.a_noised.reshape(self.n_parallels, self.parallel_size, -1)
        parallel_s0_dots_T = tree_map(
            lambda x: x.reshape(self.n_parallels, self.parallel_size, *x.shape[1:]),
            self.s0_dots_T,
        )
        parallel_elite_a = parallel_a[np.arange(self.n_parallels)[:, None], elite_idxs]
        parallel_elite_r = parallel_r[np.arange(self.n_parallels)[:, None], elite_idxs]
        parallel_elite_s0_dots_T = tree_map(
            lambda x: x[np.arange(self.n_parallels)[:, None], elite_idxs],
            parallel_s0_dots_T,
        )

        best_elite_concatenated_a = np.concatenate(
            [self.parallel_best_a, parallel_elite_a], axis=1
        )
        best_elite_concatenated_r = np.concatenate(
            [self.parallel_best_r, parallel_elite_r], axis=1
        )
        best_elite_concatenated_s0_dots_T = tree_map(
            lambda x, y: np.concatenate([x, y], axis=1),
            self.parallel_best_s0_dots_T,
            parallel_elite_s0_dots_T,
        )
        argsorted_best_elite_concatenated_r = np.argsort(
            best_elite_concatenated_r, axis=1
        )[..., ::-1]
        self.parallel_best_a = best_elite_concatenated_a[
            np.arange(self.n_parallels)[:, None],
            argsorted_best_elite_concatenated_r[:, : self.n_elites],
        ]
        self.parallel_best_r = best_elite_concatenated_r[
            np.arange(self.n_parallels)[:, None],
            argsorted_best_elite_concatenated_r[:, : self.n_elites],
        ]
        self.parallel_best_s0_dots_T = tree_map(
            lambda x: x[
                np.arange(self.n_parallels)[:, None],
                argsorted_best_elite_concatenated_r[:, : self.n_elites],
            ],
            best_elite_concatenated_s0_dots_T,
        )

        self.climb_i += 1

        # self.writer.add_scalar(
        #     "train/avg_best_reward",
        #     np.mean(self.parallel_best_r),
        #     self.global_simsteps_elapsed,
        # )

    def augment(self):
        # self.logger.info(f"Augmenting...")

        # Do hindsight relabeling on best to create augmented data
        combos = np.stack([np.zeros(64), np.arange(64) + 1], axis=-1).astype(int)

        xpos = self.parallel_best_s0_dots_T.pipeline_state.x.pos
        xpos = xpos[..., :-1].reshape(*xpos.shape[:3], -1)
        xpos_reshaped = xpos.reshape(-1, *xpos.shape[2:])
        self.augmented_o0 = xpos_reshaped[:, combos[:, 0]]
        self.augmented_o0 = self.augmented_o0.reshape(-1, *self.augmented_o0.shape[2:])
        self.augmented_g = xpos_reshaped[:, combos[:, 1]][..., 2:]
        self.augmented_g = self.augmented_g.reshape(-1, *self.augmented_g.shape[2:])
        a = self.parallel_best_a
        a = a.reshape(-1, *a.shape[2:])
        self.augmented_a = a.repeat(64, 0)

        # Symmetry augmentation
        o0_refl_wrt_y = self.augmented_o0 * 1
        o0_refl_wrt_y[..., 0] *= -1
        g_refl_wrt_y = self.augmented_g * 1
        g_refl_wrt_y[..., 0] *= -1
        a_refl_wrt_y = self.augmented_a * 1
        a_refl_wrt_y[..., 0] *= -1

        o0_refl_wrt_x = self.augmented_o0 * 1
        o0_refl_wrt_x[..., 1] *= -1
        g_refl_wrt_x = self.augmented_g * 1
        g_refl_wrt_x[..., 1] *= -1
        a_refl_wrt_x = self.augmented_a * 1
        a_refl_wrt_x[..., 1] *= -1

        o0_refl_wrt_xy = self.augmented_o0 * 1
        o0_refl_wrt_xy[..., :2] *= -1
        g_refl_wrt_xy = self.augmented_g * 1
        g_refl_wrt_xy[..., :2] *= -1
        a_refl_wrt_xy = self.augmented_a * 1
        a_refl_wrt_xy[..., :2] *= -1

        # self.augmented_o0 = np.concatenate(
        #     [
        #         self.augmented_o0,
        #         o0_refl_wrt_y,
        #         o0_refl_wrt_x,
        #         o0_refl_wrt_xy,
        #     ],
        #     axis=0,
        # )
        # self.augmented_g = np.concatenate(
        #     [
        #         self.augmented_g,
        #         g_refl_wrt_y,
        #         g_refl_wrt_x,
        #         g_refl_wrt_xy,
        #     ],
        #     axis=0,
        # )
        # self.augmented_a = np.concatenate(
        #     [
        #         self.augmented_a,
        #         a_refl_wrt_y,
        #         a_refl_wrt_x,
        #         a_refl_wrt_xy,
        #     ],
        #     axis=0,
        # )

        # Prepare experience buffer (histories)
        if self.o0_hist is None:
            self.o0_hist = self.augmented_o0
        else:
            self.o0_hist = np.concatenate([self.o0_hist, self.augmented_o0], axis=0)
        if self.g_hist is None:
            self.g_hist = self.augmented_g
        else:
            self.g_hist = np.concatenate([self.g_hist, self.augmented_g], axis=0)
        if self.a_hist is None:
            self.a_hist = self.augmented_a
        else:
            self.a_hist = np.concatenate([self.a_hist, self.augmented_a], axis=0)
        # xpos = self.parallel_best_s0_dots_T.pipeline_state.x.pos
        # xpos = xpos.reshape((-1, *xpos.shape[2:]))
        # if self.x0_dots_T_hist is None:
        #     self.x0_dots_T_hist = xpos
        # else:
        #     self.x0_dots_T_hist = np.concatenate(
        #         [
        #             self.x0_dots_T_hist,
        #             xpos,
        #         ],
        #         axis=0,
        #     )
        # a = self.parallel_best_a
        # a = a.reshape((-1, *a.shape[2:]))
        # if self.a_hist is None:
        #     self.a_hist = a
        # else:
        #     self.a_hist = np.concatenate([self.a_hist, a], axis=0)

        # Pruning
        buffer_size = 64 * self.n_parallels * self.n_elites * 20
        o0g = np.concatenate([self.o0_hist, self.g_hist], axis=-1)
        o0g_var = np.var(o0g, axis=0)

        if self.o0_hist.shape[0] > buffer_size:

            if self.args.prune_strat == "fifo":
                self.o0_hist = self.o0_hist[-buffer_size:]
                self.g_hist = self.g_hist[-buffer_size:]
                self.a_hist = self.a_hist[-buffer_size:]
            elif self.args.prune_strat == "random":
                n_trials = 1000
                best_var_reduction = np.inf
                best_survivor_idxs = None
                for _ in range(n_trials):
                    survivor_idxs = np.random.choice(
                        self.o0_hist.shape[0], size=buffer_size, replace=False
                    )
                    candidate_var = np.var(o0g[survivor_idxs], axis=0)
                    var_reduction = np.mean(o0g_var - candidate_var)
                    if var_reduction < best_var_reduction:
                        best_var_reduction = var_reduction
                        best_survivor_idxs = survivor_idxs
                        print(best_var_reduction)

                self.o0_hist = self.o0_hist[best_survivor_idxs]
                self.g_hist = self.g_hist[best_survivor_idxs]
                self.a_hist = self.a_hist[best_survivor_idxs]
            elif self.args.prune_strat == "goal_dist":
                deltas = self.g_hist[:, None] - self.g_all[None]
                goal_dists = np.linalg.norm(deltas, axis=-1)
                min_goal_dists = np.min(goal_dists, axis=-1)
                elite_idxs = np.argsort(min_goal_dists)[:buffer_size]
                self.o0_hist = self.o0_hist[elite_idxs]
                self.g_hist = self.g_hist[elite_idxs]
                self.a_hist = self.a_hist[elite_idxs]

        # self.x0_dots_T_hist = self.x0_dots_T_hist[-buffer_size:]
        # self.a_hist = self.a_hist[-buffer_size:]

        # self.writer.add_scalar(
        #     "train/buffer_size",
        #     # self.x0_dots_T_hist.shape[0],
        #     self.o0_hist.shape[0],
        #     self.global_simsteps_elapsed,
        # )

        # for j in range(o0g_var.shape[-1]):
        #     self.writer.add_scalar(
        #         f"train/o0g_var_{j:02d}",
        #         o0g_var[j],
        #         self.global_simsteps_elapsed,
        #     )

    def fit(self):
        combos = np.stack([np.zeros(64), np.arange(64) + 1], axis=-1).astype(int)

        fit_batch_size = 512
        n_global_steps = 1000
        eval_every = 100

        device = torch.device("cuda")
        self.proposal_model = self.proposal_model.to(device)
        adam = torch.optim.Adam(params=self.proposal_model.parameters(), lr=3e-4)
        pbar = tqdm(total=n_global_steps)
        for i in range(n_global_steps):
            adam.zero_grad()

            sample_idxs = np.random.randint(
                0, self.x0_dots_T_hist.shape[0], (fit_batch_size,)
            )
            combo_idxs = np.random.randint(0, combos.shape[0], (fit_batch_size,))
            chosen_combos = combos[combo_idxs]
            chosen_o0g = np.take_along_axis(
                self.x0_dots_T_hist[sample_idxs], chosen_combos[..., None], 1
            )
            chosen_o0 = chosen_o0g[:, 0]
            chosen_g = chosen_o0g[:, 1, -2:]
            chosen_o0g = np.concatenate([chosen_o0, chosen_g], axis=-1)
            chosen_a = self.a_hist[sample_idxs]

            o0g = torch.as_tensor(chosen_o0g, dtype=torch.float, device=device)
            a = torch.as_tensor(chosen_a, dtype=torch.float, device=device)

            if i == 0:
                seen_o0g = o0g * 1
                seen_a = a * 1

            if i % eval_every == 0:
                with torch.no_grad():
                    dist = self.proposal_model.forward(seen_o0g)
                    log_probs = dist.log_prob(seen_a).sum(-1)
                    seen_loss = -log_probs.mean()

            dist = self.proposal_model.forward(o0g)

            log_probs = dist.log_prob(a).sum(-1)
            loss = -log_probs.mean()
            loss.backward()
            adam.step()
            pbar.update(1)
            pbar.set_postfix({"loss": f"{seen_loss.item():.2e}"})

        pbar.close()
        self.proposal_model = self.proposal_model.cpu()

    def test(self):
        # self.logger.info(f"Testing...")
        o0_test = self.s0_test.pipeline_state.x.pos[:, :, :-1].reshape(
            -1, self.obs_size
        )
        # a_proposals = self.proposal_model.predict(o0_test, self.g_test)
        a_proposals = self.deployment_model.predict(o0_test, self.g_test)
        if self.deployment_model.n_proposals > 1:
            max_r_proposal = np.ones((self.env_container.batch_size,)) * -np.inf
            max_a_proposal = np.zeros(
                (self.env_container.batch_size, a_proposals.shape[-1])
            )

            for a in a_proposals:
                aa = self.action_populator.populate(a)
                s0_dots_T_proposals = self.rollouter.single_action_rollout(
                    self.s0_test, aa, 64
                )
                # self.global_simsteps_elapsed += self.env_container.batch_size * 64
                r, rr, rrr = self.evaluator.evaluate(
                    s0_dots_T_proposals.pipeline_state.x.pos[:, :, :, :-1],
                    self.g_test[:, None],
                )
                r += rr
                r *= 1 - rrr
                update_yes = r > max_r_proposal
                max_r_proposal = np.where(update_yes, r, max_r_proposal)
                max_a_proposal = np.where(update_yes[:, None], a, max_a_proposal)

            a = max_a_proposal
        else:
            a = a_proposals[0]
        aa = self.action_populator.populate(a)
        s0_dots_T_test = self.rollouter.single_action_rollout(self.s0_test, aa, 64)
        r_test, success_yes, failure_yes = self.evaluator.evaluate(
            s0_dots_T_test.pipeline_state.x.pos[:, :, :, :-1], self.g_test[:, None]
        )
        # Don't apply success bonus for testing
        r_test *= 1 - failure_yes
        success_yes *= 1 - failure_yes
        #
        # self.logger.info(f"Test avg. reward: {r_test.mean()}")
        # self.logger.info(f"Test success rate: {success_yes.mean()}")
        # self.writer.add_scalar(
        #     "test/avg_reward", r_test.mean(), self.global_simsteps_elapsed
        # )
        # self.writer.add_scalar(
        #     "test/success_rate", success_yes.mean(), self.global_simsteps_elapsed
        # )


class NopeOptimizer(GCSLOptimizer):
    def __init__(
        self,
        args,
        env_container: EnvContainer,
        action_populator: ActionPopulator,
        rollouter: BilliardsRollouter,
        evaluator: Evaluator,
        proposal_model: GCFitPredict,
    ):
        super().__init__(
            args, env_container, action_populator, rollouter, evaluator, proposal_model
        )

    def sample(self, deterministic=False):
        o0 = np.array(
            self.s0.pipeline_state.x.pos[..., :-1],
        )
        o0 = o0.reshape(o0.shape[0], -1)
        g = self.g
        # o0g = torch.as_tensor(np.concatenate([o0, g], axis=-1), dtype=torch.float)
        o0 = torch.as_tensor(o0)
        g = torch.as_tensor(g)
        a = self.proposal_model.predict(o0, g).detach().cpu().numpy()
        if deterministic:
            self.a_noised = a
        else:
            noise = np.random.normal(0, 1, size=a.shape) * 0.1
            self.a_noised = a + noise
        self.a_noised = unit_triangle_wave_np(self.a_noised)

        aa = self.action_populator.populate(self.a_noised)
        self.s0_dots_T = self.rollouter.single_action_rollout(self.s0, aa, 64)
        self.global_simsteps_elapsed += self.env_container.batch_size * 64

    def fit(self):
        combos = np.stack([np.zeros(64), np.arange(64) + 1], axis=-1).astype(int)
        chosen_o0g = self.x0_dots_T_hist[:, combos]
        chosen_o0 = chosen_o0g[..., 0, :]
        chosen_o0 = chosen_o0.reshape(-1, chosen_o0.shape[-1])
        chosen_g = chosen_o0g[..., 1, -2:]
        chosen_g = chosen_g.reshape(-1, chosen_g.shape[-1])
        chosen_a = self.a_hist[:, None][:, combos[:, 0]]
        chosen_a = chosen_a.reshape(-1, chosen_a.shape[-1])
        self.proposal_model.fit(chosen_o0, chosen_g, chosen_a)
