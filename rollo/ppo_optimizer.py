import itertools
import os

import imageio
import jax
import numpy as np
import pyvista as pv
import torch
from gymnasium import spaces
from jax import tree_map
from torch import nn

from mlexp_utils.dirs import proj_dir
from rollo.action_populators import BilliardsActionPopulator, ActionPopulator
from rollo.env_containers import EnvContainer
from rollo.evaluators import Evaluator
from rollo.models import MyKNeighbors
from rollo.rollouters import BilliardsRollouter
from stable_baselines3 import PPO
from utils.train_utils import unit_triangle_wave_np
from viz.visual_data_pv import XMLVisualDataContainer


class PPOOptimizer:
    def __init__(
        self,
        args,
        logger,
        writer,
        env_container: EnvContainer,
        action_populator: ActionPopulator,
        rollouter: BilliardsRollouter,
        evaluator: Evaluator,
        proposal_model: nn.Module,
    ):
        self.o0_g = None
        self.action_size = 2
        self.g_hist = None
        self.o0_hist = None
        self.args = args
        self.logger = logger
        self.writer = writer

        self.env_container = env_container
        self.action_populator = action_populator
        self.rollouter = rollouter
        self.evaluator = evaluator
        self.proposal_model = proposal_model

        self.obs_size = self.env_container.env.observation_size
        self.deployment_model = MyKNeighbors(1, self.obs_size, 2)

        self.n_parallels = 8
        self.parallel_size = 8

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
        self.n_elites = 2
        # self.n_elites = 8

        self.x0_dots_T_hist = None
        self.a_hist = None

        self.augmented_o0 = None
        self.augmented_g = None
        self.augmented_a = None
        self.out_dir = os.path.join(
            proj_dir, "out", self.args.run_name, self.args.out_name
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.pl = pv.Plotter(off_screen=True, window_size=(608, 608))
        # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
        self.pl.add_axes()
        plane = pv.Plane(center=(0, 0, 0), i_size=3, j_size=3)
        self.pl.add_mesh(plane)

        xml_path = os.path.join(
            # proj_dir, "vendor", "toy", "src", "brax", "envs", "assets", "ant.xml"
            proj_dir,
            "vendor",
            "toy",
            "src",
            "brax",
            "envs",
            "assets",
            "billiards.xml",
        )
        visual_data = XMLVisualDataContainer(xml_path)

        deco_actor = self.pl.add_mesh(visual_data.meshes[0], color="green")
        self.char_actors = []
        for mesh in visual_data.meshes[1:]:
            actor = self.pl.add_mesh(mesh, color="green")
            self.char_actors.append(actor)

        self.pl.enable_shadows()

        self.outer_i = 0
        self.jax_rng = jax.random.PRNGKey(self.args.seed)

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
        self.s0_test = self.env_container.jit_env_reset(rng=self.jax_rng)
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)
        self.g_test_idxs = np.random.choice(
            self.g_all.shape[0], size=self.env_container.batch_size
        )
        self.g_test = self.g_all[self.g_test_idxs]
        self.global_simsteps_elapsed = 0

        self.ppo = PPO("MlpPolicy", self.env_container.env)

    def reset(self):
        self.parallel_best_a = np.zeros(
            (self.n_parallels, self.n_elites, self.action_size)
        )
        self.parallel_best_r = np.ones((self.n_parallels, self.n_elites)) * -np.inf

        self.s0 = self.env_container.jit_env_reset(rng=self.jax_rng)
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)
        self.parallel_s0 = tree_map(lambda x: x[: self.n_parallels], self.s0)
        self.parallel_best_s0_dots_T = tree_map(
            lambda x: x[:, None, None].repeat(self.n_elites, 1).repeat(65, 2),
            self.parallel_s0,
        )
        # reassign
        self.s0 = tree_map(lambda x: x.repeat(self.parallel_size, 0), self.parallel_s0)

        self.parallel_g_idxs = np.random.choice(
            self.g_all.shape[0], size=self.n_parallels
        )
        # self.parallel_g_idxs = np.arange(self.n_parallels)
        # self.parallel_g_idxs = np.array([0, 1, 2, 3, 4, 5, 0, 1])
        self.g_idxs = self.parallel_g_idxs.repeat(self.parallel_size, 0)
        self.g = self.g_all[self.g_idxs]
        self.climb_i = 0

    def sample(self, noise_std: float):
        self.logger.info(f"Sampling action")
        # Worry about devices later
        o0 = np.array(
            self.s0.pipeline_state.x.pos[:, :, :-1].reshape(-1, self.obs_size)
        )
        g = self.g
        self.o0_g = np.concatenate([o0, g], axis=-1)
        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = torch.tensor(self.o0_g)
            actions, self.values, self.log_probs = self.ppo.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        if isinstance(self.ppo.action_space, spaces.Box):
            if self.ppo.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions = self.ppo.policy.unscale_action(clipped_actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(
                    actions, self.ppo.action_space.low, self.ppo.action_space.high
                )

        self.a_noised = clipped_actions

        self.logger.info(f"Rolling out")
        aa = self.action_populator.populate(self.a_noised)
        self.s0_dots_T = self.rollouter.single_action_rollout(self.s0, aa, 64)
        self.global_simsteps_elapsed += self.env_container.batch_size * 64

        # Dump debug image here
        xpos = np.array(self.s0_dots_T.pipeline_state.x.pos)

        # Set ball init positions
        # for actor in self.char_actors:
        #     m = np.eye(4)
        #     m[:3, 3] = xpos[0, 0, 0]
        #     actor.user_matrix = m

        # Draw trajectory as tubes
        self.traj_splines = []
        self.traj_actors = []
        for i in range(self.env_container.batch_size):
            spline = pv.Spline(xpos[i, :, -1], n_points=65).tube(radius=0.01)
            spline["scalars"] = np.zeros(spline.n_points)
            self.traj_splines.append(spline)
            if spline.n_points > 0:
                spline_actor = self.pl.add_mesh(spline, scalars="scalars", clim=[0, 1])
                self.traj_actors.append(spline_actor)

        self.pl.camera.position = (-3.5, -3.5, 3.5)
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.render()

        img = np.array(self.pl.screenshot())
        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # plt.imshow(img)
        # plt.pause(0.1)
        # plt.show()
        # plt.close()
        w = imageio.get_writer(
            f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_00_sample.png"
        )
        w.append_data(img)
        w.close()

    def evaluate(self):
        self.logger.info(f"Evaluating...")
        r, rr, rrr = self.evaluator.evaluate(
            self.s0_dots_T.pipeline_state.x.pos[:, :, :, :-1],
            self.g[:, None],
        )
        self.r = r * 1
        self.r += rr
        self.r *= 1 - rrr

        # May have to happen inside `evaluate()`
        self.ppo.rollout_buffer.reset()  # to be on-policy
        self.ppo.rollout_buffer.add(
            self.o0_g,  # type: ignore[arg-type]
            self.a_noised,
            self.r,
            self._last_episode_starts,  # type: ignore[arg-type]
            self.values,
            self.log_probs,
        )
        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
        self.ppo.rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=np.zeros(self.env_container.batch_size, dtype=bool),
        )

        shaped_r = self.r
        for i, spline_mesh in enumerate(self.traj_splines):
            spline_mesh["scalars"] = shaped_r[i].repeat(spline_mesh.n_points)

        self.pl.camera.position = (-3.5, -3.5, 3.5)
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.render()

        img = np.array(self.pl.screenshot())
        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # plt.imshow(img)
        # plt.pause(0.1)
        # plt.show()
        # plt.close()
        w = imageio.get_writer(
            f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_01_evaluate.png"
        )
        w.append_data(img)
        w.close()

    def filter(self):
        self.logger.info(f"Filtering...")

        # If this works out of the box then we good
        self.ppo.train()

        # # Debug visual: destroy tubes, rebuild
        # for actor in self.traj_actors:
        #     self.pl.remove_actor(actor)
        # self.pl.meshes.clear()
        #
        # self.traj_splines = []
        # self.traj_actors = []
        # shaped_r = self.parallel_best_r
        # for i in range(self.n_parallels):
        #     for j in range(self.n_elites):
        #         spline = pv.Spline(
        #             self.parallel_best_s0_dots_T.pipeline_state.x.pos[i, j, :, -1],
        #             n_points=65,
        #         ).tube(radius=0.01)
        #         spline["scalars"] = shaped_r[i, j].repeat(spline.n_points)
        #         self.traj_splines.append(spline)
        #         if spline.n_points > 0:
        #             spline_actor = self.pl.add_mesh(
        #                 spline, scalars="scalars", clim=[0, 1]
        #             )
        #             self.traj_actors.append(spline_actor)
        #
        # self.pl.camera.position = (-3.5, -3.5, 3.5)
        # self.pl.camera.focal_point = (0, 0, 0)
        # self.pl.render()
        #
        # img = np.array(self.pl.screenshot())
        # # import matplotlib.pyplot as plt
        # #
        # # plt.figure()
        # # plt.imshow(img)
        # # plt.pause(0.1)
        # # plt.show()
        # # plt.close()
        # w = imageio.get_writer(
        #     f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_02_filter.png"
        # )
        # w.append_data(img)
        # w.close()
        #
        # for actor in self.traj_actors:
        #     self.pl.remove_actor(actor)
        # self.pl.meshes.clear()

        self.climb_i += 1

    def augment(self):
        self.logger.info(f"Augmenting...")

        self.logger.info(f"PPO doesn't use hindsight")

        self.writer.add_scalar(
            "train/buffer_size",
            # self.x0_dots_T_hist.shape[0],
            self.o0_hist.shape[0],
            self.global_simsteps_elapsed,
        )

        # for j in range(o0g_var.shape[-1]):
        #     self.writer.add_scalar(
        #         f"train/o0g_var_{j:02d}",
        #         o0g_var[j],
        #         self.global_simsteps_elapsed,
        #     )

    def fit(self):
        self.logger.info(f"Fitting...")
        # self.proposal_model.fit(self.augmented_o0, self.augmented_g, self.augmented_a)
        # self.proposal_model.fit(self.x0_dots_T_hist, self.a_hist)
        self.proposal_model.fit(self.o0_hist, self.g_hist, self.a_hist)
        self.deployment_model.fit(self.o0_hist, self.g_hist, self.a_hist)

    def test(self):
        self.logger.info(f"Testing...")
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

        self.logger.info(f"Test avg. reward: {r_test.mean()}")
        self.logger.info(f"Test success rate: {success_yes.mean()}")
        self.writer.add_scalar(
            "test/avg_reward", r_test.mean(), self.global_simsteps_elapsed
        )
        self.writer.add_scalar(
            "test/success_rate", success_yes.mean(), self.global_simsteps_elapsed
        )

        # Debug visual: destroy tubes, rebuild
        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()
        for actor in self.char_actors:
            self.pl.remove_actor(actor)

        self.traj_splines = []
        self.traj_actors = []
        shaped_r = r_test + success_yes
        for i in range(self.env_container.batch_size):
            spline = pv.Spline(
                s0_dots_T_test.pipeline_state.x.pos[i, :, -1],
                n_points=65,
            ).tube(radius=0.01)
            spline["scalars"] = shaped_r[i].repeat(spline.n_points)
            self.traj_splines.append(spline)
            if spline.n_points > 0:
                spline_actor = self.pl.add_mesh(spline, scalars="scalars", clim=[0, 1])
                self.traj_actors.append(spline_actor)

        self.pl.camera.position = (-3.5, -3.5, 3.5)
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.render()

        img = np.array(self.pl.screenshot())
        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # plt.imshow(img)
        # plt.pause(0.1)
        # plt.show()
        # plt.close()
        w = imageio.get_writer(
            f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_03_test.png"
        )
        w.append_data(img)
        w.close()

        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()
