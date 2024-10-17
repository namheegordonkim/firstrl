import itertools
import os

import imageio
import jax
import numpy as np
import pyvista as pv
from jax import tree_map
from torch import nn

from mlexp_utils.dirs import proj_dir
from rollo.action_populators import BilliardsActionPopulator, ActionPopulator
from rollo.env_containers import EnvContainer
from rollo.evaluators import Evaluator
from rollo.models import MyKNeighbors
from rollo.rollouters import BilliardsRollouter
from utils.train_utils import unit_triangle_wave_np
from viz.visual_data_pv import XMLVisualDataContainer


class NopeRLOptimizer:
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

        self.n_parallels = self.args.n_parallels
        self.parallel_size = self.args.parallel_size

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
        self.s0_test = self.env_container.jit_env_reset(rng=jax.random.PRNGKey(0))
        self.jax_rng, _ = jax.random.split(self.jax_rng, 2)
        np.random.seed(0)
        self.g_test_idxs = np.random.choice(
            self.g_all.shape[0], size=self.env_container.batch_size
        )
        np.random.seed(self.args.seed)
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

        if self.outer_i == 0 or self.args.s0_strat == "naive":
            # Just select the first n_parallels; should preserve uniformness
            self.parallel_s0 = tree_map(lambda x: x[: self.n_parallels], self.s0)
            self.parallel_g_idxs = self.g_idxs[: self.n_parallels]

        else:
            # Roll out the sampled combos and then choose the poorest performing ones
            o0 = np.array(
                self.s0.pipeline_state.x.pos[:, :, :-1].reshape(-1, self.obs_size)
            )
            a_proposals = self.deployment_model.predict(o0, self.g)
            a = a_proposals[0]
            aa = self.action_populator.populate(a)
            s0_dots_T = self.rollouter.single_action_rollout(self.s0, aa, 64)
            self.global_simsteps_elapsed += self.env_container.batch_size * 64
            r, rr, rrr = self.evaluator.evaluate(
                s0_dots_T.pipeline_state.x.pos[:, :, :, :-1],
                self.g[:, None],
            )
            r += rr
            r *= 1 - rrr
            argsort_r = np.argsort(r)

            if self.args.s0_strat == "try":
                self.parallel_s0 = tree_map(
                    lambda x: x[argsort_r[: self.n_parallels]], self.s0
                )
                self.parallel_g_idxs = self.g_idxs[argsort_r[: self.n_parallels]]

            elif self.args.s0_strat == "hybrid":
                self.parallel_s0 = tree_map(
                    lambda x: np.concatenate(
                        [
                            x[: self.n_parallels // 2],
                            x[argsort_r[: self.n_parallels // 2]],
                        ],
                        axis=0,
                    ),
                    self.s0,
                )
                self.parallel_g_idxs = np.concatenate(
                    [
                        self.g_idxs[: self.n_parallels // 2],
                        self.g_idxs[argsort_r[: self.n_parallels // 2]],
                    ],
                    axis=0,
                )

        # self.parallel_g_idxs = np.arange(self.n_parallels)
        # self.parallel_g_idxs = np.array([0, 1, 2, 3, 4, 5, 0, 1])
        # reassign
        self.s0 = tree_map(lambda x: x.repeat(self.parallel_size, 0), self.parallel_s0)
        self.g_idxs = self.parallel_g_idxs.repeat(self.parallel_size, 0)
        self.g = self.g_all[self.g_idxs]
        self.climb_i = 0

        self.parallel_best_s0_dots_T = tree_map(
            lambda x: x[:, None, None].repeat(self.n_elites, 1).repeat(65, 2),
            self.parallel_s0,
        )

    def sample(self, noise_std: float):
        if self.args.strawman == 0:
            self.logger.info("Strawmanning")
            self.a = np.random.uniform(
                -1, 1, (self.env_container.batch_size, self.action_size)
            )
        else:
            self.writer.add_scalar(
                "train/noise_std", noise_std, self.global_simsteps_elapsed
            )

            if self.climb_i == 0:
                if self.outer_i > 0:
                    self.logger.info(f"Using proposal to warm-start")
                    # Worry about devices later
                    o0 = np.array(
                        self.s0.pipeline_state.x.pos[:, :, :-1].reshape(
                            -1, self.obs_size
                        )
                    )
                    g = self.g
                    a_proposals = self.proposal_model.predict(o0, g)
                    if self.args.n_proposals > 1:
                        max_r_proposal = (
                            np.ones((self.env_container.batch_size,)) * -np.inf
                        )
                        max_a_proposal = np.zeros(
                            (self.env_container.batch_size, a_proposals.shape[-1])
                        )

                        for a in a_proposals:
                            aa = self.action_populator.populate(a)
                            s0_dots_T_proposals = self.rollouter.single_action_rollout(
                                self.s0, aa, 64
                            )
                            self.global_simsteps_elapsed += (
                                self.env_container.batch_size * 64
                            )
                            r, rr, rrr = self.evaluator.evaluate(
                                s0_dots_T_proposals.pipeline_state.x.pos[:, :, :, :-1],
                                self.g[:, None],
                            )
                            r += rr
                            r *= 1 - rrr
                            update_yes = r > max_r_proposal
                            max_r_proposal = np.where(update_yes, r, max_r_proposal)
                            max_a_proposal = np.where(
                                update_yes[:, None], a, max_a_proposal
                            )

                        self.a = max_a_proposal
                    else:
                        self.a = a_proposals[0]
                    # a = self.proposal_model.forward(o0, g)
                    # self.a = a.cpu().detach().numpy()
                else:
                    self.logger.info(f"Uniform random init")
                    self.a = np.random.uniform(
                        -1, 1, (self.env_container.batch_size, self.action_size)
                    )
            else:
                self.logger.info(f"Hehe")
                self.a = self.parallel_best_a[:, 0].repeat(self.parallel_size, 0)

        self.logger.info(f"Sampling and adding noise")
        noise = np.random.normal(0, noise_std, size=self.a.shape)
        self.a_noised = self.a + noise
        self.a_noised = unit_triangle_wave_np(self.a_noised)

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
        # w = imageio.get_writer(
        #     f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_00_sample.png"
        # )
        # w.append_data(img)
        # w.close()

    def evaluate(self):
        self.logger.info(f"Evaluating...")
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

        # self.r = self.evaluator.evaluate(
        #     self.s0_dots_T.pipeline_state.x.pos[:, :, 0, :-1],
        #     self.g_all[None].repeat(self.env_container.batch_size, axis=0),
        # )
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
        # w = imageio.get_writer(
        #     f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_01_evaluate.png"
        # )
        # w.append_data(img)
        # w.close()

    def filter(self):
        self.logger.info(f"Filtering...")
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

        # Debug visual: destroy tubes, rebuild
        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()

        self.traj_splines = []
        self.traj_actors = []
        shaped_r = self.parallel_best_r
        for i in range(self.n_parallels):
            for j in range(self.n_elites):
                spline = pv.Spline(
                    self.parallel_best_s0_dots_T.pipeline_state.x.pos[i, j, :, -1],
                    n_points=65,
                ).tube(radius=0.01)
                spline["scalars"] = shaped_r[i, j].repeat(spline.n_points)
                self.traj_splines.append(spline)
                if spline.n_points > 0:
                    spline_actor = self.pl.add_mesh(
                        spline, scalars="scalars", clim=[0, 1]
                    )
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
        # w = imageio.get_writer(
        #     f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_02_filter.png"
        # )
        # w.append_data(img)
        # w.close()

        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()

        self.climb_i += 1

        self.writer.add_scalar(
            "train/avg_best_reward",
            np.mean(self.parallel_best_r),
            self.global_simsteps_elapsed,
        )

    def augment(self):
        self.logger.info(f"Augmenting...")

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

        self.writer.add_scalar(
            "train/buffer_size",
            # self.x0_dots_T_hist.shape[0],
            self.o0_hist.shape[0],
            self.global_simsteps_elapsed,
        )

        for j in range(o0g_var.shape[-1]):
            self.writer.add_scalar(
                f"train/o0g_var_{j:02d}",
                o0g_var[j],
                self.global_simsteps_elapsed,
            )

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
        # w = imageio.get_writer(
        #     f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_03_test.png"
        # )
        # w.append_data(img)
        # w.close()

        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()
