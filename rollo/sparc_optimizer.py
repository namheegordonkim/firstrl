import os

import imageio
import jax
import numpy as np
import torch
import pyvista as pv
from jax import tree_map

from mlexp_utils.dirs import proj_dir
from rollo.action_populators import ActionPopulator, BilliardsActionPopulator
from rollo.env_containers import EnvContainer
from rollo.evaluators import Evaluator
from rollo.rollouters import BilliardsRollouter
from torch import nn

from viz.visual_data_pv import XMLVisualDataContainer


class SparcOptimizer:
    def __init__(
        self,
        args,
        logger,
        env_container: EnvContainer,
        action_populator: BilliardsActionPopulator,
        rollouter: BilliardsRollouter,
        evaluator: Evaluator,
        proposal_model: nn.Module,
    ):
        self.args = args
        self.logger = logger

        self.env_container = env_container
        self.action_populator = action_populator
        self.rollouter = rollouter
        self.evaluator = evaluator
        self.proposal_model = proposal_model

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

    def reset(self):
        self.parallel_best_a = np.zeros((self.n_parallels, self.n_elites, 1))
        self.parallel_best_r = np.ones((self.n_parallels, self.n_elites)) * -np.inf

        self.s0 = self.env_container.jit_env_reset(rng=jax.random.PRNGKey(seed=0))
        self.parallel_s0 = tree_map(lambda x: x[: self.n_parallels], self.s0)
        self.parallel_best_s0_dots_T = tree_map(
            lambda x: x[:, None, None].repeat(self.n_elites, 1).repeat(65, 2),
            self.parallel_s0,
        )
        # reassign
        self.s0 = tree_map(lambda x: x.repeat(self.parallel_size, 0), self.parallel_s0)

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
        self.parallel_g_idxs = np.random.choice(
            self.g_all.shape[0], size=self.n_parallels
        )
        self.g_idxs = self.parallel_g_idxs.repeat(self.parallel_size, 0)
        self.g = self.g_all[self.g_idxs]
        self.climb_i = 0

    def sample(self):
        if self.climb_i == 0:
            self.logger.info(f"Using proposal to warm-start")
            # Worry about devices later
            o0 = torch.as_tensor(
                np.array(self.s0.pipeline_state.x.pos[:, 0, :-1]), dtype=torch.float
            )
            g = torch.as_tensor(self.g, dtype=torch.float)
            a = self.proposal_model.forward(o0, g)
            self.a = a.cpu().detach().numpy()
        else:
            self.logger.info(f"Hehe")
            self.a = self.parallel_best_a[:, 0].repeat(self.parallel_size, 0)

        self.logger.info(f"Sampling and adding noise")
        noise = np.random.normal(0, 0.1, size=self.a.shape)
        self.a_noised = self.a + noise

        self.logger.info(f"Rolling out")
        aa = self.action_populator.populate(self.a_noised)
        self.s0_dots_T = self.rollouter.single_action_rollout(self.s0, aa, 64)

        # Dump debug image here
        xpos = np.array(self.s0_dots_T.pipeline_state.x.pos)

        # Set ball init positions
        for actor in self.char_actors:
            m = np.eye(4)
            m[:3, 3] = xpos[0, 0, 0]
            actor.user_matrix = m

        # Draw trajectory as tubes
        self.traj_splines = []
        self.traj_actors = []
        for i in range(self.env_container.batch_size):
            spline = pv.Spline(xpos[i, :, 0], n_points=2).tube(radius=0.01)
            spline["scalars"] = np.zeros(spline.n_points)
            self.traj_splines.append(spline)
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
        self.r = self.evaluator.evaluate(
            self.s0_dots_T.pipeline_state.x.pos[:, :, 0, :-1],
            self.g[:, None],
        )
        # self.r = self.evaluator.evaluate(
        #     self.s0_dots_T.pipeline_state.x.pos[:, :, 0, :-1],
        #     self.g_all[None].repeat(self.env_container.batch_size, axis=0),
        # )
        shaped_r = np.exp(-((self.r / (0.3)) ** 2))
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
        shaped_r = np.exp(-((self.parallel_best_r / (0.3)) ** 2))
        for i in range(self.n_parallels):
            for j in range(self.n_elites):
                spline = pv.Spline(
                    self.parallel_best_s0_dots_T.pipeline_state.x.pos[i, j, :, 0],
                    n_points=2,
                ).tube(radius=0.01)
                spline["scalars"] = shaped_r[i, j].repeat(spline.n_points)
                self.traj_splines.append(spline)
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
            f"{self.out_dir}/{self.outer_i:02d}_{self.climb_i:02d}_02_filter.png"
        )
        w.append_data(img)
        w.close()

        for actor in self.traj_actors:
            self.pl.remove_actor(actor)
        self.pl.scene_meshes.clear()

        self.climb_i += 1

    def augment(self):
        self.logger.info(f"Augmenting...")

    def fit(self):
        self.logger.info(f"Fitting...")
