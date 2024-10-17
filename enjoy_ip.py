from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import numpy as np
import pyvista as pv
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from torch import nn

from rollo.env_containers import EnvContainer
from rollo.learners import ReinforceLearner, Learner
from rollo.rollouters import PolicyRollouter
from rollo.torch_nets import ProbMLP, MLP
from viz.visual_data import XMLVisualDataContainer
import matplotlib


@dataclass
class VisualArray:
    meshes: np.ndarray
    actors: np.ndarray


class AppState:
    def __init__(
        self,
        scene_visuals: VisualArray,
        trail_visuals: VisualArray,
        guide_visuals: VisualArray,
        learner: ReinforceLearner,
    ):
        self.show_axes = False
        self.pose_idx = 0

        self.scene_meshes = scene_visuals.meshes
        self.scene_actors = scene_visuals.actors
        self.trail_meshes = trail_visuals.meshes
        self.trail_actors = trail_visuals.actors
        self.guide_meshes = guide_visuals.meshes
        self.guide_actors = guide_visuals.actors

        self.optimizer = learner
        self.env_container = learner.env_container
        self.rollouter = learner.rollouter

        self.trajectory = None
        self.trajectory_t = 0

        self.first_time = True

        self.play_mode = False
        self.deterministic = False
        self.show_trails = True
        self.show_guides = True
        self.color_code = 0

        self.stepping = False
        self.step_i = 0
        self.n_steps = 10
        self.n_epochs = 10

        self.rollout_length = 64


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )

        changed, app_state.play_mode = imgui.checkbox("Play Mode", app_state.play_mode)
        changed, app_state.rollout_length = imgui.slider_int("Rollout Length", app_state.rollout_length, 1, 1000)
        changed, app_state.deterministic = imgui.checkbox("Deterministic", app_state.deterministic)

        clicked = imgui.button("Reset")
        if app_state.first_time or clicked:
            app_state.optimizer.reset()

        sample_clicked = imgui.button("Sample")
        if sample_clicked:
            print("Sampling")
            app_state.optimizer.sample(app_state.rollout_length, app_state.deterministic)

        imgui.same_line()

        evaluate_clicked = imgui.button("Evaluate")
        if evaluate_clicked:
            print("Evaluating")
            app_state.optimizer.evaluate()

        imgui.same_line()
        weigh_clicked = imgui.button("Weigh")
        if weigh_clicked:
            print("Weighing")
            app_state.optimizer.weigh()
        imgui.same_line()
        clicked = imgui.button("Learn")
        if clicked:
            print("Learning")
            app_state.optimizer.learn(app_state.n_epochs)

        changed, app_state.n_epochs = imgui.slider_int("# Epochs", app_state.n_epochs, 1, 100)
        changed, app_state.n_steps = imgui.slider_int("# Steps", app_state.n_steps, 1, 100)
        clicked = imgui.button("Step")
        if clicked and not app_state.stepping:
            app_state.stepping = True
            app_state.step_i = 0

        changed, app_state.show_trails = imgui.checkbox("Show Trails", app_state.show_trails)

        imgui.text("Trail Color Code")
        cc_radio_clicked1 = imgui.radio_button("Body", app_state.color_code == 0)
        if cc_radio_clicked1:
            app_state.color_code = 0
        cc_radio_clicked2 = imgui.radio_button("Step Reward", app_state.color_code == 1)
        if cc_radio_clicked2:
            app_state.color_code = 1
        cc_radio_clicked3 = imgui.radio_button("Cumulative Reward", app_state.color_code == 2)
        if cc_radio_clicked3:
            app_state.color_code = 2
        cc_radio_clicked4 = imgui.radio_button("Estimated Value", app_state.color_code == 3)
        if cc_radio_clicked4:
            app_state.color_code = 3
        cc_radio_clicked5 = imgui.radio_button("Advantage", app_state.color_code == 4)
        if cc_radio_clicked5:
            app_state.color_code = 4

        cc_radio_clicked = np.any([cc_radio_clicked1, cc_radio_clicked2, cc_radio_clicked3, cc_radio_clicked4, cc_radio_clicked5])
        imgui.same_line()

        imgui.end()

        # Stepping
        if app_state.stepping:
            app_state.optimizer.sample(app_state.rollout_length, app_state.deterministic)
            app_state.optimizer.evaluate()
            app_state.optimizer.weigh()
            app_state.optimizer.learn(app_state.n_epochs)

        if app_state.play_mode:
            transition = app_state.rollouter.rollout(
                app_state.env_container.env_state,
                app_state.optimizer.policy,
                1,
                app_state.deterministic,
            )
            app_state.env_container.env_state, _ = jax.tree.map(lambda x: x[:, -1], transition)
            # Animating
            for i in range(len(app_state.scene_actors)):
                pos = np.array(app_state.env_container.env_state.pipeline_state.x.pos[0, i])
                quat = np.array(app_state.env_container.env_state.pipeline_state.x.rot[0, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m

        else:
            # Guide and trails
            for i in range(app_state.env_container.batch_size):
                for j in range(2):
                    if app_state.show_trails and app_state.optimizer.tau is not None:
                        if app_state.stepping or sample_clicked:
                            pos = np.array(app_state.optimizer.tau.pipeline_state.x.pos[i, :, j])
                            if j == 1:
                                quat = np.array(app_state.optimizer.tau.pipeline_state.x.rot[i, :, j])
                                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]
                                offset = np.array([[0, 0, 0.6]]).repeat(quat.shape[0], 0)
                                pos += Rotation.from_quat(quat).apply(offset)
                            app_state.trail_meshes[i, j].points = pos
                            # app_state.trail_meshes[i, j].point_data_to_cell_data()
                            app_state.trail_meshes[i, j].lines = pv.MultipleLines(points=pos).lines

                        app_state.trail_actors[i, j].SetVisibility(True)
                    else:
                        app_state.trail_actors[i, j].SetVisibility(False)

            # Animating
            for i in range(len(app_state.scene_actors)):
                pos = np.array(app_state.env_container.env_state.pipeline_state.x.pos[0, i])
                quat = np.array(app_state.env_container.env_state.pipeline_state.x.rot[0, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m

        # Coloring
        if app_state.show_trails and (app_state.stepping or cc_radio_clicked or sample_clicked or evaluate_clicked or weigh_clicked):
            for i in range(len(app_state.trail_meshes)):
                for j in range(2):
                    color = {
                        0: [1.0, 1.0, 1.0],
                        1: [1.0, 0.0, 0.0],
                    }[j]
                    colors = np.array([color]).repeat(app_state.trail_meshes[i, j].n_points, 0)

                    if app_state.color_code == 1 and app_state.optimizer.tau is not None:
                        rewards = np.array(app_state.optimizer.tau.reward)
                        colors = matplotlib.colormaps["viridis"](rewards[i])[..., :3]
                    elif app_state.color_code == 2 and app_state.optimizer.R is not None:
                        color = matplotlib.colormaps["viridis"](app_state.optimizer.R[i, [0]] / app_state.rollout_length)[..., :3]
                        colors = np.array([color]).repeat(app_state.trail_meshes[i, j].n_points, 0)
                    elif app_state.color_code == 3 and app_state.optimizer.V is not None:
                        colors = matplotlib.colormaps["viridis"](app_state.optimizer.V[i] / app_state.rollout_length)[..., :3]
                    elif app_state.color_code == 4 and app_state.optimizer.A is not None:
                        colors = matplotlib.colormaps["viridis"](app_state.optimizer.A[i])[..., :3]

                    app_state.trail_meshes[i, j].point_data["color"] = colors

        if app_state.stepping:
            app_state.step_i += 1
            if app_state.step_i >= app_state.n_steps:
                app_state.stepping = False

        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main(args):
    pl = ImguiPlotter()
    plane = pv.Plane(center=(0, 0, -0.5), direction=(0, 0, 1), i_size=10, j_size=10)
    pl.add_mesh(plane, show_edges=True)
    pl.add_axes()

    env_name = "inverted_pendulum"
    backend = "mjx"
    batch_size = 128
    env_container = EnvContainer(env_name, backend, batch_size)

    mjcf_path = "brax/envs/assets/inverted_pendulum.xml"
    visual = XMLVisualDataContainer(mjcf_path)
    n = len(visual.meshes)
    scene_meshes = np.empty((n,), dtype=object)
    scene_actors = np.empty((n,), dtype=object)
    for j, mesh in enumerate(visual.meshes):
        # Assign white color to each mesh
        color = {
            0: [1.0, 1.0, 1.0],
            1: [1.0, 0.0, 0.0],
        }[j]
        mesh.cell_data["color"] = np.array([color]).repeat(mesh.n_cells, 0)
        actor = pl.add_mesh(mesh, scalars="color", rgb=True, show_scalar_bar=False)
        scene_meshes[j] = mesh
        scene_actors[j] = actor
    scene_visuals = VisualArray(scene_meshes, scene_actors)

    policy = ProbMLP(
        input_size=env_container.env.observation_size,
        output_size=env_container.env.action_size,
        hidden_size=64,
    )
    value = MLP(
        input_size=env_container.env.observation_size,
        output_size=1,
        hidden_size=64,
    )
    rollouter = PolicyRollouter(env_container)
    learner = ReinforceLearner(args, env_container, rollouter, policy, value)

    trail_meshes = np.empty((batch_size, 2), dtype=object)
    trail_actors = np.empty((batch_size, 2), dtype=object)
    guide_meshes = np.empty((batch_size,), dtype=object)
    guide_actors = np.empty((batch_size,), dtype=object)
    for i in range(batch_size):
        for j in range(2):
            color = {
                0: [1.0, 1.0, 1.0],
                1: [1.0, 0.0, 0.0],
            }[j]
            trail_mesh = pv.MultipleLines(points=np.zeros((2, 3)))
            trail_mesh.point_data["color"] = np.array([color]).repeat(trail_mesh.n_points, 0) * 1
            trail_actor = pl.add_mesh(trail_mesh, rgb=True, scalars="color", show_scalar_bar=False)
            trail_meshes[i, j] = trail_mesh
            trail_actors[i, j] = trail_actor
            trail_actor.SetVisibility(False)
        guide_mesh = pv.MultipleLines(points=np.zeros((3, 3)))
        guide_mesh.cell_data["color"] = np.array([[0.0, 0.0, 1.0]]).repeat(guide_mesh.n_cells, 0) * 1
        guide_actor = pl.add_mesh(guide_mesh, rgb=True, scalars="color", show_scalar_bar=False)
        guide_actor.SetVisibility(False)
        guide_meshes[i] = guide_mesh
        guide_actors[i] = guide_actor

    trail_visuals = VisualArray(trail_meshes, trail_actors)
    guide_visuals = VisualArray(guide_meshes, guide_actors)

    # Run the GUI
    app_state = AppState(scene_visuals, trail_visuals, guide_visuals, learner)
    setup_and_run_gui(pl, app_state)
    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
