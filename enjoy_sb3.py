# @title Import MuJoCo, MJX, and Brax
import os
from argparse import ArgumentParser

import jax
import numpy as np
import torch.random
from tensorboardX import SummaryWriter

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
from rollo.env_containers import EnvContainer
from utils.vecenv import MyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)


# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"


class LogStdAnnealCallback(BaseCallback):

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        model = self.model
        ratio = self.num_timesteps / (self.total_timesteps - 100)
        ratio = min(1.0, ratio)
        init_std = torch.exp(torch.tensor(model.policy.log_std_init))
        bottom_std = torch.exp(torch.tensor(-5.0))
        std = init_std * (1 - ratio) + (bottom_std - init_std) * ratio
        model.policy.log_std.data[:] = torch.log(std)
        return True


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12345,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    # logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    logdir = os.path.join(proj_dir, "logdir", args.run_name, args.out_name)
    os.makedirs(logdir, exist_ok=True)
    outdir = os.path.join(proj_dir, "out", args.run_name, args.out_name)
    os.makedirs(outdir, exist_ok=True)
    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    env_name = "inverted_pendulum"
    backend = "mjx"
    batch_size = 1024

    episode_length = args.episode_length
    train_env_container = EnvContainer(
        env_name, backend, batch_size, True, episode_length=episode_length
    )
    eval_env_container = EnvContainer(
        env_name,
        backend,
        batch_size,
        False,
        episode_length=1000,
    )
    train_vecenv = MyVecEnv(train_env_container, args.seed)
    train_vecenv.seed(args.seed)
    eval_vecenv = MyVecEnv(eval_env_container, args.seed)
    eval_vecenv.seed(0)
    eval_callback = EvalCallback(
        eval_vecenv, n_eval_episodes=1, eval_freq=1000, log_path=logdir
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=outdir, name_prefix="rl_model"
    )
    n_episodes = args.n_episodes
    total_timesteps = batch_size * episode_length * n_episodes
    logstd_anneal_callback = LogStdAnnealCallback(total_timesteps)

    # model = PPO("MlpPolicy", "CartPole-v1", tensorboard_log=logdir).learn(10000)
    if args.checkpoint_path is not None:
        model = PPO.load(args.checkpoint_path, env=train_vecenv)
        model.tensorboard_log = logdir
        model.seed = args.seed
        if not args.continue_yes:
            model.policy.log_std.data[:] = args.log_std_init
            model.num_timesteps = 0
        # print("hehe")
    else:
        model = PPO(
            "MlpPolicy",
            train_vecenv,
            policy_kwargs={"log_std_init": args.log_std_init, "net_arch": [64, 64]},
            learning_rate=3e-4,
            max_grad_norm=0.1,
            batch_size=16384,
            n_epochs=10,
            n_steps=64,
            tensorboard_log=logdir,
            seed=args.seed,
        )
    # model.policy.log_std.requires_grad_(False)
    model.learn(
        total_timesteps,
        # callback=[eval_callback, checkpoint_callback, logstd_anneal_callback],
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="",
        progress_bar=True,
    )

    model.save(f"{outdir}/model.zip")
    logger.info(f"Saved to {outdir}/model.zip")

    logger.info("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    parser.add_argument("--env_name", type=str, default="walker2d")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--log_std_init", type=float, default=-2.0)
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
