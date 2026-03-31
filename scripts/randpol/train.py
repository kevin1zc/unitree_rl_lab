#!/usr/bin/env python3

"""Train a RANDPOL policy on Unitree Isaac Lab tasks."""

import gymnasium as gym
import inspect
import os
import pathlib
import shutil
import sys
import warnings
from datetime import datetime

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RANDPOL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment. Pass an explicit value for multi-seed studies.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
cli_args.add_randpol_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.randpol.config import RandpolRunnerCfg
from unitree_rl_lab.randpol.runner import RandpolOnPolicyRunner
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_iteration_summary(iteration: int, total_iterations: int, stats, start_time: datetime) -> None:
    iteration_time = stats.collection_time + stats.learning_time
    steps_per_second = int(stats.num_transitions / max(iteration_time, 1.0e-8))
    remaining_iterations = max(0, total_iterations - iteration - 1)
    eta_seconds = remaining_iterations * iteration_time
    elapsed_seconds = (datetime.now() - start_time).total_seconds()

    print("\n" + "#" * 80)
    print(f"{'Learning iteration ' + str(iteration) + '/' + str(total_iterations):^80}")
    print()
    print(f"{'Total steps:':>34} {stats.total_steps}")
    print(f"{'Steps per second:':>34} {steps_per_second}")
    print(f"{'Collection time:':>34} {stats.collection_time:.3f}s")
    print(f"{'Learning time:':>34} {stats.learning_time:.3f}s")
    print(f"{'Returns time:':>34} {stats.returns_time:.3f}s")
    print(f"{'Value fit time:':>34} {stats.value_fit_time:.3f}s")
    print(f"{'Policy update time:':>34} {stats.policy_update_time:.3f}s")
    print(f"{'Mean value loss:':>34} {stats.value_loss:.4f}")
    print(f"{'Mean policy loss:':>34} {stats.policy_loss:.4f}")
    print(f"{'Mean reward:':>34} {stats.mean_reward:.2f}")
    print(f"{'Mean episode length:':>34} {stats.mean_episode_length:.2f}")
    print(f"{'Mean action std:':>34} {stats.mean_action_std:.2f}")

    for key in sorted(stats.extras):
        print(f"{key + ':':>34} {stats.extras[key]:.4f}")

    print("-" * 80)
    print(f"{'Iteration time:':>34} {iteration_time:.2f}s")
    print(f"{'Time elapsed:':>34} {_format_duration(elapsed_seconds)}")
    print(f"{'ETA:':>34} {_format_duration(eta_seconds)}")
    print()
    print("#" * 80)


def _configure_randpol_env_cfg(env_cfg, agent_cfg: RandpolRunnerCfg) -> None:
    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is None or agent_cfg.obs_history_length <= 0:
        return
    for group_name in ("policy", "critic"):
        group_cfg = getattr(observations_cfg, group_name, None)
        if group_cfg is None:
            continue
        if hasattr(group_cfg, "history_length"):
            group_cfg.history_length = agent_cfg.obs_history_length
        if hasattr(group_cfg, "flatten_history_dim"):
            group_cfg.flatten_history_dim = True


@hydra_task_config(args_cli.task, "randpol_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RandpolRunnerCfg):
    """Train with a RANDPOL agent."""
    agent_cfg = cli_args.update_randpol_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    _configure_randpol_env_cfg(env_cfg, agent_cfg)

    if args_cli.seed is None:
        print(f"[WARN] No explicit --seed was provided. Reusing configured seed: {agent_cfg.seed}")
    print(f"[INFO] Using seed: {agent_cfg.seed}")

    log_root_path = os.path.join("logs", "randpol", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    if f"seed{agent_cfg.seed}" not in log_dir:
        log_dir += f"_seed{agent_cfg.seed}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner = RandpolOnPolicyRunner(env, agent_cfg)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    metadata = {
        "task": args_cli.task,
        "agent_cfg": agent_cfg.to_dict(),
        "env_cfg_class": env_cfg.__class__.__name__,
    }
    with open(os.path.join(log_dir, "params", "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    train_start_time = datetime.now()
    for iteration in range(agent_cfg.max_iterations):
        stats = runner.learn_iteration()
        _print_iteration_summary(iteration, agent_cfg.max_iterations, stats, train_start_time)
        writer.add_scalar("train/value_loss", stats.value_loss, iteration)
        writer.add_scalar("train/policy_loss", stats.policy_loss, iteration)
        writer.add_scalar("train/returns_time", stats.returns_time, iteration)
        writer.add_scalar("train/value_fit_time", stats.value_fit_time, iteration)
        writer.add_scalar("train/policy_update_time", stats.policy_update_time, iteration)
        if stats.num_completed_episodes > 0 and not math.isnan(stats.mean_reward):
            writer.add_scalar("train/mean_reward", stats.mean_reward, iteration)
            writer.add_scalar("train/mean_episode_length", stats.mean_episode_length, iteration)
        writer.add_scalar("train/steps_per_second", stats.num_transitions / max(stats.collection_time + stats.learning_time, 1.0e-8), iteration)
        writer.add_scalar("train/mean_action_std", stats.mean_action_std, iteration)
        for key, value in stats.extras.items():
            writer.add_scalar(key, value, iteration)

        if iteration % agent_cfg.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, "checkpoints", f"model_{iteration}.pt")
            torch.save(runner.state_dict(), checkpoint_path)

    if agent_cfg.max_iterations <= 0:
        warnings.warn("max_iterations <= 0, skipping final checkpoint save.", stacklevel=1)
    else:
        checkpoint_path = os.path.join(log_dir, "checkpoints", f"model_{agent_cfg.max_iterations - 1}.pt")
        torch.save(runner.state_dict(), checkpoint_path)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
