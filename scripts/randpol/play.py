#!/usr/bin/env python3

"""Play a trained RANDPOL checkpoint on a Unitree Isaac Lab task."""

import argparse
import pathlib
import sys
import time

import gymnasium as gym

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained RANDPOL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video while playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument(
    "--experiment_name", type=str, default=None, help="Experiment folder name under logs/randpol/."
)
parser.add_argument("--load_run", type=str, default=".*", help="Regex for the run directory to load from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to load.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--print-terrain-levels",
    action="store_true",
    default=False,
    help="Print terrain level statistics after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.randpol.runner import RandpolOnPolicyRunner
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _default_experiment_name(task_name: str) -> str:
    return task_name.lower().replace("-", "_").removesuffix("_play")


def _extract_actor_obs(obs_dict: dict[str, torch.Tensor], actor_obs_group: str) -> torch.Tensor:
    if actor_obs_group in obs_dict:
        return obs_dict[actor_obs_group]
    if "policy" in obs_dict:
        return obs_dict["policy"]
    return obs_dict[next(iter(obs_dict.keys()))]


def _print_terrain_levels(env) -> None:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is None:
        return
    terrain_levels = terrain_levels.detach().cpu()
    if terrain_levels.numel() == 0:
        return
    print(
        "[INFO] Terrain levels in play: "
        f"mean={float(terrain_levels.float().mean()):.3f}, "
        f"min={int(terrain_levels.min())}, "
        f"max={int(terrain_levels.max())}"
    )


def _restore_agent_cfg_from_checkpoint(agent_cfg, checkpoint_state: dict) -> None:
    cfg_state = checkpoint_state.get("cfg")
    if not isinstance(cfg_state, dict):
        return
    for key, value in cfg_state.items():
        if hasattr(agent_cfg, key):
            setattr(agent_cfg, key, value)
    if hasattr(agent_cfg, "activation") and "activation" not in cfg_state:
        agent_cfg.activation = "elu"
    if hasattr(agent_cfg, "obs_history_length") and "obs_history_length" not in cfg_state:
        agent_cfg.obs_history_length = 0


def _configure_randpol_env_cfg(env_cfg, agent_cfg) -> None:
    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is None or getattr(agent_cfg, "obs_history_length", 0) <= 0:
        return
    for group_name in ("policy", "critic"):
        group_cfg = getattr(observations_cfg, group_name, None)
        if group_cfg is None:
            continue
        if hasattr(group_cfg, "history_length"):
            group_cfg.history_length = agent_cfg.obs_history_length
        if hasattr(group_cfg, "flatten_history_dim"):
            group_cfg.flatten_history_dim = True


def _configure_multi_terrain_play_view(env_cfg) -> None:
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator is None:
        return

    num_terrain_types = len(terrain_generator.sub_terrains)
    if num_terrain_types <= 1:
        return

    # Use equal proportions in play so every active terrain family gets a visible column.
    for sub_terrain_cfg in terrain_generator.sub_terrains.values():
        sub_terrain_cfg.proportion = 1.0

    terrain_generator.num_cols = max(int(terrain_generator.num_cols), num_terrain_types)
    terrain_generator.num_rows = max(int(terrain_generator.num_rows), 3)
    env_cfg.scene.num_envs = max(int(env_cfg.scene.num_envs), int(terrain_generator.num_cols))

    span_x = terrain_generator.size[0] * terrain_generator.num_rows + 2.0 * terrain_generator.border_width
    span_y = terrain_generator.size[1] * terrain_generator.num_cols + 2.0 * terrain_generator.border_width
    span = max(span_x, span_y)
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.eye = (0.75 * span, 0.75 * span, 0.55 * span)
    print(
        "[INFO] Configured play view for multiple terrain types: "
        f"rows={terrain_generator.num_rows}, cols={terrain_generator.num_cols}, "
        f"viewer_eye={env_cfg.viewer.eye}, equalized_proportions=True"
    )


def _resolve_checkpoint_path(agent_cfg) -> str:
    experiment_name = args_cli.experiment_name or agent_cfg.experiment_name or _default_experiment_name(args_cli.task)
    log_root_path = os.path.abspath(os.path.join("logs", "randpol", experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, run_dir=args_cli.load_run, other_dirs=["checkpoints"], checkpoint=".*")


def main():
    """Play with a trained RANDPOL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "randpol_cfg_entry_point")
    if agent_cfg.experiment_name == "":
        agent_cfg.experiment_name = _default_experiment_name(args_cli.task)

    resume_path = _resolve_checkpoint_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    checkpoint_state = torch.load(resume_path, map_location="cpu")
    _restore_agent_cfg_from_checkpoint(agent_cfg, checkpoint_state)
    agent_cfg.device = env_cfg.sim.device
    _configure_randpol_env_cfg(env_cfg, agent_cfg)
    _configure_multi_terrain_play_view(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = RandpolOnPolicyRunner(env, agent_cfg, update_normalization=False)
    runner.load_state_dict(checkpoint_state)
    policy = runner.get_inference_policy()
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")

    obs_dict, _ = env.reset()
    if args_cli.print_terrain_levels:
        _print_terrain_levels(env)
    dt = env.unwrapped.step_dt
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actor_obs = _extract_actor_obs(obs_dict, agent_cfg.actor_obs_group)
            actions = policy(actor_obs)
            obs_dict, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
