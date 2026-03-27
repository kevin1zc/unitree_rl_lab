from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mean_episode_reward(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str) -> torch.Tensor:
    return torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s


def _curriculum_update_due(env: ManagerBasedRLEnv) -> bool:
    return env.common_step_counter % env.max_episode_length == 0


def _level_from_symmetric_range(value_range: Sequence[float]) -> float:
    return float(abs(value_range[1]))


def _at_or_above_limit(current_level: float, limit_level: float, tol: float = 1.0e-6) -> bool:
    return current_level + tol >= limit_level


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def forward_uniform_lin_yaw_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta = 0.2
            current_level = float(ranges.lin_vel_x[1])
            target_level = min(current_level + delta, float(limit_ranges.lin_vel_x[1]))
            ranges.lin_vel_x = [float(limit_ranges.lin_vel_x[0]), target_level]
            yaw_limit = min(target_level, float(limit_ranges.ang_vel_z[1]))
            ranges.ang_vel_z = [-yaw_limit, yaw_limit]

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def symmetric_uniform_lin_yaw_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta = 0.2
            current_level = float(abs(ranges.lin_vel_x[1]))
            target_level = min(current_level + delta, float(limit_ranges.lin_vel_x[1]))
            ranges.lin_vel_x = [-target_level, target_level]
            yaw_limit = min(target_level, float(limit_ranges.ang_vel_z[1]))
            ranges.ang_vel_z = [-yaw_limit, yaw_limit]

    return torch.tensor(abs(ranges.lin_vel_x[1]), device=env.device)


def staged_lin_vel_x_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    step: float = 0.2,
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)
    current_level = _level_from_symmetric_range(ranges.lin_vel_x)
    limit_level = float(limit_ranges.lin_vel_x[1])

    if _curriculum_update_due(env) and not _at_or_above_limit(current_level, limit_level):
        if reward > reward_term.weight * 0.8:
            next_level = min(current_level + float(step), limit_level)
            ranges.lin_vel_x = [-next_level, next_level]

    return torch.tensor(_level_from_symmetric_range(ranges.lin_vel_x), device=env.device)


def staged_forward_lin_vel_x_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    step: float = 0.2,
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)
    current_level = float(ranges.lin_vel_x[1])
    limit_level = float(limit_ranges.lin_vel_x[1])

    if _curriculum_update_due(env) and not _at_or_above_limit(current_level, limit_level):
        if reward > reward_term.weight * 0.8:
            next_level = min(current_level + float(step), limit_level)
            ranges.lin_vel_x = [float(limit_ranges.lin_vel_x[0]), next_level]

    return torch.tensor(float(ranges.lin_vel_x[1]), device=env.device)


def staged_velocity_opt_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    lin_reward_term_name: str = "track_lin_vel_xy",
    yaw_reward_term_name: str = "track_ang_vel_z",
    step_x: float = 0.1,
    step_y: float = 0.1,
    step_yaw: float = 0.1,
) -> dict[str, float]:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    lin_reward_term = env.reward_manager.get_term_cfg(lin_reward_term_name)
    yaw_reward_term = env.reward_manager.get_term_cfg(yaw_reward_term_name)
    lin_reward = _mean_episode_reward(env, env_ids, lin_reward_term_name)
    yaw_reward = _mean_episode_reward(env, env_ids, yaw_reward_term_name)

    forward_cap = float(ranges.lin_vel_x[1])
    backward_cap = max(0.0, -float(ranges.lin_vel_x[0]))
    yaw_cap = _level_from_symmetric_range(ranges.ang_vel_z)
    left_cap = max(0.0, float(ranges.lin_vel_y[1]))
    right_cap = max(0.0, -float(ranges.lin_vel_y[0]))

    forward_limit = float(limit_ranges.lin_vel_x[1])
    backward_limit = max(0.0, -float(limit_ranges.lin_vel_x[0]))
    yaw_limit = _level_from_symmetric_range(limit_ranges.ang_vel_z)
    left_limit = max(0.0, float(limit_ranges.lin_vel_y[1]))
    right_limit = max(0.0, -float(limit_ranges.lin_vel_y[0]))

    if _curriculum_update_due(env):
        if not _at_or_above_limit(forward_cap, forward_limit):
            if lin_reward > lin_reward_term.weight * 0.8:
                next_forward = min(forward_cap + float(step_x), forward_limit)
                ranges.lin_vel_x = [0.0, next_forward]
                forward_cap = next_forward
        elif not _at_or_above_limit(backward_cap, backward_limit):
            if lin_reward > lin_reward_term.weight * 0.8:
                next_backward = min(backward_cap + float(step_x), backward_limit)
                ranges.lin_vel_x = [-next_backward, forward_limit]
                backward_cap = next_backward
        elif not _at_or_above_limit(yaw_cap, yaw_limit):
            if yaw_reward > yaw_reward_term.weight * 0.8:
                next_yaw = min(yaw_cap + float(step_yaw), yaw_limit)
                ranges.ang_vel_z = [-next_yaw, next_yaw]
                yaw_cap = next_yaw
        elif not _at_or_above_limit(left_cap, left_limit):
            if lin_reward > lin_reward_term.weight * 0.8:
                next_left = min(left_cap + float(step_y), left_limit)
                ranges.lin_vel_y = [-right_cap, next_left]
                left_cap = next_left
        elif not _at_or_above_limit(right_cap, right_limit):
            if lin_reward > lin_reward_term.weight * 0.8:
                next_right = min(right_cap + float(step_y), right_limit)
                ranges.lin_vel_y = [-next_right, left_limit]
                right_cap = next_right

    return {
        "forward": float(ranges.lin_vel_x[1]),
        "backward": max(0.0, -float(ranges.lin_vel_x[0])),
        "yaw": float(_level_from_symmetric_range(ranges.ang_vel_z)),
        "left": max(0.0, float(ranges.lin_vel_y[1])),
        "right": max(0.0, -float(ranges.lin_vel_y[0])),
    }


def staged_forward_then_lin_vel_y_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    step: float = 0.1,
    lateral_start_forward_level: float = 0.5,
) -> dict[str, float]:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)
    forward_level = float(ranges.lin_vel_x[1])
    forward_limit = float(limit_ranges.lin_vel_x[1])
    lateral_level = _level_from_symmetric_range(ranges.lin_vel_y)
    lateral_limit = float(limit_ranges.lin_vel_y[1])

    if _curriculum_update_due(env) and reward > reward_term.weight * 0.8:
        if not _at_or_above_limit(forward_level, forward_limit):
            next_forward_level = min(forward_level + float(step), forward_limit)
            ranges.lin_vel_x = [0.0, next_forward_level]
            forward_level = next_forward_level
        if forward_level >= float(lateral_start_forward_level) and not _at_or_above_limit(lateral_level, lateral_limit):
            next_lateral_level = min(lateral_level + float(step), lateral_limit)
            ranges.lin_vel_y = [-next_lateral_level, next_lateral_level]

    return {
        "forward": float(ranges.lin_vel_x[1]),
        "lateral": float(_level_from_symmetric_range(ranges.lin_vel_y)),
    }


def staged_ang_vel_z_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    step: float = 0.2,
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    lin_vel_x_level = _level_from_symmetric_range(ranges.lin_vel_x)
    lin_vel_x_limit = float(limit_ranges.lin_vel_x[1])
    yaw_level = _level_from_symmetric_range(ranges.ang_vel_z)
    yaw_limit = float(limit_ranges.ang_vel_z[1])

    if _curriculum_update_due(env) and _at_or_above_limit(lin_vel_x_level, lin_vel_x_limit):
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward = _mean_episode_reward(env, env_ids, reward_term_name)
        if not _at_or_above_limit(yaw_level, yaw_limit) and reward > reward_term.weight * 0.8:
            next_level = min(yaw_level + float(step), yaw_limit)
            ranges.ang_vel_z = [-next_level, next_level]

    return torch.tensor(_level_from_symmetric_range(ranges.ang_vel_z), device=env.device)


def staged_lin_vel_y_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    step: float = 0.2,
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    lin_vel_x_level = _level_from_symmetric_range(ranges.lin_vel_x)
    lin_vel_x_limit = float(limit_ranges.lin_vel_x[1])
    yaw_level = _level_from_symmetric_range(ranges.ang_vel_z)
    yaw_limit = float(limit_ranges.ang_vel_z[1])
    lin_vel_y_level = _level_from_symmetric_range(ranges.lin_vel_y)
    lin_vel_y_limit = float(limit_ranges.lin_vel_y[1])

    if (
        _curriculum_update_due(env)
        and _at_or_above_limit(lin_vel_x_level, lin_vel_x_limit)
        and _at_or_above_limit(yaw_level, yaw_limit)
    ):
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward = _mean_episode_reward(env, env_ids, reward_term_name)
        if not _at_or_above_limit(lin_vel_y_level, lin_vel_y_limit) and reward > reward_term.weight * 0.8:
            next_level = min(lin_vel_y_level + float(step), lin_vel_y_limit)
            ranges.lin_vel_y = [-next_level, next_level]

    return torch.tensor(_level_from_symmetric_range(ranges.lin_vel_y), device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
