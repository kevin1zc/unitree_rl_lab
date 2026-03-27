from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class StagedVelocityOptCommand(UniformVelocityCommand):
    """Biased sampler with concurrent x/y/yaw support."""

    cfg: "StagedVelocityOptCommandCfg"

    def _sample_biased_axis(self, num_envs: int, low: float, high: float) -> torch.Tensor:
        values = torch.zeros(num_envs, device=self.device)
        pos_cap = max(0.0, float(high))
        neg_cap = max(0.0, -float(low))
        if pos_cap <= 1.0e-6 and neg_cap <= 1.0e-6:
            return values

        magnitudes = torch.rand(num_envs, device=self.device).pow(float(self.cfg.magnitude_exponent))
        if pos_cap > 1.0e-6 and neg_cap > 1.0e-6:
            is_positive = torch.rand(num_envs, device=self.device) < 0.5
            if torch.any(is_positive):
                values[is_positive] = magnitudes[is_positive] * pos_cap
            if torch.any(~is_positive):
                values[~is_positive] = -magnitudes[~is_positive] * neg_cap
        elif pos_cap > 1.0e-6:
            values = magnitudes * pos_cap
        else:
            values = -magnitudes * neg_cap
        return values

    def _sample_uniform_axis(self, num_envs: int, low: float, high: float) -> torch.Tensor:
        low = float(low)
        high = float(high)
        if abs(high - low) <= 1.0e-6:
            return torch.full((num_envs,), low, device=self.device)
        return torch.empty(num_envs, device=self.device).uniform_(low, high)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = int(env_ids.numel())
        if num_envs == 0:
            return

        self.vel_command_b[env_ids, :] = 0.0
        self.is_standing_env[env_ids] = torch.rand(num_envs, device=self.device) <= self.cfg.rel_standing_envs
        if self.cfg.heading_command:
            raise NotImplementedError("StagedVelocityOptCommand does not support heading_command=True.")

        ranges = self.cfg.ranges
        self.vel_command_b[env_ids, 0] = self._sample_biased_axis(num_envs, *ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = self._sample_biased_axis(num_envs, *ranges.lin_vel_y)
        yaw_values = self._sample_uniform_axis(num_envs, *ranges.ang_vel_z)
        if float(self.cfg.yaw_overlay_prob) < 1.0:
            yaw_active = torch.rand(num_envs, device=self.device) <= float(self.cfg.yaw_overlay_prob)
            yaw_values = yaw_values * yaw_active
        self.vel_command_b[env_ids, 2] = yaw_values


@configclass
class StagedVelocityOptCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = StagedVelocityOptCommand
    yaw_overlay_prob: float = 0.5
    magnitude_exponent: float = 0.5
