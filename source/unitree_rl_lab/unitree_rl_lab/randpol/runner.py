from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
import time
import warnings

import numpy as np
import torch
from torch import nn

from .config import RandpolRunnerCfg
from .models import RandomFeatureGaussianPolicy, RandomFeatureValue, fit_ridge_closed_form


@dataclass(slots=True)
class RandpolIterationStats:
    value_loss: float
    policy_loss: float
    mean_reward: float
    mean_episode_length: float
    num_completed_episodes: int
    num_transitions: int
    total_steps: int
    collection_time: float
    learning_time: float
    returns_time: float
    value_fit_time: float
    policy_update_time: float
    mean_action_std: float
    extras: dict[str, float]


class RunningMeanStd:
    """Minimal running mean/std tracker for online normalization."""

    def __init__(self, shape: torch.Size | tuple[int, ...], device: torch.device, epsilon: float) -> None:
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(float(epsilon), device=device, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float32)
        if values.ndim == 0:
            values = values.reshape(1)
        batch_count = values.shape[0] if values.ndim > 0 else 1
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        self._update_from_moments(
            batch_mean=batch_mean,
            batch_var=batch_var,
            batch_count=torch.tensor(float(batch_count), device=values.device, dtype=torch.float64),
        )

    def normalize(self, values: torch.Tensor, epsilon: float, clip: float) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.var + float(epsilon))
        return normalized.clamp(-float(clip), float(clip))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state_dict["mean"].to(device=self.mean.device, dtype=self.mean.dtype))
        self.var.copy_(state_dict["var"].to(device=self.var.device, dtype=self.var.dtype))
        self.count.copy_(state_dict["count"].to(device=self.count.device, dtype=self.count.dtype))

    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count).to(dtype=self.mean.dtype)
        m_a = self.var * self.count.to(dtype=self.var.dtype)
        m_b = batch_var * batch_count.to(dtype=batch_var.dtype)
        correction = delta.square() * (self.count * batch_count / total_count).to(dtype=batch_var.dtype)
        new_var = (m_a + m_b + correction) / total_count.to(dtype=batch_var.dtype)

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1.0e-12))
        self.count.copy_(total_count)


class RandpolInferenceModule(nn.Module):
    """Deterministic inference module with baked-in observation normalization."""

    def __init__(
        self,
        policy_state: dict[str, torch.Tensor],
        actor_obs_rms_state: dict[str, torch.Tensor],
        cfg: dict,
    ) -> None:
        super().__init__()
        obs_dim = int(policy_state["trunk.net.0.weight"].shape[1])
        action_dim = int(policy_state["head.weight"].shape[0])
        feature_dim = int(policy_state["head.weight"].shape[1])

        self.normalize_obs = bool(cfg.get("normalize_obs", True))
        self.norm_epsilon = float(cfg.get("norm_epsilon", 1.0e-8))
        self.obs_norm_clip = float(cfg.get("obs_norm_clip", 10.0))

        self.register_buffer(
            "obs_mean",
            actor_obs_rms_state.get("mean", torch.zeros(obs_dim, dtype=torch.float32)).to(dtype=torch.float32),
        )
        self.register_buffer(
            "obs_var",
            actor_obs_rms_state.get("var", torch.ones(obs_dim, dtype=torch.float32)).to(dtype=torch.float32),
        )

        self.policy = RandomFeatureGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dims=cfg.get("hidden_dims", []),
            activation=cfg.get("activation", "elu"),
            init_dist=cfg.get("init_dist", "uniform"),
            init_log_std=float(cfg.get("init_log_std", 0.0)),
            log_std_min=float(cfg.get("log_std_min", -20.0)),
            log_std_max=float(cfg.get("log_std_max", 2.0)),
        )
        self.policy.load_state_dict(policy_state)
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized_obs = obs
        if self.normalize_obs:
            normalized_obs = (normalized_obs - self.obs_mean) / torch.sqrt(self.obs_var + self.norm_epsilon)
            normalized_obs = torch.clamp(normalized_obs, -self.obs_norm_clip, self.obs_norm_clip)
        return self.policy.act_inference(normalized_obs)


class RandpolRolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        actor_obs_dim: int,
        critic_feature_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.actor_obs = torch.zeros(num_steps, num_envs, actor_obs_dim, device=device)
        self.critic_features = torch.zeros(num_steps, num_envs, critic_feature_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.old_log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def add(
        self,
        step: int,
        actor_obs: torch.Tensor,
        critic_features: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.actor_obs[step].copy_(actor_obs)
        self.critic_features[step].copy_(critic_features)
        self.actions[step].copy_(actions)
        self.old_log_probs[step].copy_(old_log_probs)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)
        self.values[step].copy_(values)

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.rewards.shape[0])):
            next_values = last_values if step == self.rewards.shape[0] - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * not_done - self.values[step]
            advantage = delta + gamma * gae_lambda * not_done * advantage
            self.advantages[step] = advantage
            self.returns[step] = advantage + self.values[step]


class RandpolOnPolicyRunner:
    def __init__(self, env, cfg: RandpolRunnerCfg, *, update_normalization: bool = True):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.update_normalization = update_normalization

        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Configured device '{cfg.device}' is unavailable.")

        self.obs_dict, _ = self.env.reset()
        actor_obs = self._extract_group_obs(self.obs_dict, cfg.actor_obs_group)
        critic_obs = self._extract_group_obs(self.obs_dict, cfg.critic_obs_group, fallback=actor_obs)
        self.num_envs = actor_obs.shape[0]
        self.actor_obs_dim = actor_obs.shape[-1]
        self.critic_obs_dim = critic_obs.shape[-1]
        self.action_dim = int(self.env.unwrapped.action_manager.total_action_dim)
        self.actor_obs_rms = RunningMeanStd(
            shape=actor_obs.shape[1:],
            device=self.device,
            epsilon=self.cfg.norm_epsilon,
        )
        self.critic_obs_rms = RunningMeanStd(
            shape=critic_obs.shape[1:],
            device=self.device,
            epsilon=self.cfg.norm_epsilon,
        )
        self.reward_rms = RunningMeanStd(shape=(), device=self.device, epsilon=self.cfg.norm_epsilon)
        self.discounted_returns = torch.zeros(self.num_envs, device=self.device)
        self._update_observation_normalizers(self.obs_dict)

        self.policy_model = RandomFeatureGaussianPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.action_dim,
            feature_dim=cfg.feature_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            init_dist=cfg.init_dist,
            init_log_std=cfg.init_log_std,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(self.device)
        self.value_model = RandomFeatureValue(
            obs_dim=self.critic_obs_dim,
            feature_dim=cfg.feature_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            init_dist=cfg.init_dist,
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.top_parameters(), lr=cfg.policy_lr)
        self._clamp_policy_std()

        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.iteration = 0

    @staticmethod
    def _extract_group_obs(obs_dict, group_name: str, fallback: torch.Tensor | None = None) -> torch.Tensor:
        if group_name in obs_dict:
            return obs_dict[group_name]
        if fallback is not None:
            return fallback
        if "policy" in obs_dict:
            return obs_dict["policy"]
        first_key = next(iter(obs_dict.keys()))
        return obs_dict[first_key]

    def _update_observation_normalizers(self, obs_dict) -> None:
        if not self.update_normalization or not self.cfg.normalize_obs:
            return
        actor_obs = self._extract_group_obs(obs_dict, self.cfg.actor_obs_group)
        critic_obs = self._extract_group_obs(obs_dict, self.cfg.critic_obs_group, fallback=actor_obs)
        self.actor_obs_rms.update(actor_obs)
        self.critic_obs_rms.update(critic_obs)

    def _normalize_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_obs:
            return actor_obs
        return self.actor_obs_rms.normalize(actor_obs, epsilon=self.cfg.norm_epsilon, clip=self.cfg.obs_norm_clip)

    def _normalize_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_obs:
            return critic_obs
        return self.critic_obs_rms.normalize(critic_obs, epsilon=self.cfg.norm_epsilon, clip=self.cfg.obs_norm_clip)

    def _normalize_rewards(self, rewards: torch.Tensor, finished: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_reward:
            return rewards
        if self.update_normalization:
            self.discounted_returns = self.discounted_returns * self.cfg.gamma + rewards
            self.reward_rms.update(self.discounted_returns)
        normalized_rewards = rewards / torch.sqrt(self.reward_rms.var + float(self.cfg.norm_epsilon))
        normalized_rewards = normalized_rewards.clamp(-float(self.cfg.reward_norm_clip), float(self.cfg.reward_norm_clip))
        self.discounted_returns = self.discounted_returns.masked_fill(finished, 0.0)
        return normalized_rewards

    def _policy_log_std_lower_bound(self) -> float:
        if self.cfg.min_policy_std <= 0:
            return float(self.cfg.log_std_min)
        return max(float(self.cfg.log_std_min), math.log(float(self.cfg.min_policy_std)))

    def _clamp_policy_std(self) -> None:
        lower = self._policy_log_std_lower_bound()
        with torch.no_grad():
            self.policy_model.log_std.clamp_(lower, self.cfg.log_std_max)

    @staticmethod
    def _gaussian_kl_divergence(
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ) -> torch.Tensor:
        old_var = old_std.square()
        new_var = new_std.square()
        return (
            torch.log(new_std / old_std)
            + (old_var + (old_mean - new_mean).square()) / (2.0 * new_var)
            - 0.5
        ).sum(dim=-1)

    def _policy_learning_rate(self) -> float:
        return float(self.policy_optimizer.param_groups[0]["lr"])

    def _set_policy_learning_rate(self, value: float) -> None:
        clamped_value = min(max(float(value), self.cfg.policy_lr_min), self.cfg.policy_lr_max)
        for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = clamped_value

    @staticmethod
    def _project_linear_params(weight: torch.Tensor, bias: torch.Tensor | None, coef_bound: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        if coef_bound <= 0:
            return weight, bias
        width = max(1, weight.shape[1])
        per_feature_bound = float(coef_bound) / float(width)
        projected_weight = weight.clamp(-per_feature_bound, per_feature_bound)
        projected_bias = None if bias is None else bias.clamp(-float(coef_bound), float(coef_bound))
        return projected_weight, projected_bias

    def _update_episode_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> tuple[list[float], list[float]]:
        self.episode_returns += rewards
        self.episode_lengths += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return [], []

        finished_returns = self.episode_returns[done_ids].detach().cpu().tolist()
        finished_lengths = self.episode_lengths[done_ids].detach().cpu().tolist()
        self.episode_returns[done_ids] = 0.0
        self.episode_lengths[done_ids] = 0
        return finished_returns, finished_lengths

    @staticmethod
    def _flatten_log_value(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().to(dtype=torch.float32).mean().item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.astype(np.float32).mean())
        if isinstance(value, (float, int, np.floating, np.integer, bool)):
            return float(value)
        return None

    def _merge_env_log(self, aggregated_logs: dict[str, list[float]], extras: dict | None) -> None:
        if not isinstance(extras, dict):
            return
        log_dict = extras.get("log") or extras.get("episode")
        if not isinstance(log_dict, dict):
            return
        for key, value in log_dict.items():
            scalar_value = self._flatten_log_value(value)
            if scalar_value is None:
                continue
            aggregated_logs.setdefault(key, []).append(scalar_value)

    def _apply_value_head_bound(
        self, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mode = str(self.cfg.value_bound_mode).lower()
        if self.cfg.value_coef_bound <= 0 or mode in {"off", "none", "warn"}:
            return weight, bias
        if mode == "clip":
            return self._project_linear_params(weight, bias, self.cfg.value_coef_bound)
        raise ValueError(f"Unsupported value_bound_mode: {self.cfg.value_bound_mode!r}")

    def _collect_rollout(self) -> tuple[RandpolRolloutStorage, list[float], list[float], dict[str, float]]:
        storage = RandpolRolloutStorage(
            num_steps=self.cfg.num_steps_per_env,
            num_envs=self.num_envs,
            actor_obs_dim=self.actor_obs_dim,
            critic_feature_dim=self.cfg.feature_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        finished_returns: list[float] = []
        finished_lengths: list[float] = []
        aggregated_logs: dict[str, list[float]] = {}

        for step in range(self.cfg.num_steps_per_env):
            actor_obs_raw = self._extract_group_obs(self.obs_dict, self.cfg.actor_obs_group)
            critic_obs_raw = self._extract_group_obs(self.obs_dict, self.cfg.critic_obs_group, fallback=actor_obs_raw)
            actor_obs = self._normalize_actor_obs(actor_obs_raw)
            critic_obs = self._normalize_critic_obs(critic_obs_raw)

            with torch.no_grad():
                actions, old_log_probs = self.policy_model.act(actor_obs)
                critic_features = self.value_model.extract_features(critic_obs)
                values = self.value_model.head(critic_features).squeeze(-1)

            env_actions = actions
            if self.cfg.clip_actions is not None:
                env_actions = env_actions.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)

            next_obs, raw_rewards, terminated, truncated, extras = self.env.step(env_actions)
            finished = terminated | truncated
            rewards = self._normalize_rewards(raw_rewards, finished)
            timeout_mask = truncated.to(dtype=rewards.dtype)
            if timeout_mask.any():
                rewards = rewards + self.cfg.gamma * values * timeout_mask
            dones = finished.to(dtype=torch.float32)
            returns, lengths = self._update_episode_stats(raw_rewards, finished)
            finished_returns.extend(returns)
            finished_lengths.extend(lengths)
            self._merge_env_log(aggregated_logs, extras)

            storage.add(
                step=step,
                actor_obs=actor_obs,
                critic_features=critic_features,
                actions=actions,
                old_log_probs=old_log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
            )
            self._update_observation_normalizers(next_obs)
            self.obs_dict = next_obs

        with torch.no_grad():
            actor_obs_raw = self._extract_group_obs(self.obs_dict, self.cfg.actor_obs_group)
            critic_obs_raw = self._extract_group_obs(self.obs_dict, self.cfg.critic_obs_group, fallback=actor_obs_raw)
            actor_obs = self._normalize_actor_obs(actor_obs_raw)
            critic_obs = self._normalize_critic_obs(critic_obs_raw)
            last_values = self.value_model(critic_obs)

        returns_start = time.perf_counter()
        storage.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        returns_time = time.perf_counter() - returns_start
        mean_logs = {key: float(np.mean(values)) for key, values in aggregated_logs.items() if values}
        mean_logs["Diagnostics/returns_time"] = returns_time
        return storage, finished_returns, finished_lengths, mean_logs, returns_time

    def _fit_value(self, critic_features: torch.Tensor, returns: torch.Tensor) -> float:
        with torch.no_grad():
            return_mean = returns.mean()
            return_std = returns.std().clamp(min=1e-6)
            normalized_returns = (returns - return_mean) / return_std
            weight, bias = fit_ridge_closed_form(critic_features, normalized_returns, l2_reg=self.cfg.ridge_lambda)
            weight = weight * return_std
            bias = bias * return_std + return_mean

            if self.cfg.value_ema_tau < 1.0:
                tau = self.cfg.value_ema_tau
                old_weight = self.value_model.head.weight.data
                old_bias = self.value_model.head.bias.data
                weight = (1.0 - tau) * old_weight + tau * weight
                bias = (1.0 - tau) * old_bias + tau * bias

            weight, bias = self._apply_value_head_bound(weight, bias)
            self.value_model.set_head(weight, bias)
            predictions = self.value_model.head(critic_features).squeeze(-1)
            return float(torch.mean((predictions - returns) ** 2).item())

    def _update_policy_exact(
        self,
        actor_obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[float, float, float]:
        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp(min=1e-8))

        snapshot = {
            key: value.clone()
            for key, value in self.policy_model.state_dict().items()
            if "trunk" not in key
        }
        optimizer_snapshot = copy.deepcopy(self.policy_optimizer.state_dict())
        with torch.no_grad():
            old_mean, old_std = self.policy_model.distribution_params(actor_obs)

        policy_loss = 0.0
        approx_kl = 0.0
        update_skipped = 0.0
        had_error = False

        new_log_probs, entropy = self.policy_model.evaluate_actions(actor_obs, actions)
        if not torch.isfinite(new_log_probs).all() or not torch.isfinite(entropy).all():
            had_error = True
        else:
            actor_loss = -(advantages * new_log_probs).mean()
            loss = actor_loss - self.cfg.entropy_coef * entropy.mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.policy_grad_clip > 0:
                nn.utils.clip_grad_norm_(self.policy_model.top_parameters(), self.cfg.policy_grad_clip)
            self.policy_optimizer.step()
            self._clamp_policy_std()
            policy_loss = float(actor_loss.detach().item())

            if not torch.isfinite(self.policy_model.head.weight).all() or not torch.isfinite(self.policy_model.log_std).all():
                had_error = True
            else:
                with torch.no_grad():
                    new_mean, new_std = self.policy_model.distribution_params(actor_obs)
                    approx_kl = float(
                        self._gaussian_kl_divergence(old_mean, old_std, new_mean, new_std).mean().item()
                    )
                if self.cfg.policy_kl_threshold > 0 and approx_kl > self.cfg.policy_kl_threshold:
                    had_error = True

        if had_error:
            current_state = self.policy_model.state_dict()
            current_state.update(snapshot)
            self.policy_model.load_state_dict(current_state)
            self.policy_optimizer.load_state_dict(optimizer_snapshot)
            self._clamp_policy_std()
            update_skipped = 1.0
            self._set_policy_learning_rate(self._policy_learning_rate() * self.cfg.policy_lr_scale_down)
        elif self.cfg.policy_kl_threshold > 0 and approx_kl < 0.5 * self.cfg.policy_kl_threshold:
            self._set_policy_learning_rate(self._policy_learning_rate() * self.cfg.policy_lr_scale_up)

        return policy_loss, approx_kl, update_skipped

    def _update_policy_clipped(
        self,
        actor_obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[float, float, float]:
        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp(min=1e-8))

        batch_size = actor_obs.shape[0]
        num_minibatches = max(1, min(int(self.cfg.num_minibatches), batch_size))
        minibatch_size = math.ceil(batch_size / num_minibatches)
        num_epochs = max(1, int(self.cfg.policy_update_epochs))

        losses: list[float] = []
        approx_kls: list[float] = []

        for _ in range(num_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            should_stop = False
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                mb_obs = actor_obs[indices]
                mb_actions = actions[indices]
                mb_advantages = advantages[indices]
                mb_old_log_probs = old_log_probs[indices]

                new_log_probs, entropy = self.policy_model.evaluate_actions(mb_obs, mb_actions)
                if not torch.isfinite(new_log_probs).all() or not torch.isfinite(entropy).all():
                    continue

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * mb_advantages
                clipped = ratio.clamp(1.0 - self.cfg.surrogate_clip, 1.0 + self.cfg.surrogate_clip) * mb_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                loss = actor_loss - self.cfg.entropy_coef * entropy.mean()

                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.policy_grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy_model.top_parameters(), self.cfg.policy_grad_clip)
                self.policy_optimizer.step()
                self._clamp_policy_std()

                approx_kl = float((ratio - 1.0 - log_ratio).mean().item())
                losses.append(float(actor_loss.detach().item()))
                approx_kls.append(approx_kl)

                if self.cfg.policy_kl_threshold > 0 and approx_kl > self.cfg.policy_kl_threshold:
                    should_stop = True
                    break

            if should_stop:
                break

        mean_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
        if self.cfg.policy_kl_threshold > 0:
            if mean_kl > self.cfg.policy_kl_threshold:
                self._set_policy_learning_rate(self._policy_learning_rate() * self.cfg.policy_lr_scale_down)
            elif mean_kl < 0.5 * self.cfg.policy_kl_threshold:
                self._set_policy_learning_rate(self._policy_learning_rate() * self.cfg.policy_lr_scale_up)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        return mean_loss, mean_kl, 0.0

    def _update_policy(
        self,
        actor_obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor | None = None,
    ) -> tuple[float, float, float]:
        mode = str(getattr(self.cfg, "policy_update_mode", "clipped")).lower()
        if mode in {"clipped", "clip", "ppo", "hybrid"}:
            if old_log_probs is None:
                raise ValueError("old_log_probs must be provided for clipped policy updates.")
            return self._update_policy_clipped(actor_obs, actions, advantages, old_log_probs)
        if mode in {"exact", "score_function", "pg"}:
            return self._update_policy_exact(actor_obs, actions, advantages)
        raise ValueError(f"Unsupported policy_update_mode: {self.cfg.policy_update_mode!r}")

    def learn_iteration(self) -> RandpolIterationStats:
        collection_start = time.perf_counter()
        storage, finished_returns, finished_lengths, extras, returns_time = self._collect_rollout()
        collection_time = time.perf_counter() - collection_start

        actor_obs = storage.actor_obs.reshape(-1, self.actor_obs_dim)
        critic_features = storage.critic_features.reshape(-1, self.cfg.feature_dim)
        actions = storage.actions.reshape(-1, self.action_dim)
        old_log_probs = storage.old_log_probs.reshape(-1)
        returns = storage.returns.reshape(-1)
        advantages = storage.advantages.reshape(-1)

        learning_start = time.perf_counter()
        value_start = time.perf_counter()
        value_loss = self._fit_value(critic_features, returns)
        value_fit_time = time.perf_counter() - value_start
        policy_start = time.perf_counter()
        policy_loss, _, _ = self._update_policy(actor_obs, actions, advantages, old_log_probs)
        policy_update_time = time.perf_counter() - policy_start
        learning_time = time.perf_counter() - learning_start
        extras["Diagnostics/value_fit_time"] = value_fit_time
        extras["Diagnostics/policy_update_time"] = policy_update_time
        self.iteration += 1

        if finished_returns:
            mean_reward = float(np.mean(finished_returns))
            mean_episode_length = float(np.mean(finished_lengths))
        else:
            mean_reward = float("nan")
            mean_episode_length = float("nan")

        return RandpolIterationStats(
            value_loss=value_loss,
            policy_loss=policy_loss,
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            num_completed_episodes=len(finished_returns),
            num_transitions=self.cfg.num_steps_per_env * self.num_envs,
            total_steps=self.iteration * self.cfg.num_steps_per_env * self.num_envs,
            collection_time=collection_time,
            learning_time=learning_time,
            returns_time=returns_time,
            value_fit_time=value_fit_time,
            policy_update_time=policy_update_time,
            mean_action_std=float(torch.exp(self.policy_model.log_std.detach()).mean().item()),
            extras=extras,
        )

    def get_inference_policy(self):
        def inference_policy(actor_obs: torch.Tensor) -> torch.Tensor:
            normalized_actor_obs = self._normalize_actor_obs(actor_obs)
            return self.policy_model.act_inference(normalized_actor_obs)

        return inference_policy

    def _export_state(self) -> dict:
        return {
            "policy_model": {
                key: value.detach().to(device="cpu", dtype=torch.float32).clone()
                for key, value in self.policy_model.state_dict().items()
            },
            "actor_obs_rms": {
                key: value.detach().to(device="cpu").clone()
                for key, value in self.actor_obs_rms.state_dict().items()
            },
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg.__dict__),
        }

    @staticmethod
    def build_inference_module_from_state(state_dict: dict) -> RandpolInferenceModule:
        return RandpolInferenceModule(
            policy_state=state_dict["policy_model"],
            actor_obs_rms_state=state_dict.get("actor_obs_rms", {}),
            cfg=state_dict.get("cfg", {}),
        )

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> str:
        export_dir = os.path.abspath(path)
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, filename)
        module = self.build_inference_module_from_state(self._export_state()).cpu().eval()
        dummy_input = torch.zeros(1, int(module.obs_mean.numel()), dtype=torch.float32)
        traced = torch.jit.trace(module, dummy_input)
        traced.save(export_path)
        return export_path

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> str:
        export_dir = os.path.abspath(path)
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, filename)
        module = self.build_inference_module_from_state(self._export_state()).cpu().eval()
        dummy_input = torch.zeros(1, int(module.obs_mean.numel()), dtype=torch.float32)
        torch.onnx.export(
            module,
            dummy_input,
            export_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
            dynamo=False,
        )
        return export_path

    def state_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "actor_obs_rms": self.actor_obs_rms.state_dict(),
            "critic_obs_rms": self.critic_obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "discounted_returns": self.discounted_returns.clone(),
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg.__dict__),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration = int(state_dict.get("iteration", 0))
        self.policy_model.load_state_dict(state_dict["policy_model"])
        self.value_model.load_state_dict(state_dict["value_model"])
        if "policy_optimizer" in state_dict:
            self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self._clamp_policy_std()
        if "actor_obs_rms" in state_dict:
            self.actor_obs_rms.load_state_dict(state_dict["actor_obs_rms"])
        if "critic_obs_rms" in state_dict:
            self.critic_obs_rms.load_state_dict(state_dict["critic_obs_rms"])
        if "reward_rms" in state_dict:
            self.reward_rms.load_state_dict(state_dict["reward_rms"])
        if "discounted_returns" in state_dict:
            loaded_discounted_returns = state_dict["discounted_returns"].to(
                device=self.device, dtype=self.discounted_returns.dtype
            )
            if loaded_discounted_returns.shape == self.discounted_returns.shape:
                self.discounted_returns.copy_(loaded_discounted_returns)
            else:
                self.discounted_returns.zero_()
                warnings.warn(
                    (
                        "Skipping discounted_returns restore because checkpoint env count "
                        f"{tuple(loaded_discounted_returns.shape)} does not match current env count "
                        f"{tuple(self.discounted_returns.shape)}."
                    ),
                    stacklevel=2,
                )
