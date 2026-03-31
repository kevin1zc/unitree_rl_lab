# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from unitree_rl_lab.randpol.config import RandpolRunnerCfg


@configclass
class BaseRANDPOLRunnerCfg(RandpolRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1000
    save_interval = 100
    experiment_name = ""
    actor_obs_group = "policy"
    critic_obs_group = "critic"
    obs_history_length = 0
    feature_dim = 400
    hidden_dims = [500]
    activation = "elu"
    init_dist = "uniform"
    min_policy_std = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    ridge_lambda = 1.0e-2
    policy_update_mode = "clipped"
    policy_update_epochs = 5
    num_minibatches = 4
    surrogate_clip = 0.2
    policy_lr = 3.0e-4
    policy_kl_threshold = 2.0e-2
    policy_lr_scale_down = 0.5
    policy_lr_scale_up = 1.02
    policy_lr_min = 3.0e-5
    policy_lr_max = 3.0e-4
    entropy_coef = 0.01
    policy_grad_clip = 0.5
    normalize_advantages = True
    normalize_obs = True
    normalize_reward = True
    obs_norm_clip = 10.0
    reward_norm_clip = 10.0
    norm_epsilon = 1.0e-8
    value_coef_bound = 4000.0
    value_bound_mode = "warn"
    value_ema_tau = 0.25
