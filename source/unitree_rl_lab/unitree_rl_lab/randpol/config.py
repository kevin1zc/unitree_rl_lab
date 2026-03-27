from isaaclab.utils import configclass


@configclass
class RandpolRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"

    num_steps_per_env: int = 100
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = ""
    run_name: str = ""

    actor_obs_group: str = "policy"
    critic_obs_group: str = "critic"
    obs_history_length: int = 0
    clip_actions: float | None = None

    feature_dim: int = 800
    hidden_dims: list[int] = [500]
    activation: str = "elu"
    init_dist: str = "uniform"
    init_log_std: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    min_policy_std: float = 0.2

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ridge_lambda: float = 1.0e-2
    policy_update_mode: str = "clipped"
    policy_update_epochs: int = 5
    num_minibatches: int = 4
    surrogate_clip: float = 0.2
    policy_lr: float = 3.0e-4
    policy_kl_threshold: float = 2.0e-2
    policy_lr_scale_down: float = 0.5
    policy_lr_scale_up: float = 1.02
    policy_lr_min: float = 3.0e-5
    policy_lr_max: float = 3.0e-4
    entropy_coef: float = 0.01
    policy_grad_clip: float = 0.5
    normalize_advantages: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True
    obs_norm_clip: float = 10.0
    reward_norm_clip: float = 10.0
    norm_epsilon: float = 1.0e-8
    value_coef_bound: float = 4000.0
    value_bound_mode: str = "warn"
    value_ema_tau: float = 0.25
