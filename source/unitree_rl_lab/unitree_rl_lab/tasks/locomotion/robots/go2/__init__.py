import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "randpol_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.randpol_cfg:BaseRANDPOLRunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-ForwardYaw-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotForwardYawEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotForwardYawPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "randpol_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.randpol_cfg:BaseRANDPOLRunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Opt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotVelocityOptEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotVelocityOptPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "randpol_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.randpol_cfg:BaseRANDPOLRunnerCfg",
    },
)
