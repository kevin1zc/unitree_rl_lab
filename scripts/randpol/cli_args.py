import argparse
import random

def add_randpol_args(parser: argparse.ArgumentParser) -> None:
    """Add RANDPOL-specific CLI arguments."""
    arg_group = parser.add_argument_group("randpol", description="Arguments for the RANDPOL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--num_steps_per_env", type=int, default=None, help="Rollout steps per environment.")
    arg_group.add_argument("--feature_dim", type=int, default=None, help="Random feature dimension.")
    arg_group.add_argument("--policy_lr", type=float, default=None, help="Policy learning rate.")


def update_randpol_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update RANDPOL configuration from CLI arguments."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if hasattr(args_cli, "max_iterations") and args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.feature_dim is not None:
        agent_cfg.feature_dim = args_cli.feature_dim
    if args_cli.policy_lr is not None:
        agent_cfg.policy_lr = args_cli.policy_lr
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name

    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    return agent_cfg
