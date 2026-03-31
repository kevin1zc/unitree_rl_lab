#!/usr/bin/env python3

"""Export RANDPOL checkpoints to deployable JIT and ONNX policies."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
from torch import nn


def _load_models_module():
    models_path = SOURCE_ROOT / "unitree_rl_lab" / "randpol" / "models.py"
    spec = importlib.util.spec_from_file_location("randpol_export_models", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load RANDPOL models from {models_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_models = _load_models_module()
RandomFeatureGaussianPolicy = _models.RandomFeatureGaussianPolicy


CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


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


def _build_inference_module_from_state(state_dict: dict) -> RandpolInferenceModule:
    return RandpolInferenceModule(
        policy_state=state_dict["policy_model"],
        actor_obs_rms_state=state_dict.get("actor_obs_rms", {}),
        cfg=state_dict.get("cfg", {}),
    )


def _checkpoint_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if not match:
        return (-1, path.name)
    return (int(match.group(1)), path.name)


def _resolve_run_dirs(
    policy_root: pathlib.Path,
    run_name: str,
    export_all_runs: bool,
) -> list[pathlib.Path]:
    if (policy_root / "params" / "deploy.yaml").exists():
        return [policy_root]

    if run_name:
        run_dir = policy_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Requested run does not exist: {run_dir}")
        return [run_dir]

    run_dirs = sorted([entry for entry in policy_root.iterdir() if entry.is_dir()])
    if export_all_runs:
        return run_dirs
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {policy_root}")
    return [run_dirs[-1]]


def _resolve_checkpoint(run_dir: pathlib.Path, checkpoint_name: str) -> pathlib.Path:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_name:
        checkpoint_path = pathlib.Path(checkpoint_name)
        if not checkpoint_path.is_absolute():
            checkpoint_path = checkpoint_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_paths = sorted(
        [entry for entry in checkpoint_dir.glob("model_*.pt") if entry.is_file()],
        key=_checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No RANDPOL checkpoints found under: {checkpoint_dir}")
    return checkpoint_paths[-1]


def _export_run(run_dir: pathlib.Path, checkpoint_name: str, force: bool) -> pathlib.Path:
    deploy_cfg = run_dir / "params" / "deploy.yaml"
    if not deploy_cfg.exists():
        raise FileNotFoundError(f"Missing deploy config: {deploy_cfg}")

    checkpoint_path = _resolve_checkpoint(run_dir, checkpoint_name)
    export_dir = run_dir / "exported"
    jit_path = export_dir / "policy.pt"
    onnx_path = export_dir / "policy.onnx"

    if jit_path.exists() and onnx_path.exists() and not force:
        print(f"[SKIP] {run_dir.name}: {export_dir} already has policy.pt and policy.onnx")
        return onnx_path

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    module = _build_inference_module_from_state(checkpoint_state).cpu().eval()
    dummy_input = torch.zeros(1, int(module.obs_mean.numel()), dtype=torch.float32)

    export_dir.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(module, dummy_input)
    traced.save(jit_path)
    torch.onnx.export(
        module,
        dummy_input,
        onnx_path,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    print(f"[OK] Exported {checkpoint_path.name} -> {jit_path.name}, {onnx_path.name}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RANDPOL checkpoints to deployable policies.")
    parser.add_argument(
        "--policy-root",
        type=pathlib.Path,
        required=True,
        help="Experiment root or specific run directory.",
    )
    parser.add_argument("--run", type=str, default="", help="Specific run directory to export.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint filename or absolute path.")
    parser.add_argument("--all-runs", action="store_true", help="Export every run under the policy root.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing exported artifacts.")
    args = parser.parse_args()

    policy_root = args.policy_root.expanduser().resolve()
    run_dirs = _resolve_run_dirs(policy_root, args.run, args.all_runs)
    for run_dir in run_dirs:
        _export_run(run_dir, args.checkpoint, args.force)


if __name__ == "__main__":
    main()
