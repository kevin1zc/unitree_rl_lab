#!/usr/bin/env python3

"""Export RANDPOL checkpoints to deployable JIT and ONNX policies."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.randpol.runner import RandpolOnPolicyRunner


CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


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
    module = RandpolOnPolicyRunner.build_inference_module_from_state(checkpoint_state).cpu().eval()
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
