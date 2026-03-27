# Go2 Input Controls

This document describes the keyboard-to-joystick mapping used by:

```bash
cd /home/zhuochen/Softwares/unitree/unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl --network lo --domain-id 1
```

It also documents the optional Xbox controller mapping used on this machine through:

```text
/dev/input/js0
```

By default, `go2_ctrl` now uses the newest exported policy run under:

```text
/home/zhuochen/Softwares/unitree/unitree_rl_lab/logs/randpol/unitree_go2_forwardyaw_velocity
```

## Keyboard State Control

- `f`: enter `FixStand`
  - maps to joystick `[L2 + A]`
- `v`: start the policy from `FixStand`
  - maps to joystick `[Start]`
- `b`: return to `Passive`
  - maps to joystick `[L2 + B]`

## Keyboard Velocity Commands

- Hold `w`: ramp forward velocity
- Hold `s`: ramp backward velocity
- Hold `a`: ramp left velocity
- Hold `d`: ramp right velocity
- Hold `Left Arrow`: ramp yaw-left velocity
- Hold `Right Arrow`: ramp yaw-right velocity
- `Space`: clear all velocity commands

Velocity commands now ramp while the corresponding key remains active and clear when the hold window expires. This is closer to joystick behavior than the previous instant-saturation version.

Different axes can be active at the same time. For example, forward velocity from `w` can be combined with yaw velocity from `Left Arrow` or `Right Arrow`.

The ramp behavior is tuned in:

```text
/home/zhuochen/Softwares/unitree/unitree_rl_lab/deploy/robots/go2/config/config.yaml
```

Relevant parameters:

- `hold_timeout`
- `linear_ramp_rate`
- `yaw_ramp_rate`

While the command is changing, `go2_ctrl` also prints the current smoothed command in the console:

```text
Keyboard velocity command: forward=..., lateral=..., yaw=...
```

## Sim Notes

- Keyboard control only works after `go2_ctrl` has connected to `rt/lowstate`.
- The DDS domain and interface must match the simulator.
- In this workspace, `/home/zhuochen/Softwares/unitree/unitree_mujoco/simulate/config.yaml` is currently set to:
  - `domain_id: 1`
  - `interface: "lo"`
- So with the current simulator config, use:

```bash
./go2_ctrl --network lo --domain-id 1
```

- If you instead start Mujoco with:

```bash
./unitree_mujoco -i 0 -n lo -r go2 -s scene.xml
```

- then use:

```bash
./go2_ctrl --network lo --domain-id 0
```

- State-control keys (`f`, `v`, `b`) are one-shot triggers.
- Velocity keys ramp up while active instead of jumping straight to full scale.
- To pin a specific policy run, add:

```bash
./go2_ctrl --network lo --domain-id 1 --policy-run 2026-03-11_23-46-58
```

- To use a different experiment root or a specific exported run directory, add:

```bash
./go2_ctrl --network lo --domain-id 1 --policy-dir /path/to/policy/root
```

- If you train a new `randpol` run and want it to become deployable for `go2_ctrl`, export it with:

```bash
/home/zhuochen/Softwares/isaacsim-5.1.0/python.sh \
  /home/zhuochen/Softwares/unitree/unitree_rl_lab/scripts/randpol/export_deploy.py \
  --policy-root /home/zhuochen/Softwares/unitree/unitree_rl_lab/logs/randpol/unitree_go2_forwardyaw_velocity
```

## Sim2Real

- For the real robot on this machine, use `eno1` and DDS domain `0`.
- Before launching `go2_ctrl`, it is safer to verify read-only communication first:

```bash
cd /home/zhuochen/Softwares/unitree
/tmp/go2_lowstate_probe --domain-id 0 --network eno1 --samples 3
```

- If that prints live `tick`, joint, and IMU values, the DDS link is working.
- Then the real-robot controller command is:

```bash
cd /home/zhuochen/Softwares/unitree/unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl --network eno1 --domain-id 0
```

- `go2_ctrl` is an active controller and may publish commands. The probe above is passive.

## Xbox Controller

- The currently connected controller is detected as `Xbox Wireless Controller` on `/dev/input/js0`.
- On this machine, `go2_ctrl` now reads it directly through the Linux joystick device.
- Mapping:
  - left stick: proportional x-y velocity command
  - right stick x: proportional yaw velocity command
  - `LT + A`: enter `FixStand`
  - `LT + B`: return to `Passive`
  - `X` or `Menu`: start the policy
- Tuning lives in:

```text
/home/zhuochen/Softwares/unitree/unitree_rl_lab/deploy/robots/go2/config/config.yaml
```

- Relevant gamepad parameters:
  - `device`
  - `deadzone`
  - `axis_smooth`
  - `trigger_threshold`

## Troubleshooting

- If you only see `Waiting for connection rt/lowstate`, the controller is not connected yet, so keyboard input will do nothing.
- First make sure Mujoco is already running.
- Then make sure `--domain-id` matches the simulator's `domain_id`.
- In the current workspace, the default simulator config uses `domain_id: 1`, so `f`, `v`, and `b` will not work unless you start `go2_ctrl` with `--domain-id 1`.
- For Sim2Real on this Ubuntu machine, `ufw` must be disabled or DDS traffic may be blocked even if ping works.
- Terminal keyboard input still does not provide perfect key-release events, so release is inferred using the configured `hold_timeout`.
