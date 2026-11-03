# Verification record

Date: 2026-11-02 (Asia/Shanghai)

## Passed locally

| Check | Command | Result |
|---|---|---|
| Python syntax | `python -m compileall -q reproduction` | PASS |
| Adapter and runner contracts | `python -m unittest discover -s reproduction/tests -v` | PASS, 5 tests |
| Synthetic DIMA CPU path | `smoke_test.py --device cpu` | PASS (saved result) |
| Synthetic DIMA CUDA path | `smoke_test.py --device cuda` | PASS on RTX 3050 Laptop 4 GB (saved result) |
| Bi-DexHands command parsing | `train_bidex.py --help` | PASS |
| MAMuJoCo command matrix generation | `run_mamujoco_matrix.py` without `--execute` | PASS, 28 commands printed |

The synthetic check covers DIMA denoising loss/backpropagation, single-current-agent action masks, Appendix E.5 action schedules, Euler/Heun numerical paths, FSQ reconstruction/backpropagation, reward/termination Transformer loss/backpropagation and causality, plus Actor/Critic PPO gradients. It uses random tensors; it is not an environment rollout or a paper-score reproduction.

## Blocked locally

The Windows host has CUDA PyTorch but lacks `einops`, `ray`, `wandb`, old Gym, MuJoCo 2.1/mujoco-py, TensorBoard and Isaac Gym. It has no installed Linux/WSL distribution. Thus MAMuJoCo and Bi-DexHands initialization, long training, real checkpoint evaluation, multi-seed aggregation, baseline comparisons and paper figures/tables have **not** been run.

`train.py --help` and `evaluate_mamujoco.py --help` currently load runtime modules before argument parsing; in a complete environment this is normal, but on this preflight host they fail early because dependencies are absent. Use `requirements-*.txt` and the setup guidance before executing them.

## Reproduction status

`CANNOT_VERIFY`: no scientific-return claim has been reproduced. The delivery supplies a source-faithful starting point, explicitly recorded paper/code deviations and an executable protocol pending a compatible Linux simulator environment.
