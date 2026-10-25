# Go2 Pedipulation Implementation Plan

> Execute inline using executing-plans, with TDD and verification-before-completion.

**Goal:** Add independent quadruped and biped right-front-foot manipulation tasks.
**Architecture:** Shared body-frame position command and reward module, two small derived task factories. Retain existing actor sensor inputs and add three command values.
**Tech Stack:** mjlab 1.2, MuJoCo 3.5, PyTorch, RSL-RL 5.

## Constraints
- Quad inherits DreamWaQ rough terrain and estimator, 5000 iterations / 4096 environments.
- Biped inherits StandUp V2.1, 6000 iterations / 4096 environments.
- Actor receives velocity3 + foot-position3 commands; no new privileged measurements.
- Existing remote queue remains unchanged.

## Task 1: Workspace and command
- [x] Write tests/test_pedipulation.py: task registration, smooth references at stage boundaries, Gaussian error reward, reachable FK target witnesses.
- [x] Run unittest tests.test_pedipulation and confirm missing feature failure.
- [x] Create src/tasks/go2/pedipulation/workspace.py: MuJoCo FK bank from nominal FR joint perturbations.
- [x] Create commands.py: FootPositionCommandCfg and FootPositionCommand, timer stages and set_target(target_b, env_ids) interface.
- [x] Verify command reset, timer stages, body frame transforms and target bank.

## Task 2: Tasks and rewards
- [x] Create mdp.py: actor extensions, position/return rewards, support reward, stage-aware base reward adapters.
- [x] Create env_cfg.py: quad_pedipulation_env_cfg and biped_pedipulation_env_cfg derived factories.
- [x] Create __init__.py: register Unitree-Go2-Quad-Pedipulation and Unitree-Go2-Biped-Pedipulation with isolated experiments.
- [x] Update src/tasks/__init__.py TASK_IDS.
- [x] Generalize dreamwaq/rl.py encoder, decoder and export dimensions from input tensors; preserve 45/225 defaults.
- [x] Verify original task configurations are unchanged and new biped symmetry loss disabled.

## Task 3: Integration and documentation
- [x] Run full unittest suite; adjust existing registration expectations for two added tasks.
- [x] Run scripts/check_tasks.py for both tasks: finite observations, resets, two PPO updates, checkpoint roundtrip and ONNX validation.
- [x] Exercise active reach/hold/return phases explicitly with a long-lived environment, beyond initial delay.
- [x] Verify original DreamWaQ with short GPU check after dimension generalization.
- [x] Add docs/pedipulation.md with command convention, observation inputs, reward behavior and training commands.
- [x] Review diff and git diff --check; record test evidence and limitations.
