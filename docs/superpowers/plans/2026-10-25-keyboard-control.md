# Go2 keyboard control implementation plan

**Goal:** Run the downloaded policies with concurrent keyboard locomotion and FR position commands.
**Architecture:** Play-only command subclasses own persistent commands. A native viewer subclass routes key events through the existing main-thread action queue. `play.py` opts in before constructing the environment.
**Tech stack:** Existing Python, Torch, MuJoCo and mjlab; no added packages.

## Constraints
- Preserve training and normal play, actor dimensions, downloaded checkpoint compatibility.
- Native single-environment keyboard control only; no browser or external hardware interface.
- Clear code with necessary comments, no broad defensive exception handling.

## Task 1: Persistent play commands and key mapping
- [x] Add `tests/test_keyboard_control.py`: real CPU command instances in a small fake scene; assert persistence through compute, limits, source four-column compatibility, projected targets, warm-up delay, hold/return/re-target continuity, reset, independent dog/foot keys and unchanged base configurations.
- [x] Run `python -m unittest discover -s tests -p test_keyboard_control.py -v`; confirm the missing feature fails.
- [x] Add `src/utils/keyboard_control.py`: `KeyboardVelocityCommandCfg`, `KeyboardFootPositionCommandCfg`, `configure_keyboard_commands(cfg)`, `KeyboardController(env).handle_key(key)`.
- [x] Re-run the targeted suite and fix behavior until green.

## Task 2: Native viewer and CLI
- [x] Add tests for A consuming only the teleop action, non-teleop passthrough, observation refresh and startup validation.
- [x] Add `src/utils/keyboard_viewer.py`: `KeyboardMujocoViewer(env, policy)`; main-thread action dispatch, overlay and command refresh.
- [x] Update `scripts/play.py` with `keyboard: bool = False`; validate native backend/single environment, configure commands before environment construction and select the keyboard viewer.
- [x] Run targeted tests and `python scripts/play.py Unitree-Go2-Quad-Pedipulation --help`.

## Task 3: Validation and usage
- [x] Add a bounded CUDA smoke check `scripts/check_keyboard_control.py` that loads downloaded Flat/Quad/Biped checkpoints, issues velocity and FR commands, checks actor command observations and finite actions, holds beyond six seconds, returns and resets. Provide native viewer mode for checking the callback/overlay path without leaving a process running.
- [x] Run the full unittest suite, CUDA checks and short native viewer check with `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.
- [x] Write `docs/keyboard-control.md` with commands, keys, limits, coordinate convention and measured validation scope. Link from README.
- [x] Review the changed files and report the exact launch command and test results.

Completed: 78 tests; seven downloaded policies in CUDA; both Pedipulation native viewers; independent review passed after the history-cache fix. Evidence: `logs/keyboard_validation/2026-10-25/verification.json`.
