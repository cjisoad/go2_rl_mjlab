# Play strip implementation plan

**Goal:** Replace the large keyboard-play terrain grid with a repeatable wide strip containing each original terrain type.
**Architecture:** A play-only composite SubTerrainCfg reuses the native terrain functions. `configure_play_terrain` changes only the freshly loaded play configuration; `play.py` selects auto/original/strip. Keyboard graphics start with debug/shadows/reflections off.
**Tech stack:** Existing mjlab/MuJoCo, numpy, torch and GLFW; no new dependencies.

- [x] Write failing tests in `tests/test_play_terrain.py` for mode selection, original-type preservation, fixed start/yaw, no observation changes, real terrain ray coverage/seams and deterministic generation.
- [x] Run the tests and observe the missing feature failure.
- [x] Implement `src/utils/play_terrain.py`: `StripTerrainCfg.function` composes existing functions at fixed X offsets; `configure_play_terrain(cfg, mode, keyboard)` prepares only play configuration.
- [x] Add `PlayConfig.play_terrain` and call the helper before environment construction. Keep the loaded actor inputs and physical settings intact.
- [x] Default keyboard debug/shadows/reflections off, print/show current strip section, update the existing keyboard smoke check to exercise play defaults.
- [x] Run unit tests, compile both source strip variants, inspect rays and scene layout, check CUDA/actor compatibility and render Rough/DreamWaQ/Quad.
- [x] Measure step/frame times on the same local NVIDIA WSLg stack; compare against `/tmp/rough-profile.json` captured before the change.
- [x] Update keyboard docs with controls, original-map switch, course size/order/difficulty, measured performance and test scope. Preserve evidence in logs/play_strip_validation.

Completed and verified: 86 tests; native Rough/DreamWaQ/Quad with downloaded checkpoints; fixed start pose, full source terrain types, render filtering and unchanged actor inputs. Evidence: `logs/play_strip_validation/2026-10-25/verification.json`.
