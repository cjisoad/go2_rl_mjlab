# Go2 keyboard control design

Approved in conversation on 2026-10-25. Control the robot and its FR leg, not a separate arm.

Follow mjlab's explicit boolean CLI convention (`--keyboard True`). Use `scripts/play.py --keyboard True --viewer native --num-envs 1`. Plain play and training are unchanged. Keyboard mode replaces automatic command generators only in its fresh play configuration. Supported velocity tasks: Flat, Rough, DreamWaQ, AMP, StandUp and both Pedipulation tasks; Backflip and Tracking have no compatible velocity input and report a clear error.

W/S increment body forward velocity by +/-0.1 m/s; A/D increment left velocity by +/-0.1 m/s; Q/E increment turn velocity by +/-0.1 rad/s. X zeros velocity. Values persist between presses and are clamped to the task's play command ranges. StandUp/Biped retain their source upright velocity convention and zero lateral range.

I/K, J/L and U/O move the requested FR goal along body +X/-X, +Y/-Y and +Z/-Z by 0.01 m per key event. H returns the FR leg home without stopping locomotion. Requests are clamped to the workspace bounding box and projected to the union of home-to-training-target segments; those are the actual reference trajectories used in training. Keep the requested cursor separate from the projected goal so repeated small key increments are retained. Move from the current reference to the new goal with a one-second quintic blend, then hold indefinitely. Respect the task's initial stand-up delay; do not restart that delay on each key press. Reset clears velocity and returns the reference home.

The render callback only enqueues keys; the simulation thread changes tensors. Consume A before the viewer's existing show-all binding. Keep Space/Enter, speed and debug keys working. Display speed, requested/projected/current FR references, coordinate convention and key help. Recompute the cached observation after a key changes commands without pushing history a second time. Actor input shapes and policy checkpoints are unchanged.

Validate unit behavior, original task regressions, real CUDA environments with downloaded policies, and the native viewer using a short programmatic key sequence. Do not claim measured tracking success from command plumbing tests.

## Shortcut conflict correction

User requested disabling MuJoCo shortcuts. The passive Simulate window applies built-in keys independently of the Python callback, so keyboard mode now creates its own GLFW window with MuJoCo MjvScene/MjrContext rendering. Only repository robot/playback keys are installed; mouse camera movement/zoom and debug plots remain available. The normal non-keyboard viewer is unchanged. Regression forbids calling launch_passive; native smoke verifies the installed GLFW callback and unchanged render flags after all robot-control keys.
