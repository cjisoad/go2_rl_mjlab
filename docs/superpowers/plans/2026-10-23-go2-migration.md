# Go2 mjlab migration implementation plan

Goal: retain tracking and Go2 training, migrate DreamWaQ, AMP, stand-up V2.1 and backflip V1.6 from the specified remote workspace.

User approved the design on 2026-10-23 and selected backflip V1.6. Implement directly in `/home/e980/unitree_rl_mjlab`, on branch `codex/go2-methods-migration` with baseline commit `1425b15`. Baseline bundle is in this chat's work directory. The remote is read-only.

Architecture: independent `src/tasks/go2/{dreamwaq,amp,standup,backflip}` packages, each registering its own mjlab task. Reuse the existing Go2 model and mjlab managers. Keep the installed mjlab 1.2.0, mujoco-warp 3.5.0 and RSL-RL 5.0.1. Adapt the source algorithm extensions through RSL-RL's callable class paths; never shadow the installed `rsl_rl` package. No Isaac runtime dependency.

Global constraints:
- Retain all tracking code and its G1/G1-23DoF assets and motion utilities.
- Retain Go2 Flat/Rough tasks and necessary training/playback/export utilities.
- Remove unrelated robot velocity tasks/assets and physical deployment/simulator programs.
- Preserve method-specific observations, histories, rewards, domain randomization, episode timing and curriculum; document simulator adaptations.
- Explicit joint and foot ordering, wxyz quaternion conventions, base frame velocities and terminal observation handling are required.
- Keep code readable with necessary comments; avoid broad exception handlers and speculative compatibility scaffolding.
- All new task defaults use tensorboard to permit local unattended training.
- Use tests first for numerical and algorithm behaviors. Verify actual CUDA environment stepping and short PPO updates; check checkpoint resume and policy export.

Tasks:
1. Capture current remote source and AMP demonstration data plus licenses and hashes. Restore backflip V1.6 source using the last V1.6 run's recorded base commit and diff, compare all V1.6 parameter snapshots.
2. Prune unrelated code, make task imports explicit and keep tracking dependencies. Verify retained registry/imports. Fix package discovery and training/playback routing only where needed.
3. Migrate DreamWaQ: 45-dimensional proprioception and 5-step history, 19-dimensional context estimator with velocity supervision, reconstruction and KL terms, native RSL-RL actor/PPO extension. Task `Unitree-Go2-DreamWaQ`. Test latent/velocity gradients, history reset, rollout update, checkpoint and export.
4. Migrate AMP: explicit source feature ordering, source demonstration dataset, discriminator/style reward, replay and discriminator training with terminal transitions. Task `Unitree-Go2-AMP`. Test dataset feature correspondence, reward and discriminator updates, save/load.
5. Migrate current standup V2.1: source reward and action/command/reset/curriculum logic adapted to mjlab. Task `Unitree-Go2-StandUp`. Test source numerical semantics and environment startup/step.
6. Migrate backflip V1.6: restore archived source/config, preserve phase timing and pitch progression, delayed action and contact rewards. Task `Unitree-Go2-Backflip-V1.6`. Test phase state/reset and reward semantics, environment startup/step.
7. Integrate all registrations and run focused unit tests, GPU smoke tests, short training updates, checkpoint restore and inference/export tests. Review all changes for missing source behaviors and unrelated scope.
8. Write Chinese usage documentation, provenance manifest, removal summary and verification results. Leave all changes in the requested WSL repository and deliver a concise report in this chat's outputs directory.

Execution: use specialized implementation agents for independent method packages, as required by the applicable subagent-driven-development / dispatching-parallel-agents skills. Parent handles shared integration and backflip. Agents own disjoint paths and must not edit shared code or run GPU jobs concurrently. Capture reports in work/ for durable progress; run a separate final review.
