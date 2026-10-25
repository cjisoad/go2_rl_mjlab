# Go2 AMP

Native mjlab task `Unitree-Go2-AMP`. Run the repository train/play scripts with
that task identifier; defaults log to TensorBoard in experiment `go2_amp`.

AMP states are 30 scalars: absolute joint positions, velocities (FL, FR, RL, RR;
hip, thigh, calf), then body-frame linear and angular velocity. Policy states are
45 scalars; privileged critic states are 60. The discriminator scores consecutive
states with a 256/128 ReLU network and minibatch standard deviation. The reward
is `0.6 * task_reward + 0.4 * softplus(discriminator_logit)`.

`algorithm.py` extends installed RSL-RL 5 PPO. An Adam pre-step hook computes the
independent discriminator gradients, sharing each PPO minibatch's optimizer step
and adaptive learning rate. Source trunk/head weight decay is 0.001/0.1; BCE and
R1 regularization coefficient 10 are preserved. Raw statistics update after the
optimizer step. Replay is a persistent 100,000-pair circular buffer with source
permutation/modulo sampling. Checkpoints save replay, discriminator, normalizer,
and the shared optimizer; runner saves also export `policy.onnx`.

`amp_terminal_capture` is a zero-valued reward term with nonzero manager weight.
It captures pre-reset AMP states immediately before mjlab's automatic reset.
Completed transitions use that capture; new episodes start from reset states.
This hook follows mjlab 1.2's reward-before-reset ordering.

The source loader, discriminator, replay implementation and BSD license are
retained locally; dataset weights and interpolation conventions are unchanged.
No Isaac dependency or shadow RSL-RL package is used. Demonstrations are bundled
under `src/assets/motions/go2/amp` with metadata and provenance.

CPU validation: `PYTHONPATH=. python tests/test_amp.py`. GPU rollout and learning
quality validation remain integration steps. The plane terrain matches the
source configuration's sole enabled flat subterrain. Native friction uses a
single MuJoCo sliding coefficient rather than separate static/dynamic buckets;
Source restitution draws (0 to 0.15) are adapted to MuJoCo contact damping ratios. Self-collisions are enabled. Source USD collision geometry and solver behavior cannot be exactly
reproduced in the existing MJCF. Source added-base-mass randomization (-1 to +3 kg) is ported using native body_mass with unchanged inertia, matching the source point-mass behavior.
