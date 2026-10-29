"""Episode-level backflip outcomes; privileged state is used only for evaluation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BackflipEvaluationCfg:
    contact_force: float = 1.0
    airborne_duration: float = 0.06
    takeoff_clearance: float = 0.06
    rotation_tolerance: float = math.pi / 6.0
    upright_tilt: float = math.radians(20.0)
    min_height: float = 0.26
    max_height: float = 0.40
    max_horizontal_speed: float = 0.5
    max_vertical_speed: float = 0.3
    max_angular_speed: float = 1.0
    stable_duration: float = 0.2


class BackflipEvaluator:
    """Track one attempt per environment, without changing rewards or observations."""

    def __init__(self, num_envs, device, step_dt, cfg=None):
        self.cfg = cfg if cfg is not None else BackflipEvaluationCfg()
        self.step_dt = step_dt
        self.airborne_steps = math.ceil(self.cfg.airborne_duration / step_dt)
        self.required_stable_steps = math.ceil(self.cfg.stable_duration / step_dt)
        self.air_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_steps = torch.zeros_like(self.air_steps)
        self.observed_steps = torch.zeros_like(self.air_steps)
        self.rotation = torch.zeros(num_envs, device=device)
        self.ground_height = torch.zeros_like(self.rotation)
        self.seen_ground = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.takeoff = torch.zeros_like(self.seen_ground)
        self.inverted = torch.zeros_like(self.seen_ground)
        self.full_flip = torch.zeros_like(self.seen_ground)
        self.failed = torch.zeros_like(self.seen_ground)
        self._buffers = (
            self.air_steps,
            self.stable_steps,
            self.observed_steps,
            self.rotation,
            self.ground_height,
            self.seen_ground,
            self.takeoff,
            self.inverted,
            self.full_flip,
            self.failed,
        )
        self.last_step = -1

    @property
    def success(self):
        # Deliberately not latched: losing stability before timeout revokes success.
        return (self.stable_steps >= self.required_stable_steps) & ~self.failed

    def update(
        self,
        *,
        episode_steps,
        height,
        linear_velocity,
        angular_velocity,
        gravity,
        feet_contact,
        nonfoot_contact,
    ):
        cfg = self.cfg
        reset = episode_steps <= 1
        for buffer in self._buffers:
            buffer[reset] = 0
        self.ground_height[reset] = height[reset]
        self.observed_steps += 1

        grounded = feet_contact.any(dim=1)
        self.seen_ground |= grounded
        before_takeoff = grounded & ~self.takeoff
        self.ground_height[before_takeoff] = height[before_takeoff]
        self.air_steps.copy_(torch.where(grounded, 0, self.air_steps + 1))
        # Accumulate from the first airborne frame, including the confirmation delay.
        attempting = self.takeoff | (self.seen_ground & ~grounded)
        self.rotation.copy_(torch.where(before_takeoff, 0.0, self.rotation))
        self.rotation += torch.where(
            attempting, -angular_velocity[:, 1] * self.step_dt, 0.0
        )
        self.takeoff |= (
            self.seen_ground
            & (self.air_steps >= self.airborne_steps)
            & (height - self.ground_height >= cfg.takeoff_clearance)
        )
        self.failed |= attempting & nonfoot_contact
        self.inverted |= self.takeoff & (gravity[:, 2] > 0.5)
        rotation_ok = (self.rotation - 2.0 * math.pi).abs() <= cfg.rotation_tolerance
        self.full_flip |= self.takeoff & self.inverted & rotation_ok & ~self.failed

        # Foot order is FL, FR, RL, RR. Three feet necessarily include both ends.
        supported = feet_contact.sum(dim=1) >= 3
        upright = gravity[:, 2] < -math.cos(cfg.upright_tilt)
        height_ok = (height >= cfg.min_height) & (height <= cfg.max_height)
        quiet = (
            (linear_velocity[:, :2].norm(dim=1) < cfg.max_horizontal_speed)
            & (linear_velocity[:, 2].abs() < cfg.max_vertical_speed)
            & (angular_velocity.norm(dim=1) < cfg.max_angular_speed)
        )
        stable = (
            self.full_flip
            & rotation_ok
            & supported
            & upright
            & height_ok
            & quiet
            & ~self.failed
            & ~nonfoot_contact
        )
        self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))


def _contact_mask(env, sensor, threshold):
    # Current forces [env, geom, xyz] with one slot per geom in this task.
    return env.scene[sensor].data.force.norm(dim=-1) > threshold


def backflip_metric(env, stage, nonfoot_sensors):
    """Emit one terminal outcome, compensating MetricsManager's step averaging."""
    if not hasattr(env, "_backflip_evaluator"):
        env._backflip_evaluator = BackflipEvaluator(
            env.num_envs, env.device, env.step_dt
        )
    evaluator = env._backflip_evaluator
    if evaluator.last_step != env.common_step_counter:
        robot = env.scene["robot"].data
        threshold = evaluator.cfg.contact_force
        nonfoot = torch.stack(
            [
                _contact_mask(env, name, threshold).any(dim=1)
                for name in nonfoot_sensors
            ],
            dim=1,
        ).any(dim=1)
        evaluator.update(
            episode_steps=env.episode_length_buf,
            height=robot.root_link_pos_w[:, 2],
            linear_velocity=robot.root_link_lin_vel_w,
            angular_velocity=robot.root_link_ang_vel_b,
            gravity=robot.projected_gravity_b,
            feet_contact=_contact_mask(env, "feet_contact", threshold),
            nonfoot_contact=nonfoot,
        )
        evaluator.last_step = env.common_step_counter
    outcome = getattr(evaluator, stage)
    # MetricsManager divides by actual calls, not the randomized episode timer.
    # Sum(outcome * N only at terminal step) / N = one outcome per episode.
    return outcome.float() * env.reset_buf * evaluator.observed_steps
