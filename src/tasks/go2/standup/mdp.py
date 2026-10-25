"""Native mjlab tensor adapters for the unmodified go2_handstand rewards."""

from __future__ import annotations

import torch
from mjlab.envs import mdp as lab_mdp
from dataclasses import dataclass, field
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from .source_config import Go2_Handstand_Cfg_Yu as SourceCfg
from .source_rewards import SourceRewards

JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
FEET_NAMES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]


class SourceState(SourceRewards):
    def __init__(self, env):
        self.env = env
        self.cfg = SourceCfg()
        self.device, self.num_envs, self.num_dof = env.device, env.num_envs, 12
        self.dt = env.step_dt
        self.robot = env.scene["robot"]
        self.joint_ids = self.robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
        self.feet_indices = [0, 1, 2, 3]
        self.site_ids = self.robot.find_sites(
            ["FL", "FR", "RL", "RR"], preserve_order=True
        )[0]
        self.feet_name_reward_indices = [0, 1]
        self.contact_foot_indices = [2, 3]
        self.penalised_contact_indices = list(range(4, 12))
        self.descire_joint_pos = torch.tensor(
            [self.cfg.init_state.descire_joint_angles[n] for n in JOINT_NAMES],
            device=self.device,
        )
        self.target_gravity = torch.tensor(
            self.cfg.asset.target_gravity, device=self.device
        )
        self.soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[
            0, self.joint_ids
        ]
        self.dof_vel_limits = torch.full((12,), 30.0, device=self.device)
        self.torque_limits = torch.tensor(
            [35.55 if "calf" in n else 23.7 for n in JOINT_NAMES], device=self.device
        )
        self.measured_heights = 0.0  # source plane has no height samples
        self.rew_hanstand = torch.zeros(1, device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_contacts = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._previous_vel = torch.zeros((self.num_envs, 12), device=self.device)
        self._step = -1

    def refresh(self):
        env, data = self.env, self.robot.data
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        # Source callback runs before rewards; mjlab updates commands afterwards.
        env.command_manager.get_term("base_velocity").before_rewards()
        self.commands = env.command_manager.get_term("base_velocity").source_commands
        self.root_states = torch.cat(
            (data.root_link_pose_w, data.root_link_vel_w), dim=-1
        )
        self.base_lin_vel, self.base_ang_vel = (
            data.root_link_lin_vel_b,
            data.root_link_ang_vel_b,
        )
        self.projected_gravity = data.projected_gravity_b
        euler = torch.stack(euler_xyz_from_quat(data.root_link_quat_w), dim=1)
        self.base_euler_xyz = torch.where(euler > torch.pi, euler - 2 * torch.pi, euler)
        self.rigid_state = torch.zeros((self.num_envs, 4, 13), device=self.device)
        self.rigid_state[:, :, :3] = data.site_pos_w[:, self.site_ids]
        self.contact_forces = contact_forces(env)
        self.feet_pos = self.rigid_state[:, self.feet_name_reward_indices, :3]
        self.dof_pos, self.dof_vel = (
            data.joint_pos[:, self.joint_ids],
            data.joint_vel[:, self.joint_ids],
        )
        self.last_dof_vel = self._previous_vel.clone()
        self._previous_vel.copy_(self.dof_vel)
        self.torques = data.actuator_force[:, self.joint_ids]
        action = env.action_manager.get_term("joint_pos")
        self.actions, self.last_actions = action.raw_action, action.previous_raw_actions
        self.episode_length_buf = env.episode_length_buf
        self.reset_buf, self.time_out_buf = env.reset_buf, env.reset_time_outs
        # Source applies pushes after reading base velocities, before reward evaluation.
        if (
            self.cfg.domain_rand.push_robots
            and env.cfg.enable_pushes
            and env.common_step_counter % 400 == 0
        ):
            vel = data.root_link_vel_w.clone()
            vel[:, :2] += torch.empty_like(vel[:, :2]).uniform_(-0.4, 0.4)
            vel[:, 3:] += torch.empty_like(vel[:, 3:]).uniform_(-0.6, 0.6)
            self.robot.write_root_link_velocity_to_sim(vel)


def source_reward(env, name: str):
    if not hasattr(env, "_standup_v2_source"):
        env._standup_v2_source = SourceState(env)
    state = env._standup_v2_source
    state.refresh()
    result = getattr(state, f"_reward_{name}")()
    # The source alive reward returns a scalar; mjlab expects one value per environment.
    return (
        torch.full((env.num_envs,), result, device=env.device)
        if isinstance(result, (float, int))
        else result
    )


@dataclass(kw_only=True)
class VelocityRanges:
    lin_vel_x: tuple = tuple(SourceCfg.commands.ranges.lin_vel_x)
    lin_vel_y: tuple = tuple(SourceCfg.commands.ranges.lin_vel_y)
    ang_vel_z: tuple = tuple(SourceCfg.commands.ranges.ang_vel_yaw)
    heading: tuple = tuple(SourceCfg.commands.ranges.heading)


class SourceVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.source_commands = torch.zeros((env.num_envs, 4), device=env.device)
        self._step = -1

    @property
    def command(self):
        return self.source_commands[:, :3]

    def _resample_command(self, env_ids):
        n = len(env_ids)
        for i, bounds in enumerate(
            (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y)
        ):
            self.source_commands[env_ids, i] = torch.empty(
                n, device=self.device
            ).uniform_(*bounds)
        if self.cfg.heading_command:
            self.source_commands[env_ids, 3] = torch.empty(
                n, device=self.device
            ).uniform_(*self.cfg.ranges.heading)
        else:
            self.source_commands[env_ids, 2] = torch.empty(
                n, device=self.device
            ).uniform_(*self.cfg.ranges.ang_vel_z)
        self.source_commands[env_ids[torch.rand(n, device=self.device) < 0.20]] = 0.0
        self.source_commands[env_ids[torch.rand(n, device=self.device) > 0.9], :2] = 0.0
        self.source_commands[env_ids, :2] *= (
            torch.norm(self.source_commands[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)

    def before_rewards(self):
        if self._step == self._env.common_step_counter:
            return
        self._step = self._env.common_step_counter
        ids = (
            (
                self._env.episode_length_buf
                % round(self.cfg.resampling_time_range[0] / self._env.step_dt)
                == 0
            )
            .nonzero()
            .flatten()
        )
        self._resample_command(ids)
        self._update_command()

    def _update_command(self):
        if self.cfg.heading_command:
            forward = torch.zeros((self.num_envs, 3), device=self.device)
            forward[:, 0] = 1.0
            forward = quat_apply(
                self._env.scene["robot"].data.root_link_quat_w, forward
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.source_commands[:, 2] = torch.clamp(
                0.5 * wrap_to_pi(self.source_commands[:, 3] - heading), -1.0, 1.0
            )

    def _update_metrics(self):
        pass

    def compute(self, dt):
        # Updated at the source callback position, not after rewards/resets.
        pass

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return super().reset(env_ids)


@dataclass(kw_only=True)
class SourceVelocityCommandCfg(CommandTermCfg):
    resampling_time_range: tuple = (10.0, 10.0)
    heading_command: bool = True
    ranges: VelocityRanges = field(default_factory=VelocityRanges)

    def build(self, env):
        return SourceVelocityCommand(self, env)


def policy_observation(env):
    robot = env.scene["robot"]
    ids = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    data = robot.data
    action = env.action_manager.get_term("joint_pos").raw_action
    obs = torch.cat(
        (
            data.root_link_ang_vel_b * 0.25,
            data.projected_gravity_b,
            env.command_manager.get_command("base_velocity")
            * torch.tensor([2.0, 2.0, 0.25], device=env.device),
            data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
            data.joint_vel[:, ids] * 0.05,
            action,
        ),
        dim=-1,
    )
    if env.cfg.observations["actor"].enable_corruption:
        noise = torch.tensor(
            [0.05] * 6 + [0.0] * 3 + [0.01] * 12 + [0.075] * 12 + [0.0] * 12,
            device=env.device,
        )
        obs += (2 * torch.rand_like(obs) - 1) * noise
    # Share the SAME noisy policy observation with the privileged observation, as in source.
    env._standup_v2_policy_obs = obs
    return obs.clamp(-100.0, 100.0)


def critic_observation(env):
    if not hasattr(env, "_standup_v2_policy_obs"):
        policy_observation(env)
    domain = getattr(
        env, "_standup_v2_domain", torch.zeros((env.num_envs, 34), device=env.device)
    )
    contact = contact_forces(env)[:, :4, 2] > 1.0
    return torch.cat(
        (
            env.scene["robot"].data.root_link_lin_vel_b * 2.0,
            env._standup_v2_policy_obs,
            domain,
            contact,
        ),
        dim=-1,
    ).clamp(-100.0, 100.0)


def reset_source(env, env_ids):
    lab_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range={},
        velocity_range={
            key: (-0.5, 0.5) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        },
    )
    robot = env.scene["robot"]
    pos = robot.data.default_joint_pos[env_ids] * torch.empty_like(
        robot.data.joint_pos[env_ids]
    ).uniform_(0.5, 1.5)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)
    state = getattr(env, "_standup_v2_source", None)
    if state is not None:
        state._previous_vel[env_ids] = 0.0
        state.feet_air_time[env_ids] = 0.0
        # Source does not reset the batch height gate or last_contacts.


def contact_forces(env):
    # Virtual source order: four feet, eight penalized thigh/calf bodies, base.
    names = [f"foot_{leg}" for leg in ("FL", "FR", "RL", "RR")]
    names += [
        f"body_{leg}_{part}"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf")
    ]
    names += ["base_contact"]
    return torch.stack([env.scene[name].data.force.sum(dim=1) for name in names], dim=1)


def base_contact(env):
    return torch.linalg.vector_norm(contact_forces(env)[:, -1], dim=-1) > 1.0


def source_timeout(env):
    return env.episode_length_buf > env.max_episode_length


@requires_model_fields(
    "geom_friction",
    "geom_solref",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "dof_armature",
    "dof_frictionloss",
    "dof_damping",
    recompute=RecomputeLevel.set_const,
)
def randomize_source_physics(env, env_ids):
    """Source startup draws, with restitution mapped to contact damping ratio."""
    robot = env.scene["robot"]
    n, dev = env.num_envs, env.device

    def sample(bounds, width=1):
        return torch.empty((n, width), device=dev).uniform_(*bounds)

    cfg = SourceCfg.domain_rand
    friction, restitution = sample(cfg.friction_range), sample(cfg.restitution_range)
    added_mass, com_offset = (
        sample(cfg.added_base_mass_range),
        sample(cfg.added_base_com_range, 3),
    )
    kp, kd = (
        sample(cfg.stiffness_multiplier_range, 12),
        sample(cfg.damping_multiplier_range, 12),
    )
    armature, joint_friction, damping = (
        sample(cfg.joint_armature_range),
        sample(cfg.joint_friction_range),
        sample(cfg.joint_damping_range),
    )
    env._standup_v2_motor_offsets = sample(cfg.motor_zero_offset_range, 12)
    env._standup_v2_domain = torch.cat(
        (
            torch.zeros_like(friction),
            added_mass,
            com_offset,
            kp,
            kd,
            armature,
            joint_friction,
            damping,
            restitution,
            restitution,
        ),
        dim=-1,
    )
    model = env.sim.model
    idx = robot.indexing
    geoms, bodies, dofs = idx.geom_ids, idx.body_ids, idx.joint_v_adr
    model.geom_friction[:, geoms, 0] = friction
    # MuJoCo has no PhysX restitution coefficient; damping ratio produces an
    # equivalent idealized rebound ratio for a linear spring-damper contact.
    loge = torch.log(restitution.clamp_min(1e-6))
    model.geom_solref[:, geoms, 1] = -loge / torch.sqrt(torch.pi**2 + loge**2)
    old_mass = model.body_mass[:, bodies].clone()
    ratio = sample(cfg.multiplied_link_mass_range, len(bodies))
    base_local = robot.find_bodies("base_link")[0][0]
    ratio[:, base_local] = (old_mass[:, base_local] + added_mass[:, 0]) / old_mass[
        :, base_local
    ]
    model.body_mass[:, bodies] *= ratio
    model.body_inertia[:, bodies] *= ratio.unsqueeze(-1)
    model.body_ipos[:, idx.root_body_id] += com_offset
    model.dof_armature[:, dofs] = armature
    model.dof_frictionloss[:, dofs] = joint_friction
    model.dof_damping[:, dofs] = damping
    ids = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    # Source policy order is independent of MJCF actuator order.
    inverse = torch.argsort(torch.tensor(ids, device=dev))
    actuator = robot.actuators[0]
    actuator.stiffness[:] = 40.0 * kp[:, inverse]
    actuator.damping[:] = kd[:, inverse]
