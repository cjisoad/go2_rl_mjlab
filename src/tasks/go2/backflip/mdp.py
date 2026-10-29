"""Archived V1.6 backflip rewards, with mjlab state/contact access.

Source: unitree_rl_lab, f36f6b9 + 2026-07-04_23-21-24 V1.6 recorded diff.
Landing targets restore nonzero stance width and rear-leg standing posture.
The orientation target is corrected to match the negative-pitch progress target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

# Both targets describe the same negative-Y full turn.
FLIP_START = 0.5
LANDING_START = 1.15
LANDING_PREP_START = 1.0


def current_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.action


def previous_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.prev_action


def backflip_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    phase = torch.pi * env.episode_length_buf[:, None] * env.step_dt / 2.0
    return torch.cat(
        [
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2.0),
            torch.cos(phase / 2.0),
            torch.sin(phase / 4.0),
            torch.cos(phase / 4.0),
        ],
        dim=-1,
    )


def _gravity_vec_w(asset: Entity, num_envs: int) -> torch.Tensor:
    gravity = asset.data.gravity_vec_w
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0)
    if gravity.ndim == 3 and gravity.shape[1] == 1:
        gravity = gravity.squeeze(1)
    return gravity.expand(num_envs, -1)


def orientation_control(
    env: ManagerBasedRlEnv,
    flip_start: float = FLIP_START,
    landing_start: float = LANDING_START,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    phase = ((current_time - flip_start) / (landing_start - flip_start)).clamp(0.0, 1.0)
    axis = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    quat_pitch = math_utils.quat_from_angle_axis(-2.0 * torch.pi * phase, axis)
    desired_projected_gravity = math_utils.quat_apply_inverse(
        quat_pitch, _gravity_vec_w(asset, env.num_envs)
    )
    return torch.sum(
        torch.square(asset.data.projected_gravity_b - desired_projected_gravity), dim=1
    )


def ang_vel_y(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    ang_vel = -asset.data.root_link_ang_vel_b[:, 1].clamp(max=7.2, min=-7.2)
    return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)


def pitch_progress(
    env: ManagerBasedRlEnv,
    flip_start: float = FLIP_START,
    landing_start: float = LANDING_START,
    target_rotation: float = 2.0 * torch.pi,
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    active = torch.logical_and(current_time > flip_start, current_time < landing_start)
    if not hasattr(env, "_backflip_pitch_rotation"):
        env._backflip_pitch_rotation = torch.zeros(env.num_envs, device=env.device)
    reset_mask = env.episode_length_buf <= 1
    env._backflip_pitch_rotation = torch.where(
        reset_mask,
        torch.zeros_like(env._backflip_pitch_rotation),
        env._backflip_pitch_rotation,
    )
    rotation_progress = -asset.data.root_link_ang_vel_b[:, 1] * env.step_dt
    env._backflip_pitch_rotation = env._backflip_pitch_rotation + torch.where(
        active, rotation_progress, torch.zeros_like(rotation_progress)
    )
    target_ratio = ((current_time - flip_start) / (landing_start - flip_start)).clamp(
        min=0.0, max=1.0
    )
    progress_ratio = (env._backflip_pitch_rotation / target_rotation).clamp(
        min=0.0, max=1.5
    )
    return torch.exp(-torch.square(progress_ratio - target_ratio) / std) * active


def ang_vel_z(
    env: ManagerBasedRlEnv,
    deadzone: float = 0.3,
    early_start: float = 0.45,
    late_start: float = 0.85,
    full_start: float = 1.2,
    early_scale: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    yaw_rate = torch.clamp(yaw_rate - deadzone, min=0.0)
    early_gate = ((current_time - early_start) / (late_start - early_start)).clamp(
        min=0.0, max=1.0
    )
    late_gate = ((current_time - late_start) / (full_start - late_start)).clamp(
        min=0.0, max=1.0
    )
    phase_scale = early_scale * early_gate + (1.0 - early_scale) * late_gate
    return yaw_rate * phase_scale


def lin_vel_z(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    lin_vel = asset.data.root_link_lin_vel_w[:, 2].clamp(max=3.0)
    return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)


def launch_lin_vel_z(
    env: ManagerBasedRlEnv,
    launch_start: float = 0.3,
    launch_end: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    lin_vel = asset.data.root_link_lin_vel_w[:, 2].clamp(max=2.0)
    active = torch.logical_and(current_time > launch_start, current_time < launch_end)
    return lin_vel * active


def crouch_base_height(
    env: ManagerBasedRlEnv,
    crouch_target_height: float = 0.28,
    stand_target_height: float = 0.3,
    crouch_end: float = 0.25,
    launch_end: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    ramp = ((current_time - crouch_end) / (launch_end - crouch_end)).clamp(
        min=0.0, max=1.0
    )
    launch_target_height = crouch_target_height + ramp * (
        stand_target_height - crouch_target_height
    )
    target_height = torch.where(
        current_time < crouch_end, crouch_target_height, launch_target_height
    )
    active = current_time < launch_end
    return torch.square(target_height - asset.data.root_link_pos_w[:, 2]) * active


def height_control(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    height_diff = torch.square(target_height - asset.data.root_link_pos_w[:, 2])
    return height_diff * torch.logical_or(
        torch.logical_and(current_time > 0.45, current_time < 0.5), current_time > 1.4
    )


def _landing_blend(
    current_time: torch.Tensor,
    start: float = LANDING_PREP_START,
    end: float = LANDING_START,
) -> torch.Tensor:
    """Smoothstep handover: zero before preparation, one from landing onward."""
    phase = ((current_time - start) / (end - start)).clamp(0.0, 1.0)
    return phase.square() * (3.0 - 2.0 * phase)


def air_tuck_pose(
    env: ManagerBasedRlEnv,
    tuck_start: float = 0.5,
    hold_start: float = 0.65,
    hold_end: float = LANDING_PREP_START,
    tuck_end: float = LANDING_START,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    target_joint_pos = torch.tensor(
        [0.0, 1.1, -1.8, 0.0, 1.1, -1.8, 0.0, 1.2, -1.8, 0.0, 1.2, -1.8],
        device=env.device,
    )
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pose_error = torch.sum(torch.square(joint_pos - target_joint_pos), dim=1)
    ramp_in = ((current_time - tuck_start) / (hold_start - tuck_start)).clamp(
        min=0.0, max=1.0
    )
    # Complementary to the rear-leg landing ramp: no abrupt target switch.
    ramp_out = 1.0 - _landing_blend(current_time, hold_end, tuck_end)
    gate = ramp_in * ramp_out
    return pose_error * gate


def front_hip_landing_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    landing_start: float = LANDING_PREP_START,
    landing_full: float = LANDING_START,
) -> torch.Tensor:
    """Keep front hips near zero through landing while allowing leg reach."""
    asset: Entity = env.scene[asset_cfg.name]
    hip_error = asset.data.joint_pos[:, asset_cfg.joint_ids].square().sum(dim=1)
    gate = _landing_blend(
        env.episode_length_buf * env.step_dt, landing_start, landing_full
    )
    return hip_error * gate


def rear_leg_landing_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    target_pose: tuple[float, ...] = (0.0, 1.0, -1.5, 0.0, 1.0, -1.5),
    hip_error_weight: float = 4.0,
    landing_start: float = LANDING_PREP_START,
    landing_full: float = LANDING_START,
) -> torch.Tensor:
    """Recover RR/RL standing posture, emphasizing hip abduction control."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = (joint_pos - joint_pos.new_tensor(target_pose)).square()
    weights = joint_pos.new_tensor(
        [hip_error_weight, 1.0, 1.0, hip_error_weight, 1.0, 1.0]
    )
    gate = _landing_blend(
        env.episode_length_buf * env.step_dt, landing_start, landing_full
    )
    return (error * weights).sum(dim=1) * gate


def actions_symmetry(env: ManagerBasedRlEnv) -> torch.Tensor:
    actions = env.action_manager.action
    actions_diff = torch.square(actions[:, 0] + actions[:, 3])
    actions_diff += torch.square(actions[:, 1:3] - actions[:, 4:6]).sum(dim=-1)
    actions_diff += torch.square(actions[:, 6] + actions[:, 9])
    actions_diff += torch.square(actions[:, 7:9] - actions[:, 10:12]).sum(dim=-1)
    return actions_diff


def gravity_y(
    env: ManagerBasedRlEnv,
    landing_start: float = 1.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    active = current_time > landing_start
    return torch.square(asset.data.projected_gravity_b[:, 1]) * active


def feet_distance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    stance_width: float = 0.30,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    foot_pos_translated = asset.data.site_pos_w[
        :, asset_cfg.site_ids, :
    ] - asset.data.root_link_pos_w.unsqueeze(1)
    root_link_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(
        -1, len(asset_cfg.site_ids), -1
    )
    foot_pos_b = math_utils.quat_apply_inverse(root_link_quat_w, foot_pos_translated)
    # Site order is FL, FR, RL, RR; width is the full left-to-right distance.
    desired_ys = foot_pos_b.new_tensor([0.5, -0.5, 0.5, -0.5]) * stance_width
    stance_diff = torch.square(desired_ys - foot_pos_b[:, :, 1]).sum(dim=1)
    active = torch.logical_or(current_time < 0.5, current_time > 1.2)
    return stance_diff * active


def _smooth_window(
    current_time: torch.Tensor, start: float, full: float, end: float
) -> torch.Tensor:
    ramp_in = ((current_time - start) / (full - start)).clamp(min=0.0, max=1.0)
    ramp_out = ((end - current_time) / (end - full)).clamp(min=0.0, max=1.0)
    return torch.minimum(ramp_in, ramp_out)


def front_leg_reach_landing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    front_feet_cfg: SceneEntityCfg,
    reach_start: float = 0.9,
    reach_full: float = 1.15,
    reach_end: float = 1.35,
    desired_x: float = 0.28,
    desired_z: float = -0.24,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    front_foot_pos_translated = asset.data.site_pos_w[
        :, front_feet_cfg.site_ids, :
    ] - asset.data.root_link_pos_w.unsqueeze(1)
    root_link_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(
        -1, len(front_feet_cfg.site_ids), -1
    )
    front_foot_pos_b = math_utils.quat_apply_inverse(
        root_link_quat_w, front_foot_pos_translated
    )
    reach_x_error = torch.square(front_foot_pos_b[:, :, 0] - desired_x)
    reach_z_error = torch.square(front_foot_pos_b[:, :, 2] - desired_z)
    front_foot_error = torch.sum(reach_x_error + 0.5 * reach_z_error, dim=1)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    front_target = torch.tensor([0.0, 0.45, -1.0, 0.0, 0.45, -1.0], device=env.device)
    front_joint_error = torch.sum(torch.square(joint_pos[:, :6] - front_target), dim=1)
    gate = _smooth_window(current_time, reach_start, reach_full, reach_end)
    return (front_foot_error + 0.15 * front_joint_error) * gate


def front_feet_first_contact(
    env: ManagerBasedRlEnv,
    front_sensor_cfg: str,
    rear_sensor_cfg: str,
    threshold: float = 1.0,
    contact_start: float = 1.05,
    contact_end: float = 1.4,
) -> torch.Tensor:
    current_time = env.episode_length_buf * env.step_dt
    active = torch.logical_and(current_time > contact_start, current_time < contact_end)
    front_contact = _contact_any(env, front_sensor_cfg, threshold)
    rear_contact = _contact_any(env, rear_sensor_cfg, threshold)
    front_any = front_contact.any(dim=1)
    rear_any = rear_contact.any(dim=1)
    return front_any.float() * torch.logical_not(rear_any).float() * active


def early_rear_feet_contact(
    env: ManagerBasedRlEnv,
    rear_sensor_cfg: str,
    threshold: float = 1.0,
    contact_start: float = 1.05,
    contact_end: float = 1.35,
) -> torch.Tensor:
    current_time = env.episode_length_buf * env.step_dt
    active = torch.logical_and(current_time > contact_start, current_time < contact_end)
    rear_contact = _contact_any(env, rear_sensor_cfg, threshold)
    return rear_contact.any(dim=1).float() * active


def landing_stability(
    env: ManagerBasedRlEnv,
    stable_start: float = 1.25,
    target_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    active = current_time > stable_start
    base_xy_vel = torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)
    yaw_roll_rate = torch.square(asset.data.root_link_ang_vel_b[:, 0]) + torch.square(
        asset.data.root_link_ang_vel_b[:, 2]
    )
    height_error = torch.square(asset.data.root_link_pos_w[:, 2] - target_height)
    return (base_xy_vel + 0.25 * yaw_roll_rate + 2.0 * height_error) * active


def feet_height_before_backflip(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    foot_height = (
        asset.data.site_pos_w[:, asset_cfg.site_ids, 2].view(env.num_envs, -1) - 0.02
    )
    return foot_height.clamp(min=0.0).sum(dim=1) * (current_time < 0.5)


def action_rate(
    env: ManagerBasedRlEnv,
    landing_start: float = LANDING_START,
    landing_scale: float = 1.0,
) -> torch.Tensor:
    """Scale action-change cost after landing without slowing the flip."""
    cost = torch.sum(
        torch.square(env.action_manager.prev_action - env.action_manager.action), dim=1
    )
    landed_phase = env.episode_length_buf * env.step_dt >= landing_start
    return cost * torch.where(landed_phase, landing_scale, 1.0)


def _contact_any(env, sensor_cfg: str, threshold: float):
    # mjlab: [env, primary geom, physics history, xyz].
    forces = env.scene[sensor_cfg].data.force_history
    return forces.norm(dim=-1).amax(dim=2) > threshold


def _nonfoot_count(env, sensor_names, threshold):
    # Count bodies, not the multiple collision geoms belonging to one calf/base.
    return torch.stack(
        [
            env.scene[name].data.force_history.sum(dim=1).norm(dim=-1).amax(dim=1)
            > threshold
            for name in sensor_names
        ],
        dim=1,
    ).sum(dim=1)


def undesired_contacts(env, sensor_cfg, threshold=1.0, landing_start=1.15):
    return _nonfoot_count(env, sensor_cfg, threshold) * (
        env.episode_length_buf * env.step_dt > landing_start
    )


def prelaunch_nonfoot_contacts(env, sensor_cfg, threshold=1.0, contact_end=0.5):
    return _nonfoot_count(env, sensor_cfg, threshold) * (
        env.episode_length_buf * env.step_dt < contact_end
    )


def stance_foot_slip(
    env, sensor_cfg, asset_cfg, threshold=1.0, stance_end=0.45,
    landing_start=LANDING_START,
):
    asset = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    # Preserve archived prelaunch shaping and its contact-history semantics.
    contact = _contact_any(env, sensor_cfg, threshold)
    relative_vel = foot_vel_w - asset.data.root_link_lin_vel_w[:, None, :2]
    prelaunch_cost = (relative_vel.square().sum(-1) * contact).sum(1)
    # After landing, penalize actual ground slip only while a foot is touching.
    current_contact = env.scene[sensor_cfg].data.force.norm(dim=-1) > threshold
    landing_cost = (foot_vel_w.square().sum(-1) * current_contact).sum(1)
    return prelaunch_cost * (current_time < stance_end) + landing_cost * (
        current_time >= landing_start
    )


def base_height(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def randomize_default_joint_pos(env, env_ids):
    robot = env.scene["robot"]
    ids = slice(None) if env_ids is None else env_ids
    robot.data.default_joint_pos[ids] += torch.empty_like(
        robot.data.default_joint_pos[ids]
    ).uniform_(-0.02, 0.02)
    action = env.action_manager.get_term("joint_pos")
    action._offset[ids] = robot.data.default_joint_pos[ids][:, action.target_ids]
    action.reset(ids)
