"""Reward adaptations for a single free FR leg; actor extensions are command-only."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.go2.dreamwaq import mdp as quad
from src.tasks.go2.standup import mdp as biped


def term(env):
    return env.command_manager.get_term("foot_position")


def quad_observation(env, noise=True):
    return torch.cat(
        (
            quad.proprioception(env, noise),
            env.command_manager.get_command("foot_position"),
        ),
        -1,
    )


def biped_observation(env):
    obs = torch.cat(
        (
            biped.policy_observation(env),
            env.command_manager.get_command("foot_position"),
        ),
        -1,
    )
    # Keep the source critic's shared noisy actor observation in the new layout.
    env._standup_v2_policy_obs = obs
    return obs


def foot_position_b(env):
    data = env.scene["robot"].data
    return quat_apply_inverse(
        data.root_link_quat_w,
        data.site_pos_w[:, term(env).site_id] - data.root_link_pos_w,
    )


def position_score(actual, target, std=0.05):
    return torch.exp(-(actual - target).square().sum(-1) / std**2)


def posture_gate(env):
    if not term(env).cfg.biped:
        return torch.ones(env.num_envs, device=env.device)
    data = env.scene["robot"].data
    # Privileged height is used only by rewards, never by actor or command clocks.
    upright = -data.projected_gravity_b[:, 0] > 0.75
    return upright * (data.root_link_pos_w[:, 2] > 0.42)


def fr_tracking(env):
    command = term(env)
    in_cycle = (command.elapsed >= command.cfg.durations[0]) & command.enabled
    return (
        position_score(foot_position_b(env), command.reference)
        * in_cycle
        * posture_gate(env)
    )


def return_weight(env):
    command = term(env)
    start = sum(command.cfg.durations[:3])
    progress = ((command.elapsed - start) / command.cfg.durations[3]).clamp(0, 1)
    # A home reward only late in return avoids fighting the outgoing reference.
    return command.returning * progress.square()


def fr_return(env):
    return (
        position_score(foot_position_b(env), term(env).home)
        * return_weight(env)
        * posture_gate(env)
    )


def foot_weights(env, sensor_name="feet_ground_contact"):
    command = term(env)
    sensor = env.scene[sensor_name]
    _, names = env.scene["robot"].find_geoms(sensor.cfg.primary.pattern)
    fr = next(i for i, name in enumerate(names) if name.startswith("FR_"))
    weights = torch.ones((env.num_envs, len(names)), device=env.device)
    weights[:, fr] = 1 - command.blend
    return weights, fr


def fr_ground_return(env):
    _, fr = foot_weights(env, "pedipulation_ground_contact")
    contact = (
        env.scene["pedipulation_ground_contact"].data.force[:, fr].norm(dim=-1) > 1.0
    )
    return contact * return_weight(env)


def support_contact(env):
    _, fr = foot_weights(env, "pedipulation_ground_contact")
    contact = env.scene["pedipulation_ground_contact"].data.force.norm(dim=-1) > 1.0
    support = [i for i in range(contact.shape[1]) if i != fr]
    return contact[:, support].float().mean(-1) * term(env).blend


def quad_reward(env, kind):
    command, data = term(env), env.scene["robot"].data
    if kind == "joint_position":
        weights = torch.ones_like(data.joint_pos)
        ids = env.scene["robot"].find_joints("FR_.*")[0]
        weights[:, ids] = 1 - command.blend[:, None]
        cost = ((data.joint_pos - data.default_joint_pos) * weights).norm(dim=-1)
        velocity = env.command_manager.get_command("twist")
        moving = (velocity.norm(dim=-1) > 0) | (
            data.root_link_lin_vel_b[:, :2].norm(dim=-1) > 0.3
        )
        return torch.where(moving, cost, 5 * cost)
    weights, _ = foot_weights(env)
    sensor = env.scene["feet_ground_contact"]
    if kind == "air_time":
        return (
            (sensor.data.last_air_time - 0.5)
            * sensor.compute_first_contact(env.step_dt)
            * weights
        ).sum(-1) * (env.command_manager.get_command("twist")[:, :2].norm(dim=-1) > 0.1)
    if kind == "air_variance":

        def variance(values):
            values = values.clamp(max=0.5)
            mean = (values * weights).sum(-1) / weights.sum(-1)
            return ((values - mean[:, None]).square() * weights).sum(-1) / (
                weights.sum(-1) - 1
            )

        return variance(sensor.data.last_air_time) + variance(
            sensor.data.last_contact_time
        )
    if kind == "slide":
        ids = env.scene["robot"].find_sites(("FR", "FL", "RR", "RL"))[0]
        speed = data.site_lin_vel_w[:, ids, :2].norm(dim=-1)
        contact = sensor.data.force_history.norm(dim=-1).max(dim=2).values > 1.0
        return (speed * contact * weights).sum(-1)
    raise ValueError(kind)


def biped_reward(env, name):
    # Calling the original adapter first retains its once-per-step state refresh.
    original = biped.source_reward(env, name)
    state = env._standup_v2_source
    free = term(env).blend
    keep = 1 - free
    ready = state.rew_hanstand > 0.70
    if name == "default_pos":
        error = (state.dof_pos - state.descire_joint_pos).abs()
        return (
            error[:, :3].sum(-1) + error[:, 3:6].sum(-1) * keep + error[:, 6:].sum(-1)
        )
    if name == "default_pos_reward":
        error = (state.dof_pos - state.descire_joint_pos).abs()
        return torch.exp(-error[:, :3].sum(-1) - error[:, 3:6].sum(-1) * keep) * ready
    if name == "default_hip_pos":
        return (
            state.dof_pos[:, [0, 6, 9]].abs().sum(-1) + state.dof_pos[:, 3].abs() * keep
        )
    if name == "symmetric_joints":
        joints = state.dof_pos.reshape(-1, 4, 3).clone()
        joints[:, [1, 3], 0] *= -1
        return (
            (joints[:, 0] - joints[:, 1]).abs().sum(-1) * keep
            + (joints[:, 2] - joints[:, 3]).abs().sum(-1)
        ) * ready
    if name == "feet_height_symmetry":
        return original * keep
    if name == "handstand_feet_height_exp":
        error = (state.feet_pos[:, :, 2] - 0.67).abs()
        return torch.exp(-10 * (error[:, 0] + error[:, 1] * keep))
    raise ValueError(name)
