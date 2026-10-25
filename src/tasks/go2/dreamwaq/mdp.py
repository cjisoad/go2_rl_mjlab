import re
import torch


def scale_frame(frame, noise=True):
    limits = frame.new_tensor(
        [0.2] * 3 + [0.05] * 3 + [0.0] * 3 + [0.01] * 12 + [1.5] * 12 + [0.0] * 12
    )
    scale = frame.new_tensor([0.2] * 3 + [1.0] * 18 + [0.05] * 12 + [1.0] * 12)
    if noise:
        frame = frame + (torch.rand_like(frame) * 2 - 1) * limits
    return frame.clamp(-100, 100) * scale


def proprioception(env, noise=True):
    data = env.scene["robot"].data
    return scale_frame(
        torch.cat(
            (
                data.root_link_ang_vel_b,
                data.projected_gravity_b,
                env.command_manager.get_command("twist"),
                data.joint_pos - data.default_joint_pos,
                data.joint_vel,
                env.action_manager.action,
            ),
            -1,
        ),
        noise,
    )


def effort(env):
    robot = env.scene["robot"]
    # Actuator space can be ordered differently from the joint observation space.
    joint_effort = torch.zeros_like(robot.data.default_joint_pos)
    for actuator in robot.actuators:
        joint_effort[:, actuator.target_ids] = robot.data.actuator_force[
            :, actuator.ctrl_ids
        ]
    return joint_effort


def scalar_reward(env, kind):
    d = env.scene["robot"].data
    cmd = env.command_manager.get_command("twist")
    if kind == "track_linear":
        return torch.exp(
            -((cmd[:, :2] - d.root_link_lin_vel_b[:, :2]) ** 2).sum(-1) / 0.25
        )
    if kind == "track_angular":
        return torch.exp(-((cmd[:, 2] - d.root_link_ang_vel_b[:, 2]) ** 2) / 0.25)
    if kind == "linear_z":
        return d.root_link_lin_vel_b[:, 2].square()
    if kind == "angular_xy":
        return d.root_link_ang_vel_b[:, :2].square().sum(-1)
    if kind == "energy":
        return (d.joint_vel.abs() * effort(env).abs()).sum(-1)
    if kind == "joint_position":
        cost = (d.joint_pos - d.default_joint_pos).norm(dim=-1)
        moving = (cmd.norm(dim=-1) > 0) | (
            d.root_link_lin_vel_b[:, :2].norm(dim=-1) > 0.3
        )
        return torch.where(moving, cost, 5 * cost)
    sensor = env.scene["feet_ground_contact"]
    if kind == "air_time":
        return (
            (sensor.data.last_air_time - 0.5)
            * sensor.compute_first_contact(env.step_dt)
        ).sum(-1) * (cmd[:, :2].norm(dim=-1) > 0.1)
    if kind == "air_variance":
        return sensor.data.last_air_time.clamp(max=0.5).var(
            -1
        ) + sensor.data.last_contact_time.clamp(max=0.5).var(-1)
    if kind == "slide":
        ids = env.scene["robot"].find_sites(("FR", "FL", "RR", "RL"))[0]
        speed = d.site_lin_vel_w[:, ids, :2].norm(dim=-1)
        return (
            speed * (sensor.data.force_history.norm(dim=-1).max(dim=2).values > 1)
        ).sum(-1)
    if kind == "undesired":
        contact_sensor = env.scene["undesired_contact"]
        match = contact_sensor.cfg.primary
        _, names = env.scene["robot"].find_geoms(match.pattern)
        names = [
            name
            for name in names
            if not any(re.fullmatch(pattern, name) for pattern in match.exclude)
        ]
        return undesired_contact_count(contact_sensor.data.force_history, names)
    raise ValueError(kind)


def command_curriculum(env, env_ids):
    term = env.command_manager.get_term("twist")
    cfg = env.reward_manager.get_term_cfg("track_linear")
    reward = (
        env.reward_manager._episode_sums["track_linear"][env_ids].mean()
        / env.cfg.episode_length_s
    )
    if (
        env.common_step_counter % env.max_episode_length == 0
        and reward > cfg.weight * 0.8
    ):
        for name, limit in [("lin_vel_x", 1.0), ("lin_vel_y", 0.4)]:
            low, high = getattr(term.cfg.ranges, name)
            setattr(
                term.cfg.ranges, name, (max(-limit, low - 0.1), min(limit, high + 0.1))
            )
    return torch.tensor(term.cfg.ranges.lin_vel_x[1], device=env.device)


def undesired_contact_count(force_history, geom_names):
    """Count source limb bodies after summing their non-foot geom force vectors.

    MuJoCo attaches the foot sphere to the calf body. Selecting bodies would
    therefore charge ordinary support as a calf collision. Geom selection and
    body grouping recover the source distinction without per-geom overcounting.
    """
    groups = {}
    for index, name in enumerate(geom_names):
        if name.endswith("_foot_collision"):
            continue
        body = re.sub(r"\d*_collision$", "", name)
        groups.setdefault(body, []).append(index)
    result = torch.zeros(
        force_history.shape[0], device=force_history.device, dtype=torch.long
    )
    for indices in groups.values():
        body_force = force_history[:, indices].sum(dim=1)
        result += body_force.norm(dim=-1).max(dim=-1).values > 1.0
    return result


def restitution_damping(restitution):
    """Map a rebound ratio to ideal spring-damper damping (MuJoCo adaptation)."""
    loge = torch.log(restitution.clamp_min(1e-12))
    damping = -loge / torch.sqrt(torch.pi**2 + loge**2)
    return torch.where(restitution <= 0, torch.ones_like(damping), damping)


from mjlab.managers.event_manager import requires_model_fields


@requires_model_fields("geom_solref")
def randomize_restitution(env, env_ids, asset_cfg):
    robot = env.scene[asset_cfg.name]
    geoms = robot.indexing.geom_ids[asset_cfg.geom_ids]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    restitution = torch.empty((len(ids), len(geoms)), device=env.device).uniform_(
        0.0, 0.15
    )
    env.sim.model.geom_solref[ids[:, None], geoms[None, :], 1] = restitution_damping(
        restitution
    )
