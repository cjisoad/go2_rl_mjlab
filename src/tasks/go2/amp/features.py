"""Source AMP state contract: FL, FR, RL, RR; q, qdot, body v, body omega."""

import torch
from mjlab.managers.event_manager import requires_model_fields

JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


def amp_features(env):
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
    data = robot.data
    return torch.cat(
        (
            data.joint_pos[:, ids],
            data.joint_vel[:, ids],
            data.root_link_lin_vel_b,
            data.root_link_ang_vel_b,
        ),
        dim=-1,
    ).clamp(-100, 100)


def capture_terminal(env):
    # Reward manager runs before automatic reset; preserve that episode's state.
    env.extras["amp_terminal"] = amp_features(env).clone()
    return torch.zeros(env.num_envs, device=env.device)


def terminal_next_observation(next_obs, dones, extras):
    result = next_obs.clone()
    if dones.any():
        result[dones.bool()] = extras["amp_terminal"][dones.bool()].to(result.device)
    return result


def track_xy(env, command_name="twist", std=0.5):
    command = env.command_manager.get_command(command_name)
    error = (
        (command[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2])
        .square()
        .sum(-1)
    )
    return torch.exp(-error / std**2)


def track_yaw(env, command_name="twist", std=0.5):
    command = env.command_manager.get_command(command_name)
    error = (command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]).square()
    return torch.exp(-error / std**2)


def joint_effort(env):
    return env.scene["robot"].data.actuator_force


def command_curriculum(env, env_ids):
    ranges = env.command_manager.get_term("twist").cfg.ranges
    term = "track_linear_velocity"
    reward = (
        env.reward_manager._episode_sums[term][env_ids].mean()
        / env.max_episode_length_s
    )
    weight = env.reward_manager.get_term_cfg(term).weight
    if env.common_step_counter % env.max_episode_length == 0 and reward > weight * 0.8:
        delta = torch.tensor([-0.1, 0.1], device=env.device)
        ranges.lin_vel_x = (
            (torch.tensor(ranges.lin_vel_x, device=env.device) + delta)
            .clamp(-1.0, 1.0)
            .tolist()
        )
        ranges.lin_vel_y = (
            (torch.tensor(ranges.lin_vel_y, device=env.device) + delta)
            .clamp(-0.4, 0.4)
            .tolist()
        )
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def restitution_damping(restitution):
    """Ideal spring-damper rebound ratio adaptation for MuJoCo contacts."""
    loge = torch.log(restitution.clamp_min(1e-12))
    ratio = -loge / torch.sqrt(torch.pi**2 + loge**2)
    return torch.where(restitution <= 0, torch.ones_like(ratio), ratio)


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
