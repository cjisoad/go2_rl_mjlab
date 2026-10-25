"""Policy metadata in action order, including explicit Python PD actuators."""

import torch


def get_policy_metadata(env, run_path: str) -> dict:
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    gains = {
        name: (actuator.cfg.stiffness, actuator.cfg.damping)
        for actuator in robot.actuators
        for name in actuator.target_names
    }
    names = action.target_names
    scale = action.scale
    if isinstance(scale, torch.Tensor):
        scale = scale[0].detach().cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": list(names),
        "joint_stiffness": [gains[name][0] for name in names],
        "joint_damping": [gains[name][1] for name in names],
        "default_joint_pos": robot.data.default_joint_pos[0, action.target_ids]
        .detach()
        .cpu()
        .tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": scale,
    }
