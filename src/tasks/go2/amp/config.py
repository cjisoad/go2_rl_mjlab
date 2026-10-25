"""Go2 AMP configuration using native mjlab terms and RSL-RL models."""

from dataclasses import dataclass
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    CurriculumTermCfg,
)
from mjlab.envs import mdp
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
from .features import (
    amp_features,
    capture_terminal,
    track_xy,
    track_yaw,
    joint_effort,
    command_curriculum,
    randomize_restitution,
)
from src.tasks.go2.actuator import Go2HVActuatorCfg
from copy import deepcopy
from mjlab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.envs.mdp import dr
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import illegal_contact
from .actions import AMPJointPositionActionCfg
from mjlab.utils.spec_config import CollisionCfg


@dataclass
class AMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "src.tasks.go2.amp.algorithm:AMPPPO"
    style_weight: float = 0.4
    amp_replay_buffer_size: int = 100000


def amp_runner_cfg():
    cfg = unitree_go2_ppo_runner_cfg()
    cfg.algorithm = AMPAlgorithmCfg(entropy_coef=0.01)
    cfg.actor.obs_normalization = False
    cfg.critic.obs_normalization = False
    cfg.logger = "tensorboard"
    cfg.experiment_name = "go2_amp"
    cfg.max_iterations = 10000
    return cfg


def amp_env_cfg(play=False):
    cfg = unitree_go2_flat_env_cfg(play=False)
    cfg.scene.num_envs = 2048 if not play else 32
    cfg.episode_length_s = 20.0
    cfg.actions["joint_pos"] = AMPJointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=0.25, use_default_offset=True
    )
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={"state": ObservationTermCfg(func=amp_features)},
        concatenate_terms=True,
        enable_corruption=False,
    )
    # Reproduce the source policy (45) and privileged critic (60) states.
    original = cfg.observations["actor"].terms
    actor = {
        name: deepcopy(original[name])
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "command",
            "joint_pos",
            "joint_vel",
            "actions",
        )
    }
    actor["base_ang_vel"].func = mdp.base_ang_vel
    actor["base_ang_vel"].params = {}
    actor["base_ang_vel"].scale = 0.2
    actor["joint_vel"].scale = 0.05
    for term in actor.values():
        term.clip = (-100, 100)
    critic = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, clip=(-100, 100))
    }
    critic.update({k: deepcopy(v) for k, v in actor.items() if k != "actions"})
    critic["joint_effort"] = ObservationTermCfg(
        func=joint_effort, scale=0.01, clip=(-100, 100)
    )
    critic["actions"] = deepcopy(actor["actions"])
    for term in critic.values():
        term.noise = None
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor, concatenate_terms=True, enable_corruption=True
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic, concatenate_terms=True, enable_corruption=False
    )
    robot = deepcopy(cfg.scene.entities["robot"])
    robot.articulation.actuators = (
        Go2HVActuatorCfg(
            target_names_expr=(".*",),
            stiffness=25.0,
            damping=0.5,
            armature=0.0,
            frictionloss=0.01,
        ),
    )
    robot.collisions = (
        CollisionCfg(
            geom_names_expr=(".*_collision.*",),
            contype=1,
            conaffinity=1,
            condim=3,
            friction=(1.0,),
        ),
    )
    robot.init_state.pos = (0.0, 0.0, 0.4)
    robot.init_state.joint_pos = {
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    }
    cfg.scene.entities["robot"] = robot
    cfg.commands["twist"].ranges.lin_vel_x = (-1.0, 1.0) if play else (-0.1, 0.1)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.4, 0.4) if play else (-0.1, 0.1)
    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands["twist"].heading_command = False
    cfg.commands["twist"].ranges.heading = None
    cfg.commands["twist"].init_velocity_prob = 0.0
    cfg.commands["twist"].resampling_time_range = (10.0, 10.0)
    cfg.commands["twist"].rel_standing_envs = 0.1
    cfg.curriculum = {"lin_vel_cmd_levels": CurriculumTermCfg(func=command_curriculum)}
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)
    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.2)
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=(".*_collision.*",)
    )
    cfg.events["foot_friction"].params["shared_random"] = False
    cfg.events["restitution"] = EventTermCfg(
        func=randomize_restitution,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision.*",))},
    )
    cfg.events["base_mass"] = EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "ranges": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_contact = ContactSensorCfg(
        name="amp_base_contact",
        primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=3,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (base_contact,)
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "base_contact": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": "amp_base_contact", "force_threshold": 1.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.8}
        ),
    }
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (5.0, 10.0)
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }
    cfg.events.pop("randomize_terrain", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.rewards = {
        name: cfg.rewards[name]
        for name in (
            "track_linear_velocity",
            "track_angular_velocity",
            "joint_acc_l2",
            "action_rate_l2",
        )
    }
    cfg.rewards["track_linear_velocity"].func = track_xy
    cfg.rewards["track_angular_velocity"].func = track_yaw
    cfg.rewards["track_linear_velocity"].weight = 1.5 / dt
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 0.5 / dt
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5
    cfg.rewards["joint_vel"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4 / dt)
    cfg.rewards["joint_acc_l2"].weight = -2.5e-8 / dt
    cfg.rewards["action_rate_l2"].weight = -0.015 / dt
    cfg.rewards["amp_terminal_capture"] = RewardTermCfg(
        func=capture_terminal, weight=1.0
    )
    return cfg
