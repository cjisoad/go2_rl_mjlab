"""Go2 backflip restored from the last archived V1.6 run (2026-07-04)."""

from copy import deepcopy

from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.go2.actuator import Go2HVActuatorCfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg

from . import mdp
from .actions import DelayedJointPositionActionCfg
from .evaluation import backflip_metric

JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
FRONT_HIPS = SceneEntityCfg(
    "robot", joint_names=("FR_hip_joint", "FL_hip_joint"), preserve_order=True
)
REAR_JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES[6:], preserve_order=True)
FEET = SceneEntityCfg("robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True)
FRONT_FEET = SceneEntityCfg("robot", site_names=("FL", "FR"), preserve_order=True)
NONFOOT_SENSORS = ("base_contact",) + tuple(
    f"{leg}_{part}_contact"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("thigh", "calf")
)


def contact_sensor(name, geoms):
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=geoms, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
        num_slots=1,
    )


def make_env_cfg(play=False):
    cfg = unitree_go2_flat_env_cfg()
    robot = get_go2_robot_cfg()
    robot.init_state = deepcopy(robot.init_state)
    robot.init_state.pos = (0.0, 0.0, 0.36)
    robot.init_state.joint_pos = {
        name: (
            0.0
            if "hip" in name
            else -1.5
            if "calf" in name
            else 0.8
            if name.startswith(("FR", "FL"))
            else 1.0
        )
        for name in JOINT_NAMES
    }
    robot.collisions = deepcopy(robot.collisions)
    for collision in robot.collisions:
        collision.conaffinity = 1
    robot.articulation = deepcopy(robot.articulation)
    robot.articulation.actuators = (
        Go2HVActuatorCfg(
            target_names_expr=(".*",), stiffness=70.0, damping=3.0, frictionloss=0.01
        ),
    )
    cfg.scene.entities = {"robot": robot}
    cfg.scene.num_envs = 32 if play else 4096
    cfg.scene.extent = 2.5
    cfg.scene.sensors = (
        contact_sensor(
            "feet_contact",
            (
                "FL_foot_collision",
                "FR_foot_collision",
                "RL_foot_collision",
                "RR_foot_collision",
            ),
        ),
        contact_sensor("front_contact", ("FL_foot_collision", "FR_foot_collision")),
        contact_sensor("rear_contact", ("RL_foot_collision", "RR_foot_collision")),
        contact_sensor("base_contact", ("base.*_collision",)),
    ) + tuple(
        contact_sensor(f"{leg}_{part}_contact", (f"{leg}_{part}.*_collision",))
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf")
    )
    cfg.actions = {
        "joint_pos": DelayedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=JOINT_NAMES,
            preserve_order=True,
            scale=0.5,
            use_default_offset=True,
        )
    }

    def obs(func, scale=1.0, noise=None, params=None):
        return ObservationTermCfg(
            func=func,
            scale=scale,
            noise=noise,
            clip=(-100.0, 100.0),
            params=deepcopy(params or {}),
        )

    actor = {
        "base_ang_vel": obs(
            env_mdp.base_ang_vel, 0.25, UniformNoiseCfg(n_min=-0.1, n_max=0.1)
        ),
        "projected_gravity": obs(
            env_mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)
        ),
        "joint_pos_rel": obs(
            env_mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": JOINTS},
        ),
        "joint_vel_rel": obs(
            env_mdp.joint_vel_rel,
            0.05,
            UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            {"asset_cfg": JOINTS},
        ),
        "current_action": obs(mdp.current_action),
        "last_action": obs(mdp.previous_action),
        "backflip_phase": obs(mdp.backflip_phase),
    }
    critic = {
        "base_height": obs(mdp.base_height),
        "base_lin_vel": obs(env_mdp.base_lin_vel, 2.0),
        **deepcopy(actor),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor, enable_corruption=not play, concatenate_terms=True
        ),
        "critic": ObservationGroupCfg(
            terms=critic, enable_corruption=False, concatenate_terms=True
        ),
    }
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.metrics = {
        f"backflip_{stage}_rate": MetricsTermCfg(
            func=backflip_metric,
            params={"stage": stage, "nonfoot_sensors": NONFOOT_SENSORS},
        )
        for stage in ("takeoff", "full_flip", "success")
    }
    cfg.events = {
        "physics_material": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "ranges": (0.2, 1.5),
                "operation": "abs",
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
            },
        ),
        "add_base_mass": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "ranges": (-1.0, 3.0),
                "operation": "add",
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
                "operation": "add",
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        ),
        "joint_default_pos": EventTermCfg(
            func=mdp.randomize_default_joint_pos, mode="startup", params={}
        ),
        "reset_base": EventTermCfg(
            func=env_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_joints": EventTermCfg(
            func=env_mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
        ),
    }

    def reward(name, weight, **params):
        return RewardTermCfg(
            func=getattr(mdp, name), weight=weight, params=deepcopy(params)
        )

    cfg.rewards = {
        "ang_vel_y": reward("ang_vel_y", 5.0),
        "ang_vel_z": reward("ang_vel_z", -1.5),
        "lin_vel_z": reward("lin_vel_z", 20.0),
        "pitch_progress": reward("pitch_progress", 12.0),
        "orientation_control": reward("orientation_control", -2.0),
        "feet_height_before_backflip": reward(
            "feet_height_before_backflip", -20.0, asset_cfg=FEET
        ),
        "crouch_base_height": reward("crouch_base_height", -6.0),
        "stance_foot_slip": reward(
            "stance_foot_slip", -0.15, sensor_cfg="feet_contact", asset_cfg=FEET,
            landing_start=mdp.LANDING_START,
        ),
        "launch_lin_vel_z": reward("launch_lin_vel_z", 8.0),
        "height_control": reward("height_control", -10.0, target_height=0.3),
        "air_tuck_pose": reward(
            "air_tuck_pose",
            -0.8,
            asset_cfg=JOINTS,
            hold_end=mdp.LANDING_PREP_START,
            tuck_end=mdp.LANDING_START,
        ),
        "front_hip_landing_pose": reward(
            "front_hip_landing_pose",
            -4.0,
            asset_cfg=FRONT_HIPS,
            landing_start=mdp.LANDING_PREP_START,
            landing_full=mdp.LANDING_START,
        ),
        "rear_leg_landing_pose": reward(
            "rear_leg_landing_pose",
            -1.0,
            asset_cfg=REAR_JOINTS,
            hip_error_weight=4.0,
            landing_start=mdp.LANDING_PREP_START,
            landing_full=mdp.LANDING_START,
        ),
        "actions_symmetry": reward("actions_symmetry", -0.1),
        "gravity_y": reward("gravity_y", -4.0),
        "feet_distance": reward(
            "feet_distance", -1.0, asset_cfg=FEET, stance_width=0.30
        ),
        "prelaunch_nonfoot_contacts": reward(
            "prelaunch_nonfoot_contacts", -4.0, sensor_cfg=NONFOOT_SENSORS
        ),
        "front_leg_reach_landing": reward(
            "front_leg_reach_landing", -0.7, asset_cfg=JOINTS, front_feet_cfg=FRONT_FEET
        ),
        "front_feet_first_contact": reward(
            "front_feet_first_contact",
            1.5,
            front_sensor_cfg="front_contact",
            rear_sensor_cfg="rear_contact",
        ),
        "early_rear_feet_contact": reward(
            "early_rear_feet_contact", -1.0, rear_sensor_cfg="rear_contact"
        ),
        "landing_stability": reward("landing_stability", -1.2),
        "undesired_contacts": reward(
            "undesired_contacts", -2.0, sensor_cfg=NONFOOT_SENSORS
        ),
        # Effective coefficient: -0.001 before landing, -0.005 afterward.
        "action_rate": reward(
            "action_rate", -0.001, landing_start=mdp.LANDING_START, landing_scale=5.0
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True)
    }
    # Update explicit PD torque at 400 Hz; keep policy control at 50 Hz.
    cfg.decimation = 8
    cfg.sim.mujoco.timestep = 0.0025
    cfg.sim.njmax = 1500
    cfg.sim.nconmax = 64
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.episode_length_s = 2.0
    return cfg


def make_rl_cfg():
    cfg = unitree_go2_ppo_runner_cfg()
    cfg.actor.obs_normalization = False
    cfg.critic.obs_normalization = False
    cfg.experiment_name = "go2_backflip_v16"
    cfg.max_iterations = 1000
    cfg.logger = "tensorboard"
    return cfg
