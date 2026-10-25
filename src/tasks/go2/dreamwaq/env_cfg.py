import copy
from mjlab.envs import mdp as native
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.terrains import TerrainGeneratorCfg
import mjlab.terrains as terrain
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
from . import mdp


def dreamwaq_env_cfg(play=False):
    cfg = unitree_go2_rough_env_cfg(play=play)
    from .actions import DreamWaQJointPositionActionCfg

    cfg.actions = {
        "joint_pos": DreamWaQJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }
    cfg.scene.num_envs = 32 if play else 4096
    frame = ObservationTermCfg(func=mdp.proprioception, params={"noise": True})
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"proprioception": frame},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "obs_history": ObservationGroupCfg(
            terms={"proprioception": copy.deepcopy(frame)},
            concatenate_terms=True,
            enable_corruption=False,
            history_length=5,
            flatten_history_dim=True,
        ),
        "privileged": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(
                    func=native.base_lin_vel, clip=(-100, 100)
                ),
                "joint_effort": ObservationTermCfg(
                    func=mdp.effort, scale=0.01, clip=(-100, 100)
                ),
                "height_scan": ObservationTermCfg(
                    func=native.height_scan,
                    params={"sensor_name": "terrain_scan", "offset": 0.5},
                    clip=(-1, 5),
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    cfg.rewards = {}
    weights = {
        "track_linear": 1.5,
        "track_angular": 0.75,
        "linear_z": -2.0,
        "angular_xy": -0.05,
        "energy": -2e-5,
        "joint_position": -0.7,
        "air_time": 0.1,
        "air_variance": -1.0,
        "slide": -0.1,
        "undesired": -1.0,
    }
    for kind, weight in weights.items():
        cfg.rewards[kind] = RewardTermCfg(
            func=mdp.scalar_reward, weight=weight, params={"kind": kind}
        )
    for name, func, weight in [
        ("joint_vel", native.joint_vel_l2, -0.001),
        ("joint_acc", native.joint_acc_l2, -2.5e-7),
        ("joint_torques", native.joint_torques_l2, -2e-4),
        ("action_rate", native.action_rate_l2, -0.1),
        ("dof_pos_limits", native.joint_pos_limits, -10.0),
        ("flat_orientation_l2", native.flat_orientation_l2, -2.5),
    ]:
        cfg.rewards[name] = RewardTermCfg(func=func, weight=weight)
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors if s.name != "nonfoot_ground_touch"
    ) + (
        ContactSensorCfg(
            name="base_contact",
            primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=3,
        ),
        ContactSensorCfg(
            name="undesired_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=".*(hip|thigh|calf).*_collision",
                entity="robot",
                exclude=(".*_foot_collision",),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=3,
        ),
    )
    for sensor in cfg.scene.sensors:
        if sensor.name == "feet_ground_contact":
            sensor.history_length = 3
    from mjlab.tasks.velocity import mdp as velocity

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=native.time_out, time_out=True),
        "base_contact": TerminationTermCfg(
            func=velocity.illegal_contact,
            params={"sensor_name": "base_contact", "force_threshold": 1.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=native.bad_orientation, params={"limit_angle": 0.8}
        ),
    }
    cmd = cfg.commands["twist"]
    cmd.resampling_time_range = (10.0, 10.0)
    cmd.rel_standing_envs = 0.1
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.ranges.lin_vel_x = (-1.0, 1.0) if play else (-0.1, 0.1)
    cmd.ranges.lin_vel_y = (-0.4, 0.4) if play else (-0.1, 0.1)
    cmd.ranges.ang_vel_z = (-1.0, 1.0)
    # One of five rows covers the same initial difficulty interval (0..0.2)
    # as two of ten rows in the source configuration.
    cfg.scene.terrain.max_init_terrain_level = 1 if play else 0
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        # MuJoCo-Warp 3.5 SAP scans candidates across all worlds with int32.
        # The source 10x20 primitive grid overflows that scan at 4096 envs.
        # Keep every terrain type and its height range, using fewer patches.
        num_rows=2 if play else 5,
        num_cols=7 if play else 10,
        curriculum=True,
        sub_terrains={
            "flat": terrain.BoxFlatTerrainCfg(proportion=0.1),
            "rough": terrain.HfRandomUniformTerrainCfg(
                proportion=0.1,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.25,
            ),
            "slope": terrain.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "slope_inv": terrain.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
            ),
            "boxes": terrain.BoxRandomGridTerrainCfg(
                proportion=0.2,
                grid_width=0.45,
                grid_height_range=(0.05, 0.2),
                platform_width=2.0,
            ),
            "stairs": terrain.BoxPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
            ),
            "stairs_inv": terrain.BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
            ),
        },
    )
    from mjlab.managers import EventTermCfg, CurriculumTermCfg, SceneEntityCfg
    from mjlab.envs.mdp import dr
    from src.tasks.go2.actuator import Go2HVActuatorCfg

    robot = copy.deepcopy(cfg.scene.entities["robot"])
    # PhysX source enables self-collisions. Keep robot masks mutually compatible.
    for collision in robot.collisions:
        collision.contype = 1
        collision.conaffinity = 1
    # Source unfiltered net contact force includes self-contact, not only terrain.
    for sensor in cfg.scene.sensors:
        if sensor.name in ("feet_ground_contact", "base_contact", "undesired_contact"):
            sensor.secondary = None
    robot.init_state.pos = (0.0, 0.0, 0.4)
    robot.init_state.joint_pos = {
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    }
    robot.articulation.actuators = (
        Go2HVActuatorCfg(
            target_names_expr=(".*",), stiffness=25.0, damping=0.5, frictionloss=0.01
        ),
    )
    cfg.scene.entities["robot"] = robot
    cfg.events = {
        "physics_material": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "ranges": (0.3, 1.2),
                "operation": "abs",
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision.*",)),
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
        "reset_base": cfg.events["reset_base"],
        "reset_joints": EventTermCfg(
            func=native.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (0.0, 0.0), "velocity_range": (-1.0, 1.0)},
        ),
        "push_robot": EventTermCfg(
            func=native.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
    }
    cfg.events["restitution"] = EventTermCfg(
        func=mdp.randomize_restitution,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision.*",))},
    )
    cfg.events["reset_base"].params["velocity_range"] = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=velocity.terrain_levels_vel, params={"command_name": "twist"}
        ),
        "lin_vel_cmd_levels": CurriculumTermCfg(func=mdp.command_curriculum),
    }
    # The source play task retains sensor noise and randomization.
    cfg.episode_length_s = 20.0
    return cfg
