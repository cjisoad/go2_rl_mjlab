"""Rear-leg upright stance/locomotion, source StandUp V2.1."""

from dataclasses import dataclass
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots import get_go2_robot_cfg
from src.tasks.go2.actuator import Go2HVActuatorCfg
from . import mdp
from .source_config import Go2_Handstand_Cfg_Yu as SourceCfg
from .actions import SourceDelayedJointPositionActionCfg


@dataclass
class StandupEnvCfg(ManagerBasedRlEnvCfg):
    enable_pushes: bool = True


def standup_env_cfg(play=False):
    robot = get_go2_robot_cfg()
    robot.init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=SourceCfg.init_state.default_joint_angles.copy(),
        joint_vel={".*": 0.0},
    )
    robot.articulation = EntityArticulationInfoCfg(
        actuators=(
            Go2HVActuatorCfg(target_names_expr=(".*",), stiffness=40.0, damping=1.0),
        ),
        soft_joint_pos_limit_factor=0.9,
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
    sensors = []

    def sensor(name, mode, pattern):
        sensors.append(
            ContactSensorCfg(
                name=name,
                primary=ContactMatch(mode=mode, pattern=pattern, entity="robot"),
                fields=("force", "found"),
                reduce="netforce",
                num_slots=1,
            )
        )

    for leg in ("FL", "FR", "RL", "RR"):
        sensor(f"foot_{leg}", "geom", f"{leg}_foot_collision")
        for part in ("thigh", "calf"):
            sensor(f"body_{leg}_{part}", "geom", f"{leg}_{part}.*_collision")
    sensor("base_contact", "body", "base_link")
    rewards = {
        name: RewardTermCfg(
            func=mdp.source_reward,
            weight=getattr(SourceCfg.rewards.scales, name),
            params={"name": name},
        )
        for name in sorted(vars(SourceCfg.rewards.scales))
        if not name.startswith("_") and getattr(SourceCfg.rewards.scales, name) != 0.0
    }
    cfg = StandupEnvCfg(
        scene=SceneCfg(
            num_envs=32 if play else 4096,
            extent=3.0,
            entities={"robot": robot},
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=tuple(sensors),
        ),
        actions={
            "joint_pos": SourceDelayedJointPositionActionCfg(
                entity_name="robot",
                actuator_names=tuple(mdp.JOINT_NAMES),
                preserve_order=True,
                scale=0.25,
                use_default_offset=True,
            )
        },
        commands={"base_velocity": mdp.SourceVelocityCommandCfg()},
        observations={
            "actor": ObservationGroupCfg(
                terms={
                    "source_policy": ObservationTermCfg(func=mdp.policy_observation)
                },
                concatenate_terms=True,
                enable_corruption=not play,
                history_length=1,
            ),
            "critic": ObservationGroupCfg(
                terms={
                    "source_critic": ObservationTermCfg(func=mdp.critic_observation)
                },
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        },
        events={
            "randomize_physics": EventTermCfg(
                func=mdp.randomize_source_physics, mode="startup"
            ),
            "reset_robot": EventTermCfg(func=mdp.reset_source, mode="reset"),
        },
        rewards=rewards,
        terminations={
            "base_contact": TerminationTermCfg(func=mdp.base_contact),
            "time_out": TerminationTermCfg(func=mdp.source_timeout, time_out=True),
        },
        curriculum={},
        sim=SimulationCfg(
            njmax=1500,
            nconmax=100,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
        viewer=ViewerConfig(entity_name="robot", body_name="base_link", distance=2.5),
    )
    cfg.enable_pushes = not play
    return cfg


def standup_play_env_cfg():
    return standup_env_cfg(play=True)
