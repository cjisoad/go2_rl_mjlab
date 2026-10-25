"""Independent task factories; base-task configuration remains unchanged."""

import copy

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch

from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg
from src.tasks.go2.dreamwaq.rl_cfg import dreamwaq_runner_cfg
from src.tasks.go2.standup.env_cfg import standup_env_cfg
from src.tasks.go2.standup.rl_cfg import standup_runner_cfg

from . import mdp
from .commands import FootPositionCommandCfg


def quad_pedipulation_env_cfg(play=False):
    cfg = dreamwaq_env_cfg(play)
    cfg.commands["foot_position"] = FootPositionCommandCfg()
    # The inherited foot sensor intentionally reports self-contact as well.
    # Landing and support rewards require actual terrain contact.
    ground = copy.deepcopy(
        next(
            sensor
            for sensor in cfg.scene.sensors
            if sensor.name == "feet_ground_contact"
        )
    )
    ground.name = "pedipulation_ground_contact"
    ground.secondary = ContactMatch(mode="body", pattern="terrain")
    ground.track_air_time = False
    cfg.scene.sensors = (*cfg.scene.sensors, ground)
    for group in ("actor", "obs_history"):
        cfg.observations[group].terms = {
            "proprioception": ObservationTermCfg(
                func=mdp.quad_observation, params={"noise": not play}
            )
        }
    for kind in ("joint_position", "air_time", "air_variance", "slide"):
        cfg.rewards[kind].func = mdp.quad_reward
    cfg.rewards.update(
        {
            "fr_tracking": RewardTermCfg(func=mdp.fr_tracking, weight=2.0),
            "fr_return": RewardTermCfg(func=mdp.fr_return, weight=1.0),
            "fr_ground_return": RewardTermCfg(func=mdp.fr_ground_return, weight=0.5),
            "support_contact": RewardTermCfg(func=mdp.support_contact, weight=0.5),
        }
    )
    return cfg


def biped_pedipulation_env_cfg(play=False):
    cfg = standup_env_cfg(play)
    cfg.commands["foot_position"] = FootPositionCommandCfg(
        biped=True, initial_delay=3.0
    )
    cfg.observations["actor"].terms = {
        "proprioception": ObservationTermCfg(func=mdp.biped_observation)
    }
    for name in (
        "default_pos",
        "default_pos_reward",
        "default_hip_pos",
        "symmetric_joints",
        "feet_height_symmetry",
        "handstand_feet_height_exp",
    ):
        cfg.rewards[name].func = mdp.biped_reward
    cfg.rewards.update(
        {
            "fr_tracking": RewardTermCfg(func=mdp.fr_tracking, weight=2.0),
            "fr_return": RewardTermCfg(func=mdp.fr_return, weight=1.0),
        }
    )
    return cfg


def quad_pedipulation_runner_cfg():
    cfg = dreamwaq_runner_cfg()
    cfg.experiment_name = "go2_quad_pedipulation"
    return cfg


def biped_pedipulation_runner_cfg():
    cfg = standup_runner_cfg()
    cfg.experiment_name = "go2_biped_pedipulation"
    # A FR-only command cannot be reflected into the same task's action space.
    cfg.algorithm.symmetry_cfg = None
    return cfg
