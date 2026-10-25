"""Native RSL-RL 5 PPO configuration with source mirror loss."""

from dataclasses import dataclass, field
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class StandupPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict = field(
        default_factory=lambda: {
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "data_augmentation_func": "src.tasks.go2.standup.symmetry:source_symmetry",
            "mirror_loss_coeff": 1.0,
        }
    )


def standup_runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False
        ),
        algorithm=StandupPpoAlgorithmCfg(entropy_coef=0.01),
        experiment_name="go2_standup",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=6000,
        clip_actions=100.0,
    )
