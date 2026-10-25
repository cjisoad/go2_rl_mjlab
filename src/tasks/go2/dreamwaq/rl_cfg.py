from dataclasses import dataclass
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlModelCfg


@dataclass
class DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "src.tasks.go2.dreamwaq.rl:DreamWaQPPO"
    beta: float = 1.0


def dreamwaq_runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="src.tasks.go2.dreamwaq.rl:DreamWaQActor",
            hidden_dims=(512, 256, 128),
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=False),
        algorithm=DreamWaQAlgorithmCfg(entropy_coef=0.01),
        obs_groups={"actor": ("actor",), "critic": ("actor", "privileged")},
        experiment_name="go2_dreamwaq",
        logger="tensorboard",
        max_iterations=5000,
        save_interval=50,
    )
