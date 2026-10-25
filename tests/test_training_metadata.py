"""Training metadata must remain serializable when RSL-RL binds runtime objects."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import yaml

from scripts import train
import src.tasks  # noqa: F401


class TrainingMetadataTests(unittest.TestCase):
    def test_standup_metadata_survives_runner_symmetry_environment_binding(self):
        cfg = train.TrainConfig.from_task("Unitree-Go2-StandUp")
        learned = []
        env = SimpleNamespace(spec=mujoco.MjSpec(), close=lambda: None)

        class Runner:
            def __init__(self, environment, agent, *args, **kwargs):
                # Native RSL-RL binds the environment into the symmetry dictionary.
                agent["algorithm"]["symmetry_cfg"]["_env"] = environment

            def add_git_repo_to_log(self, path):
                pass

            def learn(self, num_learning_iterations, **kwargs):
                learned.append(num_learning_iterations)

        with TemporaryDirectory() as directory:
            logdir = Path(directory)
            with (
                patch.dict(train.os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
                patch.object(train, "configure_torch_backends"),
                patch.object(train, "ManagerBasedRlEnv", return_value=env),
                patch.object(train, "RslRlVecEnvWrapper", return_value=env),
                patch.object(train, "load_runner_cls", return_value=Runner),
            ):
                train.run_train("Unitree-Go2-StandUp", cfg, logdir)
            metadata = yaml.full_load((logdir / "params/agent.yaml").read_text())
            symmetry = metadata["algorithm"]["symmetry_cfg"]
            self.assertTrue(symmetry["use_mirror_loss"])
            self.assertNotIn("_env", symmetry)
            self.assertEqual(metadata["max_iterations"], 6000)
            self.assertTrue((logdir / "params/env.yaml").is_file())
        self.assertEqual(learned, [6000])
