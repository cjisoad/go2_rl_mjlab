"""Deployment metadata follows policy action order and explicit PD gains."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class ExportTests(unittest.TestCase):
    def test_metadata_uses_action_order_and_nominal_python_actuator_gains(self):
        path = Path(__file__).resolve().parents[1] / "src/utils/export.py"
        self.assertTrue(path.exists(), "Policy-order metadata helper is missing")
        spec = importlib.util.spec_from_file_location("export_metadata", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        robot = SimpleNamespace(
            data=SimpleNamespace(default_joint_pos=torch.tensor([[0.8, 0.9]])),
            actuators=[
                SimpleNamespace(
                    target_names=["FL", "FR"],
                    cfg=SimpleNamespace(stiffness=70.0, damping=3.0),
                )
            ],
        )
        action = SimpleNamespace(
            target_names=["FR", "FL"], target_ids=torch.tensor([1, 0]), scale=0.5
        )
        env = SimpleNamespace(
            scene={"robot": robot},
            action_manager=SimpleNamespace(get_term=lambda name: action),
            command_manager=SimpleNamespace(active_terms=[]),
            observation_manager=SimpleNamespace(active_terms={"actor": ["state"]}),
        )
        metadata = module.get_policy_metadata(env, "local")
        self.assertEqual(metadata["joint_names"], ["FR", "FL"])
        self.assertEqual(metadata["joint_stiffness"], [70.0, 70.0])
        self.assertEqual(metadata["joint_damping"], [3.0, 3.0])
        torch.testing.assert_close(
            torch.tensor(metadata["default_joint_pos"]), torch.tensor([0.9, 0.8])
        )


if __name__ == "__main__":
    unittest.main()
