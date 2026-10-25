"""Numerical checks for the migrated Unitree Go2 actuator envelope."""

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class Go2TorqueEnvelopeTests(unittest.TestCase):
    def test_motoring_and_braking_limits_and_no_load_speed(self):
        path = ROOT / "src/tasks/go2/actuator.py"
        self.assertTrue(path.is_file(), "Native Go2HV actuator has not been migrated")
        spec = importlib.util.spec_from_file_location("go2_actuator", path)
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        effort = torch.tensor([100.0, -100.0, 100.0, -100.0, 100.0, -100.0])
        velocity = torch.tensor([0.0, 0.0, 21.75, 21.75, 30.0, -31.0])
        actual = module.clip_go2_effort(effort, velocity)
        torch.testing.assert_close(
            actual, torch.tensor([23.4, -23.4, 10.1, -11.7, 0.0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
