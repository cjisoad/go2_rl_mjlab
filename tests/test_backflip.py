"""Regression checks for archived V1.6 phase, pitch and contact semantics."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def load_mdp():
    path = Path(__file__).resolve().parents[1] / "src/tasks/go2/backflip/mdp.py"
    assert path.exists(), "Backflip V1.6 implementation is missing"
    spec = importlib.util.spec_from_file_location("backflip_mdp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackflipTests(unittest.TestCase):
    def test_orientation_tracks_negative_full_turn_over_progress_window(self):
        mdp = load_mdp()
        steps = torch.tensor([0, 25, 33, 41, 49, 58, 80])
        time = steps * 0.02
        angle = -2.0 * torch.pi * ((time - 0.5) / 0.65).clamp(0.0, 1.0)
        gravity = torch.stack([angle.sin(), torch.zeros_like(angle), -angle.cos()], -1)
        data = SimpleNamespace(
            gravity_vec_w=torch.tensor([0.0, 0.0, -1.0]),
            projected_gravity_b=gravity,
        )
        env = SimpleNamespace(
            episode_length_buf=steps,
            step_dt=0.02,
            num_envs=len(steps),
            device="cpu",
            scene={"robot": SimpleNamespace(data=data)},
        )
        # A correct backflip trajectory must not receive an orientation penalty.
        torch.testing.assert_close(
            mdp.orientation_control(env), torch.zeros(len(steps))
        )
        data.projected_gravity_b = gravity.clone()
        data.projected_gravity_b[:, 0] *= -1.0
        self.assertGreater(float(mdp.orientation_control(env)[2:5].sum()), 4.0)

    def test_source_delayed_target_is_retained_across_episode_reset(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from src.tasks.go2.backflip.actions import DelayedJointPositionAction

        action = object.__new__(DelayedJointPositionAction)
        action._raw_actions = torch.ones(2, 12)
        action._processed_actions = torch.full((2, 12), 0.7)
        action._previous_processed_actions = torch.full((2, 12), -0.2)
        action._offset = torch.zeros(2, 12)
        action.reset(torch.tensor([0]))
        torch.testing.assert_close(action._processed_actions[0], torch.full((12,), 0.7))
        torch.testing.assert_close(
            action._previous_processed_actions[0], torch.full((12,), 0.7)
        )
        self.assertEqual(float(action._raw_actions[0].sum()), 0.0)

    def test_nonfoot_force_is_summed_per_body_before_threshold(self):
        mdp = load_mdp()
        forces = torch.zeros(2, 2, 3, 3)
        forces[0, :, 0, 2] = 0.6
        forces[1, 0, 0, 2] = 2.0
        forces[1, 1, 0, 2] = -2.0
        env = SimpleNamespace(
            scene={"calf": SimpleNamespace(data=SimpleNamespace(force_history=forces))}
        )
        torch.testing.assert_close(
            mdp._nonfoot_count(env, ("calf",), 1.0), torch.tensor([1, 0])
        )

    def test_phase_is_six_source_sine_cosine_terms(self):
        mdp = load_mdp()
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([0, 50, 100]), step_dt=0.02
        )
        phase = mdp.backflip_phase(env)
        angle = torch.tensor([0.0, torch.pi / 2, torch.pi])
        expected = torch.stack(
            [
                angle.sin(),
                angle.cos(),
                (angle / 2).sin(),
                (angle / 2).cos(),
                (angle / 4).sin(),
                (angle / 4).cos(),
            ],
            -1,
        )
        torch.testing.assert_close(phase, expected)

    def test_pitch_progress_accumulates_and_resets_per_environment(self):
        mdp = load_mdp()
        data = SimpleNamespace(
            root_link_ang_vel_b=torch.tensor([[0.0, -5.0, 0.0], [0.0, -5.0, 0.0]])
        )
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([30, 0]),
            step_dt=0.02,
            num_envs=2,
            device="cpu",
            scene={"robot": SimpleNamespace(data=data)},
        )
        first = mdp.pitch_progress(env)
        torch.testing.assert_close(
            env._backflip_pitch_rotation, torch.tensor([0.1, 0.0])
        )
        env.episode_length_buf = torch.tensor([31, 30])
        mdp.pitch_progress(env)
        torch.testing.assert_close(
            env._backflip_pitch_rotation, torch.tensor([0.2, 0.1])
        )
        self.assertTrue(torch.isfinite(first).all())
        env.episode_length_buf = torch.tensor([0, 31])
        mdp.pitch_progress(env)
        torch.testing.assert_close(
            env._backflip_pitch_rotation, torch.tensor([0.0, 0.2])
        )

    def test_front_contact_reward_uses_history_and_excludes_rear_contacts(self):
        mdp = load_mdp()
        front = torch.zeros(2, 2, 3, 3)
        rear = torch.zeros_like(front)
        front[:, 0, 2, 2] = 2.0
        rear[1, 1, 1, 2] = 2.0
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([60, 60]),
            step_dt=0.02,
            scene={
                "front_contact": SimpleNamespace(
                    data=SimpleNamespace(force_history=front)
                ),
                "rear_contact": SimpleNamespace(
                    data=SimpleNamespace(force_history=rear)
                ),
            },
        )
        torch.testing.assert_close(
            mdp.front_feet_first_contact(env, "front_contact", "rear_contact"),
            torch.tensor([1.0, 0.0]),
        )

    def test_nominal_stance_width_has_zero_lateral_error(self):
        mdp = load_mdp()
        # FL/FR/RL/RR are on the two sides of a 0.30 m support width.
        data = SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            site_pos_w=torch.tensor(
                [
                    [
                        [0.0, 0.15, 0.0],
                        [0.0, -0.15, 0.0],
                        [0.0, 0.15, 0.0],
                        [0.0, -0.15, 0.0],
                    ]
                ]
            ),
        )
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            episode_length_buf=torch.tensor([1]),
            step_dt=0.02,
            scene={"robot": SimpleNamespace(data=data)},
        )
        cfg = SimpleNamespace(name="robot", site_ids=[0, 1, 2, 3])
        torch.testing.assert_close(mdp.feet_distance(env, cfg), torch.tensor([0.0]))


if __name__ == "__main__":
    unittest.main()
