"""Landing rewards recover stance and constrain hip splay with a smooth onset."""

import unittest
from types import SimpleNamespace

import torch

from src.tasks.go2.backflip import mdp
from src.tasks.go2.backflip.env_cfg import make_env_cfg


class BackflipLandingTests(unittest.TestCase):
    def environment(self, joints):
        return SimpleNamespace(
            episode_length_buf=torch.tensor([180, 200, 215, 230, 260]),
            step_dt=0.005,
            device="cpu",
            scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joints))},
        )

    def test_rear_pose_smoothly_takes_over_and_ignores_front_joints(self):
        joints = torch.tensor([9.0] * 6 + [0.5, 1.0, -1.5, 0.0, 1.0, -1.5]).repeat(5, 1)
        env = self.environment(joints)
        rear = SimpleNamespace(name="robot", joint_ids=list(range(6, 12)))
        # A 0.5 rad hip error has raw cost 4 * 0.5**2 = 1.
        torch.testing.assert_close(
            mdp.rear_leg_landing_pose(env, rear),
            torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
        )
        joints[:, 6] = 0.0
        torch.testing.assert_close(mdp.rear_leg_landing_pose(env, rear), torch.zeros(5))

    def test_hip_abduction_is_penalized_more_than_thigh_error(self):
        joints = torch.tensor([0.0, 1.0, -1.5, 0.0, 1.0, -1.5]).repeat(2, 1)
        joints[0, 0] += 0.5
        joints[1, 1] += 0.5
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([70, 70]),
            step_dt=0.02,
            device="cpu",
            scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joints))},
        )
        rear = SimpleNamespace(name="robot", joint_ids=list(range(6)))
        torch.testing.assert_close(
            mdp.rear_leg_landing_pose(env, rear), torch.tensor([1.0, 0.25])
        )

    def test_tuck_fades_out_with_complementary_landing_blend(self):
        joints = torch.tensor(
            [0.0, 1.1, -1.8, 0.0, 1.1, -1.8, 0.0, 1.2, -1.8, 0.0, 1.2, -1.8]
        ).repeat(5, 1)
        joints[:, 0] += 1.0
        env = self.environment(joints)
        all_joints = SimpleNamespace(name="robot", joint_ids=list(range(12)))
        torch.testing.assert_close(
            mdp.air_tuck_pose(env, asset_cfg=all_joints),
            torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            mdp._landing_blend(torch.tensor([1.0, 1.075, 1.15])),
            torch.tensor([0.0, 0.5, 1.0]),
        )
        # Smoothstep has near-zero slopes at both ends of the handover.
        blend = mdp._landing_blend(torch.tensor([1.0, 1.0001, 1.1499, 1.15]))
        self.assertLess(float(blend[1] - blend[0]), 0.00001)
        self.assertLess(float(blend[3] - blend[2]), 0.00001)

    def test_stance_width_gate_and_centered_feet_cost(self):
        data = SimpleNamespace(
            root_link_pos_w=torch.zeros(4, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            site_pos_w=torch.zeros(4, 4, 3),
        )
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([0, 25, 60, 61]),
            step_dt=0.02,
            scene={"robot": SimpleNamespace(data=data)},
        )
        feet = SimpleNamespace(name="robot", site_ids=[0, 1, 2, 3])
        torch.testing.assert_close(
            mdp.feet_distance(env, feet), torch.tensor([0.09, 0.0, 0.0, 0.09])
        )

    def test_reward_configuration_only_targets_rear_joints(self):
        cfg = make_env_cfg()
        term = cfg.rewards["rear_leg_landing_pose"]
        self.assertEqual(term.weight, -1.0)
        self.assertEqual(
            term.params["asset_cfg"].joint_names,
            (
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ),
        )
        self.assertEqual(cfg.rewards["feet_distance"].params["stance_width"], 0.30)

    def test_front_hip_configuration_targets_only_named_front_hips(self):
        for play in (False, True):
            cfg = make_env_cfg(play=play)
            self.assertIn("front_hip_landing_pose", cfg.rewards)
            term = cfg.rewards["front_hip_landing_pose"]
            self.assertEqual(term.weight, -4.0)
            self.assertEqual(term.params["asset_cfg"].joint_names,
                             ("FR_hip_joint", "FL_hip_joint"))
            self.assertTrue(term.params["asset_cfg"].preserve_order)
            self.assertEqual(term.params["landing_start"], 1.0)
            self.assertEqual(term.params["landing_full"], 1.15)

    def test_front_hip_penalizes_mirrored_splay_through_episode_end(self):
        self.assertTrue(hasattr(mdp, "front_hip_landing_pose"))
        # Deliberately scattered indices: thigh/calf/rear errors must be ignored.
        joints = torch.full((6, 12), 9.0)
        joints[:, 8] = -0.5
        joints[:, 2] = 0.5
        env = self.environment(joints)
        env.episode_length_buf = torch.tensor([180, 200, 215, 230, 260, 400])
        front_hips = SimpleNamespace(name="robot", joint_ids=[8, 2])
        torch.testing.assert_close(
            mdp.front_hip_landing_pose(env, front_hips),
            torch.tensor([0.0, 0.0, 0.25, 0.5, 0.5, 0.5]),
        )
        joints[:, [8, 2]] = 0.0
        torch.testing.assert_close(
            mdp.front_hip_landing_pose(env, front_hips), torch.zeros(6)
        )

    def test_action_rate_weight_changes_only_at_landing(self):
        for play in (False, True):
            term = make_env_cfg(play=play).rewards["action_rate"]
            env = SimpleNamespace(
                episode_length_buf=torch.tensor([0, 100, 229, 230, 396]),
                step_dt=0.005,
                action_manager=SimpleNamespace(
                    action=torch.ones(5, 12), prev_action=torch.zeros(5, 12)
                ),
            )
            weighted = term.func(env, **term.params) * term.weight
            torch.testing.assert_close(
                weighted, torch.tensor([-0.012, -0.012, -0.012, -0.06, -0.06])
            )

    def test_landing_slip_uses_ground_velocity_and_current_contact(self):
        n = 7
        velocity = torch.zeros(n, 4, 3)
        velocity[:, 0, 0] = 1.0
        velocity[0, 0, 0] = 0.0
        velocity[6, 0, 0] = 0.0
        # An airborne foot moving quickly must not incur slip cost.
        velocity[:, 1, 0] = 9.0
        current_force = torch.zeros(n, 4, 3)
        current_force[:, 0, 2] = 2.0
        current_force[4, 0, 2] = 0.0
        history = torch.zeros(n, 4, 3, 3)
        history[:, 0, :, 2] = 2.0  # Includes a contact that has ended in row 4.
        data = SimpleNamespace(
            site_lin_vel_w=velocity,
            root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
        )
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([88, 90, 160, 230, 260, 396, 260]),
            step_dt=0.005,
            scene={
                "robot": SimpleNamespace(data=data),
                "feet_contact": SimpleNamespace(data=SimpleNamespace(
                    force=current_force, force_history=history
                )),
            },
        )
        term = make_env_cfg().rewards["stance_foot_slip"]
        feet = SimpleNamespace(name="robot", site_ids=[0, 1, 2, 3])
        params = {**term.params, "asset_cfg": feet}
        torch.testing.assert_close(
            term.func(env, **params), torch.tensor([1., 0., 0., 1., 0., 1., 0.])
        )

if __name__ == "__main__":
    unittest.main()
