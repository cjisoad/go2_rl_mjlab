"""Backflip outcomes require ordered flight, rotation and sustained landing."""

import unittest
from types import SimpleNamespace

import torch
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg

from src.tasks.go2.backflip.evaluation import BackflipEvaluator, backflip_metric


class BackflipEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = BackflipEvaluator(2, "cpu", 0.02)
        self.step = 0

    def advance(
        self,
        pitch_rate=0.0,
        inverted=False,
        airborne=False,
        nonfoot=False,
        height=None,
        horizontal_speed=0.0,
        contacts=None,
    ):
        self.step += 1
        if height is None:
            height = 0.42 if airborne else 0.30
        feet = torch.full((2, 4), not airborne, dtype=torch.bool)
        if contacts is not None:
            feet[:] = torch.tensor(contacts)
        self.evaluator.update(
            episode_steps=torch.full((2,), self.step),
            height=torch.full((2,), height),
            linear_velocity=torch.tensor([[horizontal_speed, 0.0, 0.0]]).repeat(2, 1),
            angular_velocity=torch.tensor([[0.0, pitch_rate, 0.0]]).repeat(2, 1),
            gravity=torch.tensor([[0.0, 0.0, 1.0 if inverted else -1.0]]).repeat(2, 1),
            feet_contact=feet,
            nonfoot_contact=torch.full((2,), nonfoot),
        )

    def flight(self, direction=-1.0, inverted=True):
        self.advance()
        for i in range(25):
            self.advance(
                pitch_rate=direction * 2.0 * torch.pi / 0.5,
                inverted=inverted and 8 <= i <= 16,
                airborne=True,
            )

    def land(self, count=12, **kwargs):
        for _ in range(count):
            self.advance(**kwargs)

    def test_backflip_with_stable_landing_succeeds(self):
        self.flight()
        self.land(9)
        self.assertFalse(self.evaluator.success.any())
        self.land(1)
        self.assertTrue(self.evaluator.takeoff.all())
        self.assertTrue(self.evaluator.full_flip.all())
        self.assertTrue(self.evaluator.success.all())

    def test_standing_or_ordinary_jump_is_not_a_flip(self):
        self.land()
        self.assertFalse(self.evaluator.takeoff.any())
        for _ in range(10):
            self.advance(airborne=True)
        self.land()
        self.assertTrue(self.evaluator.takeoff.all())
        self.assertFalse(self.evaluator.full_flip.any())
        self.assertFalse(self.evaluator.success.any())

    def test_forward_flip_or_no_inversion_does_not_succeed(self):
        self.flight(direction=1.0)
        self.land()
        self.assertFalse(self.evaluator.full_flip.any())
        self.setUp()
        self.flight(inverted=False)
        self.land()
        self.assertFalse(self.evaluator.full_flip.any())

    def test_brief_or_low_clearance_contact_loss_is_not_takeoff(self):
        self.advance()
        self.advance(airborne=True)
        self.advance(airborne=True)
        self.advance()
        self.assertFalse(self.evaluator.takeoff.any())
        for _ in range(5):
            self.advance(airborne=True, height=0.32)
        self.assertFalse(self.evaluator.takeoff.any())

    def test_fall_after_success_revokes_success_and_cannot_recover(self):
        self.flight()
        self.land()
        self.assertTrue(self.evaluator.success.all())
        self.advance(nonfoot=True)
        self.assertFalse(self.evaluator.success.any())
        self.land()
        self.assertFalse(self.evaluator.success.any())

    def test_fast_or_insufficient_foot_support_is_not_stable_landing(self):
        self.flight()
        self.land(horizontal_speed=0.8)
        self.assertFalse(self.evaluator.success.any())
        self.land(contacts=[True, True, False, False])
        self.assertFalse(self.evaluator.success.any())
        self.land()
        self.assertTrue(self.evaluator.success.all())

    def test_reset_clears_only_restarted_environment(self):
        self.flight()
        self.land()
        self.evaluator.update(
            episode_steps=torch.tensor([1, self.step + 1]),
            height=torch.full((2,), 0.3),
            linear_velocity=torch.zeros(2, 3),
            angular_velocity=torch.zeros(2, 3),
            gravity=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            feet_contact=torch.ones(2, 4, dtype=torch.bool),
            nonfoot_contact=torch.zeros(2, dtype=torch.bool),
        )
        self.assertFalse(self.evaluator.takeoff[0])
        self.assertFalse(self.evaluator.full_flip[0])
        self.assertFalse(self.evaluator.success[0])
        self.assertTrue(self.evaluator.success[1])

    def test_metrics_report_episode_outcomes_not_fraction_of_successful_steps(self):
        self.check_metric_outcomes(initial_steps=0)

    def test_random_initial_episode_lengths_do_not_inflate_rates(self):
        self.check_metric_outcomes(initial_steps=40)

    def check_metric_outcomes(self, initial_steps):
        from src.tasks.go2.backflip.env_cfg import NONFOOT_SENSORS

        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.zeros(2, 3),
                root_link_lin_vel_w=torch.zeros(2, 3),
                root_link_ang_vel_b=torch.zeros(2, 3),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            )
        )
        scene = {"robot": robot}
        for name in ("feet_contact", *NONFOOT_SENSORS):
            scene[name] = SimpleNamespace(
                data=SimpleNamespace(
                    force=torch.zeros(2, 4 if name == "feet_contact" else 1, 3)
                )
            )
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            common_step_counter=0,
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            reset_buf=torch.zeros(2, dtype=torch.bool),
            scene=scene,
        )
        manager = MetricsManager(
            {
                stage: MetricsTermCfg(
                    func=backflip_metric,
                    params={"stage": stage, "nonfoot_sensors": NONFOOT_SENSORS},
                )
                for stage in ("takeoff", "full_flip", "success")
            },
            env,
        )
        for step in range(1, 51):
            env.common_step_counter = step
            env.episode_length_buf[:] = step + initial_steps
            airborne = 2 <= step <= 26
            robot.data.root_link_pos_w[:, 2] = 0.42 if airborne else 0.30
            robot.data.root_link_ang_vel_b[:, 1] = (
                -2 * torch.pi / 0.5 if airborne else 0.0
            )
            robot.data.projected_gravity_b[:, 2] = 1.0 if 10 <= step <= 18 else -1.0
            scene["feet_contact"].data.force[:, :, 2] = 0.0 if airborne else 20.0
            # Second environment jumps but never performs a backflip.
            robot.data.root_link_ang_vel_b[1] = 0.0
            robot.data.projected_gravity_b[1, 2] = -1.0
            env.reset_buf[:] = step == 50
            manager.compute()
        result = manager.reset(torch.arange(2))
        self.assertAlmostEqual(float(result["Episode_Metrics/takeoff"]), 1.0)
        self.assertAlmostEqual(float(result["Episode_Metrics/full_flip"]), 0.5)
        self.assertAlmostEqual(float(result["Episode_Metrics/success"]), 0.5)


if __name__ == "__main__":
    unittest.main()
