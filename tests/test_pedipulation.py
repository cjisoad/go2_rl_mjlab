import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict


class PedipulationTests(unittest.TestCase):
    def test_tasks_are_registered_with_independent_defaults(self):
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

        import src.tasks

        for name, iterations in (("Quad", 5000), ("Biped", 6000)):
            task = f"Unitree-Go2-{name}-Pedipulation"
            self.assertIn(task, src.tasks.TASK_IDS)
            cfg, rl = load_env_cfg(task), load_rl_cfg(task)
            self.assertEqual(cfg.scene.num_envs, 4096)
            self.assertEqual(rl.max_iterations, iterations)
            self.assertIn("foot_position", cfg.commands)
            self.assertIn("fr_tracking", cfg.rewards)
            self.assertIn("fr_return", cfg.rewards)
            if name == "Biped":
                self.assertIsNone(rl.algorithm.symmetry_cfg)

    def test_stage_reference_is_continuous_and_returns_home(self):
        from src.tasks.go2.pedipulation.commands import trajectory

        t = torch.tensor([0.0, 2.0, 2.5, 3.0, 5.0, 5.5, 6.0])
        blend, returning = trajectory(t, (2.0, 1.0, 2.0, 1.0))
        torch.testing.assert_close(
            blend, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
        )
        self.assertEqual(
            returning.tolist(), [False, False, False, False, True, True, False]
        )

    def test_position_reward_uses_body_frame_and_decreases_with_error(self):
        from src.tasks.go2.pedipulation.mdp import foot_position_b, position_score

        # A 90 degree yaw should rotate a world +Y displacement into body +X.
        quat = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
        data = SimpleNamespace(
            root_link_quat_w=quat,
            root_link_pos_w=torch.tensor([[2.0, 3.0, 1.0]]),
            site_pos_w=torch.tensor([[[2.0, 4.0, 1.0]]]),
        )
        term = SimpleNamespace(site_id=0)
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_term=lambda _: term),
        )
        torch.testing.assert_close(
            foot_position_b(env), torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6
        )
        scores = position_score(
            torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            torch.zeros(3, 3),
        )
        self.assertTrue(bool((scores[:-1] > scores[1:]).all()))
        self.assertEqual(scores[0].item(), 1.0)

    def test_workspaces_have_reachable_targets_and_distinct_homes(self):
        from src.tasks.go2.pedipulation.workspace import build_workspace

        quad, biped = build_workspace(False), build_workspace(True)
        self.assertGreater(len(quad.targets), 10)
        self.assertGreater(len(biped.targets), 10)
        self.assertFalse(
            torch.allclose(torch.tensor(quad.home), torch.tensor(biped.home))
        )
        for workspace in (quad, biped):
            self.assertEqual(workspace.targets.shape, workspace.joint_positions.shape)
            self.assertLess(float(abs(workspace.targets - workspace.home).max()), 0.20)

    def test_dreamwaq_extended_observations_train_and_export(self):
        from src.tasks.go2.dreamwaq.rl import DreamWaQActor

        obs = TensorDict(
            {
                "actor": torch.randn(2, 48),
                "obs_history": torch.randn(2, 240),
                "privileged": torch.randn(2, 202),
            },
            [2],
        )
        actor = DreamWaQActor(obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=[16])
        self.assertEqual(actor(obs).shape, (2, 12))
        sum(actor.estimator_losses(obs, 1.0, 1).values()).backward()
        export = actor.as_onnx()
        torch.testing.assert_close(export(obs["actor"], obs["obs_history"]), actor(obs))
        self.assertEqual(
            [tuple(x.shape) for x in export.get_dummy_inputs()], [(1, 48), (1, 240)]
        )

    def test_quad_actor_needs_only_deployable_inputs_and_position_command(self):
        from src.tasks.go2.pedipulation.mdp import quad_observation

        data = SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.zeros(2, 3),
            joint_pos=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
        )
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(
                get_command=lambda name: (
                    torch.ones(2, 3) if name == "foot_position" else torch.zeros(2, 3)
                )
            ),
            action_manager=SimpleNamespace(action=torch.zeros(2, 12)),
        )
        obs = quad_observation(env, noise=False)
        self.assertEqual(obs.shape, (2, 48))
        torch.testing.assert_close(obs[:, -3:], torch.ones(2, 3))


class FootCommandTests(unittest.TestCase):
    def test_manual_goal_cycles_and_reset_restores_home_reference(self):
        from src.tasks.go2.pedipulation.commands import FootPositionCommandCfg

        data = SimpleNamespace(
            root_link_pos_w=torch.zeros(2, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            site_pos_w=torch.zeros(2, 1, 3),
        )
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.ones(2) * 1000,
            scene={
                "robot": SimpleNamespace(
                    data=data, find_sites=lambda name: ([0], ["FR"])
                )
            },
        )
        command = FootPositionCommandCfg(manipulation_probability=1.0).build(env)
        env.command_manager = SimpleNamespace(get_term=lambda name: command)
        ids = torch.arange(2)
        goal = command.targets[:2]
        command.set_target(goal, ids)
        seen_hold = seen_return = False
        for _ in range(650):
            command.compute(0.02)
            self.assertTrue(bool(torch.isfinite(command.command).all()))
            seen_hold |= bool((command.blend == 1).all())
            seen_return |= bool(command.returning.all())
        self.assertTrue(seen_hold and seen_return)
        torch.testing.assert_close(command.goal, goal)
        command.reset(ids)
        torch.testing.assert_close(command.command, command.home.repeat(2, 1))
        self.assertTrue(bool((command.elapsed == 0).all()))
        command.clear_target(ids)
        command.compute(0.02)
        self.assertFalse(bool(command.enabled.any()))

    def test_return_reward_only_acts_late_in_return(self):
        from src.tasks.go2.pedipulation.mdp import return_weight

        command = SimpleNamespace(
            cfg=SimpleNamespace(durations=(2.0, 1.0, 2.0, 1.0)),
            elapsed=torch.tensor([4.0, 5.0, 5.5, 5.9]),
            returning=torch.tensor([False, True, True, True]),
        )
        env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command)
        )
        torch.testing.assert_close(
            return_weight(env), torch.tensor([0.0, 0.0, 0.25, 0.81])
        )

    def test_biped_posture_cost_keeps_other_legs_constrained(self):
        from src.tasks.go2.pedipulation.mdp import biped_reward

        joints = torch.zeros(2, 12)
        joints[:, 3:6] = 1.0
        joints[:, 6:9] = 2.0
        state = SimpleNamespace(
            dof_pos=joints,
            descire_joint_pos=torch.zeros(12),
            refresh=lambda: None,
            _reward_default_pos=lambda: joints.abs().sum(-1),
        )
        command = SimpleNamespace(blend=torch.tensor([0.0, 1.0]))
        env = SimpleNamespace(
            _standup_v2_source=state,
            command_manager=SimpleNamespace(get_term=lambda name: command),
        )
        state.rew_hanstand = torch.ones(2)
        torch.testing.assert_close(
            biped_reward(env, "default_pos"), torch.tensor([9.0, 6.0])
        )


class PedipulationRegressionTests(unittest.TestCase):
    def test_quad_home_matches_dreamwaq_joint_defaults(self):
        import re

        import mujoco
        import numpy as np

        from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg
        from src.tasks.go2.pedipulation.workspace import build_workspace

        cfg = dreamwaq_env_cfg().scene.entities["robot"]
        model = cfg.spec_fn().compile()
        data = mujoco.MjData(model)
        data.qpos[:7] = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
        for index in range(model.njnt):
            joint = model.joint(index)
            for pattern, angle in cfg.init_state.joint_pos.items():
                if re.fullmatch(pattern, joint.name):
                    data.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        expected = (
            data.site_xpos[model.site("FR").id] - data.xpos[model.body("base_link").id]
        )
        np.testing.assert_allclose(build_workspace(False).home, expected, atol=1e-8)

    def test_support_rewards_have_ground_filtered_sensor(self):
        from src.tasks.go2.pedipulation.env_cfg import quad_pedipulation_env_cfg

        cfg = quad_pedipulation_env_cfg()
        sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
        self.assertIn("pedipulation_ground_contact", sensors)
        self.assertEqual(
            sensors["pedipulation_ground_contact"].secondary.pattern, "terrain"
        )


class ActorAndContactTests(unittest.TestCase):
    def test_biped_actor_needs_no_privileged_state(self):
        from src.tasks.go2.pedipulation.mdp import biped_observation

        data = SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.zeros(2, 3),
            joint_pos=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
        )
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            scene={
                "robot": SimpleNamespace(
                    data=data, find_joints=lambda *a, **kw: (list(range(12)), [])
                )
            },
            cfg=SimpleNamespace(
                observations={"actor": SimpleNamespace(enable_corruption=False)}
            ),
            command_manager=SimpleNamespace(
                get_command=lambda name: (
                    torch.ones(2, 3) if name == "foot_position" else torch.zeros(2, 3)
                )
            ),
            action_manager=SimpleNamespace(
                get_term=lambda name: SimpleNamespace(raw_action=torch.zeros(2, 12))
            ),
        )
        obs = biped_observation(env)
        self.assertEqual(obs.shape, (2, 48))
        torch.testing.assert_close(obs[:, -3:], torch.ones(2, 3))
        torch.testing.assert_close(env._standup_v2_policy_obs, obs)

    def test_robot_self_contact_cannot_earn_ground_return_reward(self):
        from src.tasks.go2.pedipulation.mdp import fr_ground_return

        command = SimpleNamespace(
            cfg=SimpleNamespace(durations=(2.0, 1.0, 2.0, 1.0)),
            elapsed=torch.tensor([5.5]),
            returning=torch.tensor([True]),
            blend=torch.tensor([0.5]),
        )
        ground = SimpleNamespace(
            cfg=SimpleNamespace(primary=SimpleNamespace(pattern="feet")),
            data=SimpleNamespace(force=torch.zeros(1, 4, 3)),
        )
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            command_manager=SimpleNamespace(get_term=lambda _: command),
            scene={
                "robot": SimpleNamespace(
                    find_geoms=lambda _: (
                        list(range(4)),
                        [
                            "FL_foot_collision",
                            "FR_foot_collision",
                            "RL_foot_collision",
                            "RR_foot_collision",
                        ],
                    )
                ),
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(force=torch.ones(1, 4, 3) * 10)
                ),
                "pedipulation_ground_contact": ground,
            },
        )
        self.assertEqual(fr_ground_return(env).item(), 0.0)
        ground.data.force[:, 1, 2] = 10.0
        self.assertEqual(fr_ground_return(env).item(), 0.25)
