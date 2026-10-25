import unittest
from types import SimpleNamespace

import torch


def make_env(biped=False, foot=True):
    from mjlab.viewer import ViewerConfig

    from src.tasks.go2.standup.mdp import VelocityRanges
    from src.utils.keyboard_control import (
        KeyboardFootPositionCommandCfg,
        KeyboardVelocityCommandCfg,
    )

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1000]),
        cfg=SimpleNamespace(viewer=ViewerConfig(), scene=SimpleNamespace(terrain=None)),
        observation_manager=SimpleNamespace(
            _obs_buffer={"old": torch.zeros(1)}, _group_obs_term_history_buffer={}
        ),
        scene={"robot": SimpleNamespace(find_sites=lambda _: ([0], ["FR"]))},
    )
    ranges = VelocityRanges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    )
    velocity = KeyboardVelocityCommandCfg(ranges=ranges).build(env)
    terms = {"base_velocity" if biped else "twist": velocity}
    if foot:
        terms["foot_position"] = KeyboardFootPositionCommandCfg(biped=biped).build(env)
        terms["foot_position"]._update_metrics = lambda: None
    env.command_manager = SimpleNamespace(
        active_terms=list(terms),
        get_term=terms.__getitem__,
        get_command=lambda name: terms[name].command,
    )
    return env, velocity, terms.get("foot_position")


def advance(command, env, steps):
    for _ in range(steps):
        env.episode_length_buf += 1
        command.compute(env.step_dt)


class KeyboardControlTests(unittest.TestCase):
    def test_play_has_opt_in_keyboard_flag(self):
        from scripts.play import PlayConfig

        self.assertIn("keyboard", PlayConfig.__dataclass_fields__)
        self.assertFalse(PlayConfig().keyboard)

    def test_documented_cli_enables_keyboard(self):
        import mjlab
        import tyro

        from scripts.play import PlayConfig

        cfg = tyro.cli(
            PlayConfig,
            args=["--keyboard", "True", "--viewer", "native"],
            config=mjlab.TYRO_FLAGS,
        )
        self.assertTrue(cfg.keyboard)
        self.assertEqual(cfg.viewer, "native")

    def test_keyboard_rejects_browser_and_multiple_envs_before_startup(self):
        from scripts.play import PlayConfig, run_play

        with self.assertRaisesRegex(ValueError, "native"):
            run_play("Unitree-Go2-Flat", PlayConfig(keyboard=True, viewer="viser"))
        with self.assertRaisesRegex(ValueError, "num-envs 1"):
            run_play(
                "Unitree-Go2-Flat",
                PlayConfig(keyboard=True, viewer="native", num_envs=2),
            )

    def test_velocity_persists_and_source_adapter_sees_same_command(self):
        env, velocity, _ = make_env()
        velocity.nudge(0, 0.3)
        for _ in range(700):
            velocity.compute(env.step_dt)
            velocity.before_rewards()
        torch.testing.assert_close(velocity.command, torch.tensor([[0.3, 0, 0]]))
        torch.testing.assert_close(velocity.source_commands[:, :3], velocity.command)
        self.assertEqual(velocity.source_commands.shape, (1, 4))
        velocity.nudge(0, 10)
        velocity.nudge(1, -10)
        velocity.nudge(2, 10)
        torch.testing.assert_close(velocity.command, torch.tensor([[1.0, -0.4, 1.0]]))
        velocity.reset(torch.tensor([0]))
        self.assertFalse(velocity.command.any())

    def test_hold_does_not_restart_the_six_second_cycle(self):
        for biped in (False, True):
            with self.subTest(biped=biped):
                env, _, foot = make_env(biped)
                foot.nudge(2, 0.08)
                advance(foot, env, 70)
                expected = foot.goal.clone()
                self.assertGreater((expected - foot.home).norm().item(), 0.01)
                torch.testing.assert_close(foot.command, expected)
                advance(foot, env, 700)
                torch.testing.assert_close(foot.command, expected)
                self.assertEqual(foot.blend.item(), 1)
                self.assertFalse(foot.returning.any())

    def test_early_input_waits_for_initial_posture_delay(self):
        env, _, foot = make_env(True)
        env.episode_length_buf.zero_()
        foot.nudge(2, 0.08)
        advance(foot, env, 20)
        torch.testing.assert_close(foot.command, foot.home[None])
        advance(foot, env, 200)
        torch.testing.assert_close(foot.command, foot.goal)

    def test_target_projection_stays_on_training_reference_segments(self):
        _env, _, foot = make_env()
        foot.nudge(0, 10)
        foot.nudge(1, -10)
        foot.nudge(2, 10)
        segment = foot.targets - foot.home
        goal = foot.goal[0] - foot.home
        alpha = ((segment * goal).sum(-1) / segment.square().sum(-1)).clamp(0, 1)
        error = (alpha[:, None] * segment - goal).norm(dim=-1)
        self.assertLess(error.min().item(), 1e-6)
        self.assertTrue(bool((foot.cursor <= foot.upper).all()))
        self.assertTrue(bool((foot.cursor >= foot.lower).all()))

    def test_return_and_retarget_are_continuous(self):
        env, _, foot = make_env()
        foot.nudge(2, 0.08)
        advance(foot, env, 60)
        previous = foot.command.clone()
        foot.nudge(0, 0.02)
        torch.testing.assert_close(foot.command, previous)
        advance(foot, env, 60)
        previous = foot.command.clone()
        foot.return_home()
        torch.testing.assert_close(foot.command, previous)
        self.assertTrue(foot.returning.all())
        advance(foot, env, 25)
        self.assertGreater((foot.command - foot.home).norm().item(), 0)
        self.assertLess(
            (foot.command - foot.home).norm().item(),
            (previous - foot.home).norm().item(),
        )
        advance(foot, env, 40)
        torch.testing.assert_close(foot.command, foot.home[None])
        self.assertEqual(foot.blend.item(), 0)
        advance(foot, env, 400)
        torch.testing.assert_close(foot.command, foot.home[None])

    def test_reset_clears_held_target(self):
        env, _, foot = make_env()
        foot.nudge(2, 0.08)
        advance(foot, env, 60)
        foot.reset(torch.tensor([0]))
        advance(foot, env, 400)
        torch.testing.assert_close(foot.command, foot.home[None])
        torch.testing.assert_close(foot.cursor, foot.home[None])
        self.assertFalse(foot.enabled.any())

    def test_dog_and_foot_keys_are_independent_and_refresh_observations(self):
        from src.utils.keyboard_control import KeyboardController

        env, velocity, foot = make_env()
        controller = KeyboardController(env)
        controller.handle_key(ord("W"))
        controller.handle_key(ord("A"))
        controller.handle_key(ord("Q"))
        torch.testing.assert_close(velocity.command, torch.tensor([[0.1, 0.1, 0.1]]))
        self.assertIsNone(env.observation_manager._obs_buffer)
        initial = foot.cursor.clone()
        controller.handle_key(ord("I"))
        self.assertAlmostEqual((foot.cursor - initial)[0, 0].item(), 0.01, places=6)
        saved = foot.goal.clone()
        controller.handle_key(ord("X"))
        self.assertFalse(velocity.command.any())
        torch.testing.assert_close(foot.goal, saved)
        controller.handle_key(ord("W"))
        controller.handle_key(ord("H"))
        self.assertAlmostEqual(velocity.command[0, 0].item(), 0.1)
        torch.testing.assert_close(foot.goal, foot.home[None])

    def test_all_key_pairs_have_opposite_directions(self):
        from src.utils.keyboard_control import KeyboardController

        env, velocity, foot = make_env(True)
        controller = KeyboardController(env)
        for positive, negative, axis in [("W", "S", 0), ("A", "D", 1), ("Q", "E", 2)]:
            controller.handle_key(ord(positive))
            self.assertGreater(velocity.command[0, axis].item(), 0)
            controller.handle_key(ord(negative))
            self.assertAlmostEqual(velocity.command[0, axis].item(), 0)
        for positive, negative, axis in [("I", "K", 0), ("J", "L", 1), ("U", "O", 2)]:
            start = foot.cursor.clone()
            controller.handle_key(ord(positive))
            self.assertGreater(foot.cursor[0, axis].item(), start[0, axis].item())
            controller.handle_key(ord(negative))
            torch.testing.assert_close(foot.cursor, start)
        self.assertFalse(controller.handle_key(ord("Z")))

    def test_configuration_replacement_preserves_training_defaults(self):
        from mjlab.tasks.registry import load_env_cfg

        import src.tasks  # noqa: F401
        from src.utils.keyboard_control import (
            KeyboardVelocityCommandCfg,
            configure_keyboard_commands,
        )

        for name in (
            "Flat",
            "Rough",
            "DreamWaQ",
            "AMP",
            "StandUp",
            "Quad-Pedipulation",
            "Biped-Pedipulation",
        ):
            cfg = load_env_cfg("Unitree-Go2-" + name, play=True)
            key = "twist" if "twist" in cfg.commands else "base_velocity"
            for group in cfg.observations.values():
                for term in group.terms.values():
                    history_length = (
                        group.history_length
                        if group.history_length is not None
                        else term.history_length
                    )
                    if history_length > 0:
                        self.assertFalse(group.enable_corruption)
                        self.assertEqual(term.delay_max_lag, 0)
            previous_type = type(cfg.commands[key])
            configure_keyboard_commands(cfg)
            self.assertIsInstance(cfg.commands[key], KeyboardVelocityCommandCfg)
            self.assertIs(
                type(load_env_cfg("Unitree-Go2-" + name, play=True).commands[key]),
                previous_type,
            )
        with self.assertRaisesRegex(ValueError, "velocity"):
            configure_keyboard_commands(
                load_env_cfg("Unitree-Go2-Backflip-V1.6", play=True)
            )

    def test_history_actor_receives_stop_immediately_without_advancing_history(self):
        from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
        from mjlab.managers.observation_manager import ObservationManager

        from src.utils.keyboard_control import KeyboardController

        env, velocity, _foot = make_env(True)
        cfg = {
            "actor": ObservationGroupCfg(
                terms={
                    "commands": ObservationTermCfg(func=lambda env: velocity.command)
                },
                history_length=1,
            ),
            "obs_history": ObservationGroupCfg(
                terms={
                    "commands": ObservationTermCfg(func=lambda env: velocity.command)
                },
                history_length=5,
            ),
        }
        manager = ObservationManager(cfg, env)
        env.observation_manager = manager
        for _ in range(4):
            velocity.nudge(0, 0.1)
            manager.compute(update_history=True)
        history = manager._group_obs_term_history_buffer["obs_history"]["commands"]
        before = history.buffer.clone()
        length = history.current_length.clone()
        KeyboardController(env).handle_key(ord("X"))
        obs = manager.compute()
        torch.testing.assert_close(obs["actor"], torch.zeros(1, 3))
        torch.testing.assert_close(obs["obs_history"][:, -3:], torch.zeros(1, 3))
        torch.testing.assert_close(history.buffer[:, :-1], before[:, :-1])
        torch.testing.assert_close(history.current_length, length)

    def test_keyboard_setup_does_not_launch_simulate_shortcut_handler(self):
        from unittest.mock import Mock, patch

        import mujoco

        from src.utils.keyboard_viewer import KeyboardMujocoViewer

        env, _velocity, _foot = make_env()
        env.unwrapped = env
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
        )
        env.sim = SimpleNamespace(mj_model=model, mj_data=mujoco.MjData(model))
        env.reward_manager = SimpleNamespace(get_active_iterable_terms=lambda _: [])
        window = SimpleNamespace(
            cam=mujoco.MjvCamera(),
            opt=mujoco.MjvOption(),
            perturb=mujoco.MjvPerturb(),
            user_scn=mujoco.MjvScene(model, maxgeom=100),
            update_scene=Mock(),
        )
        with (
            patch(
                "mujoco.viewer.launch_passive",
                side_effect=AssertionError(
                    "MuJoCo Simulate shortcuts must not be installed"
                ),
            ),
            patch(
                "src.utils.keyboard_viewer.KeyboardWindow",
                create=True,
                return_value=window,
            ),
        ):
            viewer = KeyboardMujocoViewer(env, lambda obs: None)
            viewer.setup()
            self.assertIs(viewer.viewer, window)
            viewer._show_debug_vis = False
            viewer._update_debug_visualizers(window)
            window.update_scene.assert_called_once()

    def test_keyboard_window_ignores_releases_and_has_no_default_key_handler(self):
        import glfw

        from src.utils.keyboard_window import KeyboardWindow

        window = object.__new__(KeyboardWindow)
        keys = []
        window.key_callback = keys.append
        window._on_key(None, glfw.KEY_W, 0, glfw.RELEASE, 0)
        window._on_key(None, glfw.KEY_W, 0, glfw.PRESS, 0)
        window._on_key(None, glfw.KEY_W, 0, glfw.REPEAT, 0)
        self.assertEqual(keys, [glfw.KEY_W, glfw.KEY_W])

    def test_viewer_key_callback_only_enqueues_and_consumes_a(self):
        from src.utils.keyboard_viewer import KeyboardMujocoViewer

        env, velocity, _ = make_env()
        viewer = KeyboardMujocoViewer(env, lambda obs: None)
        viewer._safe_key_callback(ord("A"))
        self.assertFalse(velocity.command.any())
        self.assertFalse(viewer._show_all_envs)
        viewer._process_actions()
        self.assertAlmostEqual(velocity.command[0, 1].item(), 0.1)
        self.assertFalse(viewer._show_all_envs)
        viewer._safe_key_callback(32)
        viewer._process_actions()
        self.assertTrue(viewer._is_paused)


if __name__ == "__main__":
    unittest.main()
