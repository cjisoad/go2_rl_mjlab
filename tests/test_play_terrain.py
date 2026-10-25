import unittest

import mujoco
import numpy as np


class PlayTerrainTests(unittest.TestCase):
    def load(self, name):
        from mjlab.tasks.registry import load_env_cfg

        import src.tasks  # noqa: F401

        return load_env_cfg("Unitree-Go2-" + name, play=True)

    def test_keyboard_auto_uses_all_original_types_in_one_strip(self):
        from src.utils.play_terrain import StripTerrainCfg, configure_play_terrain

        for name in ("Rough", "DreamWaQ", "Quad-Pedipulation"):
            with self.subTest(name=name):
                cfg = self.load(name)
                original = cfg.scene.terrain.terrain_generator
                original_names = list(original.sub_terrains)
                observations = cfg.observations
                configure_play_terrain(cfg, "auto", keyboard=True)
                gen = cfg.scene.terrain.terrain_generator
                self.assertEqual((gen.num_rows, gen.num_cols), (1, 1))
                self.assertEqual(gen.size, (62.0, 10.0))
                strip = gen.sub_terrains["strip"]
                self.assertIsInstance(strip, StripTerrainCfg)
                self.assertEqual(list(strip.sections), original_names)
                self.assertIs(cfg.observations, observations)
                for key, section in strip.sections.items():
                    self.assertIsInstance(section, type(original.sub_terrains[key]))
                self.assertFalse(cfg.viewer.enable_shadows)
                self.assertFalse(cfg.viewer.enable_reflections)
                self.assertEqual(
                    cfg.events["reset_base"].params["pose_range"]["yaw"], (0.0, 0.0)
                )
                self.assertNotIn("randomize_terrain", cfg.events)
                self.assertFalse(cfg.curriculum)
                self.assertGreaterEqual(cfg.episode_length_s, 120)

    def test_original_and_nonkeyboard_auto_keep_original_map(self):
        from src.utils.play_terrain import configure_play_terrain

        for mode, keyboard in [("original", True), ("auto", False)]:
            cfg = self.load("Rough")
            original = cfg.scene.terrain.terrain_generator
            configure_play_terrain(cfg, mode, keyboard=keyboard)
            self.assertIs(cfg.scene.terrain.terrain_generator, original)
            self.assertEqual((original.num_rows, original.num_cols), (5, 5))

    def test_flat_tasks_keep_their_plane_and_explicit_strip_reports_error(self):
        from src.utils.play_terrain import configure_play_terrain

        for name in ("Flat", "AMP", "StandUp", "Biped-Pedipulation", "Backflip-V1.6"):
            cfg = self.load(name)
            configure_play_terrain(cfg, "auto", keyboard=True)
            self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
            with self.assertRaisesRegex(ValueError, "generated terrain"):
                configure_play_terrain(cfg, "strip", keyboard=True)

    def compile(self, name):
        from mjlab.terrains.terrain_generator import TerrainGenerator

        from src.utils.play_terrain import configure_play_terrain

        cfg = self.load(name)
        configure_play_terrain(cfg, "strip", keyboard=True)
        spec = mujoco.MjSpec()
        generator = TerrainGenerator(cfg.scene.terrain.terrain_generator)
        generator.compile(spec)
        return generator, spec.compile()

    def ray(self, model, data, x, y):
        geomid = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([x, y, 5.0]),
            np.array([0.0, 0.0, -1.0]),
            None,
            True,
            -1,
            geomid,
        )
        self.assertGreaterEqual(distance, 0, f"No ground at {x},{y}")
        return 5.0 - distance

    def test_actual_strip_has_fixed_start_and_ground_across_seams_and_width(self):
        for name in ("Rough", "DreamWaQ"):
            with self.subTest(name=name):
                generator, model = self.compile(name)
                np.testing.assert_allclose(generator.terrain_origins[0, 0], [-28, 0, 0])
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                self.assertAlmostEqual(self.ray(model, data, -28, 0), 0, places=5)
                # All segment boundaries are at zero-height native borders.
                for x in (-25, -17, -9, -1, 7, 15, 23, 31):
                    for y in (-4.5, 0, 4.5):
                        for dx in (-0.01, 0.01):
                            height = self.ray(model, data, x + dx, y)
                            self.assertAlmostEqual(height, 0, delta=0.01)
                for x in np.linspace(-30, 30, 25):
                    self.ray(model, data, float(x), 0)

    def test_nearby_rendering_keeps_physics_and_reveals_full_strip_when_zoomed_out(
        self,
    ):
        from src.utils.keyboard_window import KeyboardWindow

        _generator, model = self.compile("DreamWaQ")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        groups = model.geom_group.copy()
        ground_before = self.ray(model, data, 0, 0)
        window = object.__new__(KeyboardWindow)
        window.model, window.data = model, data
        window.cam, window.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        window.perturb = mujoco.MjvPerturb()
        window.user_scn = mujoco.MjvScene(model, maxgeom=20000)
        window.cam.lookat[:] = [-28, 0, 0.4]
        window.cam.distance = 2.0
        window.limit_terrain_view = True
        window.update_scene()
        self.assertLess(window.user_scn.ngeom, model.ngeom)
        np.testing.assert_array_equal(model.geom_group, groups)
        self.assertAlmostEqual(self.ray(model, data, 0, 0), ground_before)
        window.cam.distance = 80.0
        window.update_scene()
        self.assertEqual(window.user_scn.ngeom, model.ngeom)
        np.testing.assert_array_equal(model.geom_group, groups)

    def test_strip_is_repeatable_and_training_grid_is_unchanged(self):
        from mjlab.tasks.registry import load_env_cfg

        _first, model1 = self.compile("Rough")
        _second, model2 = self.compile("Rough")
        np.testing.assert_allclose(model1.geom_pos, model2.geom_pos)
        np.testing.assert_allclose(model1.hfield_data, model2.hfield_data)
        training = load_env_cfg("Unitree-Go2-Rough")
        self.assertEqual(training.scene.num_envs, 4096)
        self.assertEqual(training.scene.terrain.terrain_generator.num_rows, 10)
        self.assertEqual(training.scene.terrain.terrain_generator.num_cols, 20)


if __name__ == "__main__":
    unittest.main()
