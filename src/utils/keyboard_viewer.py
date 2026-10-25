"""Native viewer adapter: queue keyboard commands and show their current values."""

import mujoco
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from .keyboard_control import KeyboardController
from .keyboard_window import KeyboardWindow


class KeyboardMujocoViewer(NativeMujocoViewer):
    def __init__(self, env, policy):
        super().__init__(env, policy)
        self.keyboard = KeyboardController(env)
        self._show_debug_vis = False
        self._strip = None

    def setup(self):
        sim = self.env.unwrapped.sim
        self.mjm, self.mjd = sim.mj_model, sim.mj_data
        if self.cfg.fovy is not None:
            self.mjm.vis.global_.fovy = self.cfg.fovy
        self._term_names = [
            name
            for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
                self.env_idx
            )
        ]
        self._init_reward_plots(self._term_names)
        self.viewer = KeyboardWindow(self.mjm, self.mjd, self._safe_key_callback)
        self.pert = self.viewer.perturb
        self.vopt = self.viewer.opt
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = (
            self.cfg.enable_shadows
        )
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = (
            self.cfg.enable_reflections
        )
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 1
        self._setup_camera()
        terrain = self.env.unwrapped.cfg.scene.terrain
        if terrain is not None and terrain.terrain_generator is not None:
            self._strip = terrain.terrain_generator.sub_terrains.get("strip")
        self.viewer.limit_terrain_view = self._strip is not None
        if self._strip is not None:
            print(
                "[Terrain] +X strip: "
                + " -> ".join(name for name, _, _ in self._strip.layout)
            )
        print(
            "[Keyboard] Click the simulation window to focus it. Commands persist between presses."
        )
        print(self.keyboard.help_text)

    def _safe_key_callback(self, key):
        if key in self.keyboard.keys:
            # Keep all command changes in the viewer's action-processing phase.
            self.request_action("CUSTOM", key)
        else:
            super()._safe_key_callback(key)

    def _handle_custom_action(self, action, payload):
        if action == ViewerAction.CUSTOM:
            return self.keyboard.handle_key(payload)
        return super()._handle_custom_action(action, payload)

    def _update_debug_visualizers(self, viewer):
        # The scene is built on the main thread, after the GPU state is copied.
        # Debug geometry is appended after the robot/terrain, not overwritten.
        self._stabilize_tracking_camera()
        viewer.update_scene()
        if self._show_debug_vis:
            visualizer = MujocoNativeDebugVisualizer(
                viewer.user_scn, self.mjm, self.env_idx
            )
            self.env.unwrapped.update_visualizers(visualizer)

    def _set_status_overlay(self, viewer):
        status = self.get_status()
        state = "PAUSED" if status.paused else "RUNNING"
        if status.capped:
            state += " [CAPPED]"
        labels, values = self.keyboard.overlay()
        if self._strip is not None:
            x = self.mjd.xpos[self.viewer.cam.trackbodyid, 0]
            labels += "\nStrip section"
            values += f"\n{self._strip.section_at(x)} (x={x:+.1f} m)"
        viewer.set_texts(
            [
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    "Status\nStep\nTarget RT\nActual RT\n" + labels,
                    f"{state}\n{status.step_count}\n{status.target_realtime:.2f}x\n"
                    f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n"
                    + values,
                ),
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
                    self.keyboard.help_text,
                    "",
                ),
            ]
        )
