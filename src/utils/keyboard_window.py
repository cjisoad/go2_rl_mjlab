"""A MuJoCo render window whose keyboard is owned entirely by the repository.

MuJoCo's Simulate window handles built-in shortcuts in C++ before/alongside its
Python key callback. Creating our own GLFW window avoids installing that handler.
"""

import glfw
import mujoco
import numpy as np


class KeyboardWindow:
    def __init__(self, model, data, key_callback):
        if not glfw.init():
            raise RuntimeError("Unable to initialize GLFW; run keyboard play in WSLg.")
        self.window = glfw.create_window(1200, 900, "Go2 keyboard control", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Unable to create the keyboard control window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model, self.data = model, data
        self.key_callback = key_callback
        self.limit_terrain_view = False
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.user_scn = mujoco.MjvScene(model, maxgeom=20000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self._texts = []
        self._figures = []
        self._cursor = glfw.get_cursor_pos(self.window)
        # No Simulate, character callback, or built-in keyboard shortcuts.
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    @property
    def viewport(self):
        width, height = glfw.get_framebuffer_size(self.window)
        return mujoco.MjrRect(0, 0, max(1, width), max(1, height))

    def is_running(self):
        return bool(self.window and not glfw.window_should_close(self.window))

    def _on_key(self, window, key, scancode, action, mods):
        del scancode, mods
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        else:
            self.key_callback(key)

    def _on_mouse_move(self, window, x, y):
        previous_x, previous_y = self._cursor
        self._cursor = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        height = max(1, glfw.get_window_size(window)[1])
        mujoco.mjv_moveCamera(
            self.model,
            action,
            (x - previous_x) / height,
            (y - previous_y) / height,
            self.user_scn,
            self.cam,
        )

    def _on_scroll(self, window, x_offset, y_offset):
        del window, x_offset
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * y_offset,
            self.user_scn,
            self.cam,
        )

    def update_scene(self):
        hidden = np.empty(0, dtype=int)
        if self.limit_terrain_view:
            center = self.cam.lookat
            if self.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                center = self.data.subtree_com[self.cam.trackbodyid]
            distance = np.linalg.norm(self.data.geom_xpos[:, :2] - center[:2], axis=1)
            static = self.model.body_weldid[self.model.geom_bodyid] == 0
            finite = self.model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
            radius = max(12.0, 2.0 * self.cam.distance)
            hidden = np.flatnonzero(
                static & finite & (distance > radius + self.model.geom_rbound)
            )
        # Filter only the CPU render pass. Restore groups immediately so physics,
        # height scans, debug rendering and the next frame see the original model.
        groups = self.model.geom_group[hidden].copy()
        group_visible = self.opt.geomgroup[5]
        self.model.geom_group[hidden] = 5
        self.opt.geomgroup[5] = 0
        try:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                self.perturb,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.user_scn,
            )
        finally:
            self.model.geom_group[hidden] = groups
            self.opt.geomgroup[5] = group_visible

    def set_texts(self, texts):
        self._texts = texts

    def set_figures(self, figures):
        self._figures = figures

    def sync(self, state_only=False):
        del state_only
        glfw.poll_events()
        if not self.is_running():
            return
        glfw.make_context_current(self.window)
        viewport = self.viewport
        mujoco.mjr_render(viewport, self.user_scn, self.context)
        for font, position, left, right in self._texts:
            mujoco.mjr_overlay(font, position, viewport, left, right, self.context)
        for bounds, figure in self._figures:
            mujoco.mjr_figure(bounds, figure, self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
