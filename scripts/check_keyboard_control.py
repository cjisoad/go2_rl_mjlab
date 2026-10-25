"""Check keyboard commands with downloaded policies in CUDA and an optional native window.

Tests command/observation plumbing, not trained motion quality. Episode termination
is disabled so a fall cannot interrupt the hold/return timing checks.
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401
from src.utils.keyboard_control import KeyboardController, configure_keyboard_commands
from src.utils.play_terrain import configure_play_terrain

POLICIES = {
    "Flat": 1900,
    "Rough": 2999,
    "DreamWaQ": 4999,
    "AMP": 8000,
    "StandUp": 5999,
    "Quad-Pedipulation": 4950,
    "Biped-Pedipulation": 4900,
}


def check(task, native):
    task_id = "Unitree-Go2-" + task
    cfg = load_env_cfg(task_id, play=True)
    configure_keyboard_commands(cfg)
    configure_play_terrain(cfg, keyboard=True)
    cfg.scene.num_envs = 1
    cfg.terminations = {}
    cfg.episode_length_s = 60
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        agent_cfg = load_rl_cfg(task_id)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        terrain = cfg.scene.terrain.terrain_generator
        if terrain is not None and "strip" in terrain.sub_terrains:
            robot = env.scene["robot"]
            torch.testing.assert_close(
                robot.data.root_link_pos_w[0, :2],
                torch.tensor([-28.0, 0.0], device=env.device),
            )
            torch.testing.assert_close(
                robot.data.root_link_quat_w[0],
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device),
            )
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=env.device)
        path = (
            Path("logs/selected_policies/2026-10-25")
            / task_id
            / f"model_{POLICIES[task]}.pt"
        )
        runner.load(
            str(path), load_cfg={"actor": True}, strict=True, map_location=env.device
        )
        policy = runner.get_inference_policy(device=env.device)
        controls = KeyboardController(env)

        def step():
            with torch.no_grad():
                obs = wrapped.get_observations()
                assert all(torch.isfinite(value).all() for value in obs.values())
                action = policy(obs)
                assert torch.isfinite(action).all()
                wrapped.step(action)

        def check_velocity_observation():
            obs = wrapped.get_observations()
            scale = controls.velocity.command.new_tensor(
                [2.0, 2.0, 0.25]
                if task in ("StandUp", "Biped-Pedipulation")
                else [1.0, 1.0, 1.0]
            )
            expected = controls.velocity.command * scale
            torch.testing.assert_close(obs["actor"][:, 6:9], expected)
            if "obs_history" in obs:
                latest = obs["obs_history"].reshape(1, 5, -1)[:, -1]
                torch.testing.assert_close(latest[:, 6:9], expected)

        controls.handle_key(ord("W"))
        check_velocity_observation()
        for _ in range(200):
            step()
        torch.testing.assert_close(
            controls.velocity.command[:, 0], torch.tensor([0.1], device=env.device)
        )
        if controls.foot is not None:
            controls.handle_key(ord("U"))
            controls.handle_key(ord("U"))
            controls.handle_key(ord("I"))
            goal = controls.foot.goal.clone()
            for _ in range(400):
                step()
            torch.testing.assert_close(controls.foot.command, goal)
            obs = wrapped.get_observations()["actor"]
            torch.testing.assert_close(obs[:, -3:], goal)
            controls.handle_key(ord("H"))
            for _ in range(60):
                step()
            torch.testing.assert_close(controls.foot.command, controls.foot.home[None])
        controls.handle_key(ord("X"))
        assert not controls.velocity.command.any()
        check_velocity_observation()

        if native:
            import glfw

            from src.utils.keyboard_viewer import KeyboardMujocoViewer

            viewer = KeyboardMujocoViewer(wrapped, policy)
            try:
                viewer.setup()
                window = viewer.viewer
                installed = glfw.set_key_callback(window.window, window._on_key)
                assert installed == window._on_key, (
                    "GLFW must use only the repository key handler"
                )
                flags = window.opt.flags.copy()
                for key in "WASDQEIJKLUOHX":
                    installed(window.window, ord(key), 0, glfw.PRESS, 0)
                viewer._process_actions()
                assert (window.opt.flags == flags).all(), (
                    "Robot keys changed rendering flags"
                )
                assert not viewer._show_all_envs
                for _ in range(5):
                    assert viewer._execute_step(), viewer.get_status().last_error
                    viewer.sync_env_to_viewer()
                viewer._last_tick_time = time.perf_counter()
                viewer._stats_last_time = viewer._last_tick_time
                frames = 0
                while frames < 20:
                    frames += viewer.tick()
                    viewer._update_stats()
                status = viewer.get_status()
                print(
                    f"NATIVE PERF {task_id}: {status.smoothed_fps:.1f} FPS, "
                    f"{status.actual_realtime:.2f}x simulation time",
                    flush=True,
                )
                installed(window.window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
                viewer._process_actions()
                assert viewer._is_paused
                installed(window.window, glfw.KEY_ENTER, 0, glfw.PRESS, 0)
                viewer._process_actions()
                assert not controls.velocity.command.any()
                viewer.sync_env_to_viewer()
                distance = window.cam.distance
                window._on_scroll(window.window, 0, 1)
                assert window.cam.distance != distance
                installed(window.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
                assert not window.is_running()
            finally:
                viewer.close()
        wrapped.reset()
        assert not controls.velocity.command.any()
        if controls.foot is not None:
            torch.testing.assert_close(controls.foot.command, controls.foot.home[None])
        print(f"KEYBOARD PASS {task_id} native={native}", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=POLICIES)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    # Single-world inference and the small CPU checks do not benefit from
    # spawning a thread pool for every small tensor operation.
    torch.set_num_threads(1)
    for task in args.task or ("Flat", "Quad-Pedipulation", "Biped-Pedipulation"):
        check(task, args.native)


if __name__ == "__main__":
    main()
