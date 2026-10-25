"""Exercise a complete FR command cycle in real CUDA environments.

This checks command/reward plumbing, not a trained robot's success rate.
"""

import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401 -- register repository tasks

for posture in ("Quad", "Biped"):
    cfg = load_env_cfg(f"Unitree-Go2-{posture}-Pedipulation", play=True)
    cfg.scene.num_envs = 2
    if cfg.scene.terrain.terrain_generator:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
        cfg.scene.terrain.max_init_terrain_level = 1
    # Keep a fallen, untrained robot alive so every command stage is exercised.
    cfg.terminations = {}
    cfg.episode_length_s = 60.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        env.reset()
        command = env.command_manager.get_term("foot_position")
        command.set_target(command.targets[:2], torch.arange(2, device=env.device))
        seen = set()
        for _ in range(700):
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros((2, 12), device=env.device)
            )
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(x).all() for x in obs.values())
            if command.blend.max() == 0:
                seen.add("rest")
            if ((command.blend > 0) & (command.blend < 1) & ~command.returning).any():
                seen.add("reach")
            if command.blend.min() == 1:
                seen.add("hold")
            if command.returning.any():
                seen.add("return")
        assert seen == {"rest", "reach", "hold", "return"}, seen
        print("CYCLE PASS", posture, sorted(seen), flush=True)
    finally:
        env.close()
