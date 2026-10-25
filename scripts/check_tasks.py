"""Run one actual CUDA task through resets, learning, checkpoint and export."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["WANDB_MODE"] = "disabled"
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
os.chdir(repo)

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.tasks

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=src.tasks.TASK_IDS)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--motion-file", type=Path)
args = parser.parse_args()
cfg = load_env_cfg(args.task)
if "motion" in cfg.commands:
    if args.motion_file is None:
        parser.error("Tracking smoke tests require --motion-file")
    cfg.commands["motion"].motion_file = str(args.motion_file.expanduser().resolve())
cfg.scene.num_envs = args.num_envs
if cfg.scene.terrain and cfg.scene.terrain.terrain_generator:
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cfg.scene.terrain.max_init_terrain_level = 1
agent = load_rl_cfg(args.task)
agent.logger = "tensorboard"
agent.upload_model = False
agent.algorithm.num_learning_epochs = 1
agent.algorithm.num_mini_batches = 2
agent.num_steps_per_env = 24
logdir = repo / "logs/smoke" / args.task
logdir.mkdir(parents=True, exist_ok=True)
print("SMOKE task", args.task, "GPU", torch.cuda.get_device_name(0), flush=True)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = wrapped.get_observations()
    dims = {k: list(v.shape) for k, v in obs.items()}
    print("SMOKE observations", dims, flush=True)
    reset_count = 0
    for step in range(110):
        obs, rewards, dones, extras = wrapped.step(
            torch.zeros((args.num_envs, wrapped.num_actions), device="cuda:0")
        )
        assert all(torch.isfinite(v).all() for v in obs.values()), (
            "observations",
            step,
        )
        assert torch.isfinite(rewards).all(), ("rewards", step)
        reset_count += int(dones.sum())
    # Every task must execute its timeout reset path even in short smoke tests.
    env.episode_length_buf[-1] = env.max_episode_length
    obs, rewards, dones, extras = wrapped.step(
        torch.zeros((args.num_envs, wrapped.num_actions), device="cuda:0")
    )
    assert dones[-1], "Timeout reset not reached"
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent), str(logdir), device="cuda:0")
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
    checkpoint = logdir / "roundtrip.pt"
    runner.save(str(checkpoint))
    policy = runner.get_inference_policy("cuda:0")
    with torch.inference_mode():
        before = policy(wrapped.get_observations()).clone()
    with torch.inference_mode():
        runner.load(str(checkpoint), map_location="cuda:0")
        after = runner.get_inference_policy("cuda:0")(wrapped.get_observations())
    torch.testing.assert_close(before, after)
    runner.export_policy_to_onnx(str(logdir), "smoke_policy.onnx")
    import onnx

    model = onnx.load(str(logdir / "smoke_policy.onnx"))
    onnx.checker.check_model(model)
    result = {
        "task": args.task,
        "num_envs": args.num_envs,
        "zero_action_steps": 110,
        "natural_resets": reset_count,
        "timeout_reset": True,
        "training_iterations": 2,
        "steps_per_iteration": 24,
        "checkpoint_roundtrip": True,
        "onnx_valid": True,
        "observations": dims,
    }
    (logdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SMOKE PASS", json.dumps(result), flush=True)
finally:
    env.close()
