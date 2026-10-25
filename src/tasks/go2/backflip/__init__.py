"""Register the source V1.6 backflip as an independent Go2 task."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfg import make_env_cfg, make_rl_cfg

register_mjlab_task(
    task_id="Unitree-Go2-Backflip-V1.6",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
