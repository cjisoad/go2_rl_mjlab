"""Go2 quadruped and rear-leg biped right-front-foot manipulation tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import (
    biped_pedipulation_env_cfg,
    biped_pedipulation_runner_cfg,
    quad_pedipulation_env_cfg,
    quad_pedipulation_runner_cfg,
)

register_mjlab_task(
    task_id="Unitree-Go2-Quad-Pedipulation",
    env_cfg=quad_pedipulation_env_cfg(),
    play_env_cfg=quad_pedipulation_env_cfg(play=True),
    rl_cfg=quad_pedipulation_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="Unitree-Go2-Biped-Pedipulation",
    env_cfg=biped_pedipulation_env_cfg(),
    play_env_cfg=biped_pedipulation_env_cfg(play=True),
    rl_cfg=biped_pedipulation_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
