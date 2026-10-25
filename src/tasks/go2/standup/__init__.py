from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfg import standup_env_cfg, standup_play_env_cfg
from .rl_cfg import standup_runner_cfg

register_mjlab_task(
    task_id="Unitree-Go2-StandUp",
    env_cfg=standup_env_cfg(),
    play_env_cfg=standup_play_env_cfg(),
    rl_cfg=standup_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
