from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfg import dreamwaq_env_cfg
from .rl_cfg import dreamwaq_runner_cfg

register_mjlab_task(
    task_id="Unitree-Go2-DreamWaQ",
    env_cfg=dreamwaq_env_cfg(),
    play_env_cfg=dreamwaq_env_cfg(play=True),
    rl_cfg=dreamwaq_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
