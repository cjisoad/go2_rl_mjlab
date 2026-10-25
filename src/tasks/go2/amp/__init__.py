from mjlab.tasks.registry import register_mjlab_task
from .config import amp_env_cfg, amp_runner_cfg
from .runner import AMPOnPolicyRunner

register_mjlab_task(
    task_id="Unitree-Go2-AMP",
    env_cfg=amp_env_cfg(),
    play_env_cfg=amp_env_cfg(play=True),
    rl_cfg=amp_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
