"""Register only this repository's retained Tracking and Go2 tasks."""

from importlib import import_module

for _module in (
    "tracking.config.g1",
    "tracking.config.g1_23dof",
    "velocity.config.go2",
    "go2.dreamwaq",
    "go2.amp",
    "go2.standup",
    "go2.backflip",
    "go2.pedipulation",
):
    import_module(f"{__name__}.{_module}")

TASK_IDS = (
    "Unitree-G1-Tracking",
    "Unitree-G1-Tracking-No-State-Estimation",
    "Unitree-G1-23Dof-Tracking",
    "Unitree-G1-23Dof-Tracking-No-State-Estimation",
    "Unitree-Go2-Flat",
    "Unitree-Go2-Rough",
    "Unitree-Go2-DreamWaQ",
    "Unitree-Go2-AMP",
    "Unitree-Go2-StandUp",
    "Unitree-Go2-Backflip-V1.6",
    "Unitree-Go2-Quad-Pedipulation",
    "Unitree-Go2-Biped-Pedipulation",
)
