"""Unitree Go2 HV torque-speed envelope from the source Isaac Lab actuator.

The four migrated methods use explicit PD torque control. MuJoCo's built-in
position actuators cannot reproduce the source's different motoring/braking
limits, so this actuator computes and clips the torque every physics step.
"""

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg


def clip_go2_effort(effort: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Source Go2HV: knee 13.5 rad/s, no-load 30 rad/s, 20.2/23.4 Nm."""
    limit = torch.where(velocity * effort > 0, 20.2, 23.4)
    speed_scale = ((30.0 - velocity.abs()) / (30.0 - 13.5)).clamp(0.0, 1.0)
    limit = limit * speed_scale
    return torch.clamp(effort, -limit, limit)


class Go2HVActuator(IdealPdActuator):
    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        effort = self.stiffness * (cmd.position_target - cmd.pos)
        effort = effort + self.damping * (cmd.velocity_target - cmd.vel)
        effort = effort + cmd.effort_target
        return clip_go2_effort(effort, cmd.vel)


@dataclass(kw_only=True)
class Go2HVActuatorCfg(IdealPdActuatorCfg):
    effort_limit: float = 23.4

    def build(self, entity, target_ids, target_names) -> Go2HVActuator:
        return Go2HVActuator(self, entity, target_ids, target_names)
