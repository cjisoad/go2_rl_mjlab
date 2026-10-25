"""Source random intra-control-tick action switch, without Isaac dependencies."""

from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class SourceDelayedJointPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        return self._entity.find_joints(cfg.actuator_names, preserve_order=True)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_raw_actions = torch.zeros_like(self.raw_action)
        self._switch = torch.zeros(
            (env.num_envs, 1), device=env.device, dtype=torch.long
        )
        self._decimation = env.cfg.decimation
        self._substep = 0

    def process_actions(self, actions):
        self.previous_raw_actions.copy_(self.raw_action)
        super().process_actions(actions.clamp(-100.0, 100.0))
        self._switch.random_(0, self._decimation)
        self._substep = 0

    def apply_actions(self):
        action = torch.where(
            self._substep >= self._switch, self.raw_action, self.previous_raw_actions
        )
        offsets = getattr(self._env, "_standup_v2_motor_offsets", 0.0)
        target = self._offset + self._scale * action + offsets
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
        self._substep += 1

    def reset(self, env_ids=None):
        # Source post_physics_step retains current actions even across reset.
        pass


@dataclass(kw_only=True)
class SourceDelayedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return SourceDelayedJointPositionAction(self, env)
