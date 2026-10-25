"""A one-policy-step position-target delay, with source FR/FL/RR/RL order."""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class DelayedJointPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        # mjlab's standard JOINT resolver follows MJCF order, even with preserve_order.
        return self._entity.find_joints(cfg.actuator_names, preserve_order=True)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_processed_actions = self._processed_actions.clone()

    def process_actions(self, actions):
        self._previous_processed_actions[:] = self._processed_actions
        super().process_actions(actions)
        self._processed_actions.clamp_(-100.0, 100.0)

    def apply_actions(self):
        self._entity.set_joint_position_target(
            self._previous_processed_actions, joint_ids=self._target_ids
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        # Match V1.6: raw actions reset, but the last processed target is retained.
        self._previous_processed_actions[ids] = self._processed_actions[ids]


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return DelayedJointPositionAction(self, env)
