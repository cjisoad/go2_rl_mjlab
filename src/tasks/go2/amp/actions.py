"""Source position-target clipping absent from mjlab 1.2's base action."""

from dataclasses import dataclass
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class AMPJointPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.clamp_(-100.0, 100.0)


@dataclass(kw_only=True)
class AMPJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return AMPJointPositionAction(self, env)
