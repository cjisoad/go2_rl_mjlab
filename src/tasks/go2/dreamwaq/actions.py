"""Source joint-position target clipping, absent in mjlab's base action."""

from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


class DreamWaQJointPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.clamp_(-100.0, 100.0)


@dataclass(kw_only=True)
class DreamWaQJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return DreamWaQJointPositionAction(self, env)
