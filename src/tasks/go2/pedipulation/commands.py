"""Body-frame FR position commands with timed reach, hold and return phases."""

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .workspace import build_workspace


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x.pow(3) * (10.0 + x * (-15.0 + 6.0 * x))


def trajectory(elapsed, durations):
    rest, reach, hold, returning = durations
    hold_end = rest + reach + hold
    blend = smoothstep((elapsed - rest) / reach) * (
        1.0 - smoothstep((elapsed - hold_end) / returning)
    )
    is_return = (elapsed >= hold_end) & (elapsed < hold_end + returning)
    return blend, is_return


@dataclass(kw_only=True)
class FootPositionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (6.0, 6.0)
    durations: tuple[float, float, float, float] = (2.0, 1.0, 2.0, 1.0)
    initial_delay: float = 1.0
    manipulation_probability: float = 0.65
    biped: bool = False

    def build(self, env):
        return FootPositionCommand(self, env)


class FootPositionCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        workspace = build_workspace(cfg.biped)
        self.home = torch.tensor(
            workspace.home, device=self.device, dtype=torch.float32
        )
        self.targets = torch.tensor(
            workspace.targets, device=self.device, dtype=torch.float32
        )
        self.site_id = env.scene["robot"].find_sites("FR")[0][0]
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.enabled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.manual = torch.zeros_like(self.enabled)
        self.goal = self.home.repeat(self.num_envs, 1)
        self.reference = self.goal.clone()
        self.blend = self.elapsed.clone()
        self.returning = self.enabled.clone()
        self.metrics["position_error"] = self.elapsed.clone()

    @property
    def command(self):
        return self.reference

    def _resample_command(self, env_ids):
        self.elapsed[env_ids] = 0.0
        automatic = env_ids[~self.manual[env_ids]]
        self.goal[automatic] = self.targets[
            torch.randint(len(self.targets), (len(automatic),), device=self.device)
        ]
        self.enabled[automatic] = (
            torch.rand(len(automatic), device=self.device)
            < self.cfg.manipulation_probability
        )
        self._update_command()

    def _update_command(self):
        blend, returning = trajectory(self.elapsed, self.cfg.durations)
        self.blend = blend * self.enabled
        self.returning = returning & self.enabled
        self.reference.copy_(self.home + self.blend[:, None] * (self.goal - self.home))

    def compute(self, dt):
        # Only the command clock determines stages; contact truth is reward-only.
        ready = (
            self._env.episode_length_buf * self._env.step_dt >= self.cfg.initial_delay
        )
        self.elapsed += dt * ready
        ids = (self.elapsed >= sum(self.cfg.durations)).nonzero().flatten()
        self._resample(ids)
        self._update_command()
        self._update_metrics()

    def _update_metrics(self):
        from .mdp import foot_position_b

        self.metrics["position_error"].copy_(
            (foot_position_b(self._env) - self.reference).norm(dim=-1)
        )

    def set_target(self, target_b: torch.Tensor, env_ids: torch.Tensor):
        """Set an absolute body-frame goal (metres); execute a full timed cycle.

        The caller supplies reachable positions. Manual goals persist across
        cycles until clear_target() is called. This API never samples actor state.
        """
        self.goal[env_ids] = target_b
        self.manual[env_ids] = True
        self.enabled[env_ids] = True
        self.elapsed[env_ids] = 0.0
        self._update_command()

    def clear_target(self, env_ids: torch.Tensor):
        self.manual[env_ids] = False
        self.enabled[env_ids] = False
        self.elapsed[env_ids] = 0.0
        self._update_command()
