"""Play-only commands for simultaneous Go2 locomotion and FR leg control."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from src.tasks.go2.pedipulation.commands import (
    FootPositionCommand,
    FootPositionCommandCfg,
    smoothstep,
)

VELOCITY_KEYS = {
    ord("W"): (0, 0.1),
    ord("S"): (0, -0.1),
    ord("A"): (1, 0.1),
    ord("D"): (1, -0.1),
    ord("Q"): (2, 0.1),
    ord("E"): (2, -0.1),
}
FOOT_KEYS = {
    ord("I"): (0, 0.01),
    ord("K"): (0, -0.01),
    ord("J"): (1, 0.01),
    ord("L"): (1, -0.01),
    ord("U"): (2, 0.01),
    ord("O"): (2, -0.01),
}


@dataclass(kw_only=True)
class KeyboardVelocityCommandCfg(CommandTermCfg):
    ranges: Any
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env):
        return KeyboardVelocityCommand(self, env)


class KeyboardVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # StandUp's reward adapter consumes a four-column source command.
        # All other policies observe the first three columns of this same buffer.
        self.source_commands = torch.zeros((self.num_envs, 4), device=self.device)

    @property
    def command(self):
        return self.source_commands[:, :3]

    def nudge(self, axis, delta):
        name = ("lin_vel_x", "lin_vel_y", "ang_vel_z")[axis]
        low, high = getattr(self.cfg.ranges, name)
        self.command[:, axis].add_(delta).clamp_(low, high)

    def stop(self):
        self.source_commands.zero_()

    def reset(self, env_ids):
        self.source_commands[env_ids] = 0.0
        return super().reset(env_ids)

    def before_rewards(self):
        # Retain the StandUp callback interface without its random resampling.
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class KeyboardFootPositionCommandCfg(FootPositionCommandCfg):
    def build(self, env):
        return KeyboardFootPositionCommand(self, env)


class KeyboardFootPositionCommand(FootPositionCommand):
    """Blend to a reachable training reference, hold, and return only on request."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        points = torch.cat((self.home[None], self.targets))
        self.lower, self.upper = points.amin(0), points.amax(0)
        self.cursor = self.reference.clone()
        self._start = self.reference.clone()
        self._blend_start = self.blend.clone()
        self._blend_goal = self.blend.clone()
        self._transition_time = torch.ones_like(self.elapsed)

    def _project_target(self):
        # Training references lie on these home-to-goal segments. Projection
        # includes the outgoing/return path, rather than snapping to the bank.
        segments = self.targets - self.home
        offset = self.cursor - self.home
        fraction = (offset[:, None] * segments).sum(-1) / segments.square().sum(-1)
        points = self.home + fraction.clamp(0, 1)[..., None] * segments
        nearest = (points - self.cursor[:, None]).square().sum(-1).argmin(-1)
        return points[torch.arange(self.num_envs, device=self.device), nearest]

    def _move_to(self, goal, blend_goal):
        self._start.copy_(self.reference)
        self._blend_start.copy_(self.blend)
        self.goal.copy_(goal)
        self._blend_goal.copy_(blend_goal)
        self._transition_time.zero_()
        self._update_command()

    def nudge(self, axis, delta):
        self.cursor[:, axis].add_(delta)
        self.cursor.clamp_(self.lower, self.upper)
        goal = self._project_target()
        active = (goal - self.home).norm(dim=-1) > 1e-6
        self._move_to(goal, active.float())

    def return_home(self):
        self.cursor.copy_(self.home)
        self._move_to(self.cursor, torch.zeros_like(self.blend))

    def _resample_command(self, env_ids):
        # CommandManager.reset still clears metrics, but never samples goals.
        pass

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        self.cursor[env_ids] = self.home
        self.goal[env_ids] = self.home
        self.reference[env_ids] = self.home
        self._start[env_ids] = self.home
        for values in (self.blend, self._blend_start, self._blend_goal, self.elapsed):
            values[env_ids] = 0.0
        self._transition_time[env_ids] = 1.0
        self.enabled[env_ids] = False
        self.returning[env_ids] = False
        return extras

    def _update_command(self):
        progress = self._transition_time.clamp(0, 1)
        blend = smoothstep(progress)
        self.reference.copy_(self._start + blend[:, None] * (self.goal - self._start))
        self.blend.copy_(
            self._blend_start + blend * (self._blend_goal - self._blend_start)
        )
        self.returning.copy_(
            (self._blend_goal == 0) & (progress < 1) & (self.blend > 0)
        )
        self.enabled.copy_((self.blend > 0) | (self._blend_goal > 0))
        # Preserve stage fields consumed by the existing evaluation rewards.
        self.elapsed.copy_(
            torch.where(
                self.returning,
                sum(self.cfg.durations[:3]) + progress,
                self.cfg.durations[0] + progress,
            )
        )

    def compute(self, dt):
        ready = (
            self._env.episode_length_buf * self._env.step_dt >= self.cfg.initial_delay
        )
        self._transition_time.add_(dt * ready).clamp_(max=1.0)
        self._update_command()
        self._update_metrics()


def configure_keyboard_commands(cfg):
    """Replace only a freshly loaded play config; never edit training factories."""
    velocity_name = next(
        (name for name in ("twist", "base_velocity") if name in cfg.commands), None
    )
    if velocity_name is None:
        raise ValueError(
            "--keyboard requires a Go2 velocity task; Backflip and Tracking are unsupported."
        )
    ranges = deepcopy(cfg.commands[velocity_name].ranges)
    cfg.commands[velocity_name] = KeyboardVelocityCommandCfg(ranges=ranges)
    if "foot_position" in cfg.commands:
        cfg.commands["foot_position"] = KeyboardFootPositionCommandCfg(
            **asdict(cfg.commands["foot_position"])
        )


class KeyboardController:
    """Interpret key events on the simulation thread; velocities and goals persist."""

    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        terms = self.env.command_manager
        name = "twist" if "twist" in terms.active_terms else "base_velocity"
        self.velocity = terms.get_term(name)
        self.foot = (
            terms.get_term("foot_position")
            if "foot_position" in terms.active_terms
            else None
        )
        self.keys = set(VELOCITY_KEYS) | {ord("X")}
        if self.foot is not None:
            self.keys.update(FOOT_KEYS)
            self.keys.add(ord("H"))

    def handle_key(self, key):
        if key in VELOCITY_KEYS:
            self.velocity.nudge(*VELOCITY_KEYS[key])
        elif key == ord("X"):
            self.velocity.stop()
        elif self.foot is not None and key in FOOT_KEYS:
            self.foot.nudge(*FOOT_KEYS[key])
        elif self.foot is not None and key == ord("H"):
            self.foot.return_home()
        else:
            return False
        self._refresh_observations()
        return True

    def _refresh_observations(self):
        manager = self.env.observation_manager
        # Invalidating the cache alone leaves history_length=1 actors stale.
        # Replace only the newest frame; buffer[0] is mjlab's lag-zero view.
        # Supported play configs disable manager noise/delay on these history
        # terms; their functions already build the scaled policy/source frames.
        for group, buffers in manager._group_obs_term_history_buffer.items():
            for name, history in buffers.items():
                if not history.is_initialized:
                    continue
                cfg = manager.get_term_cfg(group, name)
                frame = cfg.func(self.env, **cfg.params).clone()
                if cfg.clip is not None:
                    frame.clamp_(*cfg.clip)
                if cfg.scale is not None:
                    frame.mul_(cfg.scale)
                history[0].copy_(frame)
        manager._obs_buffer = None

    def overlay(self):
        def coordinates(value):
            return " ".join(f"{x:+.3f}" for x in value[0].tolist())

        labels = ["Velocity (m/s,m/s,rad/s)"]
        values = [coordinates(self.velocity.command)]
        if self.foot is not None:
            labels += [
                "FR requested (body m)",
                "FR goal (body m)",
                "FR reference (body m)",
            ]
            values += [
                coordinates(x)
                for x in (self.foot.cursor, self.foot.goal, self.foot.reference)
            ]
        return "\n".join(labels), "\n".join(values)

    @property
    def help_text(self):
        text = "W/S forward | A/D left | Q/E turn | X stop\n"
        if self.foot is not None:
            text += "I/K FR x | J/L FR y | U/O FR z | H return\n"
        return text + "Space pause | Enter reset | +/- playback speed | Esc close"
