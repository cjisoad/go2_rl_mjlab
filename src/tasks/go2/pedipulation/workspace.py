"""Small reachable FR target banks, computed once with the actual Go2 model."""

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import mujoco
import numpy as np

from src.assets.robots import get_go2_robot_cfg
from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg
from src.tasks.go2.standup.source_config import Go2_Handstand_Cfg_Yu as StandupCfg

FR_JOINTS = tuple(f"FR_{part}_joint" for part in ("hip", "thigh", "calf"))


@dataclass(frozen=True)
class FootWorkspace:
    home: np.ndarray
    targets: np.ndarray
    joint_positions: np.ndarray


@lru_cache(maxsize=2)
def build_workspace(biped: bool) -> FootWorkspace:
    cfg = get_go2_robot_cfg() if biped else dreamwaq_env_cfg().scene.entities["robot"]
    model = cfg.spec_fn().compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    defaults = (
        StandupCfg.init_state.descire_joint_angles
        if biped
        else cfg.init_state.joint_pos
    )
    for index in range(model.njnt):
        joint = model.joint(index)
        for pattern, angle in defaults.items():
            if re.fullmatch(pattern, joint.name):
                data.qpos[joint.qposadr[0]] = angle
    site, base = model.site("FR").id, model.body("base_link").id
    mujoco.mj_forward(model, data)
    home = (data.site_xpos[site] - data.xpos[base]).copy()
    joints = [model.joint(name).id for name in FR_JOINTS]
    addresses = model.jnt_qposadr[joints]
    rest = data.qpos[addresses].copy()
    limits = model.jnt_range[joints]
    delta = np.array((0.25, 0.45, 0.50))
    lower = np.maximum(rest - delta, limits[:, 0] + 0.05)
    upper = np.minimum(rest + delta, limits[:, 1] - 0.05)
    targets, witnesses = [], []
    for angles in product(*(np.linspace(a, b, 7) for a, b in zip(lower, upper))):
        data.qpos[addresses] = angles
        mujoco.mj_forward(model, data)
        target = (data.site_xpos[site] - data.xpos[base]).copy()
        offset = target - home
        # Quad targets lift clear of the nominal plane. Biped targets move
        # about its folded-front-leg posture, without prescribing ground contact.
        valid = 0.04 < np.linalg.norm(offset) < 0.15
        if not biped:
            valid = valid and 0.04 < offset[2] < 0.12 and abs(offset[1]) < 0.07
        if valid:
            targets.append(target)
            witnesses.append(angles)
    if not targets:
        raise ValueError("FR joint range produced no manipulation targets")
    return FootWorkspace(home, np.asarray(targets), np.asarray(witnesses))
