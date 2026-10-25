"""Source mirror permutations for the Isaac Lab / RSL-RL symmetry interface."""

import torch
from .source_config import Go2_Handstand_PPO_Yu as SourcePPO


def mirror(values, permutation):
    indices = torch.tensor([int(abs(p)) for p in permutation], device=values.device)
    signs = torch.tensor(
        [-1.0 if p < 0 else 1.0 for p in permutation], device=values.device
    )
    return values[..., indices] * signs


def source_symmetry(obs, actions, env=None, obs_type=None):
    """Concatenate original and mirrored samples; PPO data augmentation stays disabled."""
    if obs is not None:
        if isinstance(obs, torch.Tensor):
            if obs.shape[-1] != 45:
                raise ValueError(
                    "StandUp V2 mirror loss expects the 45-dimensional policy observations"
                )
            obs = torch.cat(
                (obs, mirror(obs, SourcePPO.algorithm.obs_permutation)), dim=0
            )
        else:
            # RSL-RL >= 3 uses a TensorDict; the mirror loss only evaluates the actor.
            mirrored = obs.clone()
            mirrored["actor"] = mirror(
                obs["actor"], SourcePPO.algorithm.obs_permutation
            )
            obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        actions = torch.cat(
            (actions, mirror(actions, SourcePPO.algorithm.act_permutation)), dim=0
        )
    return obs, actions
