"""Native RSL-RL 5 PPO with the source AMP discriminator objective."""

from pathlib import Path
import torch
from rsl_rl.algorithms import PPO
from .features import JOINT_NAMES, terminal_next_observation
from .motion_loader import AMPLoader
from .discriminator import Discriminator
from .replay import ReplayBuffer

DATASETS = {
    **{f"kine2go_ai4_dog_walk_{i:02}": 1.0 for i in (0, 1, 2, 3, 4, 6)},
    **{
        f"kine2go_ai4_dog_synth_{shape}_walk": 0.5
        for shape in ("circle", "eight", "ellipse", "square")
    },
}


def load_demonstrations(device, dt):
    path = Path(__file__).resolve().parents[3] / "assets/motions/go2/amp"
    return AMPLoader(device, path, DATASETS, dt, 1, list(JOINT_NAMES))


def mix_rewards(task, style, style_weight=0.4):
    return (1 - style_weight) * task + style_weight * style


class AMPPPO(PPO):
    def __init__(
        self, *args, style_weight=0.4, amp_replay_buffer_size=100000, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.style_weight = style_weight
        self.discriminator = Discriminator(
            60, [256, 128], 1.0, device=self.device, empirical_normalization=True
        )
        self.amp_storage = ReplayBuffer(30, amp_replay_buffer_size, self.device)
        self.optimizer.add_param_group(
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3}
        )
        self.optimizer.add_param_group(
            {"params": self.discriminator.linear.parameters(), "weight_decay": 0.1}
        )
        # The pre-step hook adds independent AMP gradients after PPO backward and
        # before the same Adam step, preserving per-minibatch LR and weight decay.
        self.optimizer.register_step_pre_hook(self._amp_backward)
        self.optimizer.register_step_post_hook(self._amp_normalize)

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg.amp_data = load_demonstrations(device, env.unwrapped.step_dt)
        return alg

    def act(self, obs):
        self.amp_obs = obs["amp"].detach().clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        nxt = terminal_next_observation(obs["amp"], dones, extras)
        style = self.discriminator.predict_reward(self.amp_obs, nxt)
        self.amp_storage.insert(self.amp_obs, nxt)
        super().process_env_step(
            obs, mix_rewards(rewards, style, self.style_weight), dones, extras
        )

    def _amp_backward(self, optimizer, args, kwargs):
        policy = next(self._policy_batches)
        expert = next(self._expert_batches)
        count = len(policy[0])
        logits = self.discriminator(
            torch.cat((torch.cat(policy, -1), torch.cat(expert, -1)))
        )
        loss, penalty = self.discriminator.compute_loss(
            logits[:count], logits[count:], expert, policy, lambda_=10
        )
        (loss + penalty).backward()
        if self.is_multi_gpu:
            for param in self.discriminator.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad)
                    param.grad.div_(self.gpu_world_size)
        self._amp_loss += loss.item()
        self._amp_penalty += penalty.item()
        self._raw_batches = (*expert, *policy)

    def _amp_normalize(self, optimizer, args, kwargs):
        self.discriminator.update_normalization(*self._raw_batches)

    def update(self):
        count = self.num_learning_epochs * self.num_mini_batches
        size = (
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches
        )
        self._policy_batches = self.amp_storage.feed_forward_generator(
            count, size, allow_replacement=True
        )
        self._expert_batches = self.amp_data.feed_forward_generator(count, size)
        self._amp_loss = self._amp_penalty = 0.0
        stats = super().update()
        stats.update(
            amp=self._amp_loss / count, amp_gradient_penalty=self._amp_penalty / count
        )
        return stats

    def save(self):
        state = super().save()
        state["amp"] = {
            "discriminator": self.discriminator.state_dict(),
            "style_weight": self.style_weight,
        }
        state["amp_replay"] = {
            "states": self.amp_storage.states,
            "next_states": self.amp_storage.next_states,
            "step": self.amp_storage.step,
            "num_samples": self.amp_storage.num_samples,
        }
        return state

    def load(self, state, load_cfg=None, strict=True):
        result = super().load(state, load_cfg, strict)
        self.discriminator.load_state_dict(state["amp"]["discriminator"], strict=strict)
        self.style_weight = state["amp"]["style_weight"]
        if load_cfg is None or load_cfg.get("optimizer", False):
            replay = state["amp_replay"]
            self.amp_storage.states.copy_(replay["states"])
            self.amp_storage.next_states.copy_(replay["next_states"])
            self.amp_storage.step = replay["step"]
            self.amp_storage.num_samples = replay["num_samples"]
        return result

    def train_mode(self):
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        self.discriminator.eval()

    def broadcast_parameters(self):
        super().broadcast_parameters()
        for param in self.discriminator.parameters():
            torch.distributed.broadcast(param.data, src=0)
