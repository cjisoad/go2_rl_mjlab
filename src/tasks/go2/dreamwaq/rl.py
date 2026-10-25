# Native RSL-RL 5 DreamWaQ extension. PPO update adapted from BSD-3-Clause rsl_rl.
import copy
from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn


class DreamWaQActor(MLPModel):
    def _get_latent_dim(self):
        return self.obs_dim + 19

    def __init__(self, obs, *args, **kwargs):
        super().__init__(obs, *args, **kwargs)
        self.history_dim = obs["obs_history"].shape[-1]
        self.encoder = MLP(self.history_dim, 64, [128], "elu")
        self.mean_velocity = nn.Linear(64, 3)
        self.logvar_velocity = nn.Linear(64, 3)
        self.mean_latent = nn.Linear(64, 16)
        self.logvar_latent = nn.Linear(64, 16)
        self.decoder = MLP(19, self.obs_dim, [64, 128], "elu")
        self._sample_context = False

    def context(self, history, stochastic=False):
        encoded = self.encoder(history)
        mv, lv = (
            self.mean_velocity(encoded),
            self.logvar_velocity(encoded).clamp(-10, 10),
        )
        mz, lz = self.mean_latent(encoded), self.logvar_latent(encoded).clamp(-10, 10)
        velocity = mv + torch.randn_like(mv) * torch.exp(0.5 * lv) if stochastic else mv
        latent = mz + torch.randn_like(mz) * torch.exp(0.5 * lz) if stochastic else mz
        return torch.cat((velocity, latent), -1), velocity, mz, lz

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        self._sample_context = stochastic_output
        return super().forward(obs, masks, hidden_state, stochastic_output)

    def get_latent(self, obs, masks=None, hidden_state=None):
        context = self.context(obs["obs_history"], self._sample_context)[0]
        return torch.cat((context, super().get_latent(obs, masks, hidden_state)), -1)

    def estimator_losses(self, obs, beta, num_mini_batches):
        code, velocity, mean, logvar = self.context(obs["obs_history"], True)
        # Source supervises sampled velocity, with a summed (not averaged) latent KL.
        return {
            "velocity": nn.functional.mse_loss(
                velocity, obs["privileged"][:, :3].detach()
            )
            / num_mini_batches,
            "reconstruction": nn.functional.mse_loss(
                self.decoder(code), obs["actor"].detach()
            )
            / num_mini_batches,
            "kl": beta
            * (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum())
            / num_mini_batches,
        }

    def as_onnx(self, verbose=False):
        return DreamWaQExport(self, verbose)

    def as_jit(self):
        return DreamWaQExport(self, False)


class DreamWaQExport(nn.Module):
    is_recurrent = False

    def __init__(self, actor, verbose):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = actor.obs_dim
        self.history_dim = actor.history_dim
        self.encoder = copy.deepcopy(actor.encoder)
        self.mean_velocity = copy.deepcopy(actor.mean_velocity)
        self.mean_latent = copy.deepcopy(actor.mean_latent)
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        self.output = (
            actor.distribution.as_deterministic_output_module()
            if actor.distribution
            else nn.Identity()
        )

    def forward(self, obs, obs_history):
        encoded = self.encoder(obs_history)
        code = torch.cat((self.mean_velocity(encoded), self.mean_latent(encoded)), -1)
        return self.output(self.mlp(torch.cat((code, self.obs_normalizer(obs)), -1)))

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.obs_dim), torch.zeros(1, self.history_dim))

    @property
    def input_names(self):
        return ["obs", "obs_history"]

    @property
    def output_names(self):
        return ["actions"]

    @torch.jit.export
    def reset(self):
        pass


class DreamWaQPPO(PPO):
    def __init__(self, *args, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def clip_gradients(self):
        # Source has one actor-critic module, including its context estimator.
        nn.utils.clip_grad_norm_(
            chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm
        )

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        estimator_means = {"velocity": 0.0, "reconstruction": 0.0, "kl": 0.0}
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(
                    num_aug, 1
                )
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            )  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            estimator_losses = self.actor.estimator_losses(
                batch.observations, self.beta, self.num_mini_batches
            )
            loss = loss + sum(estimator_losses.values())
            for key, value in estimator_losses.items():
                estimator_means[key] += value.item()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(
                        batch.observations[:original_batch_size]
                    )  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            self.clip_gradients()
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        loss_dict.update(
            {
                "estimator_" + key: value / num_updates
                for key, value in estimator_means.items()
            }
        )
        return loss_dict
