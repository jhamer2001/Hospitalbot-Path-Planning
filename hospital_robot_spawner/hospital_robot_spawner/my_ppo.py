import numpy as np
import torch as th
import warnings
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union


SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        # basic AC network
        "MlpPolicy": ActorCriticPolicy,
        # AC network with image processing
        "CnnPolicy": ActorCriticCnnPolicy,
        # AC network with multiple different inputs
        # in this case it would involve lidar
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            # the environment can be of type string or an
            # instance of a gym environment
            env: Union[GymEnv, str],
            # the schedule can dynamically change the learning rate but is
            # default at 0.0003
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            # number of epochs for optimization
            n_epochs: int = 10,
            # discount factor
            gamma: float = 0.99,
            # factor for bias vs variance
            gae_lambda: float = 0.95,
            # clip range for Policy network
            clip_range: Union[float, Schedule] = 0.2,
            # turns on and off advantage normalization
            normalize_advantage: bool = True,
            # entropy coef for loss calculations
            ent_coef: float = 0.0,
            # value function coeff for loss calculations
            vf_coef: float = 0.5,
            # maximum value for gradient clipping
            max_grad_norm: float = 0.5,
            # replay buffer setup
            rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            # limiting KL divergence
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:
            assert (
                batch_size > 1
            )

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            untruncated_batches = buffer_size // batch_size
            if buffer_size % buffer_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for clipping
        # converts self.clip_range into a scheduling function
        # changes the clip range based on the current progress
        self.clip_range = get_schedule_fn(self.clip_range)

    def train(self) -> None:
        """
        Updates policies using a rollout buffer built into stable baselines3
        """
        # Turn on training mode
        self.policy.set_training_mode(True)
        # Update the optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Get the current clip range since it can be dynamically updated
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        # policy gradient losses and value losses
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs

        for epoch in range(self.n_epochs):
            # KL div added as a last ditch effort more info below
            approx_kl_divs = []

            # rollout buffer is the same thing as a replay buffer
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions
                print(actions)
                
                # check if we are in discrete time
                # flatten here makes the data one dimension
                if isinstance(self.action_space, spaces.Discrete):
                    actions = data.actions.long().flatten()

                # evaluate_actions is built into the ActorCriticPolicy setup
                # in stable baselines3
                # this actually evaluates the actions according to the current
                # policy (critic network)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations, actions)
                # makes the values one dimension for processing later
                values = values.flatten()

                # Normalize advantage for stability
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    # standard formula for normalization to help with stability
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8)

                # find the ratio between the old and new policy for J
                ratio = th.exp(log_prob - data.old_log_prob)

                # clipped loss - this is the J function broken down
                pl1 = advantages * ratio
                pl2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range)
                # this mean ensures you get a single value that represents the
                # average policy loss
                policy_loss = th.min(pl1, pl2).mean()

                # Logging - I never think to do this so thank you to the
                # reference code
                pg_losses.append(policy_loss.item())
                # lets you know how often you have to clip the values so you
                # don't take too large a step
                clip_fraction = th.mean((th.abs(ratio-1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # assign the actor network values
                values_pred = values

                # find the td error - using gae_lambda in this algorithm
                value_loss = F.mse_loss(data.returns, values_pred)
                value_losses.append(value_loss.item())

                """
                LOOK INTO THIS MORE FOR THE PAPER AND EXPLANATION IF THEY ASK
                """
                # Explore more if there is entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # this algorithm implementation uses KL divergence as a last ditch effort
                # if you get to this point it's basically TRPO
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            # count the updates
            self._n_updates += 1
            if not continue_training:
                break

        # Explained variance is a metric used to measure how well the value
        # function accounts for the variance in returns.
        # We want low variance, that is the whole point of PPO
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
            )

        # More logs I never think about
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(
                self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        
