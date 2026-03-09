# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import TransformerMemory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticTransformer(ActorCritic):
    """Actor-Critic architecture with Transformer-based temporal memory.

    This class follows the same design pattern as :class:`ActorCriticRecurrent`, but
    replaces the RNN (LSTM/GRU) backbone with a Transformer encoder that uses a
    sliding-window context buffer during inference.

    Args:
        num_actor_obs: Dimension of the actor observation vector.
        num_critic_obs: Dimension of the critic observation vector.
        num_actions: Dimension of the action vector.
        actor_hidden_dims: Hidden layer sizes for the actor MLP head.
        critic_hidden_dims: Hidden layer sizes for the critic MLP head.
        activation: Activation function name for MLP layers.
        transformer_hidden_dim: Internal dimension (d_model) of the Transformer.
        transformer_num_layers: Number of Transformer encoder layers.
        transformer_num_heads: Number of attention heads.
        transformer_context_len: Maximum context length for the sliding-window buffer.
        transformer_dropout: Dropout probability inside Transformer layers.
        init_noise_std: Initial standard deviation of action noise.
        noise_std_type: Type of noise parameterization ('scalar' or 'log').
    """

    is_recurrent = True  # shares the same rollout storage path as recurrent models

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        transformer_hidden_dim=128,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_context_len=64,
        transformer_dropout=0.0,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        # The MLP heads receive the Transformer hidden_dim as input
        super().__init__(
            num_actor_obs=transformer_hidden_dim,
            num_critic_obs=transformer_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        activation = resolve_nn_activation(activation)

        # Transformer memory modules (one for actor, one for critic)
        self.memory_a = TransformerMemory(
            input_size=num_actor_obs,
            hidden_size=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_len=transformer_context_len,
            dropout=transformer_dropout,
        )
        self.memory_c = TransformerMemory(
            input_size=num_critic_obs,
            hidden_size=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_len=transformer_context_len,
            dropout=transformer_dropout,
        )

        print(f"Actor Transformer: {self.memory_a}")
        print(f"Critic Transformer: {self.memory_c}")

    def reset(self, dones=None):
        """Reset the Transformer context buffers for done environments."""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """Compute actions from observations.

        Args:
            observations: Actor observations.
            masks: Trajectory masks for batch (training) mode.
            hidden_states: Unused, kept for API compatibility.

        Returns:
            Sampled actions.
        """
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        """Compute deterministic actions (mean) for inference.

        Args:
            observations: Actor observations.

        Returns:
            Deterministic actions.
        """
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Compute value estimates from critic observations.

        Args:
            critic_observations: Critic observations.
            masks: Trajectory masks for batch (training) mode.
            hidden_states: Unused, kept for API compatibility.

        Returns:
            Value estimates.
        """
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        """Return context buffers from both actor and critic Transformers.

        Returns:
            Tuple of (actor_context, critic_context).
        """
        return self.memory_a.hidden_states, self.memory_c.hidden_states
