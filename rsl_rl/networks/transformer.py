# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]  # type: ignore[index]
        return self.dropout(x)


class TransformerMemory(nn.Module):
    """Transformer-based temporal memory module, designed as a drop-in replacement for the
    RNN-based ``Memory`` module.

    It maintains a sliding-window context buffer during inference and processes full
    sequences during training (batch mode).

    Args:
        input_size: Dimension of per-step observations.
        hidden_size: Internal dimension (d_model) of the Transformer.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        context_len: Maximum context length (sliding window size) for inference.
        dropout: Dropout probability applied inside Transformer layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        context_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_len = context_len

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=context_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # we use (seq, batch, feature) convention
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context buffer for inference (sliding window): (context_len, num_envs, hidden_size)
        self.context_buffer: torch.Tensor | None = None

    def forward(self, input: torch.Tensor, masks: torch.Tensor | None = None, hidden_states=None) -> torch.Tensor:
        """Forward pass of the Transformer memory.

        Args:
            input: Observation tensor.
                - Batch (training) mode: shape ``(seq_len, batch, input_size)``.
                - Inference mode: shape ``(batch, input_size)`` (single step).
            masks: Optional padding masks for training. Shape ``(seq_len, batch)``
                with True for valid time-steps. If provided, the module runs in batch
                mode.
            hidden_states: Unused, kept for API compatibility with ``Memory``.

        Returns:
            Transformer output tensor.
                - Batch mode: ``(seq_len, batch, hidden_size)``
                - Inference mode: ``(1, batch, hidden_size)``
        """
        batch_mode = masks is not None
        if batch_mode:
            return self._forward_batch(input, masks)
        else:
            return self._forward_inference(input)

    def _forward_batch(self, input: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Training forward: full sequence processing.

        Args:
            input: (seq_len, batch, input_size)
            masks: (seq_len, batch) – True for valid positions
        """
        seq_len = input.size(0)

        # Project input to hidden dimension
        x = self.input_proj(input)  # (seq_len, batch, hidden_size)
        x = self.pos_encoder(x)

        # Build causal mask (upper-triangular = -inf) to enforce auto-regressive attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input.device)

        # Build key padding mask from the trajectory masks: (batch, seq_len)
        # masks has True for *valid* timesteps; key_padding_mask needs True for *padded* positions
        key_padding_mask = ~masks.permute(1, 0).bool()  # (batch, seq_len)

        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return out  # (seq_len, batch, hidden_size)

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """Inference forward: single step with sliding-window context.

        Args:
            input: (batch, input_size) – single time-step observation
        """
        batch_size = input.size(0)

        # Project the new observation
        x = self.input_proj(input).unsqueeze(0)  # (1, batch, hidden_size)

        # Append to context buffer
        if self.context_buffer is None or self.context_buffer.size(1) != batch_size:
            self.context_buffer = x
        else:
            self.context_buffer = torch.cat([self.context_buffer, x], dim=0)
            # Trim to context window
            if self.context_buffer.size(0) > self.context_len:
                self.context_buffer = self.context_buffer[-self.context_len :]

        # Apply positional encoding
        ctx = self.pos_encoder(self.context_buffer)

        # Causal mask
        ctx_len = ctx.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(ctx_len, device=input.device)

        out = self.transformer_encoder(ctx, mask=causal_mask)
        # Return only the last time-step output
        return out[-1:, :, :]  # (1, batch, hidden_size)

    def reset(self, dones: torch.Tensor | None = None, hidden_states=None):
        """Reset the context buffer for done environments.

        Args:
            dones: Boolean/int tensor of shape ``(num_envs,)`` indicating which environments to
                reset. If ``None``, all context buffers are cleared.
            hidden_states: Unused, kept for API compatibility with ``Memory``.
        """
        if dones is None:
            self.context_buffer = None
        elif self.context_buffer is not None:
            # Zero out context for finished environments
            self.context_buffer[:, dones == 1, :] = 0.0

    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        """Detach the context buffer from the computational graph.

        Args:
            dones: If provided, only detach the context of the specified environments.
        """
        if self.context_buffer is not None:
            if dones is None:
                self.context_buffer = self.context_buffer.detach()
            else:
                self.context_buffer[:, dones == 1, :] = self.context_buffer[:, dones == 1, :].detach()

    @property
    def hidden_states(self):
        """Property for API compatibility with ``Memory``. Returns the context buffer."""
        return self.context_buffer
