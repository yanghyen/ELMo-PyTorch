"""
Character / token biLM (skip + stacked LSTM) — **same architecture** as the
``SimpleLanguageModel`` inner class in ``training.py``.

This module exists only so downstream (e.g. SST-2) can ``import`` the class for
checkpoint loading **without** changing ``training.py``. If you edit the inner
class in ``training.py``, update this file to match (or load checkpoints will
break).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_activation_fn(name):
    name = (name or "relu").lower()
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return F.gelu
    logger.warning("Unknown char_cnn.activation=%s, fallback to relu", name)
    return F.relu


class SimpleLanguageModel(nn.Module):
    """
    Simplified bidirectional LM with character CNN, optional highway, skip to LSTM.

    ``forward`` returns a dict with ``lm_embeddings`` of shape
    ``(batch, 2, seq_len, 2 * hidden_dim)``: two stacked views (skip-duplicate
    and concatenated forward/backward top states).
    """

    def __init__(self, options, vocab_size):
        super().__init__()
        self.options = options

        dropout = float(options.get("dropout", 0.1))
        lstm_cfg = options.get("lstm", {})
        self.use_skip_connections = bool(
            lstm_cfg.get("use_skip_connections", True)
        )
        self.proj_clip = lstm_cfg.get("proj_clip")
        if self.proj_clip is not None:
            self.proj_clip = float(self.proj_clip)

        if options.get("char_cnn"):
            char_cfg = options["char_cnn"]
            char_vocab_size = char_cfg["n_characters"]
            char_embed_dim = char_cfg["embedding"]["dim"]
            self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
            self.char_activation = _get_activation_fn(
                char_cfg.get("activation", "relu")
            )

            filters = char_cfg.get("filters") or [[3, 256]]
            self.char_cnn_layers = nn.ModuleList(
                nn.Conv1d(char_embed_dim, int(num), int(width), padding=int(width) // 2)
                for width, num in filters
            )
            total_filters = sum(int(num) for _, num in filters)

            n_highway = int(char_cfg.get("n_highway", 0))
            self.highway_layers = nn.ModuleList(
                nn.Linear(total_filters, total_filters * 2)
                for _ in range(n_highway)
            )

            proj_dim = int(lstm_cfg["projection_dim"])
            self.projection = nn.Linear(total_filters, proj_dim)
            input_dim = proj_dim
        else:
            embed_dim = options.get("embedding_dim", 512)
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            input_dim = embed_dim

        hidden_dim = int(lstm_cfg["dim"])
        self.hidden_dim = hidden_dim
        n_layers = int(lstm_cfg["n_layers"])
        lstm_dropout = dropout if n_layers > 1 else 0.0

        self.forward_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.backward_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.skip_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        )

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        if self.options.get("char_cnn"):
            return self._forward_char(inputs)
        return self._forward_token(inputs)

    def _forward_char(self, char_ids):
        batch_size, seq_len, max_chars = char_ids.shape
        mask = (char_ids.sum(dim=-1) > 0).float()

        char_embeds = self.char_embedding(char_ids)
        char_embeds = char_embeds.transpose(2, 3)

        x = char_embeds.view(-1, char_embeds.size(2), char_embeds.size(3))
        conv_outputs = []
        for conv in self.char_cnn_layers:
            y = conv(x)
            y = self.char_activation(y)
            y = F.max_pool1d(y, kernel_size=y.size(2)).squeeze(2)
            conv_outputs.append(y)
        x = torch.cat(conv_outputs, dim=1).view(batch_size, seq_len, -1)

        for highway in self.highway_layers:
            gate_and_transform = highway(x)
            transform, gate = torch.chunk(gate_and_transform, 2, dim=-1)
            transform = F.relu(transform)
            gate = torch.sigmoid(gate)
            x = gate * transform + (1.0 - gate) * x

        token_embeddings = self.projection(x)

        return self._run_lstm(token_embeddings, mask)

    def _forward_token(self, token_ids):
        mask = (token_ids > 0).float()
        token_embeddings = self.token_embedding(token_ids)
        return self._run_lstm(token_embeddings, mask)

    def _run_lstm(self, embeddings, mask):
        embeddings = self.input_dropout(embeddings)
        forward_out, _ = self.forward_lstm(embeddings)

        backward_in = torch.flip(embeddings, dims=[1])
        backward_out, _ = self.backward_lstm(backward_in)
        backward_out = torch.flip(backward_out, dims=[1])

        if self.use_skip_connections:
            skip = embeddings if self.skip_proj is None else self.skip_proj(embeddings)
            forward_out = forward_out + skip
            backward_out = backward_out + skip
        if self.proj_clip is not None:
            forward_out = torch.clamp(
                forward_out, min=-self.proj_clip, max=self.proj_clip
            )
            backward_out = torch.clamp(
                backward_out, min=-self.proj_clip, max=self.proj_clip
            )

        bi = torch.cat([forward_out, backward_out], dim=-1)
        skip_base = embeddings if self.skip_proj is None else self.skip_proj(embeddings)
        emb2 = torch.cat([skip_base, skip_base], dim=-1)
        lm_embeddings = torch.stack([emb2, bi], dim=1)

        return {
            "lm_embeddings": lm_embeddings,
            "mask": mask,
            "forward_output": forward_out,
            "backward_output": backward_out,
        }
