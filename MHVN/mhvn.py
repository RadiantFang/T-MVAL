import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from . import rotary

from .fused_add_dropout_scale import (
     bias_dropout_add_scale_fused_train,
     bias_dropout_add_scale_fused_inference,
     modulate_fused,
)
#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6,
                                                                                                                 dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)

        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operations
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x,
                                  self.dropout)
        return x


#################################################################################
#                         Modified section starts here                         #
#################################################################################

class MLPHead(nn.Module):
    """
    A configurable MLP regression head for ensemble learning.
    The input size, hidden width, depth, and activation function can be specified.
    """

    def __init__(self, input_size, hidden_units, depth, activation_fn_str, dropout_prob=0.1):
        super().__init__()

        # Activation function mapping
        activations = {
            'tanh': nn.Tanh(),
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'SiLU': nn.SiLU()
        }
        activation_fn = activations[activation_fn_str]

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_units))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        # The total depth includes the input layer and hidden layers, but not the final output layer.
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_units, 1))

        self.prediction_head = nn.Sequential(*layers)

    def forward(self, x):
        # Average-pool across the sequence dimension to obtain a sequence-level representation.
        x_pooled = torch.mean(x, dim=1)
        activity_value = self.prediction_head(x_pooled)
        return activity_value.squeeze(-1)


class EnsembleModel(nn.Module):
    """
    An ensemble model with a shared backbone.
    It contains one backbone network and 36 different MLP heads.
    """

    def __init__(self):
        super().__init__()
        # --- 1. Define the shared backbone ---
        cond_dim = 128
        n_blocks = 12
        n_heads = 12
        hidden_size = 768
        dropout = 0.1

        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim,
                      dropout=dropout) for _ in range(n_blocks)
        ])
        self.linear = nn.Conv1d(4, hidden_size, kernel_size=9, padding=4)

        # --- 2. Define multiple MLP heads ---
        self.mlp_heads = nn.ModuleList()

        # Define the hyperparameter space
        depths = [1, 2, 4]
        hidden_units_list = [384, 768, 1536]
        activation_fns = ['GELU', 'ReLU', 'SiLU']

        # Use itertools.product to generate all hyperparameter combinations.
        for depth, hidden_units, act_fn in itertools.product(depths, hidden_units_list, activation_fns):
            head = MLPHead(
                input_size=hidden_size,
                hidden_units=hidden_units,
                depth=depth,
                activation_fn_str=act_fn
            )
            self.mlp_heads.append(head)

        print(f"Successfully created the ensemble model with 1 shared backbone and {len(self.mlp_heads)} MLP heads.")

    def forward(self, indices, sigma):
        # --- 1. Forward pass through the backbone (computed once) ---
        x = self.linear(indices.permute(0, 2, 1)).permute(0, 2, 1)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

        # Shared features produced by the backbone
        shared_features = x

        # --- 2. Feed shared features into all MLP heads ---
        predictions = []
        for head in self.mlp_heads:
            predictions.append(head(shared_features))

        # Return a list containing 36 prediction tensors.
        return predictions


class EnsembleModelMotified(nn.Module):
    """
    Final ensemble model V2.
    It hardcodes 7 validated high-performing MLP heads.
    This version returns a list of 7 prediction tensors to stay compatible with the existing training code.
    """

    def __init__(self):
        super().__init__()
        # --- 1. Define the shared backbone ---
        cond_dim = 128
        n_blocks = 12
        n_heads = 12
        hidden_size = 768  # This value must match the MLPHead input size.
        dropout = 0.1

        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim,
                      dropout=dropout) for _ in range(n_blocks)
        ])
        self.linear = nn.Conv1d(4, hidden_size, kernel_size=9, padding=4)

        # --- 2. Define the 7 fixed best-performing MLP heads ---
        self.mlp_heads = nn.ModuleList()

        # The selected top-7 elite head configurations
        optimal_head_configs = [
            {'depth': 2, 'hidden_units': 1536, 'activation_fn_str': 'ReLU'},  # Index 16
            {'depth': 2, 'hidden_units': 768, 'activation_fn_str': 'ReLU'},  # Index 13
            {'depth': 2, 'hidden_units': 768, 'activation_fn_str': 'SiLU'},  # Index 14
            {'depth': 2, 'hidden_units': 384, 'activation_fn_str': 'SiLU'},  # Index 11
            {'depth': 4, 'hidden_units': 768, 'activation_fn_str': 'GELU'},  # Index 21
            {'depth': 4, 'hidden_units': 384, 'activation_fn_str': 'GELU'},  # Index 18
            {'depth': 2, 'hidden_units': 384, 'activation_fn_str': 'ReLU'}  # Index 10
        ]

        # Instantiate the 7 fixed elite heads.
        for config in optimal_head_configs:
            head = MLPHead(
                input_size=hidden_size,
                hidden_units=config['hidden_units'],
                depth=config['depth'],
                activation_fn_str=config['activation_fn_str']
            )
            self.mlp_heads.append(head)

    def forward(self, indices, sigma):
        # --- 1. Forward pass through the backbone ---
        x = self.linear(indices.permute(0, 2, 1)).permute(0, 2, 1)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

        shared_features = x

        # --- 2. Feed shared features into all fixed best-performing MLP heads ---
        predictions = [head(shared_features) for head in self.mlp_heads]

        return predictions
