"""
nano-Llama 3.1
Simpler version you can just forward on 1 GPU, without torchrun.
Changes:
- replace ColumnParallelLinear -> Linear
- replace RowParallelLinear -> Linear
- replace VocabParallelEmbedding -> Embedding

Run example:

python llama31.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
"""

import os
import glob
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# -----------------------------------------------------------------------------
# ModelArgs

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 6
    n_heads: int = 8 
    n_kv_heads: Optional[int] = None
    in_size: int = 3 
    out_size: int = 2 
    multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # use flash attention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

@dataclass
class ModelArgsVAE:
    dim: int = 512 
    n_layers: int = 4
    n_heads: int = 4  
    n_kv_heads: Optional[int] = None
    in_size: int = 3 
    out_size: int = 2 
    multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # use flash attention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0
@dataclass
class ModelArgsV3:
    dim: int = 256  
    n_layers: int = 2
    n_heads: int = 4  
    n_kv_heads: Optional[int] = None
    in_size: int = 3 
    out_size: int = 2 
    multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # use flash attention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


# -----------------------------------------------------------------------------
# Transformer

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

def apply_rotary_emb(x, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = args.flash # use flash attention?
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1 # AK: model parallel size is 1 for 1 GPU
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output projection
        proj = self.wo(output)
        return proj

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # hidden dim gymnastics that Meta simplified only later
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.in_size = params.in_size 
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Linear(params.in_size, params.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.out_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        # for use during inference
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward(self, inputs: torch.Tensor):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        _bsz, seqlen, nchannel = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits



class TransformerVAE(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.in_size = params.in_size 
        self.n_layers = params.n_layers
    
        self.tok_embeddings = nn.Linear(params.in_size, params.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.out_size, bias=False)
        self.logvar = nn.parameter.Parameter(torch.zeros([1, 48, 3]))
        self.mu = nn.parameter.Parameter(torch.zeros([1, 48, 3]))
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        # for use during inference
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward(self, x: torch.Tensor):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        x = torch.cat([x, x, x], dim=2)
        std = torch.exp(self.logvar * 0.5) 
        eps = torch.randn([x.shape[0], 48, x.shape[2]], device=x.device, dtype=x.dtype)
        s = self.mu + eps * std
        inputs = torch.cat([x, s], dim=1)
        _bsz, seqlen, nchannel = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits, self.mu, self.logvar 
 


class TransformerVAEwithTaskID(nn.Module):
    def __init__(self):
        super().__init__()
        params = ModelArgs()
        self.params = params 
        self.in_size = params.in_size 
        self.n_layers = params.n_layers
    
        self.tok_embeddings = nn.Linear(1, params.dim-64)
        self.task_embeddings = nn.Embedding(128, 64)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.out_size, bias=False)
        self.logvar = nn.parameter.Parameter(torch.zeros([1, 48, 1]))
        self.mu = nn.parameter.Parameter(torch.zeros([1, 48, 1]))
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )


    def forward(self, x: torch.Tensor):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        #x = torch.cat([x, x, x], dim=2)
        x = x.unsqueeze(2) 
        std = torch.exp(self.logvar * 0.5) 
        eps = torch.randn([x.shape[0], 48, x.shape[2]], device=x.device, dtype=x.dtype)

        s = self.mu + eps * std
        #inputs = torch.cat([x, s], dim=1)
        #_bsz, seqlen, nchannel = inputs.shape
        #h = self.tok_embeddings(inputs)
        h1a = self.tok_embeddings(x)
        tid = torch.ones([x.shape[0], x.shape[1]], dtype=torch.long, device=x.device) * 0 
        h1b = self.task_embeddings(tid)
        h1 = torch.cat([h1a, h1b], dim=2)

        h2a = self.tok_embeddings(s)
        tid = torch.ones([x.shape[0], 48], dtype=torch.long, device=x.device) * 1
        h2b = self.task_embeddings(tid)
        h2 = torch.cat([h2a, h2b], dim=2) 
        h = torch.cat([h1, h2], dim=1)

        _bsz, seqlen, nchannel = h.shape

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        logit1, logit2 = output[:, -48:, 0], output[:, -48:, 1]
        y1 = logit1.sigmoid() * 6 
        y2 = logit2.sigmoid()
        return y1, y2, self.mu, self.logvar 
 


class TransformerVAEwithTaskIDInfer(nn.Module):
    def __init__(self):
        super().__init__()
        params = ModelArgs()
        self.params = params 
        self.in_size = params.in_size 
        self.n_layers = params.n_layers
    
        self.tok_embeddings = nn.Linear(1, params.dim-64)
        self.task_embeddings = nn.Embedding(128, 64)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.out_size, bias=False)
        self.logvar = nn.parameter.Parameter(torch.zeros([1, 48, 1]))
        self.mu = nn.parameter.Parameter(torch.zeros([1, 48, 1]))
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )


    def forward(self, x: torch.Tensor):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        #x = torch.cat([x, x, x], dim=2)
        x = x.unsqueeze(0)
        x = x.unsqueeze(2) 
        std = torch.exp(self.logvar * 0.5) 
        eps = torch.randn([x.shape[0], 48, x.shape[2]], device=x.device, dtype=x.dtype)

        s = self.mu + eps * std
        #inputs = torch.cat([x, s], dim=1)
        #_bsz, seqlen, nchannel = inputs.shape
        #h = self.tok_embeddings(inputs)
        h1a = self.tok_embeddings(x)
        tid = torch.ones([x.shape[0], x.shape[1]], dtype=torch.long, device=x.device) * 0 
        h1b = self.task_embeddings(tid)
        h1 = torch.cat([h1a, h1b], dim=2)

        h2a = self.tok_embeddings(s)
        tid = torch.ones([x.shape[0], 48], dtype=torch.long, device=x.device) * 1
        h2b = self.task_embeddings(tid)
        h2 = torch.cat([h2a, h2b], dim=2) 
        h = torch.cat([h1, h2], dim=1)

        _bsz, seqlen, nchannel = h.shape

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        logit1, logit2 = output[:, -48:, 0], output[:, -48:, 1]
        y1 = logit1.sigmoid() * 6 
        y2 = logit2.sigmoid()
        return y1, y2 
 


def main():
    model_args = ModelArgs() 
    model = Transformer(model_args)
    x = torch.randn(32, 1024+48, 3)
    y = model(x)
    print(y.shape)
    
if __name__ == "__main__":
    main()