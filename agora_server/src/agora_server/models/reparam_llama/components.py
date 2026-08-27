# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Reparametrized transformer components for swarm distribution.
# Key design: rcv (the shared orthonormal basis / rcv) is NOT stored per
# ReparametrizedLinear or per TransformerBlock. Instead, it is held once at
# the expert level (BaseExpert.rcv) and passed down through the call stack,
# keeping VRAM proportional to O(1) rcv copies per expert rather than O(N*2).

import torch
import torch.nn.functional as F

from torch import nn


def build_norm(norm_type: str, dim: int, eps: float = 1e-6, trainable=True):
    """Build the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to build.
            Supported types: layernorm, np_layernorm, rmsnorm, fused_rmsnorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.
        trainable (bool, optional): Whether the normalization parameters are trainable (used for `rmsnorm` only). Defaults to True.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, trainable=trainable)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """RMSNorm normalization layer.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6, trainable=True):
        """Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            trainable (bool, optional): Whether the scaling parameter is trainable. Default is True.
        """
        super().__init__()
        self.eps = eps
        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight if self.trainable else output

    def reset_parameters(self):
        if self.trainable:
            torch.nn.init.ones_(self.weight)  # type: ignore


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class ReparametrizedLinear(nn.Module):
    """Linear layer with output-subspace parameterization: y = x @ Z @ rcv^T.

    rcv (the orthonormal basis spanning the output subspace) is NOT stored here.
    It is passed as an argument at forward time so that a single expert-level
    rcv buffer can be shared across all layers, saving VRAM.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the low-rank factorization.
        bias: Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.Z = nn.Parameter(torch.empty(in_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, rcv: torch.Tensor) -> torch.Tensor:
        """y = x @ Z @ rcv^T where rcv is (out_features, rank)."""
        x_proj = x @ self.Z  # (..., rank)
        output = x_proj @ rcv.t()  # (..., out_features)
        if self.bias is not None:
            output = output + self.bias
        return output

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.Z, mean=0.0, std=init_std)


class ReparametrizedEmbedding(nn.Module):
    """Token embedding with subspace parameterization: output = embedding(ids, E) @ rcv^T.

    rcv is passed at forward time from the expert's shared rcv buffer.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Full hidden dimension (d).
        rank: Rank of the compressed embedding space.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.E = nn.Parameter(torch.empty(num_embeddings, rank))

    def forward(self, input_ids: torch.Tensor, rcv: torch.Tensor) -> torch.Tensor:
        """output = F.embedding(input_ids, E) @ rcv^T, giving (B, S, embedding_dim)."""
        embedding_low = F.embedding(input_ids, self.E)  # (B, S, rank)
        return embedding_low @ rcv.t()  # (B, S, embedding_dim)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.E, mean=0.0, std=init_std)


class Attention(nn.Module):
    """Multi-head attention with optional reparametrized output projection.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
    """

    def __init__(self, model_args, layer_id: int, use_compression: bool):
        """Initialize the attention module.

        Args:
            model_args (ModelArgs): Model configuration arguments.
            layer_id (int): Layer ID for the attention module.
        """
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.qk_norm = model_args.qk_norm
        self.layer_id = layer_id
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.hidden_dim // model_args.n_heads

        self.wq = nn.Linear(model_args.hidden_dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)

        if use_compression:
            rank = int(model_args.hidden_dim // model_args.compression_rate)
            self.wo = ReparametrizedLinear(model_args.n_heads * self.head_dim, model_args.hidden_dim, rank)
            self.is_reparametrized = True
        else:
            self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.hidden_dim, bias=False)
            self.is_reparametrized = False

        self.hidden_dim = model_args.hidden_dim
        self.attn_proj = model_args.attn_proj
        if self.attn_proj:
            self.register_buffer("Rk", torch.empty(0), persistent=False)
            self.register_buffer("Rq", torch.empty(0), persistent=False)

        # QK-norm
        if self.qk_norm:
            self.q_norm = build_norm(
                model_args.norm_type,
                self.head_dim,
                eps=model_args.norm_eps,
                trainable=model_args.trainable_rmsnorm,
            )
            self.k_norm = build_norm(
                model_args.norm_type,
                self.head_dim,
                eps=model_args.norm_eps,
                trainable=model_args.trainable_rmsnorm,
            )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)

        if self.is_reparametrized:
            self.wo.init_weights(init_std)
        else:
            nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.attn_proj:
            rng = torch.Generator()
            rng.manual_seed(1337 + self.layer_id)

            Rk = torch.randn(
                (self.n_kv_heads * self.head_dim, int(self.n_kv_heads * self.head_dim / 4)),
                dtype=torch.float32,
                generator=rng,
            )
            self.Rk, _ = torch.linalg.qr(Rk, mode="reduced")

            Rq = torch.randn((self.hidden_dim, int(self.hidden_dim / 4)), dtype=torch.float32, generator=rng)
            self.Rq, _ = torch.linalg.qr(Rq, mode="reduced")

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, rcv: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            rcv (torch.Tensor): Compression Matrix

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.attn_proj:
            self.Rk = self.Rk.to(xq.device)
            self.Rq = self.Rq.to(xq.device)
            xq = xq @ self.Rq[None, :, :] @ self.Rq[None, :, :].transpose(1, 2)
            xk = xk @ self.Rk[None, :, :] @ self.Rk[None, :, :].transpose(1, 2)
            xv = xv @ self.Rk[None, :, :] @ self.Rk[None, :, :].transpose(1, 2)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Normalize across the head dimension (last dimension)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)

        if self.is_reparametrized:
            return self.wo(output, rcv)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feedforward with optional reparametrized output projection.
    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        use_compression: bool = False,
        compression_rate: int = 1,
    ):
        """Initialize feed forward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
            use_compression (bool): Flag for turning on compression
            compression_rate (int): Compression rate
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        if use_compression:
            rank = int(dim // compression_rate)
            self.w2 = ReparametrizedLinear(hidden_dim, dim, rank)
            self.is_reparametrized = True
        else:
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.is_reparametrized = False

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=0.02)
        if self.is_reparametrized:
            self.w2.init_weights(init_std)
        else:
            nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, rcv: torch.Tensor | None = None) -> torch.Tensor:
        z = F.silu(self.w1(x)) * self.w3(x)
        if self.is_reparametrized:
            return self.w2(z, rcv)
        return self.w2(z)


class TransformerBlock(nn.Module):
    """Transformer block that threads rcv down to reparametrized sublayers.
    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(self, layer_id: int, model_args, use_compression: bool = True):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.hidden_dim
        self.norm_reorder = model_args.norm_reorder
        self.trainable_rmsnorm = model_args.trainable_rmsnorm
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention = Attention(model_args, layer_id, use_compression)
        self.feed_forward = FeedForward(
            dim=model_args.hidden_dim,
            hidden_dim=4 * model_args.hidden_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            use_compression=use_compression,
            compression_rate=model_args.compression_rate,
        )
        self.attention_norm = build_norm(
            model_args.norm_type,
            dim=model_args.hidden_dim,
            eps=model_args.norm_eps,
            trainable=self.trainable_rmsnorm,
        )
        self.ffn_norm = build_norm(
            model_args.norm_type,
            dim=model_args.hidden_dim,
            eps=model_args.norm_eps,
            trainable=self.trainable_rmsnorm,
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        elif model_args.constant_init:
            self.weight_init_std = 0.02
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, rcv: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            rcv (torch.Tensor): Shared orthonormal basis (hidden_dim, rank) from the expert's rcv.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        if self.norm_reorder:
            pre_attention = self.attention_norm(self.attention(x, freqs_cis, rcv))
        else:
            pre_attention = self.attention(self.attention_norm(x), freqs_cis, rcv)

        h = x + pre_attention

        if self.norm_reorder:
            ffn_out = self.ffn_norm(self.feed_forward(h, rcv))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h), rcv)

        return h + ffn_out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)
