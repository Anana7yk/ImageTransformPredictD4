import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from omegaconf import DictConfig


# ---------- RoPE utilities ----------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key.
    Args:
        q, k: [B, T, n_embd] or [B, n_head, T, head_dim]
        cos, sin: [1, T, head_dim] or [T, head_dim]
    """
    while len(cos.shape) < len(q.shape):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_rotary_embeddings(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float32, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for RoPE for a given sequence length and base.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


# ---------- Self-Attention with configurable positional encoding ----------
class SelfAttention(nn.Module):
    """
    Causal self-attention (decoder-style) with built-in future masking.
    Supports two positional encoding modes:
        - "learned": uses trainable position embeddings (added to token embeddings)
        - "rope": uses Rotary Position Embeddings (applied inside attention)
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.bias = config.bias
        self.pos_encoding = getattr(config, "pos_encoding", "learned")
        self.rope_base = getattr(config, "rope_base", 10000.0)

        if self.pos_encoding == "rope":
            assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head for RoPE"
            self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
            self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
            self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
            self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        else:  # learned
            self.attn = nn.MultiheadAttention(
                embed_dim=self.n_embd,
                num_heads=self.n_head,
                dropout=self.dropout,
                bias=self.bias,
                batch_first=True,
            )

        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal (future) mask: upper triangle = -inf
        mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float('-inf')),
            diagonal=1
        )
        self.register_buffer("future_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings [B, T, n_embd].

        Returns:
            Output after causal self-attention [B, T, n_embd].
        """
        if self.pos_encoding == "rope":
            return self._forward_rope(x)
        else:
            return self._forward_learned(x)

    def _forward_learned(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        attn_mask = self.future_mask[:T, :T].to(x.device)
        y, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        y = self.resid_dropout(y)
        return y

    def _forward_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device

        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]

        # Reshape for multi-head: [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = get_rotary_embeddings(
            seq_len=T,
            dim=self.head_dim,
            device=device,
            dtype=x.dtype,
            base=self.rope_base
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        attn_mask = self.future_mask[:T, :T]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # we provide explicit mask
        )  # [B, n_head, T, head_dim]

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


# ---------- Feed-forward network with configurable activation ----------
class SwiGLU(nn.Module):
    """SwiGLU activation: silu(gate) * up."""
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """Feed-forward network with configurable activation (GELU or SwiGLU)."""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.activation = getattr(config, "activation", "gelu")  # "gelu" or "swiglu"
        self.n_embd = config.n_embd
        self.ffn_ratio = config.ffn_ratio
        self.bias = config.bias
        self.dropout = config.dropout

        if self.activation == "swiglu":
            hidden_dim = int(self.n_embd * self.ffn_ratio)
            self.swiglu = SwiGLU(self.n_embd, hidden_dim, bias=self.bias)
        else:  # gelu
            hidden_dim = int(self.n_embd * self.ffn_ratio)
            self.net = nn.Sequential(
                nn.Linear(self.n_embd, hidden_dim, bias=self.bias),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_embd, bias=self.bias),
            )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            x = self.swiglu(x)
        else:
            x = self.net(x)
        return self.dropout_layer(x)


# ---------- Cross-attention from token sequence to images context embeddings ----------
class CrossAttention(nn.Module):
    """
    Cross-attention from token sequence to images context embeddings.
    
    The image context can be:
        - A single fused embedding from two images: [B, 1, n_embd]
        - A sequence of embeddings from two images: [B, L, n_embd], where L >= 1
    
    Used to inject visual information into the token stream.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        images_embeddings: torch.Tensor,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [B, T, n_embd].
            images_embeddings: Images context embeddings [B, L, n_embd], where L >= 1.

        Returns:
            Output of cross-attention [B, T, n_embd].
        """
        k = v = images_embeddings  # [B, L, n_embd]
        if not need_weights:
            y, _ = self.attn(query=x, key=k, value=v, need_weights=False)
            y = self.resid_dropout(y)
            return y
        else:
            y, attn_weights = self.attn(query=x,
                key=k,
                value=v,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            y = self.resid_dropout(y)
            return y, attn_weights


# ---------- Transformer decoder block ----------
class DecoderBlock(nn.Module):
    """
    Transformer decoder block:
    - Causal self-attention over tokens
    - Cross-attention to image context embeddings
    - Feed-forward network
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        images_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [B, T, n_embd].
            images_embeddings: Images context embeddings [B, L, n_embd], L >= 1.

        Returns:
            Updated token embeddings [B, T, n_embd].
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), images_embeddings)
        x = x + self.mlp(self.ln_3(x))
        return x

    def forward_with_cross_attn(
        self,
        x: torch.Tensor,
        images_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that *always* returns cross-attention weights (per-head, full T x L).
        For analysis/visualization only.
        Returns:
            x_out: [B, T, n_embd]
            cross_attn_weights: [B, n_head, T, L]
        """
        x = x + self.attn(self.ln_1(x))
        cross_out, attn_weights = self.cross_attn(
            self.ln_2(x), images_embeddings, need_weights=True
        )
        x = x + cross_out
        x = x + self.mlp(self.ln_3(x))
        return x, attn_weights


# ---------- Main autoregressive transformer decoder ----------
class TransformDecoder(nn.Module):
    """
    Autoregressive transformer decoder for generating token sequences conditioned on image context.

    Supports flexible image conditioning:
        - A single fused embedding from two images: [B, 1, n_embd]
        - A sequence of embeddings from two separate images: [B, L, n_embd], L >= 1

    Also supports two positional encoding strategies:
        - "learned": trainable position embeddings (default, backward-compatible)
        - "rope": Rotary Position Embeddings (no static position embeddings)

    And two activation functions in the feed-forward network:
        - "gelu": standard GELU MLP (default)
        - "swiglu": SwiGLU activation (as in LLaMA, Mistral)

    Fully backward-compatible with older checkpoints when config matches.
    """
    def __init__(self, config: DictConfig):
        """
        Initialize the decoder from a configuration object.
        Args:
            config (DictConfig): Must contain:
                - n_embd: int
                - n_head: int
                - n_layer: int
                - max_seq_len: int
                - dropout: float
                - bias: bool
                - vocab_size: int
                - bos_token_id: int
                - eos_token_id: int
                - pad_token_id: int
                Optional:
                - pos_encoding: str ("learned" or "rope"), default "learned"
                - activation: str ("gelu" or "swiglu"), default "gelu"
                - rope_base: float, default 10000.0 (only used if pos_encoding="rope")
        """
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.pos_encoding = getattr(config, "pos_encoding", "learned")

        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)

        if self.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(self.max_seq_len, self.n_embd)
        else:
            self.position_embedding = None

        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd, elementwise_affine=config.bias)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        images_embeddings: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.

        Args:
            idx: Input token IDs [B, T].
            images_embeddings: Images context embeddings [B, L, n_embd], where L >= 1.
            targets: Optional target token IDs [B, T] for training.

        Returns:
            logits: Predicted token logits [B, T, vocab_size].
            loss: Cross-entropy loss (None if targets not provided).
                  Padding positions (where targets == PAD_TOKEN_ID) are ignored.
        """
        _, T = idx.shape
        assert T <= self.max_seq_len, f"Input length {T} > max_seq_len {self.max_seq_len}"

        tok_emb = self.token_embedding(idx)

        if self.pos_encoding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = self.dropout(tok_emb + pos_emb)
        else:
            x = self.dropout(tok_emb)

        for block in self.blocks:
            x = block(x, images_embeddings)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def forward_with_cross_attn(
        self,
        idx: torch.Tensor,
        images_embeddings: torch.Tensor,
        return_last_step_only: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns cross-attention weights from all layers.
        Designed for analysis — NOT for training.
    
        Args:
            idx: [B, T]
            images_embeddings: [B, L, n_embd]
            return_last_step_only: if True, returns attn only for last query token (T-1)
    
        Returns:
            logits: [B, T, vocab_size]
            cross_attn_per_layer: list of [B, n_head, L] (if last_step) or [B, n_head, T, L]
        """
        _, T = idx.shape
        tok_emb = self.token_embedding(idx)
    
        if self.pos_encoding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = self.dropout(tok_emb + pos_emb)
        else:
            x = self.dropout(tok_emb)
    
        cross_attn_list = []
    
        for block in self.blocks:
            x, attn_w = block.forward_with_cross_attn(x, images_embeddings)  # [B, H, T, L]
            if return_last_step_only:
                attn_w = attn_w[:, :, -1, :]  # [B, H, L]
            cross_attn_list.append(attn_w)
    
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, cross_attn_list

    @torch.no_grad()
    def generate(
        self,
        images_embeddings: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        Generate sequences conditioned on image context.

        Args:
            images_embeddings: Images context embeddings [B, L, n_embd], L >= 1.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: If not None, only sample from top-k logits.
            do_sample: If False, use greedy decoding.
            pad_token_id, bos_token_id, eos_token_id: Token IDs.

        Returns:
            Generated token sequences [B, 1 + max_new_tokens].
        """
        B = images_embeddings.shape[0]
        device = images_embeddings.device
        total_len = 1 + max_new_tokens
        if total_len > self.max_seq_len:
            raise ValueError(f"Total length {total_len} exceeds max_seq_len {self.max_seq_len}")

        pad_token_id = pad_token_id or self.pad_token_id
        bos_token_id = bos_token_id or self.bos_token_id
        eos_token_id = eos_token_id or self.eos_token_id

        idx = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond, images_embeddings)
            next_logits = logits[:, -1, :] / temperature

            next_logits[:, pad_token_id] = -float('inf')
            next_logits[:, bos_token_id] = -float('inf')

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                v, _ = torch.topk(next_logits, k)
                next_logits[next_logits < v[:, [-1]]] = -float('inf')

            if not do_sample:
                idx_next = torch.argmax(next_logits, dim=-1)
            else:
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1).squeeze(-1)

            newly_finished = (idx_next == eos_token_id)
            finished = finished | newly_finished
            idx = torch.cat([idx, idx_next.unsqueeze(-1)], dim=1)

            if finished.all():
                break

        if idx.size(1) < total_len:
            pad = torch.full((B, total_len - idx.size(1)), pad_token_id, device=device, dtype=torch.long)
            idx = torch.cat([idx, pad], dim=1)
        else:
            idx = idx[:, :total_len]

        for i in range(B):
            end_pos = (idx[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if end_pos.numel() > 0:
                first_end = end_pos[0].item()
                idx[i, first_end + 1:] = pad_token_id

        return idx

    @torch.no_grad()
    def generate_step_with_cross_attn(
        self,
        images_embeddings: torch.Tensor,
        idx_prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Run *one* autoregressive step and return:
        - next-token logits & ID,
        - cross-attention weights (last query position, per layer).
    
        Args:
            images_embeddings: [B, L, n_embd]
            idx_prefix: [B, T] — current sequence (e.g., [BOS] or [BOS, tok1, ...])
    
        Returns:
            next_logits: [B, vocab_size]
            next_token: [B]
            cross_attn: list of [B, n_head, L], len = n_layer
        """
        logits, cross_attn = self.forward_with_cross_attn(
            idx=idx_prefix,
            images_embeddings=images_embeddings,
            return_last_step_only=True,
        )
        next_logits = logits[:, -1, :]  # [B, vocab]
        next_token = torch.argmax(next_logits, dim=-1)  # greedy
        return next_logits, next_token, cross_attn