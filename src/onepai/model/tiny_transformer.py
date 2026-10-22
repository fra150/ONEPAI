"""
TinyTransformer — un transformer REALE in puro numpy.

Forward completo: embed → (attention causale + MLP/GELU + LayerNorm)×L → unembed.
Pesi deterministici dal seed. Niente torch, niente download, niente np.random
sparso nei dati: le attivazioni — e quindi i neuroni morti, le teste fantasma,
il logit lens — emergono da una computazione vera.

API in stile transformer_lens:
    model = TinyTransformer(TinyConfig(n_layers=3, n_heads=4, d_model=32))
    logits, cache = model.run_with_cache(tokens)
    logits = model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.post", my_fn)])

Iniezione di ground-truth (per validare l'analisi dell'assenza):
    TinyConfig(inject_dead_neurons=[(1, 5)], inject_ghost_heads=[(0, 3)])
rende il neurone 5 del layer 1 davvero morto e la testa 3 del layer 0 davvero
fantasma (scores ≈ 0 → attenzione diffusa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .adapter import (
    ModelAdapter, HookPoint, gelu, layer_norm, softmax,
)

__all__ = ["TinyTransformer", "TinyConfig"]


@dataclass
class TinyConfig:
    vocab_size: int = 256
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 3
    d_mlp: int = 64
    n_ctx: int = 64
    seed: int = 0
    inject_dead_neurons: List[Tuple[int, int]] = field(default_factory=list)
    inject_suppressed_neurons: List[Tuple[int, int]] = field(default_factory=list)
    inject_ghost_heads: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model deve essere divisibile per n_heads")
        return self.d_model // self.n_heads


class TinyTransformer(ModelAdapter):
    """Transformer decoder-only reale, puro numpy, con hook bidirezionali."""

    model_name = "tiny-transformer"

    def __init__(self, config: Optional[TinyConfig] = None, **kwargs):
        self.cfg = config or TinyConfig(**kwargs)
        if config is not None and kwargs:
            for k, v in kwargs.items():
                setattr(self.cfg, k, v)
        self.model_name = f"tiny-{self.cfg.n_layers}L-{self.cfg.n_heads}H-{self.cfg.d_model}d"
        self._init_weights()

    # ------------------------------------------------------------------
    # Proprietà di forma (interfaccia ModelAdapter)
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int: return self.cfg.n_layers
    @property
    def n_heads(self) -> int: return self.cfg.n_heads
    @property
    def d_model(self) -> int: return self.cfg.d_model
    @property
    def vocab_size(self) -> int: return self.cfg.vocab_size

    # ------------------------------------------------------------------
    # Inizializzazione pesi
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed)
        d, h, dh, dm = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp

        def randn(*shape, scale):
            return (rng.randn(*shape) * scale).astype(np.float32)

        self.W_E = randn(cfg.vocab_size, d, scale=1.0 / np.sqrt(d))
        self.W_pos = randn(cfg.n_ctx, d, scale=0.02)

        self.blocks = []
        for _ in range(cfg.n_layers):
            blk = {
                "W_Q": randn(h, d, dh, scale=1.0 / np.sqrt(d)),
                "W_K": randn(h, d, dh, scale=1.0 / np.sqrt(d)),
                "W_V": randn(h, d, dh, scale=1.0 / np.sqrt(d)),
                "b_O": np.zeros(d, dtype=np.float32),
                "W_O": randn(h, dh, d, scale=1.0 / np.sqrt(h * dh)),
                "ln1_g": np.ones(d, dtype=np.float32), "ln1_b": np.zeros(d, dtype=np.float32),
                "ln2_g": np.ones(d, dtype=np.float32), "ln2_b": np.zeros(d, dtype=np.float32),
                "W_in": randn(d, dm, scale=1.0 / np.sqrt(d)),
                "b_in": np.zeros(dm, dtype=np.float32),
                "W_out": randn(dm, d, scale=1.0 / np.sqrt(dm)),
                "b_out": np.zeros(d, dtype=np.float32),
            }
            self.blocks.append(blk)

        self.lnf_g = np.ones(d, dtype=np.float32)
        self.lnf_b = np.zeros(d, dtype=np.float32)
        self.W_U = randn(d, cfg.vocab_size, scale=1.0 / np.sqrt(d))

        # --- Iniezione di ground-truth -----------------------------------
        # DEAD: niente lo guida → pre ≈ 0 (W_in e bias azzerati).
        for (layer, neuron) in cfg.inject_dead_neurons:
            if 0 <= layer < cfg.n_layers and 0 <= neuron < dm:
                self.blocks[layer]["W_in"][:, neuron] = 0.0
                self.blocks[layer]["b_in"][neuron] = 0.0     # pre ≈ 0 → gelu ≈ 0
                self.blocks[layer]["W_out"][neuron, :] = 0.0
        # SUPPRESSED: qualcosa lo schiaccia → pre fortemente NEGATIVO (gated).
        for (layer, neuron) in cfg.inject_suppressed_neurons:
            if 0 <= layer < cfg.n_layers and 0 <= neuron < dm:
                self.blocks[layer]["W_in"][:, neuron] = 0.0
                self.blocks[layer]["b_in"][neuron] = -30.0   # pre ≪ 0 ma forte
                self.blocks[layer]["W_out"][neuron, :] = 0.0
        for (layer, head) in cfg.inject_ghost_heads:
            if 0 <= layer < cfg.n_layers and 0 <= head < h:
                # W_V = W_O = 0 → z = 0 → la testa non contribuisce al residual stream.
                # (Sotto maschera causale "scores≈0" NON basta: la posizione 0 attende
                #  sempre se stessa al 100%, quindi il max-weight resterebbe 1.0.)
                self.blocks[layer]["W_V"][head] = 0.0
                self.blocks[layer]["W_O"][head] = 0.0

    # ------------------------------------------------------------------
    # Forward con hook
    # ------------------------------------------------------------------

    def run_with_hooks(
        self,
        tokens,
        fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
    ) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64).ravel()
        seq = len(tokens)
        cfg = self.cfg
        hooks: Dict[str, Callable] = dict(fwd_hooks or [])

        def H(name: str, value: np.ndarray) -> np.ndarray:
            fn = hooks.get(name)
            if fn is None:
                return value
            out = fn(value, name)
            return value if out is None else np.asarray(out, dtype=np.float32)

        # Embedding + posizionale
        resid = self.W_E[tokens] + self.W_pos[:seq]      # (seq, d_model)
        resid = H(HookPoint.EMBED, resid.astype(np.float32))

        # Maschera causale (seq_q, seq_k)
        causal = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)

        for i, blk in enumerate(self.blocks):
            # --- Attention ---
            normed = layer_norm(resid, blk["ln1_g"], blk["ln1_b"])
            Q = np.einsum("sd,hde->hse", normed, blk["W_Q"])    # (h, seq, d_head)
            K = np.einsum("sd,hde->hse", normed, blk["W_K"])
            V = np.einsum("sd,hde->hse", normed, blk["W_V"])
            scores = np.einsum("hse,hte->hst", Q, K) / np.sqrt(cfg.d_head)
            scores = scores + causal[None, :, :]
            pattern = softmax(scores, axis=-1)                  # (h, seq, seq)
            pattern = H(HookPoint.attn_pattern(i), pattern)
            z = np.einsum("hst,hte->hse", pattern, V)           # (h, seq, d_head)
            z_t = np.transpose(z, (1, 0, 2))                    # (seq, h, d_head)
            z_t = H(HookPoint.attn_z(i), z_t)
            attn_out = np.einsum("she,hed->sd", z_t, blk["W_O"]) + blk["b_O"]
            attn_out = H(HookPoint.attn_out(i), attn_out)
            resid = resid + attn_out

            # --- MLP ---
            normed2 = layer_norm(resid, blk["ln2_g"], blk["ln2_b"])
            pre = normed2 @ blk["W_in"] + blk["b_in"]           # (seq, d_mlp)
            pre = H(HookPoint.mlp_pre(i), pre)
            post = gelu(pre)
            post = H(HookPoint.mlp_post(i), post)
            mlp_out = post @ blk["W_out"] + blk["b_out"]        # (seq, d_model)
            mlp_out = H(HookPoint.mlp_out(i), mlp_out)
            resid = resid + mlp_out

            resid = H(HookPoint.resid_post(i), resid)

        normed_final = layer_norm(resid, self.lnf_g, self.lnf_b)
        normed_final = H(HookPoint.LN_FINAL, normed_final)
        logits = normed_final @ self.W_U                        # (seq, vocab)
        logits = H(HookPoint.LOGITS, logits)
        return logits

    def run_with_cache(self, tokens) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward catturando ogni hook point nella cache."""
        cache: Dict[str, np.ndarray] = {}

        def capture(value, name):
            cache[name] = np.array(value, copy=True)
            return value

        names = self.hook_names()
        logits = self.run_with_hooks(tokens, fwd_hooks=[(n, capture) for n in names])
        return logits, cache

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def apply_unembed(self, resid: np.ndarray) -> np.ndarray:
        """LayerNorm finale + unembedding → logit per il logit lens reale."""
        resid = np.asarray(resid, dtype=np.float32)
        normed = layer_norm(resid, self.lnf_g, self.lnf_b)
        return normed @ self.W_U

    # -- DLA / tuned lens -----------------------------------------------

    def unembed_matrix(self) -> np.ndarray:
        return self.W_U

    def final_norm_scale(self, resid: np.ndarray):
        """LayerNorm finale: scala = gamma / std, sottrae la media."""
        x = np.asarray(resid, dtype=np.float32)
        var = x.var(axis=-1, keepdims=True)
        return (self.lnf_g / np.sqrt(var + 1e-5)).astype(np.float32), True

    def head_unembed(self, layer: int) -> Optional[np.ndarray]:
        """W_O per testa: (n_heads, d_head, d_model)."""
        return np.asarray(self.blocks[layer]["W_O"], dtype=np.float32)

    def random_tokens(self, seq_len: int = 16, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(seed if seed is not None else self.cfg.seed + 1)
        return rng.randint(0, self.cfg.vocab_size, size=seq_len)

    def __repr__(self) -> str:
        c = self.cfg
        return (f"TinyTransformer({c.n_layers}L, {c.n_heads}H, d_model={c.d_model}, "
                f"d_mlp={c.d_mlp}, vocab={c.vocab_size})")
