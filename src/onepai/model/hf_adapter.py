"""
HFAdapter — aggancio a modelli HuggingFace reali (GPT-2 e simili).

Implementa la stessa interfaccia ModelAdapter di TinyTransformer, così
AbsenceEngine e InterventionStudy girano IDENTICI su un modello vero.

Fa da ponte tra gli hook di PyTorch (tensori) e gli hook numpy di ONEPAI:
ogni fn riceve/restituisce numpy, l'adapter converte da/verso torch.

Richiede torch + transformers (entrambi opzionali per il resto di onepai).

Convenzione hook (identica a TinyTransformer):
    embed, blocks.{i}.attn.pattern, blocks.{i}.attn.z, blocks.{i}.attn.out,
    blocks.{i}.mlp.pre, blocks.{i}.mlp.post, blocks.{i}.mlp.out,
    blocks.{i}.resid_post, ln_final, logits
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .adapter import ModelAdapter, HookPoint

__all__ = ["HFAdapter"]


class HFAdapter(ModelAdapter):
    """
    Adapter per modelli causal-LM HuggingFace in stile GPT-2.

    Esempio::

        adapter = HFAdapter("gpt2")
        tokens = adapter.encode("The capital of France is")
        logits, cache = adapter.run_with_cache(tokens)
        print(adapter.decode([adapter.capture(tokens).top_token()]))
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .hf_arch import resolve_spec, get_path

        self._torch = torch
        self._get_path = get_path
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eager: necessario per ottenere i pesi di attenzione.
        # .float(): alcuni checkpoint (es. Pythia) sono in float16 e su CPU
        # producono NaN; forziamo float32 per stabilità numerica.
        self.hf = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager")
        self.hf = self.hf.float().eval().to(device)
        for p in self.hf.parameters():
            p.requires_grad_(False)

        cfg = self.hf.config
        self.model_type = getattr(cfg, "model_type", "gpt2")
        self.spec = resolve_spec(self.model_type)
        self._n_layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
        self._n_heads = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)))
        self._d_model = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
        self._vocab = int(cfg.vocab_size)
        self._d_head = self._d_model // self._n_heads

        # Sottomoduli risolti dal registro (agnostico all'architettura)
        self._blocks = get_path(self.hf, self.spec["layers"])
        self._final_norm = get_path(self.hf, self.spec["final_norm"])
        self._unembed = get_path(self.hf, self.spec["unembed"])
        if self._blocks is None or self._unembed is None:
            raise ValueError(
                f"Impossibile risolvere i sottomoduli per {self.model_type!r}: "
                f"layers={self._blocks is not None}, unembed={self._unembed is not None}")

    def _block_mod(self, blk, role: str):
        """Sottomodulo di un blocco per ruolo (attn, mlp, mlp_pre, ...)."""
        return self._get_path(blk, self.spec[role])

    # ------------------------------------------------------------------
    # Forma
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int: return self._n_layers
    @property
    def n_heads(self) -> int: return self._n_heads
    @property
    def d_model(self) -> int: return self._d_model
    @property
    def vocab_size(self) -> int: return self._vocab

    # ------------------------------------------------------------------
    # Tokenizzazione
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        ids = self.tokenizer(text, return_tensors="np")["input_ids"][0]
        return ids.astype(np.int64)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids))

    def random_tokens(self, seq_len: int = 16, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(seed if seed is not None else 0)
        return rng.randint(0, self._vocab, size=seq_len)

    # ------------------------------------------------------------------
    # Helpers torch<->numpy
    # ------------------------------------------------------------------

    def _to_ids(self, tokens):
        torch = self._torch
        ids = np.asarray(tokens, dtype=np.int64).ravel()
        return torch.tensor(ids[None, :], device=self.device)

    @staticmethod
    def _np(t) -> np.ndarray:
        return t.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # run_with_cache
    # ------------------------------------------------------------------

    def run_with_cache(self, tokens) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        torch = self._torch
        ids = self._to_ids(tokens)
        cache: Dict[str, np.ndarray] = {}
        handles = []

        # Hook di cattura su pre (up-proj), post (attivazione), out (down-proj)
        for i, blk in enumerate(self._blocks):
            # resid_post(i) = output del blocco, catturato con un hook esplicito.
            # NON usare out.hidden_states: in transformers 5.x l'ultima voce è
            # già passata dalla norma finale (verificato: hs[-1] == norm(block[-1])),
            # quindi il logit lens applicherebbe la norma due volte sull'ultimo layer.
            handles.append(blk.register_forward_hook(
                self._cap_hook(cache, HookPoint.resid_post(i))))
            pre = self._block_mod(blk, "mlp_pre")
            act = self._block_mod(blk, "mlp_act")
            mlp = self._block_mod(blk, "mlp")
            if pre is not None:
                handles.append(pre.register_forward_hook(
                    self._cap_hook(cache, HookPoint.mlp_pre(i))))
            if act is not None:
                handles.append(act.register_forward_hook(
                    self._cap_hook(cache, HookPoint.mlp_post(i))))
            if mlp is not None:
                handles.append(mlp.register_forward_hook(
                    self._cap_hook(cache, HookPoint.mlp_out(i))))

            # Uscita dell'attention e z per testa: indispensabili per la DLA
            # (senza queste, la ricostruzione del logit perde tutta l'attenzione).
            attn = self._block_mod(blk, "attn")
            if attn is not None:
                handles.append(attn.register_forward_hook(
                    self._cap_hook(cache, HookPoint.attn_out(i))))
            proj = self._block_mod(blk, "attn_out_proj")
            if proj is not None:
                handles.append(proj.register_forward_pre_hook(
                    self._cap_z_hook(cache, i)))

        try:
            with torch.no_grad():
                out = self.hf(ids, output_attentions=True,
                              output_hidden_states=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if out.hidden_states is not None:
            cache[HookPoint.EMBED] = self._np(out.hidden_states[0][0])
        if out.attentions is not None:
            for i in range(min(self._n_layers, len(out.attentions))):
                cache[HookPoint.attn_pattern(i)] = self._np(out.attentions[i][0])  # (n_head, seq, seq)
        logits = self._np(out.logits[0])    # (seq, vocab)
        cache[HookPoint.LOGITS] = logits
        return logits, cache

    def _cap_hook(self, cache: Dict[str, np.ndarray], name: str):
        def hook(module, inp, output):
            t = output[0] if isinstance(output, tuple) else output
            cache[name] = self._np(t[0])    # rimuove batch dim
        return hook

    def _cap_z_hook(self, cache: Dict[str, np.ndarray], layer: int):
        """Cattura z (seq, n_heads, d_head) dall'input della proiezione O."""
        def hook(module, args):
            x = args[0]
            seq = x.shape[1]
            z = self._np(x[0]).reshape(seq, self._n_heads, self._d_head)
            cache[HookPoint.attn_z(layer)] = z
        return hook

    # ------------------------------------------------------------------
    # run_with_hooks (interventi)
    # ------------------------------------------------------------------

    def run_with_hooks(
        self,
        tokens,
        fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
    ) -> np.ndarray:
        torch = self._torch
        ids = self._to_ids(tokens)
        handles = []
        for name, fn in (fwd_hooks or []):
            handle = self._register_intervention(name, fn)
            if handle is not None:
                handles.append(handle)
        try:
            with torch.no_grad():
                out = self.hf(ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return self._np(out.logits[0])

    def _register_intervention(self, name: str, fn: Callable):
        """Mappa un nome-hook ONEPAI sul modulo corretto, per qualunque architettura."""
        def tensor_fn(t_np: np.ndarray) -> np.ndarray:
            return np.asarray(fn(t_np, name), dtype=np.float32)

        for i, blk in enumerate(self._blocks):
            if name == HookPoint.mlp_pre(i):
                m = self._block_mod(blk, "mlp_pre")
                return m.register_forward_hook(self._mod_hook(tensor_fn)) if m else None
            if name == HookPoint.mlp_post(i):
                m = self._block_mod(blk, "mlp_act")
                return m.register_forward_hook(self._mod_hook(tensor_fn)) if m else None
            if name == HookPoint.mlp_out(i):
                m = self._block_mod(blk, "mlp")
                return m.register_forward_hook(self._mod_hook(tensor_fn)) if m else None
            if name == HookPoint.resid_post(i):
                return blk.register_forward_hook(self._mod_hook(tensor_fn))
            if name == HookPoint.attn_out(i):
                m = self._block_mod(blk, "attn")
                return m.register_forward_hook(self._mod_hook(tensor_fn)) if m else None
            if name == HookPoint.attn_z(i):
                # ablazione per-testa: pre-hook sulla proiezione d'uscita dell'attenzione,
                # input (1, seq, n_head*d_head)
                proj = self._block_mod(blk, "attn_out_proj")
                return proj.register_forward_pre_hook(self._z_pre_hook(tensor_fn)) if proj else None
        return None

    def _mod_hook(self, tensor_fn: Callable):
        """Hook che modifica l'output di un modulo. Gestisce sia tensori che tuple
        (l'API varia tra versioni di transformers)."""
        torch = self._torch

        def hook(module, inp, output):
            is_tuple = isinstance(output, tuple)
            t = output[0] if is_tuple else output
            modified = tensor_fn(self._np(t[0]))           # (seq, d) numpy
            new_t = torch.tensor(modified, dtype=t.dtype, device=t.device)[None]
            if is_tuple:
                return (new_t,) + tuple(output[1:])
            return new_t
        return hook

    def _z_pre_hook(self, tensor_fn: Callable):
        """Pre-hook su c_proj: reshape input in (seq, n_head, d_head), applica fn."""
        torch = self._torch
        nh, dh = self._n_heads, self._d_head

        def pre_hook(module, args):
            x = args[0]                                     # (1, seq, n_head*d_head)
            seq = x.shape[1]
            z = self._np(x[0]).reshape(seq, nh, dh)
            z2 = tensor_fn(z).reshape(seq, nh * dh)
            new_x = torch.tensor(z2, dtype=x.dtype, device=x.device)[None]
            return (new_x,) + tuple(args[1:])
        return pre_hook

    # ------------------------------------------------------------------
    # Unembedding (logit lens reale)
    # ------------------------------------------------------------------

    def apply_unembed(self, resid: np.ndarray) -> np.ndarray:
        torch = self._torch
        param_dtype = self._unembed.weight.dtype     # allinea al dtype del modello
        with torch.no_grad():
            t = torch.tensor(np.asarray(resid, dtype=np.float32),
                             device=self.device).to(param_dtype)
            if self._final_norm is not None:
                t = self._final_norm(t)
            logits = self._unembed(t)
        return self._np(logits.float())

    def unembed_direction(self, token: int) -> np.ndarray:
        """Direzione di unembedding del token (riga dei pesi di unembed)."""
        return self._np(self._unembed.weight[token]).astype(np.float32)

    # ------------------------------------------------------------------
    # DLA / tuned lens
    # ------------------------------------------------------------------

    def unembed_matrix(self) -> np.ndarray:
        """(d_model, vocab): trasposta dei pesi di unembedding."""
        return self._np(self._unembed.weight).astype(np.float32).T

    def final_norm_scale(self, resid: np.ndarray):
        """
        Scala elementwise della normalizzazione finale + tipo (con/senza media).

        LayerNorm: scale = gamma / sqrt(var + eps), sottrae la media.
        RMSNorm (llama/qwen): scale = gamma / rms, NON sottrae la media.
        """
        x = np.asarray(resid, dtype=np.float32)
        norm = self._final_norm
        if norm is None:
            return np.ones_like(x), False
        gamma = (self._np(norm.weight).astype(np.float32)
                 if getattr(norm, "weight", None) is not None
                 else np.ones(x.shape[-1], dtype=np.float32))
        eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
        name = type(norm).__name__.lower()
        if "rms" in name:
            rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
            return (gamma / rms).astype(np.float32), False
        var = x.var(axis=-1, keepdims=True)
        return (gamma / np.sqrt(var + eps)).astype(np.float32), True

    def head_unembed(self, layer: int) -> Optional[np.ndarray]:
        """(n_heads, d_head, d_model): la matrice O vista per testa."""
        cached = getattr(self, "_head_unembed_cache", None)
        if cached is None:
            cached = {}
            self._head_unembed_cache = cached
        if layer in cached:
            return cached[layer]

        proj = self._block_mod(self._blocks[layer], "attn_out_proj")
        if proj is None or getattr(proj, "weight", None) is None:
            return None
        W = self._np(proj.weight).astype(np.float32)
        nh, dh, d = self._n_heads, self._d_head, self._d_model
        # Conv1D (GPT-2) tiene (in, out); nn.Linear tiene (out, in).
        W_io = W if type(proj).__name__ == "Conv1D" else W.T
        if W_io.shape != (nh * dh, d):
            return None
        out = W_io.reshape(nh, dh, d)
        cached[layer] = out
        return out

    def __repr__(self) -> str:
        return (f"HFAdapter({self.model_name!r} [{self.model_type}]: {self._n_layers}L, "
                f"{self._n_heads}H, d_model={self._d_model}, vocab={self._vocab})")
