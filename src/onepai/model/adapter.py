"""
Interfaccia astratta per modelli instrumentabili + primitive numeriche condivise.

Convenzione dei nomi degli hook (mutuata da transformer_lens):

    embed
    blocks.{i}.attn.pattern     attenzione (n_heads, seq, seq)
    blocks.{i}.attn.z           output per-testa (seq, n_heads, d_head)
    blocks.{i}.attn.out         delta residuale dall'attention (seq, d_model)
    blocks.{i}.mlp.pre          pre-attivazione MLP (seq, d_mlp)
    blocks.{i}.mlp.post         post-GELU (seq, d_mlp)
    blocks.{i}.mlp.out          delta residuale dall'MLP (seq, d_model)
    blocks.{i}.resid_post       residual stream dopo il blocco (seq, d_model)
    ln_final                    residual normalizzato finale (seq, d_model)
    logits                      logit finali (seq, vocab)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ModelAdapter", "CaptureBundle", "Intervention", "HookPoint",
    "make_hook", "gelu", "softmax", "layer_norm",
]


# ----------------------------------------------------------------------
# Primitive numeriche (condivise da TinyTransformer e dagli adapter)
# ----------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU approssimata (tanh), come in GPT-2."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numericamente stabile lungo `axis`."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def layer_norm(x: np.ndarray, gamma: Optional[np.ndarray] = None,
               beta: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm sull'ultima dimensione."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        normed = normed * gamma
    if beta is not None:
        normed = normed + beta
    return normed


class HookPoint:
    """Costanti / helper per i nomi degli hook."""
    EMBED = "embed"
    LN_FINAL = "ln_final"
    LOGITS = "logits"

    @staticmethod
    def attn_pattern(i: int) -> str: return f"blocks.{i}.attn.pattern"
    @staticmethod
    def attn_z(i: int) -> str: return f"blocks.{i}.attn.z"
    @staticmethod
    def attn_out(i: int) -> str: return f"blocks.{i}.attn.out"
    @staticmethod
    def mlp_pre(i: int) -> str: return f"blocks.{i}.mlp.pre"
    @staticmethod
    def mlp_post(i: int) -> str: return f"blocks.{i}.mlp.post"
    @staticmethod
    def mlp_out(i: int) -> str: return f"blocks.{i}.mlp.out"
    @staticmethod
    def resid_post(i: int) -> str: return f"blocks.{i}.resid_post"


# ----------------------------------------------------------------------
# Strutture dati
# ----------------------------------------------------------------------

@dataclass
class CaptureBundle:
    """Risultato strutturato di un forward instrumentato."""
    tokens: np.ndarray                                 # (seq,)
    logits: np.ndarray                                 # (seq, vocab) — logit finali
    cache: Dict[str, np.ndarray] = field(default_factory=dict)
    model_name: str = "model"
    n_layers: int = 0
    n_heads: int = 0
    d_model: int = 0
    vocab_size: int = 0

    # -- accessor comodi --------------------------------------------------

    def attention(self, layer: int) -> Optional[np.ndarray]:
        """Pattern di attenzione (n_heads, seq, seq) per un layer."""
        return self.cache.get(HookPoint.attn_pattern(layer))

    def mlp_post(self, layer: int) -> Optional[np.ndarray]:
        """Attivazioni MLP post-GELU (seq, d_mlp) per un layer."""
        return self.cache.get(HookPoint.mlp_post(layer))

    def resid_post(self, layer: int) -> Optional[np.ndarray]:
        return self.cache.get(HookPoint.resid_post(layer))

    def final_token_logits(self) -> np.ndarray:
        """Logit dell'ultima posizione (predizione del prossimo token)."""
        return self.logits[-1]

    def top_token(self, position: int = -1) -> int:
        return int(np.argmax(self.logits[position]))

    def __repr__(self) -> str:
        return (f"CaptureBundle(model={self.model_name!r}, seq={len(self.tokens)}, "
                f"layers={self.n_layers}, heads={self.n_heads}, "
                f"cache_keys={len(self.cache)})")


@dataclass
class Intervention:
    """
    Descrive una modifica controllata applicata a un hook durante il forward.

    kind:
        "ablate"  — azzera (o sostituisce con `value`) le componenti indicate
        "add"     — aggiunge `coefficient * value` (direzione da iniettare)
        "scale"   — moltiplica per `coefficient`
        "patch"   — sostituisce con `value` (es. attivazione da un altro run)
        "mean"    — sostituisce con la media lungo la dimensione seq
    """
    hook_name: str
    kind: str = "ablate"
    value: Optional[np.ndarray] = None
    indices: Optional[Tuple] = None     # quali neuroni/teste toccare (None = tutto)
    coefficient: float = 1.0
    position: Optional[int] = None       # quale posizione token (None = tutte)
    label: str = ""

    def apply(self, activation: np.ndarray) -> np.ndarray:
        """Applica l'intervento a un'attivazione catturata, restituendone una copia."""
        out = np.array(activation, dtype=np.float32, copy=True)
        sel = self._selector(out)
        if self.kind == "ablate":
            out[sel] = 0.0 if self.value is None else self.value
        elif self.kind == "add":
            if self.value is not None:
                out[sel] = out[sel] + self.coefficient * np.asarray(self.value, dtype=np.float32)
        elif self.kind == "scale":
            out[sel] = out[sel] * self.coefficient
        elif self.kind == "patch":
            if self.value is not None:
                out[sel] = np.asarray(self.value, dtype=np.float32)
        elif self.kind == "mean":
            if out.ndim >= 2:
                mean_val = out.mean(axis=0, keepdims=True)
                out[sel] = np.broadcast_to(mean_val, out.shape)[sel]
        return out

    def _selector(self, arr: np.ndarray):
        """Costruisce l'indice numpy in base a position/indices."""
        if self.position is not None and self.indices is not None:
            return (self.position,) + tuple(np.atleast_1d(self.indices))
        if self.position is not None:
            return (self.position,)
        if self.indices is not None:
            # indici sull'ultima dimensione (es. neuroni MLP)
            idx = [slice(None)] * (arr.ndim - 1) + [np.atleast_1d(self.indices)]
            return tuple(idx)
        return slice(None)

    def __repr__(self) -> str:
        tag = self.label or self.kind
        return f"Intervention({tag} @ {self.hook_name}, coef={self.coefficient})"


def make_hook(intervention: Intervention,
              capture: Optional[Dict[str, np.ndarray]] = None) -> Callable[[np.ndarray, str], np.ndarray]:
    """
    Crea una funzione-hook che applica un Intervention e (opzionalmente)
    registra il valore risultante in `capture`.
    """
    def _hook(value: np.ndarray, name: str) -> np.ndarray:
        modified = intervention.apply(value)
        if capture is not None:
            capture[name] = np.array(modified, copy=True)
        return modified
    return _hook


# ----------------------------------------------------------------------
# Interfaccia astratta
# ----------------------------------------------------------------------

class ModelAdapter(ABC):
    """
    Interfaccia comune per qualunque modello instrumentabile da ONEPAI.

    Le implementazioni (TinyTransformer, HFAdapter, ...) devono fornire:
      - run_with_cache(tokens)               → (logits, cache)
      - run_with_hooks(tokens, fwd_hooks)    → logits
      - le proprietà di forma (n_layers, ...)

    `fwd_hooks` è una lista di (hook_name, fn) dove fn(value, name) -> value.
    La fn può catturare il valore e/o restituirne uno modificato.
    """

    model_name: str = "abstract"

    # -- forma -----------------------------------------------------------

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    # -- forward ---------------------------------------------------------

    @abstractmethod
    def run_with_cache(self, tokens) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward completo, restituisce (logits, cache di tutti gli hook)."""

    @abstractmethod
    def run_with_hooks(
        self,
        tokens,
        fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
    ) -> np.ndarray:
        """Forward con hook che possono catturare e/o modificare le attivazioni."""

    # -- helper di alto livello (forniti dalla base) ---------------------

    def capture(self, tokens) -> CaptureBundle:
        """Esegue run_with_cache e impacchetta il risultato in un CaptureBundle."""
        logits, cache = self.run_with_cache(tokens)
        return CaptureBundle(
            tokens=np.asarray(tokens),
            logits=logits,
            cache=cache,
            model_name=self.model_name,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            vocab_size=self.vocab_size,
        )

    def run_with_interventions(
        self,
        tokens,
        interventions: List[Intervention],
        return_cache: bool = False,
    ):
        """
        Esegue il forward applicando una lista di Intervention.
        Se return_cache=True restituisce (logits, cache_post_intervento).
        """
        capture: Dict[str, np.ndarray] = {}
        fwd_hooks = [
            (iv.hook_name, make_hook(iv, capture if return_cache else None))
            for iv in interventions
        ]
        logits = self.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        if return_cache:
            return logits, capture
        return logits

    def logit_diff(
        self,
        tokens,
        interventions: List[Intervention],
        target_token: int,
        position: int = -1,
    ) -> float:
        """
        Effetto causale di un set di interventi sul logit di `target_token`:
        logit_intervento - logit_baseline (alla posizione data).
        """
        base_logits, _ = self.run_with_cache(tokens)
        iv_logits = self.run_with_interventions(tokens, interventions)
        return float(iv_logits[position, target_token] - base_logits[position, target_token])

    def apply_unembed(self, resid: np.ndarray) -> Optional[np.ndarray]:
        """
        Proietta un residual stream (seq, d_model) sul vocabolario applicando
        la LayerNorm finale e l'unembedding — usato dal logit lens reale.
        Restituisce (seq, vocab), oppure None se il modello non lo supporta.
        Le sottoclassi che espongono W_U / LN finale devono fare override.
        """
        return None

    def supports_logit_lens(self) -> bool:
        return self.apply_unembed(np.zeros((1, self.d_model), dtype=np.float32)) is not None

    def unembed_direction(self, token: int) -> Optional[np.ndarray]:
        """Direzione di unembedding del token (per lo steering verso quel token)."""
        W_U = getattr(self, "W_U", None)
        if W_U is None:
            return None
        return np.asarray(W_U[:, token], dtype=np.float32)

    def hook_names(self) -> List[str]:
        """Elenco dei nomi di hook disponibili (best-effort)."""
        names = [HookPoint.EMBED]
        for i in range(self.n_layers):
            names += [
                HookPoint.attn_pattern(i), HookPoint.attn_z(i), HookPoint.attn_out(i),
                HookPoint.mlp_pre(i), HookPoint.mlp_post(i), HookPoint.mlp_out(i),
                HookPoint.resid_post(i),
            ]
        names += [HookPoint.LN_FINAL, HookPoint.LOGITS]
        return names
