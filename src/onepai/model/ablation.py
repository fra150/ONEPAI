"""
Ablation — *come* si toglie un componente (e perché azzerarlo non basta).

ONEPAI ha sempre ablato i componenti mettendoli a zero. Ma il modello non ha
mai visto un MLP a zero: parte dell'effetto misurato può essere danno
collaterale, non il ruolo causale del componente. Lo standard per gli studi di
activation patching è confrontare tre modalità:

    zero      — output = 0                      (baseline storica di ONEPAI)
    mean      — output = media sul corpus       (resta in distribuzione)
    resample  — output = attivazione da un altro prompt (controfattuale)

Un soppressore è solido solo se il ripristino regge in tutte e tre: se cambia
segno o sparisce con `mean`, il numero era un artefatto del fuori-distribuzione.

Riferimento: Zhang & Nanda, "Towards Best Practices of Activation Patching in
Language Models" (2023).

Uso::

    ctrl = AblationController.build(model, "mean", reference_batch=corpus)
    hook = ctrl.hook_for(HookPoint.mlp_out(23))
    logits = model.run_with_hooks(tokens, fwd_hooks=[(HookPoint.mlp_out(23), hook)])
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .adapter import HookPoint, ModelAdapter

__all__ = ["ABLATION_MODES", "ablation_hooks", "compute_mean_activations",
           "AblationController"]


ABLATION_MODES = ("zero", "mean", "resample")


# ----------------------------------------------------------------------
# Baseline: media delle attivazioni sul corpus di riferimento
# ----------------------------------------------------------------------

def compute_mean_activations(
    model: ModelAdapter,
    token_batch: Sequence[np.ndarray],
    hook_names: Optional[Sequence[str]] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Media di ogni hook su tutte le posizioni di tutte le sequenze del batch.

    La media è dataset-wide (una posizione non è comparabile tra prompt di
    lunghezza diversa). Per un hook (seq, d) restituisce (d,); per un hook
    (seq, n_heads, d_head) restituisce (n_heads, d_head).

    Args:
        model:       un ModelAdapter (HFAdapter, TinyTransformer, ...)
        token_batch: sequenze di riferimento (es. wikitext)
        hook_names:  hook da mediare (default: tutti gli mlp/attn out + z)
        max_tokens:  limite opzionale al numero di posizioni usate per media
    """
    if hook_names is None:
        hook_names = default_ablation_hooks(model)

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for tokens in token_batch:
        _, cache = model.run_with_cache(tokens)
        for name in hook_names:
            act = cache.get(name)
            if act is None:
                continue
            act = np.asarray(act, dtype=np.float64)
            flat = act.reshape(act.shape[0], -1).sum(axis=0)
            sums[name] = sums[name] + flat if name in sums else flat
            counts[name] = counts.get(name, 0) + act.shape[0]
        if max_tokens is not None and sum(counts.values()) >= max_tokens * max(1, len(hook_names)):
            break

    return {
        name: (sums[name] / max(counts[name], 1)).astype(np.float32)
        for name in sums
    }


def default_ablation_hooks(model: ModelAdapter) -> List[str]:
    """Hook su cui ha senso ablare: MLP + attention per ogni layer, più z."""
    names: List[str] = []
    for i in range(model.n_layers):
        names += [HookPoint.mlp_out(i), HookPoint.attn_out(i), HookPoint.attn_z(i)]
    return names


def ablation_hooks(model: ModelAdapter) -> List[str]:
    """Alias pubblico di `default_ablation_hooks`."""
    return default_ablation_hooks(model)


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------

def _fit_to_shape(value: np.ndarray, shape) -> np.ndarray:
    """
    Adatta un'attivazione donatore alla forma bersaglio.

    - dimensioni finali compatibili → broadcast
    - lunghezza sequenza diversa → il donatore viene tagliato o ripetuto
      lungo le posizioni (i residui non sono allineati tra prompt diversi,
      quindi questa è una resample-ablation "grossolana", non un patch 1:1)
    """
    value = np.asarray(value, dtype=np.float32)
    if value.shape == tuple(shape):
        return value
    if value.ndim == len(shape) and value.shape[1:] == tuple(shape[1:]):
        seq_t, seq_s = shape[0], value.shape[0]
        if seq_t <= seq_s:
            return value[:seq_t]
        reps = int(np.ceil(seq_t / max(seq_s, 1)))
        return np.concatenate([value] * reps, axis=0)[:seq_t]
    return np.broadcast_to(value, shape).astype(np.float32)


class AblationController:
    """
    Costruisce hook di ablazione nella modalità scelta.

    Esempio::

        ctrl = AblationController.build(model, "mean", reference_batch=corpus)
        eff = InterventionEffect(...)
        hook = ctrl.hook_for(HookPoint.mlp_out(11))
        logits = model.run_with_hooks(tokens, fwd_hooks=[(HookPoint.mlp_out(11), hook)])
    """

    def __init__(
        self,
        model: ModelAdapter,
        mode: str = "zero",
        means: Optional[Dict[str, np.ndarray]] = None,
        donor_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        if mode not in ABLATION_MODES:
            raise ValueError(f"mode deve essere uno di {ABLATION_MODES}, non {mode!r}")
        self.model = model
        self.mode = mode
        self.means = means or {}
        self.donor_cache = donor_cache or {}
        if mode == "mean" and not self.means:
            raise ValueError("mode='mean' richiede `means` (vedi build()).")
        if mode == "resample" and not self.donor_cache:
            raise ValueError("mode='resample' richiede `donor_cache` (vedi build()).")

    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: ModelAdapter,
        mode: str = "zero",
        reference_batch: Optional[Sequence[np.ndarray]] = None,
        donor_tokens: Optional[np.ndarray] = None,
        hook_names: Optional[Sequence[str]] = None,
    ) -> "AblationController":
        """
        Costruisce il controller, calcolando solo ciò che serve.

        mode="mean"      → media le attivazioni sul `reference_batch`
        mode="resample"  → cattura una cache donatore da `donor_tokens`
        mode="zero"      → niente da preparare
        """
        means = None
        donor = None
        if mode == "mean":
            if not reference_batch:
                raise ValueError("mode='mean' richiede reference_batch.")
            means = compute_mean_activations(model, reference_batch,
                                             hook_names=hook_names)
        elif mode == "resample":
            if donor_tokens is None:
                if not reference_batch:
                    raise ValueError("mode='resample' richiede donor_tokens o reference_batch.")
                donor_tokens = reference_batch[0]
            _, donor = model.run_with_cache(donor_tokens)
            if hook_names is not None:
                donor = {k: v for k, v in donor.items() if k in set(hook_names)}
        return cls(model, mode=mode, means=means, donor_cache=donor)

    # ------------------------------------------------------------------

    def hook_for(self, hook_name: str, head: Optional[int] = None) -> Callable:
        """
        Funzione-hook `(value, name) -> value` che applica l'ablazione.

        Con `head` valorizzato tocca solo quella testa (hook attn.z).
        """
        if self.mode == "zero" and head is None:
            def hook(value: np.ndarray, name: str) -> np.ndarray:
                return np.zeros_like(value, dtype=np.float32)
            return hook

        if self.mode == "zero":
            def hook(value: np.ndarray, name: str) -> np.ndarray:  # noqa: F811
                out = np.array(value, dtype=np.float32, copy=True)
                if out.ndim == 3 and head < out.shape[1]:
                    out[:, head, :] = 0.0
                return out
            return hook

        source = self.means if self.mode == "mean" else self.donor_cache
        if hook_name not in source:
            raise KeyError(f"nessun baseline per l'hook {hook_name!r} (mode={self.mode})")
        replacement = np.asarray(source[hook_name], dtype=np.float32)

        def hook(value: np.ndarray, name: str) -> np.ndarray:  # noqa: F811
            out = np.array(value, dtype=np.float32, copy=True)
            if head is not None and out.ndim == 3:
                if head < out.shape[1]:
                    if replacement.ndim == 2 and replacement.shape[0] == out.shape[1]:
                        out[:, head, :] = replacement[head]
                    else:
                        out[:, head, :] = np.asarray(replacement).ravel()[: out.shape[2]]
                return out
            return _fit_to_shape(replacement, out.shape)

        return hook

    # ------------------------------------------------------------------

    def intervention(self, hook_name: str):
        """Intervention equivalente (solo per mode='zero' o 'mean')."""
        from .adapter import Intervention
        if self.mode == "zero":
            return Intervention(hook_name, kind="ablate")
        if self.mode == "mean":
            return Intervention(hook_name, kind="patch", value=self.means[hook_name])
        raise ValueError("resample non è esprimibile come Intervention statica: usa hook_for().")

    def __repr__(self) -> str:
        return (f"AblationController(mode={self.mode!r}, "
                f"baselines={len(self.means) or len(self.donor_cache)})")
