"""
SuppressionTracer — chi ha cancellato ciò che il modello stava per dire?

Collega le due metà di ONEPAI:
    AbsenceEngine     trova il token quasi-detto (sale e poi viene schiacciato)
    InterventionStudy trova il colpevole (quale componente, se rimosso, lo ripristina)

Logica:
  1. traiettoria della probabilità del token target, layer per layer (logit lens)
  2. picco = dove "stava per dirlo"; crush = quanto è stato soppresso
  3. per ogni componente DOPO il picco (MLP e singole teste), lo azzero e guardo
     se la probabilità FINALE del target risale: se sì, quel componente lo sopprimeva.

Restituisce la storia completa: "il token X arriva al 99% al layer 7, poi la
testa L9.H4 lo cancella" — la frase che il modello si è trattenuto dal dire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .adapter import ModelAdapter, Intervention, HookPoint
from .intervention_study import _head_ablation_hook

__all__ = ["SuppressionTrace", "Suppressor", "SuppressionTracer"]


@dataclass
class Suppressor:
    """Un componente che, se rimosso, ripristina il token soppresso."""
    component: str             # "blocks.9.mlp" | "blocks.9.head.4"
    restoration: float         # prob_finale(senza componente) - prob_finale(baseline)
    layer: int


@dataclass
class SuppressionTrace:
    """La storia completa di una predizione soppressa."""
    target_token: int
    token_str: Optional[str]
    trajectory: List[Tuple[int, float]]   # (layer, prob) per ogni layer
    peak_layer: int
    peak_prob: float
    final_prob: float
    crush: float                          # peak_prob - final_prob
    suppressors: List[Suppressor] = field(default_factory=list)

    @property
    def chief_suppressor(self) -> Optional[Suppressor]:
        return self.suppressors[0] if self.suppressors else None

    def narrate(self) -> str:
        """Una frase leggibile che racconta la soppressione."""
        tok = repr(self.token_str) if self.token_str is not None else f"token {self.target_token}"
        s = (f"{tok} sale al {self.peak_prob*100:.0f}% al layer {self.peak_layer}, "
             f"poi crolla al {self.final_prob*100:.1f}%")
        chief = self.chief_suppressor
        if chief and chief.restoration > 0:
            s += f" — soppresso da {chief.component} (+{chief.restoration*100:.0f}% se rimosso)"
        return s


class SuppressionTracer:
    """
    Traccia chi sopprime una predizione che il modello "stava per fare".

    Esempio::

        tracer = SuppressionTracer(adapter)
        traces = tracer.find_and_trace(adapter.encode("The capital of France is"))
        for t in traces:
            print(t.narrate())
    """

    def __init__(self, model: ModelAdapter):
        self.model = model
        if not model.supports_logit_lens():
            raise ValueError("Il modello deve supportare il logit lens (apply_unembed).")

    # ------------------------------------------------------------------
    # Traiettoria via logit lens
    # ------------------------------------------------------------------

    def trajectory(self, tokens, target_token: int, position: int = -1) -> List[Tuple[int, float]]:
        """Probabilità del target a ogni layer (proiezione del residual stream)."""
        _, cache = self.model.run_with_cache(tokens)
        traj = []
        for i in range(self.model.n_layers):
            resid = cache.get(HookPoint.resid_post(i))
            if resid is None:
                continue
            logits = self.model.apply_unembed(resid)[position]
            traj.append((i, float(_softmax(logits)[target_token])))
        return traj

    # ------------------------------------------------------------------
    # Traccia completa di un token
    # ------------------------------------------------------------------

    def trace(
        self,
        tokens,
        target_token: int,
        position: int = -1,
        max_suppressors: int = 6,
        include_peak_layer: bool = True,
    ) -> SuppressionTrace:
        model = self.model
        traj = self.trajectory(tokens, target_token, position)
        probs = np.array([p for _, p in traj])
        peak_idx = int(probs.argmax())
        peak_layer = traj[peak_idx][0]
        peak_prob = float(probs[peak_idx])
        final_prob = float(probs[-1])

        token_str = None
        if hasattr(model, "decode"):
            try:
                token_str = model.decode([target_token])
            except Exception:
                token_str = None

        trace = SuppressionTrace(
            target_token=target_token, token_str=token_str, trajectory=traj,
            peak_layer=peak_layer, peak_prob=peak_prob, final_prob=final_prob,
            crush=peak_prob - final_prob,
        )

        # Cerca i soppressori solo se c'è effettivamente uno schiacciamento
        if peak_prob - final_prob > 1e-3:
            start = peak_layer if include_peak_layer else peak_layer + 1
            trace.suppressors = self._find_suppressors(
                tokens, target_token, position, start, final_prob, max_suppressors)
        return trace

    def _find_suppressors(
        self, tokens, target_token, position, start_layer, baseline_final, max_suppressors,
    ) -> List[Suppressor]:
        model = self.model
        candidates: List[Suppressor] = []
        for layer in range(start_layer, model.n_layers):
            # MLP del layer
            mlp_prob = self._final_prob_after(
                tokens, [Intervention(HookPoint.mlp_out(layer), kind="ablate")],
                target_token, position)
            candidates.append(Suppressor(
                component=f"blocks.{layer}.mlp",
                restoration=mlp_prob - baseline_final, layer=layer))
            # Ogni testa del layer
            for h in range(model.n_heads):
                hook = (HookPoint.attn_z(layer), _head_ablation_hook(h))
                prob = self._final_prob_after_hooks(tokens, [hook], target_token, position)
                candidates.append(Suppressor(
                    component=f"blocks.{layer}.head.{h}",
                    restoration=prob - baseline_final, layer=layer))
        # Ordina per ripristino (i veri soppressori hanno restoration > 0)
        candidates.sort(key=lambda s: s.restoration, reverse=True)
        return candidates[:max_suppressors]

    # ------------------------------------------------------------------
    # Scoperta automatica dei token quasi-detti + traccia
    # ------------------------------------------------------------------

    def find_and_trace(
        self,
        tokens,
        position: int = -1,
        top_k: int = 3,
        crush_min: float = 0.05,
        max_suppressors: int = 6,
    ) -> List[SuppressionTrace]:
        """
        Trova i token più schiacciati (logit lens) e traccia il colpevole di ciascuno.
        """
        _, cache = self.model.run_with_cache(tokens)
        n = self.model.n_layers
        layer_probs = []
        for i in range(n):
            resid = cache.get(HookPoint.resid_post(i))
            if resid is None:
                return []
            layer_probs.append(_softmax(self.model.apply_unembed(resid)[position]))
        probs = np.stack(layer_probs, axis=0)        # (n_layers, vocab)
        peak = probs.max(axis=0)
        final = probs[-1]
        crush = peak - final
        peak_layer = probs.argmax(axis=0)
        # candidati: schiacciati e con picco prima dell'ultimo layer
        mask = (crush > crush_min) & (peak_layer < n - 1)
        cand = np.where(mask)[0]
        cand = cand[np.argsort(crush[cand])[::-1][:top_k]]
        return [self.trace(tokens, int(t), position, max_suppressors) for t in cand]

    # ------------------------------------------------------------------
    # Interni
    # ------------------------------------------------------------------

    def _final_prob_after(self, tokens, interventions, target_token, position) -> float:
        logits = self.model.run_with_interventions(tokens, interventions)
        return float(_softmax(logits[position])[target_token])

    def _final_prob_after_hooks(self, tokens, fwd_hooks, target_token, position) -> float:
        logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        return float(_softmax(logits[position])[target_token])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
