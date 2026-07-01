"""
FeatureSuppressionTracer — il ponte tra SAE e soppressione.

La domanda più profonda di ONEPAI: non "quale *token* il modello si trattiene",
ma "quale *concetto*". Una SAE dà un dizionario di feature (idealmente
monosemantiche). Questo modulo:

  1. proietta le feature della SAE a OGNI layer (feature-lens): come si accende
     e si spegne ogni feature lungo la rete;
  2. trova le feature SOPPRESSE — quelle che salgono e poi vengono schiacciate
     prima dell'output: concetti che il modello costruisce e poi abbandona;
  3. INTERPRETA ogni feature decodificando la sua direzione attraverso
     l'unembedding → i token che quella feature promuove (il suo "significato");
  4. trova CHI la sopprime, azzerando i componenti tardivi e guardando quale
     ripristina l'attivazione della feature.

Output: "il modello costruisce la feature Y (~[Paris, France, capital]) al
layer 7, poi blocks.11.mlp la cancella" — il concetto trattenuto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..model.adapter import HookPoint
from ..model.intervention_study import _head_ablation_hook

__all__ = ["FeatureTrajectory", "FeatureSuppressor", "FeatureSuppressionTracer"]


@dataclass
class FeatureSuppressor:
    component: str
    restoration: float          # attivazione_feature(senza comp) - baseline
    layer: int


@dataclass
class FeatureTrajectory:
    """La storia di una feature della SAE lungo la rete."""
    feature_idx: int
    trajectory: List[Tuple[int, float]]    # (layer, attivazione)
    peak_layer: int
    peak_act: float
    final_act: float
    crush: float
    top_tokens: List = field(default_factory=list)       # significato (token o stringhe)
    suppressors: List[FeatureSuppressor] = field(default_factory=list)

    @property
    def meaning(self) -> str:
        toks = [repr(t) if isinstance(t, str) else str(t) for t in self.top_tokens[:6]]
        return ", ".join(toks)

    def narrate(self) -> str:
        s = (f"feature #{self.feature_idx} (~{self.meaning}) sale a {self.peak_act:.2f} "
             f"al layer {self.peak_layer}, poi scende a {self.final_act:.2f}")
        if self.suppressors and self.suppressors[0].restoration > 0:
            c = self.suppressors[0]
            s += f" — soppressa da {c.component} (+{c.restoration:.2f} se rimosso)"
        return s


class FeatureSuppressionTracer:
    """
    Traccia la soppressione a livello di FEATURE (non di token).

    Richiede una SAE già addestrata sulle attivazioni di `resid_post(layer)`.

    Esempio::

        sae = SparseAutoencoder(d_in=768, d_hidden=1024, l1_coef=0.008)
        sae.fit(collect_activations(ad, batch, HookPoint.resid_post(6)))
        tracer = FeatureSuppressionTracer(ad, sae, layer=6)
        for t in tracer.find_suppressed_features(ad.encode("The capital of France is")):
            print(t.narrate())
    """

    def __init__(self, model, sae, layer: int):
        self.model = model
        self.sae = sae
        self.layer = layer
        if not model.supports_logit_lens():
            raise ValueError("Il modello deve supportare il logit lens (apply_unembed).")

    # ------------------------------------------------------------------
    # Feature-lens: attivazioni delle feature a ogni layer
    # ------------------------------------------------------------------

    def feature_acts_all_layers(self, tokens, position: int = -1) -> np.ndarray:
        """(n_layers, d_hidden): attivazione di ogni feature a ogni layer."""
        _, cache = self.model.run_with_cache(tokens)
        rows = []
        for i in range(self.model.n_layers):
            resid = cache.get(HookPoint.resid_post(i))
            if resid is None:
                continue
            f = self.sae.encode(resid[position][None])[0]
            rows.append(f)
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Interpretazione: cosa "significa" una feature
    # ------------------------------------------------------------------

    def interpret_feature(self, feature_idx: int, top_k: int = 8) -> List:
        """
        I token che la direzione della feature promuove, via unembedding.
        De-standardizza la direzione del decoder prima di proiettarla.
        """
        raw_dir = (self.sae.W_dec[feature_idx] * self.sae._std).astype(np.float32)
        logits = self.model.apply_unembed(raw_dir[None])[0]
        top = np.argsort(logits)[::-1][:top_k]
        if hasattr(self.model, "decode"):
            try:
                return [self.model.decode([int(t)]) for t in top]
            except Exception:
                pass
        return [int(t) for t in top]

    # ------------------------------------------------------------------
    # Traccia completa di una feature
    # ------------------------------------------------------------------

    def trace_feature(
        self,
        tokens,
        feature_idx: int,
        position: int = -1,
        find_suppressors: bool = True,
        max_suppressors: int = 6,
        feature_acts: Optional[np.ndarray] = None,
    ) -> FeatureTrajectory:
        F = feature_acts if feature_acts is not None else self.feature_acts_all_layers(tokens, position)
        traj_vals = F[:, feature_idx]
        traj = [(i, float(v)) for i, v in enumerate(traj_vals)]
        peak_layer = int(traj_vals.argmax())
        peak_act = float(traj_vals[peak_layer])
        final_act = float(traj_vals[-1])
        ft = FeatureTrajectory(
            feature_idx=feature_idx, trajectory=traj, peak_layer=peak_layer,
            peak_act=peak_act, final_act=final_act, crush=peak_act - final_act,
            top_tokens=self.interpret_feature(feature_idx),
        )
        if find_suppressors and ft.crush > 1e-4:
            ft.suppressors = self._find_suppressors(
                tokens, feature_idx, position, peak_layer, final_act, max_suppressors)
        return ft

    # ------------------------------------------------------------------
    # Scoperta delle feature soppresse
    # ------------------------------------------------------------------

    def find_suppressed_features(
        self,
        tokens,
        position: int = -1,
        top_k: int = 5,
        crush_min: float = 0.3,
        min_peak: float = 0.5,
        find_suppressors: bool = False,
    ) -> List[FeatureTrajectory]:
        """
        Feature che salgono (peak > min_peak) e poi crollano (crush > crush_min)
        con il picco prima dell'ultimo layer.
        """
        F = self.feature_acts_all_layers(tokens, position)
        peak = F.max(axis=0)
        final = F[-1]
        crush = peak - final
        peak_layer = F.argmax(axis=0)
        n = F.shape[0]
        mask = (crush > crush_min) & (peak > min_peak) & (peak_layer < n - 1)
        cand = np.where(mask)[0]
        cand = cand[np.argsort(crush[cand])[::-1][:top_k]]
        return [self.trace_feature(tokens, int(f), position,
                                   find_suppressors=find_suppressors, feature_acts=F)
                for f in cand]

    # ------------------------------------------------------------------
    # Chi sopprime la feature
    # ------------------------------------------------------------------

    def _find_suppressors(
        self, tokens, feature_idx, position, start_layer, baseline_final, max_suppressors,
    ) -> List[FeatureSuppressor]:
        cands: List[FeatureSuppressor] = []
        last = self.model.n_layers - 1
        for layer in range(start_layer, self.model.n_layers):
            # MLP del layer
            act = self._feature_final_after(
                tokens, [(HookPoint.mlp_out(layer), _ablate_fn)], feature_idx, position)
            cands.append(FeatureSuppressor(f"blocks.{layer}.mlp", act - baseline_final, layer))
            # teste
            for h in range(self.model.n_heads):
                act = self._feature_final_after(
                    tokens, [(HookPoint.attn_z(layer), _head_ablation_hook(h))],
                    feature_idx, position)
                cands.append(FeatureSuppressor(f"blocks.{layer}.head.{h}",
                                               act - baseline_final, layer))
        cands.sort(key=lambda s: s.restoration, reverse=True)
        return cands[:max_suppressors]

    def _feature_final_after(self, tokens, ablation_hooks, feature_idx, position) -> float:
        """Attivazione della feature all'ultimo layer dopo un'ablazione."""
        captured = {}
        last = self.model.n_layers - 1

        def cap(v, name):
            captured["r"] = np.array(v, copy=True)
            return v

        hooks = list(ablation_hooks) + [(HookPoint.resid_post(last), cap)]
        self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        resid = captured.get("r")
        if resid is None:
            return 0.0
        return float(self.sae.encode(resid[position][None])[0, feature_idx])


def _ablate_fn(value: np.ndarray, name: str) -> np.ndarray:
    return np.zeros_like(value)
