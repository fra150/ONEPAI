"""
SuppressionTracer — chi ha cancellato ciò che il modello stava per dire?

Collega le due metà di ONEPAI:
    AbsenceEngine     trova il token quasi-detto (sale e poi viene schiacciato)
    InterventionStudy trova il colpevole (quale componente, se rimosso, lo ripristina)

Logica:
  1. traiettoria della probabilità del token target, layer per layer (logit lens)
  2. picco = dove "stava per dirlo"; crush = quanto è stato soppresso
  3. per ogni componente DOPO il picco (MLP e singole teste), lo sostituisci con
     la sua baseline e guardi se la probabilità FINALE del target risale: se sì,
     quel componente lo sopprimeva.

La sostituzione avviene in una delle tre modalità di `AblationController`:
    zero      — il componente viene azzerato (storico, fuori distribuzione)
    mean      — sostituito con la media sul corpus (in distribuzione)
    resample  — sostituito con l'attivazione di un altro prompt (controfattuale)

Un soppressore trovato con `zero` vale solo se regge anche con `mean`/`resample`.

Restituisce la storia completa: "il token X arriva al 99% al layer 7, poi la
testa L9.H4 lo cancella" — la frase che il modello si è trattenuto dal dire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ablation import AblationController
from .adapter import ModelAdapter, HookPoint

__all__ = ["Suppressor", "SuppressionTrace", "CrushProfile", "SuppressionTracer"]


@dataclass
class Suppressor:
    """Un componente che, se rimosso, ripristina il token soppresso."""
    component: str             # "blocks.9.mlp" | "blocks.9.head.4"
    restoration: float         # prob_finale(senza componente) - prob_finale(baseline)
    layer: int


@dataclass
class CrushProfile:
    """
    Profilo di soppressione di TUTTI i token del vocabolario a una posizione.

    Un solo forward: il logit lens dà, per ogni token, quanto è salito e poi
    sceso. È il materiale grezzo del null model (quanto è normale un crush?).
    """
    crush: np.ndarray          # (vocab,) peak - final, >= 0
    peak_prob: np.ndarray      # (vocab,)
    final_prob: np.ndarray     # (vocab,)
    peak_layer: np.ndarray     # (vocab,) layer in cui il token tocca il picco
    final_layer: int

    def top(self, k: int = 5, min_peak_layer: Optional[int] = None,
            exclude: Sequence[int] = ()) -> List[int]:
        """Indici dei k token più schiacciati (con eventuali esclusioni)."""
        crush = self.crush.copy()
        if exclude:
            crush[np.asarray(list(exclude), dtype=int)] = -np.inf
        if min_peak_layer is not None:
            crush[self.peak_layer >= min_peak_layer] = -np.inf
        order = np.argsort(crush)[::-1][:k]
        return [int(i) for i in order if np.isfinite(crush[i])]


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
    ablation: str = "zero"                # modalità con cui sono stati cercati

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
            s += f" — soppresso da {chief.component} (+{chief.restoration*100:.0f}% se rimosso"
            s += f", ablation={self.ablation})"
        return s


class SuppressionTracer:
    """
    Traccia chi sopprime una predizione che il modello "stava per fare".

    Esempio::

        tracer = SuppressionTracer(adapter)
        traces = tracer.find_and_trace(adapter.encode("The capital of France is"))

    Con ablation in distribuzione (raccomandato per i risultati da riportare)::

        tracer = SuppressionTracer.with_ablation(adapter, "mean",
                                                 reference_batch=corpus)
    """

    def __init__(
        self,
        model: ModelAdapter,
        ablation: str = "zero",
        controller: Optional[AblationController] = None,
        lens=None,
    ):
        self.model = model
        if not model.supports_logit_lens():
            raise ValueError("Il modello deve supportare il logit lens (apply_unembed).")
        self.ablation = ablation
        # lens opzionale (es. RidgeTunedLens): se presente, le proiezioni dei
        # layer intermedi passano da lui invece che dal grezzo apply_unembed.
        self.lens = lens
        if controller is None and ablation == "zero":
            controller = AblationController(model, mode="zero")
        if controller is None:
            raise ValueError(
                f"ablation={ablation!r} richiede un AblationController: usa "
                "SuppressionTracer.with_ablation(model, mode, reference_batch=...).")
        self.controller = controller

    @classmethod
    def with_ablation(
        cls,
        model: ModelAdapter,
        mode: str = "mean",
        reference_batch: Optional[Sequence[np.ndarray]] = None,
        donor_tokens: Optional[np.ndarray] = None,
        lens=None,
    ) -> "SuppressionTracer":
        """Costruisce il tracer con baseline mean/resample calcolate sul corpus."""
        controller = AblationController.build(
            model, mode=mode, reference_batch=reference_batch,
            donor_tokens=donor_tokens)
        return cls(model, ablation=mode, controller=controller, lens=lens)

    def _layer_logits(self, resid: np.ndarray, layer: int) -> np.ndarray:
        """Logit del layer: tuned lens se disponibile, altrimenti grezzo."""
        if self.lens is not None:
            return self.lens.logits(resid, layer)
        return self.model.apply_unembed(resid)

    # ------------------------------------------------------------------
    # Profilo di crush (un forward, tutto il vocabolario)
    # ------------------------------------------------------------------

    def crush_profile(self, tokens, position: int = -1) -> CrushProfile:
        """
        Per ogni token del vocabolario: probabilità massima lungo i layer
        (logit lens) e probabilità finale. Il crush è la differenza.
        """
        _, cache = self.model.run_with_cache(tokens)
        n = self.model.n_layers
        layer_probs = []
        last_layer = -1
        for i in range(n):
            resid = cache.get(HookPoint.resid_post(i))
            if resid is None:
                continue
            layer_probs.append(_softmax(self._layer_logits(resid, i)[position]))
            last_layer = i
        if not layer_probs:
            raise RuntimeError("Nessun resid_post catturato: logit lens impossibile.")
        probs = np.stack(layer_probs, axis=0)          # (n_layers_seen, vocab)
        peak = probs.max(axis=0)
        final = probs[-1]
        return CrushProfile(
            crush=np.maximum(peak - final, 0.0),
            peak_prob=peak,
            final_prob=final,
            peak_layer=probs.argmax(axis=0),
            final_layer=last_layer,
        )

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
            logits = self._layer_logits(resid, i)[position]
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
        components: str = "all",
    ) -> SuppressionTrace:
        """
        Storia di un token: traiettoria + chi lo sopprime.

        components: "all" (MLP + teste) | "mlp" (solo MLP, molto più economico:
                    n_layers forward invece di n_layers * (1 + n_heads))
        """
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
            crush=peak_prob - final_prob, ablation=self.ablation,
        )

        # Cerca i soppressori solo se c'è effettivamente uno schiacciamento
        if peak_prob - final_prob > 1e-3:
            start = peak_layer if include_peak_layer else peak_layer + 1
            trace.suppressors = self._find_suppressors(
                tokens, target_token, position, start, final_prob, max_suppressors,
                components=components)
        return trace

    def _find_suppressors(
        self, tokens, target_token, position, start_layer, baseline_final,
        max_suppressors, components: str = "all",
    ) -> List[Suppressor]:
        model = self.model
        candidates: List[Suppressor] = []
        for layer in range(start_layer, model.n_layers):
            # MLP del layer
            hook_name = HookPoint.mlp_out(layer)
            prob = self._prob_after_hooks(
                tokens, [(hook_name, self.controller.hook_for(hook_name))],
                target_token, position)
            candidates.append(Suppressor(
                component=f"blocks.{layer}.mlp",
                restoration=prob - baseline_final, layer=layer))

            # Ogni testa del layer
            if components == "all":
                z_name = HookPoint.attn_z(layer)
                for h in range(model.n_heads):
                    prob = self._prob_after_hooks(
                        tokens, [(z_name, self.controller.hook_for(z_name, head=h))],
                        target_token, position)
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
        components: str = "all",
    ) -> List[SuppressionTrace]:
        """
        Trova i token più schiacciati (logit lens) e traccia il colpevole di ciascuno.
        """
        prof = self.crush_profile(tokens, position)
        n = self.model.n_layers
        # candidati: schiacciati sopra soglia e con picco prima dell'ultimo layer
        mask = (prof.crush > crush_min) & (prof.peak_layer < n - 1)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        cand = idx[np.argsort(prof.crush[idx])[::-1][:top_k]]
        return [self.trace(tokens, int(t), position, max_suppressors,
                           components=components) for t in cand]

    # ------------------------------------------------------------------
    # Interni
    # ------------------------------------------------------------------

    def _final_prob_after(self, tokens, interventions, target_token, position) -> float:
        logits = self.model.run_with_interventions(tokens, interventions)
        return float(_softmax(logits[position])[target_token])

    def _prob_after_hooks(self, tokens, fwd_hooks, target_token, position) -> float:
        logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        return float(_softmax(logits[position])[target_token])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
