"""
Logit Lens — Cosa "penserebbe" il modello se si fermasse a ogni layer?

Proietta il residual stream a ogni layer intermedio attraverso la matrice
di unembedding finale, rivelando come le predizioni evolvono strato per strato.

Riferimento: nostalgebraist/logit-lens (2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["LayerPrediction", "LogitLens"]


@dataclass
class LayerPrediction:
    """Snapshot di cosa il modello "direbbe" fermandosi a questo layer."""
    layer_idx: int
    layer_name: str
    position: int
    top_token_ids: List[int]       # top-k token IDs
    top_probs: List[float]          # probabilità corrispondenti
    confidence: float               # max(prob) — sicurezza della predizione
    entropy: float                  # entropia della distribuzione su vocabolario
    step: int = 0


class LogitLens:
    """
    Registra le proiezioni intermedie e analizza come le predizioni emergono.

    Uso senza modello reale (da dati già catturati)::

        lens = LogitLens(vocab_size=50257, top_k=5)
        # Se hai il residual stream e la matrice W_U:
        lens.record_from_residual("layer.3", position=7, residual=h, W_U=W_U)
        # Oppure direttamente i logits pre-calcolati:
        lens.record_logits("layer.3", position=7, logits=logit_vec)

        evolution = lens.get_prediction_evolution(position=7)
        layer_of_emergence = lens.find_emergence_layer(token_id=1234)
    """

    def __init__(self, vocab_size: int = 50257, top_k: int = 5):
        self.vocab_size = vocab_size
        self.top_k = top_k
        # layer_name → List[LayerPrediction]
        self._predictions: Dict[str, List[LayerPrediction]] = {}
        self._layer_order: List[str] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Acquisizione
    # ------------------------------------------------------------------

    def record_from_residual(
        self,
        layer_name: str,
        position: int,
        residual: np.ndarray,
        W_U: np.ndarray,
        layer_idx: Optional[int] = None,
        temperature: float = 1.0,
        step: Optional[int] = None,
    ) -> LayerPrediction:
        """
        Proietta `residual` (dim,) attraverso W_U (dim × vocab) e
        salva la distribuzione risultante.
        """
        residual = np.asarray(residual, dtype=np.float32)
        W_U = np.asarray(W_U, dtype=np.float32)
        logits = residual @ W_U          # (vocab_size,)
        return self.record_logits(layer_name, position, logits,
                                  layer_idx=layer_idx, temperature=temperature, step=step)

    def record_logits(
        self,
        layer_name: str,
        position: int,
        logits: np.ndarray,
        layer_idx: Optional[int] = None,
        temperature: float = 1.0,
        step: Optional[int] = None,
    ) -> LayerPrediction:
        """
        Salva i logit grezzi (o già proiettati) per un (layer, position).
        """
        if step is None:
            step = self._step
            self._step += 1
        logits = np.asarray(logits, dtype=np.float32)
        probs = self._softmax(logits / max(temperature, 1e-6))
        top_ids = np.argsort(probs)[::-1][: self.top_k].tolist()
        top_probs = [float(probs[i]) for i in top_ids]
        ent = float(-np.sum(probs * np.log(probs + 1e-9)))
        if layer_idx is None:
            layer_idx = len(self._layer_order)
        pred = LayerPrediction(
            layer_idx=layer_idx,
            layer_name=layer_name,
            position=position,
            top_token_ids=top_ids,
            top_probs=top_probs,
            confidence=float(probs.max()),
            entropy=ent,
            step=step,
        )
        if layer_name not in self._predictions:
            self._predictions[layer_name] = []
            self._layer_order.append(layer_name)
        self._predictions[layer_name].append(pred)
        return pred

    # ------------------------------------------------------------------
    # Analisi longitudinale
    # ------------------------------------------------------------------

    def get_prediction_evolution(self, position: int, step: int = 0) -> List[LayerPrediction]:
        """
        Sequenza ordinata di predizioni per `position` attraverso tutti i layer.
        Mostra come la risposta si forma strato dopo strato.
        """
        result = []
        for layer_name in self._layer_order:
            matching = [
                p for p in self._predictions.get(layer_name, [])
                if p.position == position and p.step == step
            ]
            if matching:
                result.append(matching[-1])
        return sorted(result, key=lambda p: p.layer_idx)

    def find_emergence_layer(self, token_id: int, position: int = 0,
                              min_prob: float = 0.05) -> Optional[str]:
        """
        Primo layer in cui `token_id` appare nei top-k con probabilità >= min_prob.
        Restituisce None se il token non emerge mai.
        """
        for pred in self.get_prediction_evolution(position):
            if token_id in pred.top_token_ids:
                idx = pred.top_token_ids.index(token_id)
                if pred.top_probs[idx] >= min_prob:
                    return pred.layer_name
        return None

    def get_confidence_trajectory(self, position: int, step: int = 0) -> List[Tuple[str, float]]:
        """
        Lista (layer_name, confidence) — quanto cresce la certezza strato per strato.
        Una traiettoria piatta indica che il layer non contribuisce alla decisione.
        """
        return [
            (p.layer_name, p.confidence)
            for p in self.get_prediction_evolution(position, step)
        ]

    def get_entropy_trajectory(self, position: int, step: int = 0) -> List[Tuple[str, float]]:
        """
        Lista (layer_name, entropy) — come decresce il disordine attraverso i layer.
        """
        return [
            (p.layer_name, p.entropy)
            for p in self.get_prediction_evolution(position, step)
        ]

    def find_critical_layers(self, position: int, step: int = 0,
                             min_confidence_jump: float = 0.1) -> List[str]:
        """
        Layer in cui la confidenza aumenta di almeno `min_confidence_jump`.
        Questi sono i layer che "fanno il lavoro vero".
        """
        traj = self.get_confidence_trajectory(position, step)
        if len(traj) < 2:
            return []
        critical = []
        for i in range(1, len(traj)):
            delta = traj[i][1] - traj[i - 1][1]
            if delta >= min_confidence_jump:
                critical.append(traj[i][0])
        return critical

    def find_prediction_flips(self, position: int, step: int = 0) -> List[Tuple[str, int, int]]:
        """
        Layer in cui il token più probabile cambia.
        Ritorna lista di (layer_name, old_top_token, new_top_token).
        """
        evolution = self.get_prediction_evolution(position, step)
        flips = []
        for i in range(1, len(evolution)):
            prev_top = evolution[i - 1].top_token_ids[0]
            curr_top = evolution[i].top_token_ids[0]
            if prev_top != curr_top:
                flips.append((evolution[i].layer_name, prev_top, curr_top))
        return flips

    def summary(self) -> Dict:
        """Statistiche globali sulla lens raccolta."""
        all_preds = [p for preds in self._predictions.values() for p in preds]
        if not all_preds:
            return {"layers": 0, "predictions": 0}
        return {
            "layers": len(self._layer_order),
            "predictions": len(all_preds),
            "mean_confidence": float(np.mean([p.confidence for p in all_preds])),
            "mean_entropy": float(np.mean([p.entropy for p in all_preds])),
            "positions_tracked": len({p.position for p in all_preds}),
        }

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def __repr__(self) -> str:
        return (f"LogitLens(layers={len(self._layer_order)}, "
                f"vocab={self.vocab_size}, top_k={self.top_k})")
