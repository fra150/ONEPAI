"""
GradientFlowTracer — Mappa il flusso dei gradienti attraverso i parametri.

Identifica:
- Parametri "morti": gradiente costantemente vicino a zero
- Layer "pozzo": assorbono il gradiente senza propagarlo
- Rapporto segnale/rumore per layer
- Path del flusso dal loss verso l'input
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["GradientRecord", "GradientFlowTracer", "LayerGradientStats"]


@dataclass
class GradientRecord:
    """Un frame di gradiente catturato per un (layer, param) specifici."""
    layer_id: str
    param_name: str
    step: int
    magnitude: float       # L2 norm del tensore gradiente
    variance: float        # varianza degli elementi del gradiente
    sign_changes: int      # quante volte il gradiente ha cambiato segno rispetto a prima
    raw_stats: Dict = field(default_factory=dict)   # min, max, mean, std


@dataclass
class LayerGradientStats:
    """Statistiche aggregate per un layer attraverso tutti gli step."""
    layer_id: str
    mean_magnitude: float
    std_magnitude: float
    dead_ratio: float          # frazione dei parametri con |grad| < dead_threshold
    signal_noise_ratio: float  # mean / std (se std > 0)
    vanishing: bool            # gradiente mediamente < vanishing_threshold
    exploding: bool            # gradiente mediamente > exploding_threshold
    observations: int


class GradientFlowTracer:
    """
    Registra i gradienti passo per passo e analizza la salute del flusso.

    Uso::

        tracer = GradientFlowTracer()
        # Dopo ogni backward pass:
        tracer.record("attn.0", "W_Q", gradient_tensor)
        tracer.record("mlp.0",  "W_1", gradient_tensor)

        dead   = tracer.identify_dead_parameters()
        sinks  = tracer.identify_gradient_sinks()
        report = tracer.flow_report()

    Integrazione PyTorch (opzionale)::

        for name, param in model.named_parameters():
            if param.grad is not None:
                layer = name.rsplit(".", 1)[0]
                tracer.record(layer, name, param.grad.detach().cpu().numpy())
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 1e3
    DEAD_THRESHOLD = 1e-8

    def __init__(self):
        # (layer_id, param_name) → List[GradientRecord]
        self._records: Dict[Tuple[str, str], List[GradientRecord]] = defaultdict(list)
        self._step = 0

    # ------------------------------------------------------------------
    # Acquisizione
    # ------------------------------------------------------------------

    def record(
        self,
        layer_id: str,
        param_name: str,
        gradient: np.ndarray,
        step: Optional[int] = None,
    ) -> GradientRecord:
        """
        Salva le statistiche del tensore gradiente per (layer_id, param_name).
        Accetta numpy array, liste, o qualunque cosa convertibile a numpy.
        """
        g = np.asarray(gradient, dtype=np.float32).ravel()
        if step is None:
            step = self._step
            self._step += 1
        key = (layer_id, param_name)
        prev = self._records[key][-1] if self._records[key] else None
        sign_changes = 0
        if prev is not None and len(g) > 0:
            # Conta sign changes rispetto al gradiente precedente (stesso param)
            prev_sign = prev.raw_stats.get("_raw_sign")
            if prev_sign is not None and len(prev_sign) == len(g):
                sign_changes = int(np.sum(np.sign(g) != np.array(prev_sign)))
        rec = GradientRecord(
            layer_id=layer_id,
            param_name=param_name,
            step=step,
            magnitude=float(np.linalg.norm(g)),
            variance=float(np.var(g)),
            sign_changes=sign_changes,
            raw_stats={
                "min": float(g.min()) if len(g) else 0.0,
                "max": float(g.max()) if len(g) else 0.0,
                "mean": float(g.mean()) if len(g) else 0.0,
                "std": float(g.std()) if len(g) else 0.0,
                "n_elements": len(g),
                "dead_fraction": float((np.abs(g) < self.DEAD_THRESHOLD).mean()) if len(g) else 0.0,
                "_raw_sign": np.sign(g).tolist(),
            },
        )
        self._records[key].append(rec)
        return rec

    # ------------------------------------------------------------------
    # Analisi parametri
    # ------------------------------------------------------------------

    def identify_dead_parameters(self, n_steps: int = 5,
                                  threshold: float = 1e-7) -> List[Tuple[str, str, float]]:
        """
        Parametri in cui il gradiente è rimasto < threshold negli ultimi n_steps.
        Ritorna lista di (layer_id, param_name, mean_magnitude).
        """
        dead = []
        for (layer_id, param_name), records in self._records.items():
            recent = records[-n_steps:]
            if not recent:
                continue
            mean_mag = float(np.mean([r.magnitude for r in recent]))
            if mean_mag < threshold:
                dead.append((layer_id, param_name, mean_mag))
        return sorted(dead, key=lambda x: x[2])

    def identify_gradient_sinks(self,
                                 ratio_threshold: float = 5.0) -> List[str]:
        """
        Layer che assorbono il gradiente:  il loro gradiente è molto più grande
        di quello del layer successivo (gradiente 'sprecato' localmente).

        ratio_threshold: se magnitude[layer_i] / magnitude[layer_i+1] > ratio_threshold
        allora layer_i è un sink.
        """
        stats = self._compute_layer_stats()
        layers = sorted(stats.keys())
        sinks = []
        for i in range(len(layers) - 1):
            current = stats[layers[i]].mean_magnitude
            nxt = stats[layers[i + 1]].mean_magnitude
            if nxt > 0 and current / nxt > ratio_threshold:
                sinks.append(layers[i])
        return sinks

    def compute_signal_noise_ratio(self, layer_id: str) -> float:
        """
        SNR = mean(|gradient|) / std(|gradient|).
        Alto SNR = gradiente consistente; basso = rumoroso e inaffidabile.
        """
        records = [r for (lid, _), recs in self._records.items()
                   if lid == layer_id for r in recs]
        if not records:
            return 0.0
        mags = np.array([r.magnitude for r in records])
        s = float(mags.std())
        return float(mags.mean() / s) if s > 1e-9 else float(mags.mean())

    # ------------------------------------------------------------------
    # Riepilogo
    # ------------------------------------------------------------------

    def flow_report(self) -> Dict:
        """
        Report completo: layer ordinati dal loss verso l'input con magnitudine,
        flag vanishing/exploding, dead parameters.
        """
        stats = self._compute_layer_stats()
        dead = self.identify_dead_parameters()
        sinks = self.identify_gradient_sinks()
        return {
            "layers": {
                lid: {
                    "mean_magnitude": s.mean_magnitude,
                    "std_magnitude": s.std_magnitude,
                    "dead_ratio": s.dead_ratio,
                    "signal_noise_ratio": s.signal_noise_ratio,
                    "vanishing": s.vanishing,
                    "exploding": s.exploding,
                    "observations": s.observations,
                }
                for lid, s in stats.items()
            },
            "dead_parameters": dead,
            "gradient_sinks": sinks,
            "total_steps": self._step,
            "health": "OK" if not dead and not sinks else "DEGRADED",
        }

    def get_magnitude_trajectory(self, layer_id: str,
                                  param_name: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        Lista (step, magnitude) per un layer (media su tutti i param se param_name=None).
        """
        matching = {
            (lid, pn): recs
            for (lid, pn), recs in self._records.items()
            if lid == layer_id and (param_name is None or pn == param_name)
        }
        if not matching:
            return []
        # Aggrega per step
        step_map: Dict[int, List[float]] = defaultdict(list)
        for recs in matching.values():
            for r in recs:
                step_map[r.step].append(r.magnitude)
        return [(s, float(np.mean(mags))) for s, mags in sorted(step_map.items())]

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _compute_layer_stats(self) -> Dict[str, LayerGradientStats]:
        layer_records: Dict[str, List[GradientRecord]] = defaultdict(list)
        for (layer_id, _), recs in self._records.items():
            layer_records[layer_id].extend(recs)
        result = {}
        for layer_id, recs in layer_records.items():
            mags = np.array([r.magnitude for r in recs])
            dead_fractions = [r.raw_stats.get("dead_fraction", 0.0) for r in recs]
            mean_mag = float(mags.mean())
            std_mag = float(mags.std())
            result[layer_id] = LayerGradientStats(
                layer_id=layer_id,
                mean_magnitude=mean_mag,
                std_magnitude=std_mag,
                dead_ratio=float(np.mean(dead_fractions)),
                signal_noise_ratio=mean_mag / std_mag if std_mag > 1e-9 else mean_mag,
                vanishing=mean_mag < self.VANISHING_THRESHOLD,
                exploding=mean_mag > self.EXPLODING_THRESHOLD,
                observations=len(recs),
            )
        return result

    def __repr__(self) -> str:
        n = len(self._records)
        return f"GradientFlowTracer(params={n}, steps={self._step})"
