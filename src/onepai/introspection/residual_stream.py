"""
ResidualStreamAnalyzer — Decompone il residual stream nei contributi layer-by-layer.

In un transformer il residual stream accumula la somma di tutti i contributi
(attention + MLP) strato per strato.  Questo modulo:
- Registra il delta (contributo) di ogni layer
- Misura quanto ogni layer "muove" lo stream rispetto al totale
- Trova direzioni dominanti (PCA)
- Rileva interferenze (layer che si cancellano a vicenda)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["ResidualComponent", "StreamSnapshot", "ResidualStreamAnalyzer"]


@dataclass
class ResidualComponent:
    """Il contributo di un singolo layer al residual stream."""
    layer_type: str           # "attention" | "mlp" | "embedding" | "other"
    layer_idx: int
    layer_name: str
    position: int             # posizione token
    contribution: np.ndarray  # vettore delta aggiunto allo stream
    magnitude: float          # ||contribution||
    direction_alignment: float = 0.0   # coseno con la direzione finale (0→1)
    step: int = 0


@dataclass
class StreamSnapshot:
    """Lo stato del residual stream in un certo momento."""
    position: int
    step: int
    state: np.ndarray            # vettore corrente dello stream
    components_so_far: int
    magnitude: float


class ResidualStreamAnalyzer:
    """
    Analizza come ogni layer contribuisce al residual stream.

    Uso::

        analyzer = ResidualStreamAnalyzer(model_dim=768)
        # Registra il delta prodotto da ogni layer:
        analyzer.record_delta("attention", 0, "attn.0", position=5,
                               delta=attn_output_vector)
        analyzer.record_delta("mlp", 0, "mlp.0", position=5,
                               delta=mlp_output_vector)
        # Analisi:
        contributions = analyzer.decompose_at_position(5)
        dominant      = analyzer.find_dominant_directions(n_components=3)
        interferences = analyzer.detect_interference_patterns(position=5)
    """

    def __init__(self, model_dim: Optional[int] = None):
        self.model_dim = model_dim
        # position → List[ResidualComponent]
        self._components: Dict[int, List[ResidualComponent]] = defaultdict(list)
        self._stream_state: Dict[int, np.ndarray] = {}   # position → current stream
        self._step = 0

    # ------------------------------------------------------------------
    # Acquisizione
    # ------------------------------------------------------------------

    def record_delta(
        self,
        layer_type: str,
        layer_idx: int,
        layer_name: str,
        position: int,
        delta: np.ndarray,
        step: Optional[int] = None,
    ) -> ResidualComponent:
        """
        Registra il contributo di un layer al residual stream nella `position` data.
        delta: vettore (model_dim,) aggiunto allo stream da questo layer.
        """
        if step is None:
            step = self._step
            self._step += 1
        d = np.asarray(delta, dtype=np.float32)
        # Aggiorna lo stato corrente dello stream
        if position not in self._stream_state:
            self._stream_state[position] = np.zeros_like(d)
        self._stream_state[position] = self._stream_state[position] + d
        comp = ResidualComponent(
            layer_type=layer_type,
            layer_idx=layer_idx,
            layer_name=layer_name,
            position=position,
            contribution=d,
            magnitude=float(np.linalg.norm(d)),
            step=step,
        )
        self._components[position].append(comp)
        return comp

    def record_from_observations(self, observations: list, position: int = 0) -> None:
        """
        Popola l'analyzer dalle Observation di ShadowRegistry.
        Usa activation_value come stima del delta.
        """
        for i, obs in enumerate(observations):
            if obs.activation_value is None:
                continue
            v = np.asarray(obs.activation_value, dtype=np.float32)
            if v.ndim == 0:
                v = v.reshape(1)
            layer_name = obs.name.split(":")[0]
            layer_type = "attention" if "attn" in layer_name.lower() else "mlp"
            self.record_delta(layer_type, i, layer_name, position, v, step=i)

    # ------------------------------------------------------------------
    # Analisi
    # ------------------------------------------------------------------

    def decompose_at_position(self, position: int) -> List[ResidualComponent]:
        """
        Lista dei contributi a `position`, ordinati per layer_idx.
        Calcola anche direction_alignment rispetto alla direzione finale.
        """
        components = self._components.get(position, [])
        if not components:
            return []
        final_state = self._stream_state.get(position, np.zeros(1))
        final_norm = np.linalg.norm(final_state) + 1e-9
        final_dir = final_state / final_norm
        for comp in components:
            if comp.contribution.shape == final_dir.shape:
                comp.direction_alignment = float(
                    np.dot(comp.contribution / (comp.magnitude + 1e-9), final_dir)
                )
        return sorted(components, key=lambda c: (c.step, c.layer_idx))

    def measure_layer_importance(self, layer_name: str, position: int) -> float:
        """
        Importanza relativa di un layer: ||delta_layer|| / ||stream_final||.
        """
        components = self._components.get(position, [])
        matching = [c for c in components if c.layer_name == layer_name]
        if not matching:
            return 0.0
        delta_mag = float(np.linalg.norm(
            sum(c.contribution for c in matching)
        ))
        final_state = self._stream_state.get(position, np.zeros(1))
        final_mag = float(np.linalg.norm(final_state)) + 1e-9
        return delta_mag / final_mag

    def find_dominant_directions(self, position: int, n_components: int = 3) -> np.ndarray:
        """
        PCA sui vettori di contributo: trova le direzioni più "spiegate".
        Ritorna array (n_components, model_dim).
        """
        components = self._components.get(position, [])
        if not components:
            return np.array([])
        matrix = np.stack([c.contribution for c in components], axis=0)  # (n_layers, dim)
        # Centra
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        # SVD
        try:
            _, _, Vt = np.linalg.svd(matrix, full_matrices=False)
            return Vt[:n_components]
        except np.linalg.LinAlgError:
            return np.array([])

    def detect_interference_patterns(
        self,
        position: int,
        cosine_threshold: float = -0.5,
    ) -> List[Tuple[str, str, float]]:
        """
        Coppie di layer i cui contributi sono antiparalleli (cosine < threshold).
        Un'interferenza significa che si cancellano a vicenda — "spreco" computazionale.
        Ritorna lista di (layer_a, layer_b, cosine_similarity).
        """
        components = self._components.get(position, [])
        interferences = []
        for i, c1 in enumerate(components):
            for c2 in components[i + 1:]:
                if c1.magnitude < 1e-9 or c2.magnitude < 1e-9:
                    continue
                if c1.contribution.shape != c2.contribution.shape:
                    continue
                cosine = float(
                    np.dot(c1.contribution, c2.contribution)
                    / (c1.magnitude * c2.magnitude)
                )
                if cosine < cosine_threshold:
                    interferences.append((c1.layer_name, c2.layer_name, cosine))
        return sorted(interferences, key=lambda x: x[2])   # più negative prima

    def layer_type_breakdown(self, position: int) -> Dict[str, Dict]:
        """
        Riassunto per tipo di layer: contribution totale e % del totale.
        """
        components = self._components.get(position, [])
        type_mags: Dict[str, List[float]] = defaultdict(list)
        for c in components:
            type_mags[c.layer_type].append(c.magnitude)
        total = sum(sum(v) for v in type_mags.values()) + 1e-9
        return {
            ltype: {
                "total_magnitude": sum(mags),
                "percent": sum(mags) / total * 100,
                "n_layers": len(mags),
                "mean_per_layer": sum(mags) / len(mags),
            }
            for ltype, mags in type_mags.items()
        }

    def stream_trajectory(self, position: int) -> List[Tuple[int, float]]:
        """
        Come cresce la magnitudine del residual stream passo dopo passo.
        Ritorna lista di (step, ||stream||).
        """
        components = self._components.get(position, [])
        if not components:
            return []
        running_sum = np.zeros_like(components[0].contribution)
        traj = []
        for comp in sorted(components, key=lambda c: (c.step, c.layer_idx)):
            running_sum = running_sum + comp.contribution
            traj.append((comp.step, float(np.linalg.norm(running_sum))))
        return traj

    def summary(self) -> Dict:
        """Statistiche globali sull'analyzer."""
        all_comps = [c for comps in self._components.values() for c in comps]
        if not all_comps:
            return {"positions": 0, "components": 0}
        return {
            "positions": len(self._components),
            "components": len(all_comps),
            "layer_types": list({c.layer_type for c in all_comps}),
            "mean_magnitude": float(np.mean([c.magnitude for c in all_comps])),
            "max_magnitude": float(np.max([c.magnitude for c in all_comps])),
        }

    def __repr__(self) -> str:
        n_pos = len(self._components)
        n_comp = sum(len(v) for v in self._components.values())
        return f"ResidualStreamAnalyzer(positions={n_pos}, components={n_comp})"
