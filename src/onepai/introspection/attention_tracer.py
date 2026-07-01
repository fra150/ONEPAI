"""
Tracciamento dei pattern di attenzione testa per testa.

Rivela:
- Teste "fantasma": quasi mai significative (max weight < soglia)
- Attenzione ombra: peso alto ma effetto nullo sull'output
- Entropia per testa: focalizzata (bassa) vs diffusa (alta)
- Flusso di attenzione aggregato tra posizioni token
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["AttentionRecord", "HeadSummary", "AttentionTracer"]


@dataclass
class AttentionRecord:
    """Un frame di attenzione catturato per una (layer, head) specifica."""
    layer_id: str
    head_idx: int
    step: int
    weights: np.ndarray        # shape: (seq_len, seq_len)  — righe=query, col=key
    query_context: Optional[Dict] = None


@dataclass
class HeadSummary:
    """Statistiche aggregate per una singola testa."""
    layer_id: str
    head_idx: int
    mean_entropy: float        # 0 = perfettamente focalizzata, log(n) = uniforme
    max_weight_ever: float     # massimo peso mai osservato in questa testa
    ghost_score: float         # 0 = attiva, 1 = fantasma completo
    shadow_score: float        # decorrelazione attenzione/effetto output
    observations: int


class AttentionTracer:
    """
    Registra e analizza i pattern di attenzione layer-per-layer.

    Esempio::

        tracer = AttentionTracer(ghost_threshold=0.1)
        tracer.record("transformer.0", head_idx=3, weights=attn_4x4)
        ghosts  = tracer.identify_ghost_heads()
        summary = tracer.summarize()
    """

    def __init__(self, ghost_threshold: float = 0.1):
        self.ghost_threshold = ghost_threshold
        # (layer_id, head_idx) → List[AttentionRecord]
        self._records: Dict[Tuple[str, int], List[AttentionRecord]] = defaultdict(list)
        self._step = 0

    # ------------------------------------------------------------------
    # Acquisizione
    # ------------------------------------------------------------------

    def record(
        self,
        layer_id: str,
        head_idx: int,
        weights: np.ndarray,
        step: Optional[int] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Salva una matrice di attenzione (seq_len × seq_len)."""
        if step is None:
            step = self._step
            self._step += 1
        w = np.asarray(weights, dtype=np.float32)
        # Normalizza le righe se non sono già una distribuzione
        row_sums = w.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        w = w / row_sums
        self._records[(layer_id, head_idx)].append(
            AttentionRecord(layer_id=layer_id, head_idx=head_idx,
                            step=step, weights=w, query_context=context)
        )

    def record_from_observations(self, observations: list) -> None:
        """
        Popola il tracer dai dati già presenti in onepai.core.Observation.
        Si aspetta che obs.metadata contenga 'attention_weights', 'head_idx'.
        """
        for obs in observations:
            meta = getattr(obs, "metadata", {}) or {}
            if "attention_weights" in meta:
                layer_id = obs.name.split(":")[0]
                head_idx = int(meta.get("head_idx", 0))
                step = int(meta.get("step", self._step))
                self.record(layer_id, head_idx, meta["attention_weights"],
                            step=step, context=meta.get("context"))

    # ------------------------------------------------------------------
    # Metriche per testa
    # ------------------------------------------------------------------

    def compute_head_entropy(self, layer_id: str, head_idx: int) -> float:
        """
        H = -sum(p * log(p+ε)) mediato su tutte le query positions e tutti gli step.
        Bassa = testa focalizzata (sa dove guardare).
        Alta = testa confusa (guarda tutto uguale).
        """
        records = self._records.get((layer_id, head_idx), [])
        if not records:
            return 0.0
        entropies = []
        for rec in records:
            w = np.clip(rec.weights, 1e-9, 1.0)
            h = -np.sum(w * np.log(w), axis=-1)   # (seq_len,)
            entropies.append(float(np.mean(h)))
        return float(np.mean(entropies))

    def compute_shadow_score(self, layer_id: str, head_idx: int,
                             output_deltas: Optional[np.ndarray] = None) -> float:
        """
        Stima quanto questa testa è "ombra": alto peso ma basso effetto sull'output.
        Se output_deltas è None restituisce 0 (impossibile calcolare senza ground truth).
        output_deltas: array (n_records,) — variazione dell'output al frame corrispondente.
        """
        records = self._records.get((layer_id, head_idx), [])
        if not records or output_deltas is None:
            return 0.0
        max_weights = np.array([float(r.weights.max()) for r in records])
        deltas = np.asarray(output_deltas, dtype=np.float32)[:len(max_weights)]
        if len(deltas) < 2:
            return 0.0
        # Correlazione di Pearson tra |weight| e |delta|
        corr = float(np.corrcoef(max_weights[:len(deltas)], deltas)[0, 1])
        if np.isnan(corr):
            return 0.0
        # Alta correlazione → testa attiva, zero/neg correlazione → testa ombra
        return float(np.clip(1.0 - corr, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Identificazione pattern
    # ------------------------------------------------------------------

    def identify_ghost_heads(self) -> List[HeadSummary]:
        """
        Teste "fantasma": il massimo peso mai osservato è < ghost_threshold.
        Queste teste non hanno mai guardato significativamente da nessuna parte.
        """
        return [s for s in self.summarize() if s.max_weight_ever < self.ghost_threshold]

    def identify_diffuse_heads(self, entropy_ratio: float = 0.8) -> List[HeadSummary]:
        """
        Teste "diffuse": entropia media > entropy_ratio * log(seq_len).
        Guardano tutto in modo uniforme senza focalizzarsi.
        """
        result = []
        for (layer_id, head_idx), records in self._records.items():
            if not records:
                continue
            n = records[0].weights.shape[-1]
            max_entropy = float(np.log(n + 1e-9))
            head_entropy = self.compute_head_entropy(layer_id, head_idx)
            if head_entropy > entropy_ratio * max_entropy:
                result.append(self._build_summary(layer_id, head_idx))
        return result

    # ------------------------------------------------------------------
    # Flusso di attenzione
    # ------------------------------------------------------------------

    def get_attention_flow(self, from_pos: int, to_pos: int) -> float:
        """
        Flusso medio di attenzione da `from_pos` a `to_pos`
        sommato su tutte le teste e tutti gli step.
        """
        total = 0.0
        count = 0
        for records in self._records.values():
            for rec in records:
                if from_pos < rec.weights.shape[0] and to_pos < rec.weights.shape[1]:
                    total += float(rec.weights[from_pos, to_pos])
                    count += 1
        return total / count if count > 0 else 0.0

    def build_influence_matrix(self, seq_len: int) -> np.ndarray:
        """
        Matrice (seq_len × seq_len): M[i,j] = attenzione media da i verso j
        aggregata su tutte le teste e tutti gli step.
        """
        mat = np.zeros((seq_len, seq_len), dtype=np.float32)
        counts = np.zeros((seq_len, seq_len), dtype=np.int32)
        for records in self._records.values():
            for rec in records:
                sl = min(rec.weights.shape[0], seq_len)
                sk = min(rec.weights.shape[1], seq_len)
                mat[:sl, :sk] += rec.weights[:sl, :sk]
                counts[:sl, :sk] += 1
        mask = counts > 0
        mat[mask] /= counts[mask]
        return mat

    def get_layer_attention_summary(self, layer_id: str) -> Dict:
        """
        Riassunto rapido di tutte le teste di un dato layer.
        """
        heads = [k[1] for k in self._records if k[0] == layer_id]
        if not heads:
            return {"layer": layer_id, "heads": 0, "ghost_heads": 0}
        summaries = [self._build_summary(layer_id, h) for h in heads]
        ghosts = sum(1 for s in summaries if s.max_weight_ever < self.ghost_threshold)
        return {
            "layer": layer_id,
            "heads": len(heads),
            "ghost_heads": ghosts,
            "mean_entropy": float(np.mean([s.mean_entropy for s in summaries])),
            "most_active_head": min(summaries, key=lambda s: s.ghost_score).head_idx,
        }

    # ------------------------------------------------------------------
    # Riepilogo globale
    # ------------------------------------------------------------------

    def summarize(self) -> List[HeadSummary]:
        """Tutte le teste registrate, ordinate da più fantasma a più attiva."""
        results = [
            self._build_summary(layer_id, head_idx)
            for (layer_id, head_idx) in self._records
        ]
        return sorted(results, key=lambda h: h.ghost_score, reverse=True)

    def _build_summary(self, layer_id: str, head_idx: int) -> HeadSummary:
        records = self._records[(layer_id, head_idx)]
        max_ever = max((float(r.weights.max()) for r in records), default=0.0)
        mean_ent = self.compute_head_entropy(layer_id, head_idx)
        ghost_score = float(np.clip(1.0 - max_ever / (self.ghost_threshold + 1e-9), 0.0, 1.0))
        return HeadSummary(
            layer_id=layer_id, head_idx=head_idx,
            mean_entropy=mean_ent, max_weight_ever=max_ever,
            ghost_score=ghost_score, shadow_score=0.0,
            observations=len(records),
        )

    def __repr__(self) -> str:
        n_heads = len(self._records)
        n_rec = sum(len(v) for v in self._records.values())
        return f"AttentionTracer(heads={n_heads}, records={n_rec})"
