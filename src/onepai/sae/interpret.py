"""
FeatureInterpreter — interpretazione canonica delle feature: max-activating examples.

La lente dell'unembedding ("cosa promuoverebbe questa direzione") è veloce ma
approssimativa. Il metodo canonico del campo è guardare QUANDO la feature si
attiva: i passaggi del corpus che la accendono di più.

Questo modulo indicizza un corpus e per ogni feature mantiene i top-k contesti
che la attivano. L'interpretazione "duale" mette a confronto le due lenti:

    unembed lens : cosa la feature SCRIVE verso l'output
    max-activating: quando la feature SI ACCENDE (contesto reale evidenziato)

Se le due lenti raccontano la stessa storia, la feature è probabilmente
monosemantica; se divergono, è polisemantica o va guardata meglio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model.adapter import HookPoint

__all__ = ["FeatureExample", "DualInterpretation", "FeatureInterpreter"]


@dataclass
class FeatureExample:
    """Un passaggio del corpus che attiva fortemente una feature."""
    seq_idx: int
    position: int
    activation: float
    context: str          # testo con il token attivante evidenziato [[così]]
    token: str            # il token alla posizione attivante


@dataclass
class DualInterpretation:
    """Le due lenti su una feature, affiancate."""
    feature_idx: int
    unembed_tokens: List          # cosa promuove (lente unembedding)
    examples: List[FeatureExample]  # quando si accende (max-activating)
    firing_rate: float            # frazione di posizioni del corpus in cui spara

    def narrate(self) -> str:
        promo = ", ".join(repr(t) for t in self.unembed_tokens[:5])
        when = "; ".join(e.context for e in self.examples[:3])
        return (f"feature #{self.feature_idx} — promuove: {promo} | "
                f"si accende su: {when}")


class FeatureInterpreter:
    """
    Indicizza un corpus e interpreta le feature con il metodo canonico.

    Esempio::

        interp = FeatureInterpreter(model, sae, layer=6, k=8)
        interp.index_corpus(token_batch)          # una passata sul corpus
        dual = interp.dual_interpretation(222)     # entrambe le lenti
        for ex in dual.examples:
            print(ex.activation, ex.context)
    """

    def __init__(self, model, sae, layer: int, k: int = 8):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.k = k
        d = sae.d_hidden
        # top-k per feature: valori e riferimenti (seq_idx, position)
        self._top_vals = np.full((d, k), -np.inf, dtype=np.float32)
        self._top_refs = np.full((d, k, 2), -1, dtype=np.int64)
        self._sequences: List[np.ndarray] = []
        self._n_positions = 0
        self._fire_counts = np.zeros(d, dtype=np.int64)

    # ------------------------------------------------------------------
    # Indicizzazione
    # ------------------------------------------------------------------

    def index_corpus(self, token_batch: Sequence[np.ndarray],
                     skip_first: int = 1, fire_threshold: float = 1e-6) -> "FeatureInterpreter":
        """
        Passa il corpus nel modello, codifica con la SAE e aggiorna i top-k.
        """
        hook = HookPoint.resid_post(self.layer)
        for tokens in token_batch:
            tokens = np.asarray(tokens)
            seq_idx = len(self._sequences)
            self._sequences.append(tokens)
            _, cache = self.model.run_with_cache(tokens)
            resid = cache.get(hook)
            if resid is None:
                continue
            F = self.sae.encode(resid)                    # (seq, d_hidden)
            if skip_first > 0:
                F = F.copy()
                F[:skip_first] = 0.0
            self._n_positions += max(F.shape[0] - skip_first, 0)
            self._fire_counts += (F[skip_first:] > fire_threshold).sum(axis=0)
            self._merge_topk(F, seq_idx)
        return self

    def _merge_topk(self, F: np.ndarray, seq_idx: int) -> None:
        """Fonde le attivazioni di una sequenza nei top-k per feature."""
        seq_len, d = F.shape
        # candidati: (k + seq_len) valori per feature
        cand_vals = np.concatenate([self._top_vals, F.T], axis=1)         # (d, k+seq)
        new_refs = np.stack([
            np.full((d, seq_len), seq_idx, dtype=np.int64),
            np.tile(np.arange(seq_len, dtype=np.int64), (d, 1)),
        ], axis=2)                                                        # (d, seq, 2)
        cand_refs = np.concatenate([self._top_refs, new_refs], axis=1)    # (d, k+seq, 2)
        # scegli i k migliori per feature
        order = np.argsort(cand_vals, axis=1)[:, ::-1][:, :self.k]        # (d, k)
        rows = np.arange(d)[:, None]
        self._top_vals = np.take_along_axis(cand_vals, order, axis=1)
        self._top_refs = cand_refs[rows, order]

    # ------------------------------------------------------------------
    # Interpretazione
    # ------------------------------------------------------------------

    def top_examples(self, feature_idx: int, window: int = 7) -> List[FeatureExample]:
        """I passaggi del corpus che attivano di più la feature."""
        out = []
        for j in range(self.k):
            val = float(self._top_vals[feature_idx, j])
            if not np.isfinite(val) or val <= 0:
                continue
            seq_idx, pos = (int(x) for x in self._top_refs[feature_idx, j])
            if seq_idx < 0 or seq_idx >= len(self._sequences):
                continue
            tokens = self._sequences[seq_idx]
            out.append(FeatureExample(
                seq_idx=seq_idx, position=pos, activation=val,
                context=self._render_context(tokens, pos, window),
                token=self._decode_one(tokens[pos]),
            ))
        return out

    def firing_rate(self, feature_idx: int) -> float:
        if self._n_positions == 0:
            return 0.0
        return float(self._fire_counts[feature_idx]) / self._n_positions

    def unembed_lens(self, feature_idx: int, top_k: int = 8) -> List:
        """La lente rapida: cosa promuove la direzione della feature."""
        raw_dir = (self.sae.W_dec[feature_idx] * self.sae._std).astype(np.float32)
        logits = self.model.apply_unembed(raw_dir[None])
        if logits is None:
            return []
        top = np.argsort(logits[0])[::-1][:top_k]
        return [self._decode_one(int(t)) for t in top]

    def dual_interpretation(self, feature_idx: int, window: int = 7) -> DualInterpretation:
        """Le due lenti affiancate: cosa promuove + quando si accende."""
        return DualInterpretation(
            feature_idx=feature_idx,
            unembed_tokens=self.unembed_lens(feature_idx),
            examples=self.top_examples(feature_idx, window),
            firing_rate=self.firing_rate(feature_idx),
        )

    # ------------------------------------------------------------------
    # Interni
    # ------------------------------------------------------------------

    def _decode_one(self, token) -> str:
        if hasattr(self.model, "decode"):
            try:
                return self.model.decode([int(token)])
            except Exception:
                pass
        return str(int(token))

    def _render_context(self, tokens: np.ndarray, pos: int, window: int) -> str:
        lo = max(0, pos - window)
        hi = min(len(tokens), pos + window + 1)
        if hasattr(self.model, "decode"):
            try:
                before = self.model.decode(tokens[lo:pos]) if pos > lo else ""
                at = self.model.decode([tokens[pos]])
                after = self.model.decode(tokens[pos + 1:hi]) if pos + 1 < hi else ""
                return f"{before}[[{at}]]{after}".replace("\n", " ")
            except Exception:
                pass
        ids = tokens[lo:hi].tolist()
        rel = pos - lo
        parts = [f"[[{t}]]" if i == rel else str(t) for i, t in enumerate(ids)]
        return " ".join(parts)

    def __repr__(self) -> str:
        return (f"FeatureInterpreter(layer={self.layer}, k={self.k}, "
                f"sequences={len(self._sequences)}, positions={self._n_positions})")
