"""
AbsenceEngine — l'interpretabilità dell'assenza, su un modello reale.

La tesi originale di ONEPAI: non basta dire "questo neurone è spento".
Conta *perché* è assente. Distinguiamo:

    DEAD        — niente lo guida: pre-attivazione ≈ 0 sempre (peso inerte)
    SUPPRESSED  — qualcosa lo schiaccia: pre-attivazione forte ma NEGATIVA,
                  gated a zero dalla GELU. Il modello *sceglie* di spegnerlo.

E due forme di assenza più profonde:

    CRUSHED     — token che il logit lens vede salire strato dopo strato e poi
                  venire schiacciati prima dell'output: ciò che il modello
                  "stava per dire" ma ha soppresso.
    ABORTED     — direzioni nel residual stream aggiunte da un layer e poi
                  cancellate (antiparallele) da un layer successivo: percorsi
                  di pensiero iniziati e abbandonati.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapter import ModelAdapter, HookPoint
from ..introspection import ResidualStreamAnalyzer

__all__ = ["NeuronAbsence", "CrushedToken", "AbortedDirection",
           "AbsenceReport", "AbsenceEngine"]


@dataclass
class NeuronAbsence:
    """Un neurone assente, con la diagnosi del *perché*."""
    layer: str
    neuron_idx: int
    kind: str                  # "dead" | "suppressed"
    fired_ratio: float         # frazione di campioni in cui ha superato la soglia
    mean_pre: float            # pre-attivazione media (segno conta)
    mean_abs_pre: float        # |pre| media (forza del segnale in ingresso)
    suppression_score: float   # 0 = morto puro, 1 = soppresso con forza

    @property
    def name(self) -> str:
        return f"{self.layer}.{self.neuron_idx}"


@dataclass
class CrushedToken:
    """Un token quasi-detto: probabilità salita e poi schiacciata."""
    token_id: int
    peak_layer: int
    peak_prob: float
    final_prob: float
    crush: float               # peak_prob - final_prob (quanto è stato soppresso)


@dataclass
class AbortedDirection:
    """Un contributo aggiunto da un layer e poi cancellato da un altro."""
    layer_a: str
    layer_b: str
    cosine: float              # ≈ -1 = perfettamente antiparallelo


@dataclass
class AbsenceReport:
    model_name: str
    dead_neurons: List[NeuronAbsence] = field(default_factory=list)
    suppressed_neurons: List[NeuronAbsence] = field(default_factory=list)
    crushed_tokens: List[CrushedToken] = field(default_factory=list)
    aborted_directions: List[AbortedDirection] = field(default_factory=list)
    n_sequences: int = 0

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "dead": len(self.dead_neurons),
            "suppressed": len(self.suppressed_neurons),
            "crushed_tokens": len(self.crushed_tokens),
            "aborted_directions": len(self.aborted_directions),
            "sequences": self.n_sequences,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"AbsenceReport(dead={s['dead']}, suppressed={s['suppressed']}, "
                f"crushed={s['crushed_tokens']}, aborted={s['aborted_directions']})")


class AbsenceEngine:
    """
    Analizza le forme dell'assenza in un modello reale.

    Esempio::

        engine = AbsenceEngine(model)
        report = engine.analyze([model.random_tokens(16, seed=k) for k in range(40)])
        print(report.summary())
        for n in report.suppressed_neurons:
            print(n.name, "soppresso, forza", round(n.suppression_score, 2))
    """

    def __init__(
        self,
        model: ModelAdapter,
        fire_threshold: float = 0.05,      # post-GELU sopra cui il neurone "spara"
        fired_tolerance: float = 0.02,     # fired_ratio sotto cui è assente
        suppression_pre: float = 1.0,      # |pre| sopra cui il segnale è "forte"
        crush_min: float = 0.02,           # soglia minima di schiacciamento token
        max_lens_sequences: int = 25,      # quante sequenze usare per crushed/aborted
    ):
        self.model = model
        self.fire_threshold = fire_threshold
        self.fired_tolerance = fired_tolerance
        self.suppression_pre = suppression_pre
        self.crush_min = crush_min
        self.max_lens_sequences = max_lens_sequences

    # ------------------------------------------------------------------

    def analyze(self, token_batch: Sequence[np.ndarray]) -> AbsenceReport:
        model = self.model
        report = AbsenceReport(model_name=model.model_name, n_sequences=len(token_batch))

        # Accumula pre/post per layer su tutto il batch
        pre_acc: Dict[int, List[np.ndarray]] = defaultdict(list)
        post_acc: Dict[int, List[np.ndarray]] = defaultdict(list)
        crush_acc: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        supports_lens = model.supports_logit_lens()

        for s, tokens in enumerate(token_batch):
            _, cache = model.run_with_cache(tokens)
            for i in range(model.n_layers):
                pre = cache.get(HookPoint.mlp_pre(i))
                post = cache.get(HookPoint.mlp_post(i))
                if pre is not None:
                    pre_acc[i].append(np.asarray(pre))
                if post is not None:
                    post_acc[i].append(np.asarray(post))

            # Crushed predictions (logit lens reale) su un sottoinsieme
            if supports_lens and s < self.max_lens_sequences:
                self._collect_crushed(cache, model, crush_acc)

        # --- Classifica i neuroni assenti ---
        for i in sorted(pre_acc):
            pre = np.concatenate(pre_acc[i], axis=0)    # (N, d_mlp)
            post = np.concatenate(post_acc[i], axis=0)
            self._classify_layer(f"blocks.{i}.mlp", pre, post, report)

        # --- Crushed tokens aggregati ---
        report.crushed_tokens = self._aggregate_crushed(crush_acc)

        # --- Aborted directions (su una sequenza rappresentativa) ---
        if len(token_batch) > 0:
            report.aborted_directions = self._find_aborted(token_batch[0])

        return report

    # ------------------------------------------------------------------
    # Classificazione neuroni: dead vs suppressed
    # ------------------------------------------------------------------

    def _classify_layer(self, layer: str, pre: np.ndarray, post: np.ndarray,
                        report: AbsenceReport) -> None:
        d_mlp = pre.shape[1]
        for n in range(d_mlp):
            fired_ratio = float((post[:, n] > self.fire_threshold).mean())
            if fired_ratio >= self.fired_tolerance:
                continue  # il neurone spara abbastanza: non è assente
            mean_pre = float(pre[:, n].mean())
            mean_abs_pre = float(np.abs(pre[:, n]).mean())
            # Soppresso: segnale forte ma spinto in negativo (gated dalla GELU)
            if mean_abs_pre >= self.suppression_pre and mean_pre < 0:
                score = float(np.clip(mean_abs_pre / (self.suppression_pre * 3), 0, 1))
                report.suppressed_neurons.append(NeuronAbsence(
                    layer=layer, neuron_idx=n, kind="suppressed",
                    fired_ratio=fired_ratio, mean_pre=mean_pre,
                    mean_abs_pre=mean_abs_pre, suppression_score=score,
                ))
            else:
                report.dead_neurons.append(NeuronAbsence(
                    layer=layer, neuron_idx=n, kind="dead",
                    fired_ratio=fired_ratio, mean_pre=mean_pre,
                    mean_abs_pre=mean_abs_pre, suppression_score=0.0,
                ))

    # ------------------------------------------------------------------
    # Crushed predictions: token quasi-detti
    # ------------------------------------------------------------------

    def _collect_crushed(self, cache, model, crush_acc) -> None:
        """Per la posizione finale, traiettoria di prob per layer via logit lens."""
        layer_probs = []
        for i in range(model.n_layers):
            resid = cache.get(HookPoint.resid_post(i))
            if resid is None:
                return
            logits = model.apply_unembed(resid)[-1]      # (vocab,)
            layer_probs.append(_softmax(logits))
        probs = np.stack(layer_probs, axis=0)            # (n_layers, vocab)
        peak_layer = probs.argmax(axis=0)                # per token
        peak_prob = probs.max(axis=0)
        final_prob = probs[-1]
        crush = peak_prob - final_prob
        # token con picco prima dell'ultimo layer e schiacciamento significativo
        n_layers = probs.shape[0]
        for tok in np.where((crush > self.crush_min) & (peak_layer < n_layers - 1))[0]:
            crush_acc[int(tok)].append(
                (int(peak_layer[tok]), float(peak_prob[tok]), float(final_prob[tok]))
            )

    def _aggregate_crushed(self, crush_acc) -> List[CrushedToken]:
        out = []
        for tok, entries in crush_acc.items():
            peaks = [e[1] for e in entries]
            finals = [e[2] for e in entries]
            peak_layers = [e[0] for e in entries]
            mean_peak = float(np.mean(peaks))
            mean_final = float(np.mean(finals))
            out.append(CrushedToken(
                token_id=tok,
                peak_layer=int(np.round(np.mean(peak_layers))),
                peak_prob=mean_peak, final_prob=mean_final,
                crush=mean_peak - mean_final,
            ))
        return sorted(out, key=lambda c: c.crush, reverse=True)[:20]

    # ------------------------------------------------------------------
    # Aborted directions: contributi cancellati tra layer
    # ------------------------------------------------------------------

    def _find_aborted(self, tokens) -> List[AbortedDirection]:
        _, cache = self.model.run_with_cache(tokens)
        ra = ResidualStreamAnalyzer(model_dim=self.model.d_model)
        for i in range(self.model.n_layers):
            attn_out = cache.get(HookPoint.attn_out(i))
            mlp_out = cache.get(HookPoint.mlp_out(i))
            if attn_out is not None:
                ra.record_delta("attention", i, f"blocks.{i}.attn",
                                position=0, delta=attn_out[-1], step=i)
            if mlp_out is not None:
                ra.record_delta("mlp", i, f"blocks.{i}.mlp",
                                position=0, delta=mlp_out[-1], step=i)
        interferences = ra.detect_interference_patterns(0, cosine_threshold=-0.4)
        return [AbortedDirection(layer_a=a, layer_b=b, cosine=c)
                for a, b, c in interferences]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
