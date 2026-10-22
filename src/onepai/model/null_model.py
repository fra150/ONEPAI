"""
NullModelStudy — quanto è *normale* quello che abbiamo chiamato "pensiero soppresso"?

Un crush del 46% → 0% su un prompt scelto a mano è un aneddoto finché non sai
qual è la distribuzione di base. Se il 90% dei prompt di wikitext mostra token
che salgono e crollano nell'ultimo MLP, allora non hai trovato un pensiero
soppresso: hai misurato la calibrazione normale dell'ultimo layer.

Questo modulo misura la distribuzione nulla:

    per ogni prompt del corpus
        crush_profile()          → per ogni token del vocabolario: peak, final, crush
        top_crush                → quanto è forte il token più schiacciato
        n_crushed                → quanti token superano la soglia
        chief suppressor (MLP)   → chi lo sopprime, in modalità zero/mean/resample

e risponde a tre domande:
    1. il crush del prompt "speciale" è fuori dal range normale? (percentile)
    2. l'MLP dell'ultimo layer è arbitro anche su testo arbitrario? (frequenza)
    3. il chief suppressor cambia tra zero-ablation e mean-ablation? (robustezza)

Uso::

    study = NullModelStudy(tracer, crush_threshold=0.3)
    result = study.run(token_batches, prompts=texts)
    print(result.narrate())
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .suppression import SuppressionTracer

__all__ = ["PromptRecord", "NullModelResult", "NullModelStudy"]


@dataclass
class PromptRecord:
    """Cosa è successo su un singolo prompt."""
    prompt: str
    n_tokens: int
    top_token: int = -1
    top_token_str: str = ""
    top_crush: float = 0.0
    top_peak_layer: int = -1
    top_peak_prob: float = 0.0
    top_final_prob: float = 0.0
    mean_crush: float = 0.0
    n_crushed: int = 0                     # token sopra soglia con picco non finale
    chief: str = ""
    chief_restoration: float = 0.0
    chief_is_last_mlp: bool = False

    def as_row(self) -> Dict[str, str]:
        return {
            "Prompt": self.prompt[:64] + ("…" if len(self.prompt) > 64 else ""),
            "Top token": repr(self.top_token_str) if self.top_token_str else "—",
            "Crush": f"{self.top_crush:.2f}",
            "Peak": f"{self.top_peak_prob:.0f}% @L{self.top_peak_layer}",
            "Final": f"{self.top_final_prob*100:.1f}%",
            "Chief suppressor": (f"{self.chief} (+{self.chief_restoration*100:.0f}%)"
                                 if self.chief else "—"),
        }


@dataclass
class NullModelResult:
    """Distribuzione nulla del crush su un corpus di prompt arbitrari."""
    mode: str
    crush_threshold: float
    records: List[PromptRecord] = field(default_factory=list)
    chief_counts: Counter = field(default_factory=Counter)
    model_name: str = "model"
    n_layers: int = 0

    # -- aggregati ------------------------------------------------------

    @property
    def n_prompts(self) -> int:
        return len(self.records)

    @property
    def top_crushes(self) -> np.ndarray:
        return np.array([r.top_crush for r in self.records], dtype=np.float32)

    @property
    def mean_top_crush(self) -> float:
        c = self.top_crushes
        return float(c.mean()) if c.size else 0.0

    @property
    def median_top_crush(self) -> float:
        c = self.top_crushes
        return float(np.median(c)) if c.size else 0.0

    @property
    def p90_top_crush(self) -> float:
        c = self.top_crushes
        return float(np.percentile(c, 90)) if c.size else 0.0

    @property
    def frac_prompts_above(self) -> float:
        """Frazione di prompt con almeno un token sopra soglia di crush."""
        if not self.records:
            return 0.0
        return float(np.mean([r.top_crush >= self.crush_threshold for r in self.records]))

    @property
    def median_n_crushed(self) -> float:
        if not self.records:
            return 0.0
        return float(np.median([r.n_crushed for r in self.records]))

    @property
    def traced(self) -> List[PromptRecord]:
        return [r for r in self.records if r.chief]

    @property
    def last_mlp_name(self) -> str:
        return f"blocks.{self.n_layers - 1}.mlp"

    @property
    def last_mlp_rate(self) -> float:
        """Frazione dei prompt tracciati in cui il chief è l'MLP dell'ultimo layer."""
        t = self.traced
        if not t:
            return 0.0
        return float(np.mean([r.chief_is_last_mlp for r in t]))

    @property
    def modal_chief(self) -> str:
        return self.chief_counts.most_common(1)[0][0] if self.chief_counts else ""

    def percentile_of(self, crush: float) -> float:
        """In che percentile sta un crush rispetto alla distribuzione nulla."""
        c = self.top_crushes
        if not c.size:
            return 0.0
        return float((c < crush).mean() * 100.0)

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "mode": self.mode,
            "prompts": self.n_prompts,
            "threshold": self.crush_threshold,
            "mean_top_crush": round(self.mean_top_crush, 3),
            "median_top_crush": round(self.median_top_crush, 3),
            "p90_top_crush": round(self.p90_top_crush, 3),
            "frac_prompts_above": round(self.frac_prompts_above, 3),
            "median_n_crushed": self.median_n_crushed,
            "traced": len(self.traced),
            "modal_chief": self.modal_chief,
            "last_mlp_rate": round(self.last_mlp_rate, 3),
        }

    def narrate(self) -> str:
        s = self.summary()
        return (
            f"[{self.mode}-ablation] Su {s['prompts']} prompt arbitrari di corpus, "
            f"il {s['frac_prompts_above']*100:.0f}% ha almeno un token con crush "
            f">= {self.crush_threshold:.2f} (top-crush mediano {s['median_top_crush']:.2f}, "
            f"p90 {s['p90_top_crush']:.2f}). Tra i {s['traced']} prompt tracciati, "
            f"il chief suppressor è {s['modal_chief']} nel "
            f"{self.chief_counts.most_common(1)[0][1]/max(s['traced'],1)*100:.0f}% dei casi "
            f"e l'MLP dell'ultimo layer nel {s['last_mlp_rate']*100:.0f}%.")

    def histogram(self, bins: int = 12) -> Dict[str, List]:
        """Istogramma del top-crush per il grafico."""
        c = self.top_crushes
        if not c.size:
            return {}
        counts, edges = np.histogram(c, bins=bins, range=(0.0, 1.0))
        centers = (edges[:-1] + edges[1:]) / 2
        return {"count": [(float(x), float(y)) for x, y in zip(centers, counts)]}


class NullModelStudy:
    """
    Misura la distribuzione di base del crush su prompt arbitrari.

    Esempio::

        tracer = SuppressionTracer.with_ablation(ad, "mean", reference_batch=corpus)
        study = NullModelStudy(tracer, crush_threshold=0.3)
        result = study.run(token_batches)
    """

    def __init__(
        self,
        tracer: SuppressionTracer,
        crush_threshold: float = 0.3,
        components: str = "mlp",
        max_suppressors: int = 3,
    ):
        self.tracer = tracer
        self.model = tracer.model
        self.crush_threshold = crush_threshold
        self.components = components
        self.max_suppressors = max_suppressors

    # ------------------------------------------------------------------

    def analyze(self, tokens, prompt: str = "", trace_chief: bool = True) -> PromptRecord:
        """Profilo di crush di un prompt + (opzionale) il chief suppressor."""
        tokens = np.asarray(tokens)
        prof = self.tracer.crush_profile(tokens)
        n = self.model.n_layers

        # Un token che tocca il picco all'ultimo layer non è "quasi-detto e poi
        # soppresso": sta semplicemente emergendo. Va escluso dai candidati.
        eligible = prof.peak_layer < (n - 1)
        crush = np.where(eligible, prof.crush, 0.0)
        top = int(np.argmax(crush))
        n_crushed = int(np.sum(crush >= self.crush_threshold))

        rec = PromptRecord(
            prompt=prompt,
            n_tokens=int(tokens.shape[0]),
            top_token=top,
            top_crush=float(crush[top]),
            top_peak_layer=int(prof.peak_layer[top]),
            top_peak_prob=float(prof.peak_prob[top]),
            top_final_prob=float(prof.final_prob[top]),
            mean_crush=float(crush[eligible].mean()) if eligible.any() else 0.0,
            n_crushed=n_crushed,
        )
        if hasattr(self.model, "decode") and rec.top_crush > 0:
            try:
                rec.top_token_str = self.model.decode([top])
            except Exception:
                rec.top_token_str = ""
        if trace_chief and rec.top_crush > 1e-3:
            tr = self.tracer.trace(
                tokens, top, max_suppressors=self.max_suppressors,
                components=self.components)
            if tr.suppressors:
                rec.chief = tr.suppressors[0].component
                rec.chief_restoration = tr.suppressors[0].restoration
                rec.chief_is_last_mlp = rec.chief == f"blocks.{n - 1}.mlp"
        return rec

    # ------------------------------------------------------------------

    def run(
        self,
        token_batches: Sequence[np.ndarray],
        prompts: Optional[Sequence[str]] = None,
        trace_chief: bool = True,
        verbose: bool = False,
    ) -> NullModelResult:
        """Analizza un corpus di prompt e aggrega la distribuzione nulla."""
        result = NullModelResult(
            mode=self.tracer.ablation,
            crush_threshold=self.crush_threshold,
            model_name=self.model.model_name,
            n_layers=self.model.n_layers,
        )
        for i, tokens in enumerate(token_batches):
            prompt = prompts[i] if prompts is not None and i < len(prompts) else f"prompt[{i}]"
            rec = self.analyze(tokens, prompt=prompt, trace_chief=trace_chief)
            result.records.append(rec)
            if rec.chief:
                result.chief_counts[rec.chief] += 1
            if verbose:
                print(f"  [{i+1}/{len(token_batches)}] top crush {rec.top_crush:.2f} "
                      f"@L{rec.top_peak_layer} {rec.top_token_str!r} -> {rec.chief or '—'}")
        return result
