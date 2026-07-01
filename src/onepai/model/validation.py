"""
SuppressionValidator — da caso di studio a legge (o quasi).

"blocks.11.mlp sopprime ' France'" misurato su UNA frase è un aneddoto.
Questo modulo lo mette alla prova:

    TREATMENT — N prompt dello stesso tipo (es. "The capital of X is" con
                target X: la ripetizione che il modello considera e sopprime).
                Per ciascuno: c'è crush? chi è il top suppressor?
    CONTROL   — prompt in cui il target è la RISPOSTA GIUSTA (es.
                "Paris is the capital of" con target ' France'): qui la
                soppressione NON deve avvenire (crush ≈ 0).

Metriche:
    consistency        — frazione dei casi treatment in cui il soppressore
                         modale è il top-1 (e top-3)
    mean_crush_treat   — quanto forte è la soppressione nei treatment
    mean_crush_control — deve restare bassa nei controlli
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .suppression import SuppressionTracer

__all__ = ["ValidationCase", "ValidationResult", "SuppressionValidator"]


@dataclass
class ValidationCase:
    prompt: str
    target_str: str
    target_token: int
    peak_layer: int = -1
    peak_prob: float = 0.0
    final_prob: float = 0.0
    crush: float = 0.0
    top_suppressor: str = ""
    top_restoration: float = 0.0
    suppressors: List[Tuple[str, float]] = field(default_factory=list)   # top-3
    skipped: Optional[str] = None       # motivo se il caso è stato saltato


@dataclass
class ValidationResult:
    treatment: List[ValidationCase] = field(default_factory=list)
    control: List[ValidationCase] = field(default_factory=list)
    modal_suppressor: str = ""
    consistency_top1: float = 0.0
    consistency_top3: float = 0.0
    mean_crush_treatment: float = 0.0
    mean_crush_control: float = 0.0

    def summary(self) -> dict:
        return {
            "treatment_cases": len([c for c in self.treatment if not c.skipped]),
            "control_cases": len([c for c in self.control if not c.skipped]),
            "modal_suppressor": self.modal_suppressor,
            "consistency_top1": round(self.consistency_top1, 2),
            "consistency_top3": round(self.consistency_top3, 2),
            "mean_crush_treatment": round(self.mean_crush_treatment, 3),
            "mean_crush_control": round(self.mean_crush_control, 3),
        }

    def narrate(self) -> str:
        s = self.summary()
        return (f"Su {s['treatment_cases']} prompt, il soppressore modale è "
                f"{self.modal_suppressor} (top-1 nel {self.consistency_top1*100:.0f}% "
                f"dei casi, top-3 nel {self.consistency_top3*100:.0f}%). "
                f"Crush medio: {s['mean_crush_treatment']} nei treatment vs "
                f"{s['mean_crush_control']} nei controlli.")


class SuppressionValidator:
    """
    Valida la coerenza del soppressore su più prompt + controlli.

    Esempio (soppressione della ripetizione)::

        v = SuppressionValidator(adapter)
        treatment = [(f"The capital of {c} is", f" {c}") for c in countries]
        control   = [("Paris is the capital of", " France"), ...]
        result = v.validate(treatment, control)
        print(result.narrate())
    """

    def __init__(self, model, crush_threshold: float = 0.1,
                 max_suppressors: int = 3):
        self.model = model
        self.tracer = SuppressionTracer(model)
        self.crush_threshold = crush_threshold
        self.max_suppressors = max_suppressors

    # ------------------------------------------------------------------

    def run_case(self, prompt: str, target_str: str,
                 find_suppressors: bool = True) -> ValidationCase:
        """Analizza un singolo (prompt, target). Salta i target multi-token."""
        enc = self.model.encode(target_str)
        if len(enc) != 1:
            return ValidationCase(prompt=prompt, target_str=target_str,
                                  target_token=-1,
                                  skipped=f"target multi-token ({len(enc)})")
        target = int(enc[0])
        tokens = self.model.encode(prompt)
        tr = self.tracer.trace(tokens, target,
                               max_suppressors=self.max_suppressors
                               if find_suppressors else 0)
        case = ValidationCase(
            prompt=prompt, target_str=target_str, target_token=target,
            peak_layer=tr.peak_layer, peak_prob=tr.peak_prob,
            final_prob=tr.final_prob, crush=tr.crush,
        )
        if tr.suppressors:
            case.top_suppressor = tr.suppressors[0].component
            case.top_restoration = tr.suppressors[0].restoration
            case.suppressors = [(s.component, s.restoration)
                                for s in tr.suppressors[:3]]
        return case

    # ------------------------------------------------------------------

    def validate(
        self,
        treatment: Sequence[Tuple[str, str]],
        control: Sequence[Tuple[str, str]] = (),
        verbose: bool = False,
    ) -> ValidationResult:
        result = ValidationResult()

        for prompt, target in treatment:
            case = self.run_case(prompt, target, find_suppressors=True)
            result.treatment.append(case)
            if verbose and not case.skipped:
                print(f"  [T] {prompt!r} -> crush {case.crush:.2f}, "
                      f"top: {case.top_suppressor}")

        for prompt, target in control:
            # nei controlli non serve cercare il colpevole: conta solo il crush
            case = self.run_case(prompt, target, find_suppressors=False)
            result.control.append(case)
            if verbose and not case.skipped:
                print(f"  [C] {prompt!r} -> crush {case.crush:.2f}")

        self._aggregate(result)
        return result

    # ------------------------------------------------------------------

    def _aggregate(self, result: ValidationResult) -> None:
        valid_t = [c for c in result.treatment
                   if not c.skipped and c.crush >= self.crush_threshold]
        valid_c = [c for c in result.control if not c.skipped]

        if valid_t:
            result.mean_crush_treatment = float(np.mean([c.crush for c in valid_t]))
            counts = Counter(c.top_suppressor for c in valid_t if c.top_suppressor)
            if counts:
                result.modal_suppressor = counts.most_common(1)[0][0]
                n = len(valid_t)
                result.consistency_top1 = counts[result.modal_suppressor] / n
                in_top3 = sum(
                    1 for c in valid_t
                    if result.modal_suppressor in [s for s, _ in c.suppressors]
                )
                result.consistency_top3 = in_top3 / n
        if valid_c:
            result.mean_crush_control = float(np.mean([c.crush for c in valid_c]))
