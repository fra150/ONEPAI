"""
InterventionStudy — osservazione controllata: cosa succede se sottraggo o aggiungo?

Due sweep sistematici sul modello reale:

    ABLATION  ("cosa sottratta")  — azzera ogni componente (MLP, attention, testa)
                                    uno alla volta e misura l'effetto causale sul
                                    logit di un token target.
    STEERING  ("direzione futura") — aggiunge una direzione al residual stream con
                                    coefficienti crescenti e traccia la risposta.

L'effetto è sempre misurato come   logit_target(intervento) - logit_target(baseline).
Effetto negativo = il componente *sosteneva* il token (toglierlo lo abbassa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .adapter import ModelAdapter, Intervention, HookPoint

__all__ = ["InterventionEffect", "SteeringResponse",
           "InterventionStudyResult", "InterventionStudy"]


@dataclass
class InterventionEffect:
    component: str            # "blocks.0.mlp" | "blocks.1.attn" | "blocks.1.head.3"
    kind: str                # "ablate"
    effect: float            # logit_target(intervento) - baseline  (segno conta)

    @property
    def abs_effect(self) -> float:
        return abs(self.effect)


@dataclass
class SteeringResponse:
    hook_name: str
    label: str
    coefficients: List[float]
    target_logits: List[float]
    baseline: float

    def as_series(self) -> List:
        return list(zip(self.coefficients, self.target_logits))


@dataclass
class InterventionStudyResult:
    model_name: str
    target_token: int
    ablation_effects: List[InterventionEffect] = field(default_factory=list)
    head_effects: List[InterventionEffect] = field(default_factory=list)
    steering: List[SteeringResponse] = field(default_factory=list)

    def top_components(self, k: int = 5) -> List[InterventionEffect]:
        allfx = self.ablation_effects + self.head_effects
        return sorted(allfx, key=lambda e: e.abs_effect, reverse=True)[:k]

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "target_token": self.target_token,
            "components_tested": len(self.ablation_effects) + len(self.head_effects),
            "steering_sweeps": len(self.steering),
            "most_causal": self.top_components(1)[0].component if self.ablation_effects or self.head_effects else None,
        }

    # -- report -----------------------------------------------------------

    def to_report(self, title: str = "ONEPAI · Intervention Study"):
        from ..visualization import HTMLReport, ActivationHeatmap, line_chart

        report = HTMLReport(title=title, model_name=self.model_name)
        report.add_summary({
            "Target token": self.target_token,
            "Components tested": len(self.ablation_effects) + len(self.head_effects),
            "Steering sweeps": len(self.steering),
        })

        # Heatmap effetti per componente (layer x tipo)
        if self.ablation_effects:
            layers = sorted({e.component.rsplit(".", 1)[0] for e in self.ablation_effects})
            kinds = ["attn", "mlp"]
            mat = np.zeros((len(layers), len(kinds)), dtype=np.float32)
            for e in self.ablation_effects:
                base, kind = e.component.rsplit(".", 1)
                if base in layers and kind in kinds:
                    mat[layers.index(base), kinds.index(kind)] = e.effect
            svg = ActivationHeatmap(28).render(
                mat, title="Ablation effect on target logit",
                x_labels=kinds, y_labels=layers, diverging=True)
            report.add_section("Ablation Sweep", svg, icon="▽",
                               note="Blu = toglierlo abbassa il token (lo sosteneva). "
                                    "Rosso = lo alzava.")

        # Tabella teste piu' causali
        if self.head_effects:
            rows = [{"Head": e.component, "Effect on target": f"{e.effect:+.3f}"}
                    for e in sorted(self.head_effects, key=lambda e: e.abs_effect, reverse=True)[:12]]
            report.add_table("Head Ablation", rows, icon="◉")

        # Curve di steering
        if self.steering:
            series = {sw.label: sw.as_series() for sw in self.steering}
            svg = line_chart(series, title="Steering response (target logit vs coefficient)",
                             x_label="coefficient", y_label="target logit")
            report.add_section("Steering Sweep", svg, icon="◈",
                               note="Aggiungere una direzione e osservare il logit del target.")
        return report


class InterventionStudy:
    """
    Esegue sweep di ablazione e steering su un ModelAdapter.

    Esempio::

        study = InterventionStudy(model)
        result = study.run_full(tokens, target_token=42)
        print(result.summary())
        result.to_report().save("intervention.html")
    """

    def __init__(self, model: ModelAdapter):
        self.model = model

    # ------------------------------------------------------------------

    def baseline_logit(self, tokens, target_token: int, position: int = -1) -> float:
        logits, _ = self.model.run_with_cache(tokens)
        return float(logits[position, target_token])

    def ablation_sweep(
        self,
        tokens,
        target_token: int,
        position: int = -1,
    ) -> List[InterventionEffect]:
        """Azzera MLP e attention di ogni layer; misura l'effetto sul target."""
        base = self.baseline_logit(tokens, target_token, position)
        effects = []
        for i in range(self.model.n_layers):
            for kind, hook in (("attn", HookPoint.attn_out(i)), ("mlp", HookPoint.mlp_out(i))):
                iv = Intervention(hook, kind="ablate")
                logits = self.model.run_with_interventions(tokens, [iv])
                effect = float(logits[position, target_token]) - base
                effects.append(InterventionEffect(
                    component=f"blocks.{i}.{kind}", kind="ablate", effect=effect))
        return sorted(effects, key=lambda e: e.abs_effect, reverse=True)

    def head_ablation_sweep(
        self,
        tokens,
        target_token: int,
        position: int = -1,
    ) -> List[InterventionEffect]:
        """Azzera il contributo z di ogni singola testa; misura l'effetto."""
        base = self.baseline_logit(tokens, target_token, position)
        effects = []
        for i in range(self.model.n_layers):
            for h in range(self.model.n_heads):
                hook = (HookPoint.attn_z(i), _head_ablation_hook(h))
                logits = self.model.run_with_hooks(tokens, fwd_hooks=[hook])
                effect = float(logits[position, target_token]) - base
                effects.append(InterventionEffect(
                    component=f"blocks.{i}.head.{h}", kind="ablate", effect=effect))
        return sorted(effects, key=lambda e: e.abs_effect, reverse=True)

    def steering_sweep(
        self,
        tokens,
        hook_name: str,
        direction: np.ndarray,
        target_token: int,
        coefficients: Optional[Sequence[float]] = None,
        position: int = -1,
        label: str = "",
    ) -> SteeringResponse:
        """
        Aggiunge `coef * direction` al hook e traccia il logit del target.
        """
        if coefficients is None:
            coefficients = [-4, -2, -1, 0, 1, 2, 4]
        direction = np.asarray(direction, dtype=np.float32)
        base = self.baseline_logit(tokens, target_token, position)
        out_logits = []
        for coef in coefficients:
            if coef == 0:
                out_logits.append(base)
                continue
            iv = Intervention(hook_name, kind="add", value=direction, coefficient=float(coef))
            logits = self.model.run_with_interventions(tokens, [iv])
            out_logits.append(float(logits[position, target_token]))
        return SteeringResponse(
            hook_name=hook_name, label=label or hook_name,
            coefficients=list(coefficients), target_logits=out_logits, baseline=base)

    def target_direction(self, target_token: int) -> Optional[np.ndarray]:
        """Direzione di unembedding del token target (se il modello la espone)."""
        return self.model.unembed_direction(target_token)

    def run_full(
        self,
        tokens,
        target_token: int,
        position: int = -1,
        include_steering: bool = True,
    ) -> InterventionStudyResult:
        """Sweep completo: ablazione componenti + teste + (opz.) steering."""
        result = InterventionStudyResult(
            model_name=self.model.model_name, target_token=target_token)
        result.ablation_effects = self.ablation_sweep(tokens, target_token, position)
        result.head_effects = self.head_ablation_sweep(tokens, target_token, position)
        if include_steering:
            direction = self.target_direction(target_token)
            if direction is not None:
                last_layer = self.model.n_layers - 1
                result.steering.append(self.steering_sweep(
                    tokens, HookPoint.resid_post(last_layer), direction, target_token,
                    label=f"resid_post[{last_layer}] += W_U[{target_token}]"))
        return result


def _head_ablation_hook(head: int) -> Callable:
    """Hook che azzera il contributo della testa `head` su attn.z (seq, n_heads, d_head)."""
    def hook(value: np.ndarray, name: str) -> np.ndarray:
        out = np.array(value, dtype=np.float32, copy=True)
        if out.ndim == 3 and head < out.shape[1]:
            out[:, head, :] = 0.0
        return out
    return hook
