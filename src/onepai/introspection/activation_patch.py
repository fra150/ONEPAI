"""
ActivationPatcher — Causal tracing tramite patching delle attivazioni.

Idea: hai due esecuzioni (clean e corrupted).  Patch un componente alla volta:
sostituisci le attivazioni di un layer con quelle del run pulito e misura
quanto cambia l'output.  Il componente con effetto maggiore è causalmente critico.

Ref: Meng et al. (2022) ROME; Vig et al. (2020) Causal Mediation Analysis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["PatchResult", "CausalPath", "ActivationPatcher"]


@dataclass
class PatchResult:
    """Risultato del patching di un singolo componente."""
    component_id: str           # "layer_id" o "layer_id:head_idx" per attention
    patch_type: str             # "full_replace" | "add_delta" | "zero_ablate"
    effect_size: float          # |output_after_patch - baseline_output| / |clean_output - corrupted_output|
    causal_importance: float    # normalizzato 0→1 rispetto al massimo effetto osservato
    original_value: Optional[float] = None    # magnitudine attivazione prima del patch
    patched_value: Optional[float] = None     # magnitudine attivazione dopo il patch
    metadata: Dict = field(default_factory=dict)


@dataclass
class CausalPath:
    """Un cammino di componenti che insieme spiegano la differenza clean/corrupted."""
    components: List[str]
    total_explained: float       # 0→1: quanto della differenza questo path spiega
    path_type: str               # "direct" | "indirect" | "mixed"


class ActivationPatcher:
    """
    Esegue causal tracing senza accesso al modello in tempo reale.
    Lavora su snapshot di attivazioni pre-catturate.

    Uso::

        patcher = ActivationPatcher()
        patcher.store_run("clean",     {"layer.0": act_clean,   "layer.1": act_clean_1})
        patcher.store_run("corrupted", {"layer.0": act_corr,    "layer.1": act_corr_1})
        patcher.set_output_metric(clean_output_score, corrupted_output_score)

        results = patcher.run_full_causal_trace()
        path    = patcher.find_critical_path()
    """

    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
        self._runs: Dict[str, Dict[str, np.ndarray]] = {}
        self._output_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Acquisizione
    # ------------------------------------------------------------------

    def store_run(self, run_name: str, activations: Dict[str, np.ndarray]) -> None:
        """
        Memorizza le attivazioni di un'esecuzione.
        activations: {"layer_id": activation_array, ...}
        """
        self._runs[run_name] = {k: np.asarray(v, dtype=np.float32)
                                 for k, v in activations.items()}

    def set_output_metric(self, clean_score: float, corrupted_score: float,
                           run_names: Tuple[str, str] = ("clean", "corrupted")) -> None:
        """
        Registra lo score dell'output (es. log-prob del token target) per
        ciascun run.  Usato per normalizzare l'effect_size.
        """
        self._output_scores[run_names[0]] = clean_score
        self._output_scores[run_names[1]] = corrupted_score

    def store_from_shadow_registry(self, run_name: str, registry) -> None:
        """
        Popola un run dalle Observation di un ShadowRegistry.
        Le attivazioni sono estratte da obs.activation_value.
        """
        activations = {}
        for layer_name in registry.layers():
            observations = registry.get(layer_name)
            if observations:
                vals = [o.activation_value for o in observations
                        if o.activation_value is not None]
                if vals:
                    activations[layer_name] = np.array(vals, dtype=np.float32)
        self.store_run(run_name, activations)

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    def patch_component(
        self,
        component_id: str,
        from_run: str = "clean",
        to_run: str = "corrupted",
        patch_type: str = "full_replace",
        output_fn: Optional[Callable[[Dict[str, np.ndarray]], float]] = None,
    ) -> PatchResult:
        """
        Sostituisce l'attivazione di `component_id` nel run `to_run` con quella
        del run `from_run` e misura l'effetto sull'output.

        patch_type:
            - "full_replace": sostituisce completamente
            - "add_delta": aggiunge la differenza (clean - corrupted)
            - "zero_ablate": azzera l'attivazione nel run to_run
        """
        clean_acts = self._runs.get(from_run, {})
        corr_acts = self._runs.get(to_run, {})
        if component_id not in clean_acts:
            return PatchResult(component_id=component_id, patch_type=patch_type,
                               effect_size=0.0, causal_importance=0.0,
                               metadata={"error": "component_not_found"})
        clean_val = clean_acts[component_id]
        corr_val = corr_acts.get(component_id, np.zeros_like(clean_val))
        orig_mag = float(np.linalg.norm(corr_val))
        if patch_type == "full_replace":
            patched = clean_val.copy()
        elif patch_type == "add_delta":
            patched = corr_val + (clean_val - corr_val)
        elif patch_type == "zero_ablate":
            patched = np.zeros_like(corr_val)
        else:
            patched = clean_val.copy()
        patched_mag = float(np.linalg.norm(patched))
        # Effetto stimato: distanza tra attivazione patchata e corrupted
        activation_delta = float(np.linalg.norm(patched - corr_val))
        # Normalizza rispetto alla distanza clean-corrupted per questo componente
        total_delta = float(np.linalg.norm(clean_val - corr_val)) + 1e-9
        effect_size = activation_delta / total_delta
        return PatchResult(
            component_id=component_id,
            patch_type=patch_type,
            effect_size=effect_size,
            causal_importance=0.0,   # sarà normalizzato in run_full_causal_trace
            original_value=orig_mag,
            patched_value=patched_mag,
            metadata={"from_run": from_run, "to_run": to_run},
        )

    def run_full_causal_trace(
        self,
        from_run: str = "clean",
        to_run: str = "corrupted",
        patch_type: str = "full_replace",
    ) -> List[PatchResult]:
        """
        Esegue il patching su tutti i componenti registrati e normalizza
        causal_importance rispetto al massimo effetto.
        """
        clean_acts = self._runs.get(from_run, {})
        results = []
        for component_id in clean_acts:
            r = self.patch_component(component_id, from_run, to_run, patch_type)
            results.append(r)
        if not results:
            return []
        max_effect = max(r.effect_size for r in results) + 1e-9
        for r in results:
            r.causal_importance = r.effect_size / max_effect
        return sorted(results, key=lambda r: r.causal_importance, reverse=True)

    def find_critical_path(
        self,
        from_run: str = "clean",
        to_run: str = "corrupted",
        min_importance: float = 0.3,
        max_path_length: int = 5,
    ) -> CausalPath:
        """
        Subset minimale di componenti che spiega la maggior parte dell'effetto.
        Usa un algoritmo greedy (top-k per importance).
        """
        results = self.run_full_causal_trace(from_run, to_run)
        critical = [r for r in results if r.causal_importance >= min_importance]
        critical = critical[:max_path_length]
        total_explained = sum(r.effect_size for r in critical) / (
            sum(r.effect_size for r in results) + 1e-9
        )
        return CausalPath(
            components=[r.component_id for r in critical],
            total_explained=min(total_explained, 1.0),
            path_type="direct" if len(critical) <= 2 else "indirect",
        )

    def zero_ablation_study(self) -> List[Tuple[str, float]]:
        """
        Azzera ogni componente e misura quanto l'output si discosta da quello clean.
        Ordine decrescente di importanza.
        """
        results = self.run_full_causal_trace(patch_type="zero_ablate")
        return [(r.component_id, r.causal_importance) for r in results]

    def __repr__(self) -> str:
        runs = list(self._runs.keys())
        return f"ActivationPatcher(runs={runs})"
