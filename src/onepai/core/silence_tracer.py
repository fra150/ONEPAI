# -----------------------------------------------------------------------------
# File: onepai/core/silence_tracer.py
# -----------------------------------------------------------------------------

"""Identifica e analizza i neuroni e i layer cronicamente silenti."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

from onepai.core.shadow_mapper import ShadowMap

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SilentEntity:
    """Rappresenta un'entità (neurone/layer) identificata come silente."""
    name: str  # e.g., "layer1.0.relu" o "layer1.0.relu.42"
    silence_ratio: float
    total_observations: int
    entity_type: str  # "neuron" o "layer"


@dataclass(slots=True)
class SilenceReport:
    """Report che riassume le entità silenti trovate in una ShadowMap."""
    dead_neurons: List[SilentEntity] = field(default_factory=list)
    silent_layers: List[SilentEntity] = field(default_factory=list)
    model_name: str | None = None
    analyzed_observations: int = 0
    dead_neuron_threshold: float = 0.0


class SilenceTracer:
    """Analizza una ShadowMap per trovare neuroni e layer "morti" o silenti."""

    def trace(self, shadow_map: ShadowMap, dead_neuron_threshold: float = 0.99) -> SilenceReport:
        """
        Analizza la mappa delle ombre per generare un report sui silenzi.

        Args:
            shadow_map: La mappa delle ombre da analizzare.
            dead_neuron_threshold: Rapporto di silenzio sopra il quale un neurone
                                   è considerato "morto".

        Returns:
            Un SilenceReport con i risultati dell'analisi.
        """
        if not (0 <= dead_neuron_threshold <= 1):
            raise ValueError("La soglia per i neuroni morti deve essere tra 0 e 1.")

        logger.info(f"Inizio tracciamento silenzi con soglia: {dead_neuron_threshold}")

        report = SilenceReport(
            model_name=shadow_map.model_name,
            analyzed_observations=shadow_map.observation_count,
            dead_neuron_threshold=dead_neuron_threshold,
        )

        for layer_name, neurons in shadow_map.layer_activity.items():
            if not neurons:
                continue

            layer_total_silences = 0
            layer_total_observations = 0

            for neuron_idx, activity in neurons.items():
                layer_total_silences += activity.silence_count
                layer_total_observations += activity.total_observations

                if activity.silence_ratio >= dead_neuron_threshold:
                    report.dead_neurons.append(SilentEntity(
                        name=f"{layer_name}.{neuron_idx}",
                        silence_ratio=activity.silence_ratio,
                        total_observations=activity.total_observations,
                        entity_type="neuron"
                    ))

            # Calcola il rapporto di silenzio aggregato per il layer
            layer_silence_ratio = layer_total_silences / layer_total_observations if layer_total_observations > 0 else 0

            if layer_silence_ratio >= dead_neuron_threshold:
                report.silent_layers.append(SilentEntity(
                    name=layer_name,
                    silence_ratio=layer_silence_ratio,
                    total_observations=layer_total_observations,
                    entity_type="layer"
                ))

        # Ordina i risultati per leggibilità
        report.dead_neurons.sort(key=lambda x: x.silence_ratio, reverse=True)
        report.silent_layers.sort(key=lambda x: x.silence_ratio, reverse=True)

        logger.info(f"Tracciamento completato. Trovati {len(report.dead_neurons)} neuroni morti "
                    f"e {len(report.silent_layers)} layer silenti.")

        return report