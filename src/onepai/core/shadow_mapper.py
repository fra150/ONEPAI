# -----------------------------------------------------------------------------
# File: onepai/core/shadow_mapper.py
# -----------------------------------------------------------------------------

"""Costruisce una mappa concettuale dell'invisibile (ombre/silenzi)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .observer import Observation
from .registry import ShadowRegistry

logger = logging.getLogger(__name__)

__all__ = ["NeuronActivity", "ShadowMap", "ShadowMapper"]


@dataclass(slots=True)
class NeuronActivity:
    """Statistiche di attività/silenzio per un singolo neurone."""

    silence_count: int = 0
    total_observations: int = 0

    @property
    def silence_ratio(self) -> float:
        """Frazione di osservazioni in cui il neurone è rimasto silente."""
        if self.total_observations == 0:
            return 0.0
        return self.silence_count / self.total_observations


@dataclass(slots=True)
class ShadowMap:
    """Mappa delle ombre: attività per layer/neurone di un modello osservato."""

    model_name: Optional[str] = None
    observation_count: int = 0
    # layer_name -> {neuron_idx -> NeuronActivity}
    layer_activity: Dict[str, Dict[int, NeuronActivity]] = field(default_factory=dict)

    @property
    def layers(self) -> List[str]:
        """Nomi dei layer presenti nella mappa."""
        return list(self.layer_activity.keys())

    def neuron(self, layer: str, idx: int) -> NeuronActivity:
        """Restituisce (creandola se serve) l'attività di un neurone."""
        return self.layer_activity.setdefault(layer, {}).setdefault(idx, NeuronActivity())


class ShadowMapper:
    """Traduce osservazioni grezze in una :class:`ShadowMap` dell'invisibile.

    Un'attivazione è considerata "silenzio" quando il suo valore assoluto non
    supera ``silence_threshold`` (oppure quando ``Observation.data`` è ``None``).
    """

    def __init__(
        self,
        registry: Optional[ShadowRegistry] = None,
        silence_threshold: float = 1e-6,
    ) -> None:
        self.registry = registry
        self.silence_threshold = silence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_shadow_map(
        self,
        observations: Optional[Iterable[Observation]] = None,
        model_name: Optional[str] = None,
    ) -> ShadowMap:
        """Costruisce una ShadowMap dalle osservazioni fornite (o dal registry)."""
        shadow_map = ShadowMap(model_name=model_name)

        if observations is None:
            observations = self._registry_observations()

        for obs in observations:
            self._accumulate(shadow_map, obs)

        logger.info(
            "ShadowMap creata: %d layer, %d osservazioni",
            len(shadow_map.layer_activity),
            shadow_map.observation_count,
        )
        return shadow_map

    # Alias storico mantenuto per compatibilità con vecchie API/documentazione.
    map_observations = create_shadow_map

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _registry_observations(self) -> List[Observation]:
        if self.registry is None:
            return []
        records: List[Observation] = []
        for layer in self.registry.layers():
            records.extend(self.registry.get(layer))
        return records

    def _accumulate(self, shadow_map: ShadowMap, obs: Observation) -> None:
        shadow_map.observation_count += 1
        values = self._to_array(obs.data)

        if values is None:
            # Silenzio totale: l'intero layer non ha prodotto output.
            activity = shadow_map.neuron(obs.name, 0)
            activity.silence_count += 1
            activity.total_observations += 1
            return

        for idx, value in enumerate(values):
            activity = shadow_map.neuron(obs.name, idx)
            activity.total_observations += 1
            if abs(float(value)) <= self.silence_threshold:
                activity.silence_count += 1

    @staticmethod
    def _to_array(data: Any) -> Optional[np.ndarray]:
        """Normalizza tensori/array/list in un array 1-D, None per silenzio."""
        if data is None:
            return None
        if hasattr(data, "detach"):  # torch.Tensor senza importare torch
            data = data.detach().cpu().numpy()
        try:
            arr = np.asarray(data, dtype=float)
        except (TypeError, ValueError):
            return None
        return arr.reshape(-1)
