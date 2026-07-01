# -----------------------------------------------------------------------------
# File: onepai/core/registry.py
# -----------------------------------------------------------------------------

"""Indice in-memory delle tracce shadow raccolte."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .observer import Observation

__all__ = ["ShadowRegistry"]


class ShadowRegistry:
    """Registrazione veloce di observation indicizzate per layer.

    ``db_path`` è opzionale: se fornito viene conservato come destinazione di
    persistenza futura, ma l'indice resta in memoria.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else None
        self._by_layer: Dict[str, List[Observation]] = defaultdict(list)

    def add(self, obs: Observation) -> None:
        layer = obs.name.split(":", 1)[0]
        self._by_layer[layer].append(obs)

    def get(self, layer: str) -> List[Observation]:
        return self._by_layer.get(layer, [])

    def layers(self) -> List[str]:
        """Nomi dei layer con almeno una observation registrata."""
        return list(self._by_layer.keys())

    def summary(self) -> Dict[str, int]:
        """Restituisce conteggio record per layer."""
        return {k: len(v) for k, v in self._by_layer.items()}

    def clear(self) -> None:
        self._by_layer.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_layer.values())
