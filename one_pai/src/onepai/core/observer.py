# -----------------------------------------------------------------------------
# File: onepai/core/observer.py
# -----------------------------------------------------------------------------

"""Cattura di attivazioni, gradienti e silenzi durante l'inferenza."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Protocol

logger = logging.getLogger(__name__)


class LayerLike(Protocol):
    """Interfaccia minima per layer di rete (compatibile con PyTorch/TF)."""

def register_forward_hook(self, hook): ...  # type: ignore[override]


@dataclass(slots=True)
class Observation:
    """Un singolo campione di attivazione o silenzio."""

    name: str
    timestamp: datetime
    data: Any  # tipicamente tensor oppure None per silenzio
    meta: Dict[str, Any] = field(default_factory=dict)


class Observer:
    """Osservatore di rete che intercetta attivazioni e silenzi."""

    def __init__(self) -> None:
        self._records: List[Observation] = []
        self._attached: bool = False
        self._hooks: List[Any] = []

    def _hook_fn(self, name: str) -> Any:
        """Crea una funzione di hook per un layer specifico."""
        def hook(model: Any, input: Any, output: Any) -> None:
            # Esempio: cattura l'output. Potrebbe essere più complesso.
            obs = Observation(name=name, timestamp=datetime.utcnow(), data=output)
            self._records.append(obs)
            logger.debug(f"Osservato {name} con output di tipo {type(output)}")

        return hook

    def attach(self, model: Any, layers: Iterable[str]) -> None:
        """Aggancia l'osservatore ai layer specificati del modello."""
        if self._attached:
            raise RuntimeError("Observer già agganciato. Sganciare prima.")

        for name, layer in model.named_modules():
            if name in layers:
                handle = layer.register_forward_hook(self._hook_fn(name))
                self._hooks.append(handle)
                logger.info(f"Observer agganciato al layer: {name}")

        self._attached = True

    def detach(self) -> None:
        """Sgancia l'osservatore da tutti i layer."""
        if not self._attached:
            return

        for handle in self._hooks:
            handle.remove()

        self._hooks.clear()
        self._attached = False
        logger.info("Observer sganciato da tutti i layer.")

    def get_records(self) -> List[Observation]:
        """Restituisce una copia delle osservazioni raccolte."""
        return list(self._records)

    def clear(self) -> None:
        """Pulisce le osservazioni raccolte."""
        self._records.clear()

    def __enter__(self) -> Observer:
        # L'aggancio deve essere fatto esternamente prima del blocco with
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()