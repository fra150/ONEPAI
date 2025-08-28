# -----------------------------------------------------------------------------
# File: onepai/core/main.py
# -----------------------------------------------------------------------------

"""Punto di ingresso principale e classe orchestrator per ONEPAI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from onepai.core.archive import TreasureArchive
from onepai.core.crypto import TreasureCipher
from onepai.core.observer import Observer
from onepai.core.registry import ShadowRegistry
from onepai.core.shadow_mapper import ShadowMapper
from onepai.core.silence_tracer import SilenceTracer

logger = logging.getLogger(__name__)


class ONEPAI:
    """Orchestratore centrale per tutte le operazioni di ONEPAI."""

    def __init__(
        self,
        db_path: str | Path,
        vault_path: str | Path,
        password: Optional[str] = None,
    ) -> None:
        """
        Inizializza il sistema ONEPAI.

        Args:
            db_path: Percorso del database per ShadowRegistry (SQLite).
            vault_path: Percorso del file del vault per TreasureArchive.
            password: Password per cifrare il vault. Se non fornita, il vault
                      non sarà cifrato.
        """
        logger.info(f"Inizializzazione di ONEPAI con db='{db_path}' e vault='{vault_path}'")
        self.db_path = Path(db_path)
        self.vault_path = Path(vault_path)

        self.registry = ShadowRegistry(self.db_path)
        self.cipher = TreasureCipher(password) if password else None
        self.archive = TreasureArchive(self.vault_path, self.cipher)

        self.observer = Observer()
        self.mapper = ShadowMapper()
        self.tracer = SilenceTracer()

        logger.info("ONEPAI inizializzato con successo.")

    def __repr__(self) -> str:
        return f"ONEPAI(db='{self.db_path}', vault='{self.vault_path}', encrypted={self.cipher is not None})"