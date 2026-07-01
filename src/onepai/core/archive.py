# File: onepai/core/archive.py

"""
Serializzazione append-only del file *.onepai* con checksum SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, Optional

from .observer import Observation

_FILE_MAGIC = b"ONEPAI0"  # 7-byte magic header + version 0
_HEADER_FMT = ">I32s"  # payload_size (uint32 big-endian) + sha256 (32 bytes)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

__all__ = ["Archive", "TreasureArchive"]


class Archive:
    """Gestisce la persistenza immutabile di Observation su disco."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, observations: Iterable[Observation]) -> None:
        """Aggiunge *observations* alla fine dell'archivio."""

        with self.path.open("ab") as fh:
            for obs in observations:
                self._write_record(fh, obs)

    def iter_records(self) -> Iterable[Observation]:
        """Itera sulle observation salvate (streaming)."""

        with self.path.open("rb") as fh:
            self._verify_magic(fh)
            while True:
                header = fh.read(_HEADER_SIZE)
                if not header:
                    break
                size, expected_hash = struct.unpack(_HEADER_FMT, header)
                payload = fh.read(size)
                if hashlib.sha256(payload).digest() != expected_hash:
                    raise IOError("Checksum mismatch – archivio corrotto")
                yield self._decode(payload)

    # ------------------------------------------------------------------
    # Codifica (override nelle sottoclassi per cifratura ecc.)
    # ------------------------------------------------------------------

    def _encode(self, obs: Observation) -> bytes:
        import dataclasses
        return json.dumps(dataclasses.asdict(obs), default=str).encode()

    def _decode(self, payload: bytes) -> Observation:
        return Observation(**json.loads(payload))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_file(self) -> None:
        with self.path.open("wb") as fh:
            fh.write(_FILE_MAGIC)

    @staticmethod
    def _verify_magic(fh: BinaryIO) -> None:
        magic = fh.read(len(_FILE_MAGIC))
        if magic != _FILE_MAGIC:
            raise IOError("Invalid .onepai file: bad magic header")

    def _write_record(self, fh: BinaryIO, obs: Observation) -> None:
        payload = self._encode(obs)
        digest = hashlib.sha256(payload).digest()
        fh.write(struct.pack(_HEADER_FMT, len(payload), digest))
        fh.write(payload)


class TreasureArchive(Archive):
    """Archivio "scrigno" con cifratura opzionale dei payload.

    Il checksum è calcolato sul payload memorizzato (cifrato se ``cipher`` è
    presente), così l'integrità è verificabile anche senza la chiave.
    """

    DEFAULT_PATH = Path("data/treasures/default.onepai")

    def __init__(self, path: str | Path | None = None, cipher: Optional[Any] = None):
        self.cipher = cipher
        super().__init__(Path(path) if path is not None else self.DEFAULT_PATH)

    # ------------------------------------------------------------------
    # Codifica cifrata
    # ------------------------------------------------------------------

    def _encode(self, obs: Observation) -> bytes:
        payload = super()._encode(obs)
        if self.cipher is not None:
            payload = self.cipher.encrypt(payload)
        return payload

    def _decode(self, payload: bytes) -> Observation:
        if self.cipher is not None:
            payload = self.cipher.decrypt(payload)
        return super()._decode(payload)

    # ------------------------------------------------------------------
    # API di consultazione usata dalla CLI
    # ------------------------------------------------------------------

    def list_treasures(self) -> List[str]:
        """Elenca i nomi delle observation contenute nello scrigno."""
        return [obs.name for obs in self.iter_records()]

    def search_treasures(self, query: str) -> List[Observation]:
        """Cerca observation il cui nome contiene *query* (case-insensitive)."""
        query_lower = query.lower()
        return [obs for obs in self.iter_records() if query_lower in obs.name.lower()]
