# File: onepai/core/archive.py

"""
Serializzazione append-only del file *.onepai* con checksum SHA-256.
"""

from __future__ import annotations 
 
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

from .observer import Observation

_FILE_MAGIC = b"ONEPAI0"  # 7-byte magic header + version 0
_HEADER_FMT = ">I32s"  # payload_size (uint32 big-endian) + sha256 (32 bytes)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT) 


class Archive:
    """Gestisce la persistenza immutabile di Observation su disco."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
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
                yield Observation(**json.loads(payload))  # type: ignore[arg-type]

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
        payload = json.dumps(obs.__dict__, default=str).encode()
        digest = hashlib.sha256(payload).digest()
        fh.write(struct.pack(_HEADER_FMT, len(payload), digest))
        fh.write(payload)
