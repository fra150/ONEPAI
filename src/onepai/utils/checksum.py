"""Checksum e verifica integrità per archivi .onepai."""

from __future__ import annotations

import hashlib
import hmac
import struct
from pathlib import Path
from typing import Union

__all__ = ["sha256_bytes", "sha256_file", "sha256_stream",
           "hmac_sign", "hmac_verify", "CRCRecord"]


def sha256_bytes(data: bytes) -> bytes:
    """SHA-256 di un buffer in memoria. Ritorna digest (32 byte)."""
    return hashlib.sha256(data).digest()


def sha256_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """SHA-256 di un file su disco. Ritorna hex string (64 caratteri)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(stream, chunk_size: int = 65536) -> bytes:
    """SHA-256 di un file-like object già aperto. Non fa seek."""
    h = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)
    return h.digest()


def hmac_sign(data: bytes, key: bytes) -> bytes:
    """HMAC-SHA256 di data con key. Ritorna 32 byte."""
    return hmac.new(key, data, hashlib.sha256).digest()


def hmac_verify(data: bytes, key: bytes, signature: bytes) -> bool:
    """Verifica HMAC a tempo costante (resiste a timing attacks)."""
    expected = hmac_sign(data, key)
    return hmac.compare_digest(expected, signature)


class CRCRecord:
    """Struttura semplice per validare blocchi con CRC-32."""
    MAGIC = struct.pack(">H", 0xCA5E)
    FMT = ">HI"   # magic(2) + crc32(4)
    HEADER_SIZE = struct.calcsize(FMT)

    @staticmethod
    def pack(payload: bytes) -> bytes:
        """Aggiunge header con CRC al payload."""
        import zlib
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(CRCRecord.FMT, 0xCA5E, crc)
        return header + payload

    @staticmethod
    def unpack(data: bytes) -> bytes:
        """Verifica CRC e ritorna il payload, lancia ValueError se corrotto."""
        import zlib
        if len(data) < CRCRecord.HEADER_SIZE:
            raise ValueError("CRCRecord troppo corto")
        magic, crc = struct.unpack(CRCRecord.FMT, data[:CRCRecord.HEADER_SIZE])
        if magic != 0xCA5E:
            raise ValueError(f"Magic errato: {magic:#06x}")
        payload = data[CRCRecord.HEADER_SIZE:]
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if actual != crc:
            raise ValueError(f"CRC mismatch: atteso {crc:#010x}, trovato {actual:#010x}")
        return payload
