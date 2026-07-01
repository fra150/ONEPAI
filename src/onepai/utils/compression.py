"""Compressione/decompressione per payload di archivi .onepai."""

from __future__ import annotations

import zlib
import lzma
from typing import Literal

__all__ = ["compress", "decompress", "CompressionMethod"]

CompressionMethod = Literal["zlib", "lzma", "none"]
_MAGIC_ZLIB = b"\x78\x9c"[:1]   # primo byte degli header zlib comuni
_MAGIC_LZMA = b"\xfd7zXZ\x00"


def compress(data: bytes, method: CompressionMethod = "zlib",
             level: int = 6) -> bytes:
    """
    Comprime `data` con il metodo scelto.
    Prepone un byte di tipo: b'Z'=zlib, b'L'=lzma, b'N'=none.
    """
    if method == "zlib":
        return b"Z" + zlib.compress(data, level=level)
    if method == "lzma":
        return b"L" + lzma.compress(data, preset=min(level, 9))
    return b"N" + data


def decompress(data: bytes) -> bytes:
    """
    Decomprime un buffer prodotto da compress().
    Rileva automaticamente il metodo dal primo byte.
    """
    if not data:
        return data
    tag = data[:1]
    payload = data[1:]
    if tag == b"Z":
        return zlib.decompress(payload)
    if tag == b"L":
        return lzma.decompress(payload)
    if tag == b"N":
        return payload
    # Fallback: tenta decompress zlib per compatibilità con archivi vecchi
    try:
        return zlib.decompress(data)
    except zlib.error:
        return data


def compress_ratio(original: bytes, method: CompressionMethod = "zlib") -> float:
    """Ritorna il rapporto di compressione (< 1 = risparmio)."""
    if not original:
        return 1.0
    compressed = compress(original, method)
    return len(compressed) / len(original)
