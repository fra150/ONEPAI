# -----------------------------------------------------------------------------
# File: onepai/core/crypto.py
# -----------------------------------------------------------------------------

"""Primitive crittografiche semplificate basate su *Fernet*."""

from __future__ import annotations

from base64 import urlsafe_b64encode
from os import urandom

from cryptography.fernet import Fernet

# Re-export per compatibilità: i moduli storici importano TreasureCipher da qui.
from .treasure_cipher import TreasureCipher

__all__ = ["generate_key", "encrypt_bytes", "decrypt_bytes", "TreasureCipher"]


def generate_key() -> bytes:
    """Genera una chiave casuale a 256 bit url-safe."""
    return urlsafe_b64encode(urandom(32))


def _get_fernet(key: bytes | str) -> Fernet:
    return Fernet(key if isinstance(key, bytes) else key.encode())


def encrypt_bytes(data: bytes, key: bytes | str) -> bytes:
    """Restituisce *ciphertext* cifrato con *key*."""
    return _get_fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes | str) -> bytes:
    """Decifra *token* cifrato con *key*."""
    return _get_fernet(key).decrypt(token)
