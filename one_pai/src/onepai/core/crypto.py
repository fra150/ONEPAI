# ----------------------------------------------------------------------------- 
 # File: onepai/core/crypto.py 
 # ----------------------------------------------------------------------------- 
 
 """Primitive crittografiche semplificate basate su *Fernet* (AES‑128‑GCM).""" 
 
 from __future__ import annotations 
 
 from base64 import urlsafe_b64encode, urlsafe_b64decode 
 from os import urandom 
 from typing import Tuple 
 
 from cryptography.fernet import Fernet 
 
 __all__ = ["generate_key", "encrypt_bytes", "decrypt_bytes"] 
 
 
 def generate_key() -> bytes: 
     """Genera un random key a 256 bit url‑safe.""" 
 
     return urlsafe_b64encode(urandom(32)) 
 
 
 def _get_fernet(key: bytes | str) -> Fernet: 
     return Fernet(key if isinstance(key, bytes) else key.encode()) 
 
 
 def encrypt_bytes(data: bytes, key: bytes | str) -> bytes: 
     """Restituisce *ciphertext* cifrato con *key*.""" 
 
     return _get_fernet(key).encrypt(data) 
 
 
 def decrypt_bytes(token: bytes, key: bytes | str) -> bytes: 
     """Decifra *token* cifrato con *key*.""" 
 
     return _get_fernet(key).decrypt(token)
    # ------------------------------------------------------------------ 
    # Public API 
    # ------------------------------------------------------------------ 

    def encrypt_file(self, src: Path, dst: Path | None = None) -> Path: 
        dst = dst or src.with_suffix(src.suffix + ".enc") 
        data = src.read_bytes() 
        dst.write_bytes(encrypt_bytes(data, self.key)) 
        return dst 

    def decrypt_file(self, src: Path, dst: Path | None = None) -> Path: 
        dst = dst or src.with_suffix(src.suffix + ".dec") 
        data = decrypt_bytes(src.read_bytes(), self.key) 
        dst.write_bytes(data) 
        return dst