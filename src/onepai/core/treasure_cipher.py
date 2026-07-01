# -----------------------------------------------------------------------------
# File: onepai/core/treasure_cipher.py
# -----------------------------------------------------------------------------

"""Cifratura avanzata per i tesori ONEPAI con AES-256-GCM."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

logger = logging.getLogger(__name__)

__all__ = ["TreasureCipher"]


class TreasureCipher:
    """Cifratura AES-256-GCM per proteggere i tesori neurali."""

    def __init__(self, password: Optional[str] = None) -> None:
        """
        Inizializza il cifratore per i tesori.

        Args:
            password: Password per derivare la chiave di cifratura.
                     Se None, genera una chiave casuale.
        """
        if password:
            # Deriva una chiave dalla password usando PBKDF2
            import hashlib
            self.key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                b'onepai_salt_2024',  # Salt fisso per consistenza
                100000,  # 100k iterazioni
                32  # 256 bit
            )
            logger.info("Chiave derivata da password per TreasureCipher")
        else:
            # Genera una chiave casuale
            self.key = secrets.token_bytes(32)
            logger.info("Chiave casuale generata per TreasureCipher")

    def encrypt(self, data: bytes) -> bytes:
        """
        Cifra i dati usando AES-256-GCM.

        Args:
            data: Dati da cifrare

        Returns:
            Dati cifrati con nonce e tag inclusi
        """
        try:
            # Genera un nonce casuale
            nonce = get_random_bytes(12)  # 96 bit per GCM
            
            # Crea il cifratore
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            
            # Cifra i dati
            ciphertext, tag = cipher.encrypt_and_digest(data)
            
            # Combina nonce + tag + ciphertext
            encrypted_data = nonce + tag + ciphertext
            
            logger.debug(f"Dati cifrati: {len(data)} -> {len(encrypted_data)} bytes")
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Errore durante la cifratura: {e}")
            raise

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decifra i dati usando AES-256-GCM.

        Args:
            encrypted_data: Dati cifrati con nonce e tag

        Returns:
            Dati decifrati
        """
        try:
            if len(encrypted_data) < 28:  # 12 (nonce) + 16 (tag) = 28 byte minimi
                raise ValueError("Dati cifrati troppo corti")
            
            # Estrai componenti
            nonce = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Crea il decifratore
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            
            # Decifra e verifica
            data = cipher.decrypt_and_verify(ciphertext, tag)
            
            logger.debug(f"Dati decifrati: {len(encrypted_data)} -> {len(data)} bytes")
            return data
            
        except Exception as e:
            logger.error(f"Errore durante la decifratura: {e}")
            raise

    def encrypt_file(self, src_path: Path, dst_path: Optional[Path] = None) -> Path:
        """
        Cifra un file.

        Args:
            src_path: Percorso del file sorgente
            dst_path: Percorso del file cifrato (opzionale)

        Returns:
            Percorso del file cifrato
        """
        src_path = Path(src_path)
        if dst_path is None:
            dst_path = src_path.with_suffix(src_path.suffix + '.encrypted')
        else:
            dst_path = Path(dst_path)

        try:
            # Leggi il file
            data = src_path.read_bytes()
            
            # Cifra
            encrypted_data = self.encrypt(data)
            
            # Scrivi il file cifrato
            dst_path.write_bytes(encrypted_data)
            
            logger.info(f"File cifrato: {src_path} -> {dst_path}")
            return dst_path
            
        except Exception as e:
            logger.error(f"Errore durante la cifratura del file: {e}")
            raise

    def decrypt_file(self, src_path: Path, dst_path: Optional[Path] = None) -> Path:
        """
        Decifra un file.

        Args:
            src_path: Percorso del file cifrato
            dst_path: Percorso del file decifrato (opzionale)

        Returns:
            Percorso del file decifrato
        """
        src_path = Path(src_path)
        if dst_path is None:
            # Rimuovi l'estensione .encrypted se presente
            if src_path.suffix == '.encrypted':
                dst_path = src_path.with_suffix('')
            else:
                dst_path = src_path.with_suffix('.decrypted')
        else:
            dst_path = Path(dst_path)

        try:
            # Leggi il file cifrato
            encrypted_data = src_path.read_bytes()
            
            # Decifra
            data = self.decrypt(encrypted_data)
            
            # Scrivi il file decifrato
            dst_path.write_bytes(data)
            
            logger.info(f"File decifrato: {src_path} -> {dst_path}")
            return dst_path
            
        except Exception as e:
            logger.error(f"Errore durante la decifratura del file: {e}")
            raise

    def get_key_fingerprint(self) -> str:
        """
        Restituisce un'impronta della chiave per identificazione.

        Returns:
            Fingerprint esadecimale della chiave
        """
        import hashlib
        return hashlib.sha256(self.key).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"TreasureCipher(fingerprint={self.get_key_fingerprint()})"