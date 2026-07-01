"""
Lettore di file .onepai (Lo Scrigno).
"""
from typing import Iterator, Dict, Any
from onepai.core.crypto import decrypt_bytes

class TreasureReader:
    """
    Una classe per leggere e decifrare un file .onepai.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"INFO: Lettore inizializzato per lo scrigno: {file_path}")

    def read_all(self, limit: int = None) -> Iterator[Dict[str, Any]]:
        """
        Legge tutti i frammenti dal file, li decifra e li restituisce.

        Args:
            limit: Limita il numero di frammenti da leggere (opzionale).

        Yields:
            Un dizionario che rappresenta un singolo frammento di potenziale.
        """
        count = 0
        with open(self.file_path, "rb") as f:
            while chunk := f.read(1024):  # Legge a blocchi di 1KB (esempio)
                if limit and count >= limit:
                    break
                
                decrypted_data = decrypt_bytes(chunk)
                fragment = {"timestamp": "...", "data": decrypted_data}
                yield fragment
                count += 1