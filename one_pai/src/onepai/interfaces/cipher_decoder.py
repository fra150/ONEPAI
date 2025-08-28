"""
Funzioni di alto livello per decodificare parti dello Scrigno.
"""
from typing import Dict, Any

def decode_fragment(fragment: Dict[str, Any]) -> str:
    """
    Trasforma un frammento del tesoro in una stringa leggibile.

    Args:
        fragment: Un dizionario rappresentante un frammento.

    Returns:
        Una stringa formattata che descrive il frammento.
    """
    timestamp = fragment.get("timestamp", "N/A")
    chosen = fragment.get("chosen_path", {}).get("action", "N/A")
    aborted_count = len(fragment.get("aborted_thoughts", []))

    return (
        f"[{timestamp}] - SCELTA: {chosen} | "
        f"PENSIERI ABORTITI: {aborted_count}"
    )