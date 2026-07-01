# -----------------------------------------------------------------------------
# File: circuit_tracer/utils/__init__.py
# -----------------------------------------------------------------------------

"""
Utility per il circuit tracer.

Questo package fornisce:
- Utility per Hugging Face
- Gestione device
- Creazione file di grafo
- Offloading su disco
"""

import torch


def get_default_device() -> str:
    """
    Determina il miglior device disponibile.
    
    Returns:
        str: Nome del device ('cuda', 'mps', o 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


__all__ = ["get_default_device"]