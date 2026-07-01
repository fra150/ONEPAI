# -----------------------------------------------------------------------------
# File: circuit_tracer/replacement_model.py
# -----------------------------------------------------------------------------

"""
ReplacementModel per il circuit tracer.

Questo modulo fornisce la classe ReplacementModel che è il wrapper principale
per l'analisi di attribuzione con transcoder.
"""

# Re-export della classe ReplacementModel dal modulo models
from .models import ReplacementModel, create_model

__all__ = ["ReplacementModel", "create_model"]