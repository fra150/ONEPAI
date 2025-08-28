# -----------------------------------------------------------------------------
# File: onepai/neural/__init__.py
# -----------------------------------------------------------------------------

"""
Modulo Neural ONEPAI
====================

Sistema di analisi neurale per l'intelligenza ombra, includendo:
- Rilevatore di aborti (abortion_detector)
- Ombra dei circuiti (circuit_shadow)
- Rete invisibile (invisible_net)
- Pesi fantasma (weight_ghost)

Author: Francesco Bulla (Brainverse)
"""

from .abortion_detector import AbortionDetector, AbortedThought
from .circuit_shadow import CircuitShadow
from .invisible_net import InvisibleNet, InvisiblePath
from .weight_ghost import WeightGhost

__version__ = "1.0.0"
__author__ = "Francesco Bulla (Brainverse)"
__description__ = "Sistema di analisi neurale per l'intelligenza ombra ONEPAI"

__all__ = [
    # Classi principali
    'AbortionDetector',
    'CircuitShadow',
    'InvisibleNet',
    'WeightGhost',
    
    # Strutture dati
    'AbortedThought',
    'InvisiblePath',
    
    # Metadati
    '__version__',
    '__author__',
    '__description__'
]