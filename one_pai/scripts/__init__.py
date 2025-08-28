# -----------------------------------------------------------------------------
# File: onepai/analysis/__init__.py
# -----------------------------------------------------------------------------

"""
ONEPAI Analysis Package
=======================

Moduli per l'analisi avanzata dei pattern di assenza, silenzi cognitivi,
e dinamiche nascoste nelle reti neurali.

Author: Francesco Bulla (Brainverse)
"""

from .absence_patterns import find_recurrent_absences
from .silence_metrics import (
    calculate_silence_entropy,
    measure_void_depth,
    analyze_suppression_patterns
)
from .cognitive_dissonance import (
    detect_conflicting_thoughts,
    measure_internal_tension,
    analyze_decision_contradictions
)
from .dream_logic import (
    extract_surreal_patterns,
    analyze_impossible_connections,
    map_unconscious_associations
)

__version__ = "0.1.0"
__author__ = "Francesco Bulla"
__description__ = "Advanced analysis tools for mapping the invisible in AI"

__all__ = [
    # Absence Patterns
    "find_recurrent_absences",
    
    # Silence Metrics
    "calculate_silence_entropy",
    "measure_void_depth", 
    "analyze_suppression_patterns",
    
    # Cognitive Dissonance
    "detect_conflicting_thoughts",
    "measure_internal_tension",
    "analyze_decision_contradictions",
    
    # Dream Logic
    "extract_surreal_patterns",
    "analyze_impossible_connections",
    "map_unconscious_associations"
]