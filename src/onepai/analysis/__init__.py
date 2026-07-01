"""
Modulo Analysis ONEPAI
======================

Strumenti di analisi del non-manifesto: pattern di assenza, dissonanza
cognitiva, logica onirica e metriche del silenzio.
"""

from .absence_patterns import find_recurrent_absences
from .cognitive_dissonance import (
    CognitiveDissonance,
    ConflictingThought,
    analyze_decision_contradictions,
    detect_conflicting_thoughts,
    measure_internal_tension,
)
from .dream_logic import (
    DreamSequence,
    LogicJump,
    analyze_non_linear_patterns,
    detect_dream_sequences,
    measure_surreal_connections,
)
from .silence_metrics import (
    analyze_suppression_patterns,
    calculate_silence_entropy,
    calculate_silence_ratio,
    calculate_void_entropy,
    measure_void_depth,
)

__all__ = [
    # Pattern di assenza
    "find_recurrent_absences",
    # Dissonanza cognitiva
    "CognitiveDissonance",
    "ConflictingThought",
    "analyze_decision_contradictions",
    "detect_conflicting_thoughts",
    "measure_internal_tension",
    # Logica onirica
    "DreamSequence",
    "LogicJump",
    "analyze_non_linear_patterns",
    "detect_dream_sequences",
    "measure_surreal_connections",
    # Metriche del silenzio
    "analyze_suppression_patterns",
    "calculate_silence_entropy",
    "calculate_silence_ratio",
    "calculate_void_entropy",
    "measure_void_depth",
]
