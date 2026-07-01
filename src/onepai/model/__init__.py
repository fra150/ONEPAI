"""
onepai.model — Aggancio a modelli reali (niente più np.random).

Interfaccia in stile transformer_lens: run_with_cache / run_with_hooks.
Gli hook sono bidirezionali: catturano (per l'analisi dell'assenza) e
modificano (per gli interventi controllati: ablazione, add-direction, patch).

Componenti:
    ModelAdapter      — interfaccia astratta per qualunque modello
    CaptureBundle     — risultato strutturato di un forward instrumentato
    Intervention      — descrive una modifica controllata a un hook
    HookPoint         — costanti dei nomi degli hook (convenzione TL)
    TinyTransformer   — transformer reale in puro numpy (zero dipendenze esterne)
    ModelInstrumentor — versa le catture reali nei tracer ONEPAI
"""

from .adapter import (
    ModelAdapter, CaptureBundle, Intervention, HookPoint,
    make_hook, gelu, softmax, layer_norm,
)
from .tiny_transformer import TinyTransformer, TinyConfig
from .instrumentor import ModelInstrumentor, InstrumentationResult
from .absence import (
    AbsenceEngine, AbsenceReport, NeuronAbsence, CrushedToken, AbortedDirection,
)
from .intervention_study import (
    InterventionStudy, InterventionStudyResult, InterventionEffect, SteeringResponse,
)
from .suppression import SuppressionTracer, SuppressionTrace, Suppressor
from .validation import SuppressionValidator, ValidationCase, ValidationResult

# HFAdapter è opzionale: richiede torch + transformers.
try:
    from .hf_adapter import HFAdapter
    _HAS_HF = True
except Exception:  # pragma: no cover - dipende dall'ambiente
    HFAdapter = None  # type: ignore
    _HAS_HF = False

__all__ = [
    "ModelAdapter", "CaptureBundle", "Intervention", "HookPoint",
    "make_hook", "gelu", "softmax", "layer_norm",
    "TinyTransformer", "TinyConfig",
    "ModelInstrumentor", "InstrumentationResult",
    "AbsenceEngine", "AbsenceReport", "NeuronAbsence", "CrushedToken", "AbortedDirection",
    "InterventionStudy", "InterventionStudyResult", "InterventionEffect", "SteeringResponse",
    "SuppressionTracer", "SuppressionTrace", "Suppressor",
    "SuppressionValidator", "ValidationCase", "ValidationResult",
    "HFAdapter",
]
