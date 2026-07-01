"""
onepai.introspection — Analisi profonda di come un modello pensa, step by step.

Moduli:
    attention_tracer    — Traccia pattern testa-per-testa, identifica teste fantasma
    logit_lens          — Proietta lo stato intermedio sul vocabolario a ogni layer
    gradient_flow       — Mappa il flusso dei gradienti attraverso i parametri
    activation_patch    — Causal tracing tramite patching delle attivazioni
    token_influence     — Attribution map: quali token influenzano l'output
    residual_stream     — Decompone il residual stream nei contributi layer-by-layer
    feature_attribution — Integrated gradients, SmoothGrad, Gradient×Input
"""

from .attention_tracer import AttentionRecord, AttentionTracer, HeadSummary
from .logit_lens import LayerPrediction, LogitLens
from .gradient_flow import GradientRecord, GradientFlowTracer
from .activation_patch import PatchResult, ActivationPatcher
from .token_influence import TokenInfluenceMap, TokenInfluencer
from .residual_stream import ResidualComponent, ResidualStreamAnalyzer
from .feature_attribution import Attribution, FeatureAttributor

__all__ = [
    "AttentionRecord", "AttentionTracer", "HeadSummary",
    "LayerPrediction", "LogitLens",
    "GradientRecord", "GradientFlowTracer",
    "PatchResult", "ActivationPatcher",
    "TokenInfluenceMap", "TokenInfluencer",
    "ResidualComponent", "ResidualStreamAnalyzer",
    "Attribution", "FeatureAttributor",
]
