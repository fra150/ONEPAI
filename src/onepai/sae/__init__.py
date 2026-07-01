"""
onepai.sae — Sparse Autoencoders: il dizionario delle feature nascoste.

Una SAE decompone le attivazioni di un modello in tante feature sparse e
(idealmente) monosemantiche — il modo in cui il campo apre la "sovrapposizione"
e rende leggibile l'invisibile. ONEPAI ci aggiunge la sua tesi: quali feature
sono *assenti* (dead) o *soppresse*.

Componenti:
    SparseAutoencoder  — encoder/decoder con penalità L1, addestrabile (numpy puro)
    collect_activations— raccoglie attivazioni reali da un ModelAdapter
    FeatureReport      — analisi delle feature: vive, morte, sparsità, varianza spiegata
"""

from .autoencoder import SparseAutoencoder, FeatureReport
from .collect import collect_activations
from .feature_suppression import (
    FeatureSuppressionTracer, FeatureTrajectory, FeatureSuppressor,
)
from .interpret import FeatureInterpreter, FeatureExample, DualInterpretation

__all__ = [
    "SparseAutoencoder", "FeatureReport", "collect_activations",
    "FeatureSuppressionTracer", "FeatureTrajectory", "FeatureSuppressor",
    "FeatureInterpreter", "FeatureExample", "DualInterpretation",
]
