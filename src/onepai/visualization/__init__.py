"""
onepai.visualization — Rendering visivo dei dati di introspezione.

Moduli:
    heatmap      — Heatmap SVG/PNG (attivazioni, silenzio, gradienti)
    graph_viz    — Grafi neurali e circuiti ombra
    timeline_viz — Serie temporali di attivazioni e confidenza
    html_report  — Report HTML autocontenuto (zero dipendenze frontend)
"""

from .heatmap import ActivationHeatmap, SilenceHeatmap, AttentionHeatmap
from .graph_viz import NeuralGraphRenderer
from .timeline_viz import (
    ActivationTimeline, ConfidenceTimeline, GradientTimeline, line_chart,
)
from .html_report import HTMLReport

__all__ = [
    "ActivationHeatmap",
    "SilenceHeatmap",
    "AttentionHeatmap",
    "NeuralGraphRenderer",
    "ActivationTimeline",
    "ConfidenceTimeline",
    "GradientTimeline",
    "line_chart",
    "HTMLReport",
]
