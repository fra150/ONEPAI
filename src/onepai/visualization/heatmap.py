"""
Heatmap visualizer — genera SVG inline per matrici 2D.

Non dipende da matplotlib: produce SVG puro che funziona in browser,
Jupyter, o come file standalone.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["ActivationHeatmap", "SilenceHeatmap", "AttentionHeatmap"]

# Palette: blu → bianco → rosso (divergente) oppure bianco → rosso (sequenziale)
_SEQUENTIAL_PALETTE = [
    (0.95, 0.95, 1.00),   # quasi bianco
    (0.70, 0.85, 0.95),
    (0.30, 0.65, 0.90),
    (0.05, 0.42, 0.75),
    (0.00, 0.20, 0.55),
]

_DIVERGING_PALETTE = [
    (0.02, 0.20, 0.55),   # blu scuro  (negativo)
    (0.40, 0.65, 0.90),
    (0.95, 0.95, 0.95),   # bianco     (zero)
    (0.95, 0.55, 0.35),
    (0.60, 0.05, 0.05),   # rosso scuro (positivo)
]

_SILENCE_PALETTE = [
    (0.95, 0.98, 0.95),   # quasi bianco (attivo)
    (0.65, 0.88, 0.65),
    (0.20, 0.65, 0.20),
    (0.05, 0.40, 0.05),
    (0.00, 0.18, 0.00),   # verde scuro (silenzio totale)
]


def _interpolate_color(palette: list, t: float) -> Tuple[int, int, int]:
    """Interpola nella palette per t ∈ [0,1]."""
    t = float(np.clip(t, 0.0, 1.0))
    n = len(palette) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    frac = idx - lo
    r = palette[lo][0] + frac * (palette[hi][0] - palette[lo][0])
    g = palette[lo][1] + frac * (palette[hi][1] - palette[lo][1])
    b = palette[lo][2] + frac * (palette[hi][2] - palette[lo][2])
    return (int(r * 255), int(g * 255), int(b * 255))


def _matrix_to_svg(
    matrix: np.ndarray,
    palette: list,
    cell_size: int = 24,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "",
    diverging: bool = False,
    fmt: str = ".2f",
) -> str:
    """Genera un SVG rettangolare da una matrice 2D."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    rows, cols = matrix.shape
    label_w = 80 if y_labels else 10
    label_h = 60 if x_labels else 10
    title_h = 30 if title else 5
    svg_w = label_w + cols * cell_size + 20
    svg_h = title_h + label_h + rows * cell_size + 20
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    if diverging:
        abs_max = max(abs(vmin), abs(vmax)) + 1e-9
        vmin, vmax = -abs_max, abs_max
    span = (vmax - vmin) + 1e-9
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="ui-monospace, monospace" font-size="10">',
    ]
    if title:
        lines.append(
            f'<text x="{svg_w//2}" y="{title_h - 5}" text-anchor="middle" '
            f'font-size="13" font-weight="500" fill="#38e1ff" '
            f'letter-spacing="0.05em">{title}</text>'
        )
    # X labels (colonne)
    if x_labels:
        for j, lbl in enumerate(x_labels[:cols]):
            x = label_w + j * cell_size + cell_size // 2
            y = title_h + label_h - 5
            lines.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" '
                f'transform="rotate(-45 {x} {y})" fill="#8493b8">{lbl[:8]}</text>'
            )
    # Celle
    for i in range(rows):
        for j in range(cols):
            val = float(matrix[i, j])
            t = (val - vmin) / span
            r, g, b = _interpolate_color(palette, t)
            cx = label_w + j * cell_size
            cy = title_h + label_h + i * cell_size
            fill = f"#{r:02x}{g:02x}{b:02x}"
            text_val = format(val, fmt) if cell_size >= 20 else ""
            # Luminosità per scegliere colore testo
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_col = "#0a0f1c" if luminance > 128 else "#e8eeff"
            lines.append(
                f'<rect x="{cx}" y="{cy}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="rgba(8,12,24,0.55)" stroke-width="0.6" rx="2">'
                f'<title>{val:.4f}</title></rect>'
            )
            if text_val:
                lines.append(
                    f'<text x="{cx + cell_size//2}" y="{cy + cell_size//2 + 4}" '
                    f'text-anchor="middle" fill="{text_col}" font-size="8">{text_val}</text>'
                )
        # Y label (riga)
        if y_labels and i < len(y_labels):
            cy = title_h + label_h + i * cell_size + cell_size // 2 + 4
            lines.append(
                f'<text x="{label_w - 5}" y="{cy}" text-anchor="end" fill="#9fb0d6">'
                f'{y_labels[i][:12]}</text>'
            )
    lines.append('</svg>')
    return "\n".join(lines)


class ActivationHeatmap:
    """
    Visualizza una matrice di attivazioni (layer × neuron o step × layer).
    """

    def __init__(self, cell_size: int = 24):
        self.cell_size = cell_size

    def render(
        self,
        matrix: np.ndarray,
        title: str = "Activation Heatmap",
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        diverging: bool = True,
    ) -> str:
        """Ritorna SVG come stringa."""
        return _matrix_to_svg(
            np.asarray(matrix, dtype=np.float32),
            palette=_DIVERGING_PALETTE if diverging else _SEQUENTIAL_PALETTE,
            cell_size=self.cell_size,
            x_labels=x_labels,
            y_labels=y_labels,
            title=title,
            diverging=diverging,
        )

    def render_from_registry(self, registry, max_layers: int = 20) -> str:
        """Genera heatmap direttamente da un ShadowRegistry."""
        layers = registry.layers()[:max_layers]
        rows = []
        for layer in layers:
            obs = registry.get(layer)
            vals = []
            for o in obs:
                # Supporta sia attributo diretto che data/meta
                v = getattr(o, "activation_value", None)
                if v is None and o.data is not None:
                    try:
                        v = float(o.data)
                    except (TypeError, ValueError):
                        pass
                if v is not None:
                    vals.append(float(v))
            if vals:
                rows.append(float(np.mean(vals)))
        if not rows:
            return "<svg><text>No activation data</text></svg>"
        mat = np.array(rows).reshape(-1, 1)
        return self.render(mat, title="Registry Activations", y_labels=layers)


class SilenceHeatmap:
    """
    Visualizza le silence ratio (0=attivo, 1=silenzioso) come griglia verde.
    """

    def __init__(self, cell_size: int = 28):
        self.cell_size = cell_size

    def render(
        self,
        silence_map: dict,
        title: str = "Silence Map",
        max_neurons: int = 50,
    ) -> str:
        """
        silence_map: {layer_name: {neuron_idx: silence_ratio}} oppure
                     {layer_name: silence_ratio_scalar}.
        """
        layers = list(silence_map.keys())
        # Trova la dimensione massima
        max_n = 1
        normalized: dict = {}
        for layer, value in silence_map.items():
            if isinstance(value, dict):
                max_n = max(max_n, len(value))
                normalized[layer] = value
            else:
                normalized[layer] = {0: float(value)}
        max_n = min(max_n, max_neurons)
        matrix = np.zeros((len(layers), max_n), dtype=np.float32)
        for i, layer in enumerate(layers):
            for j, ratio in normalized[layer].items():
                if j < max_n:
                    matrix[i, j] = float(ratio)
        x_labels = [str(j) for j in range(max_n)]
        return _matrix_to_svg(
            matrix,
            palette=_SILENCE_PALETTE,
            cell_size=self.cell_size,
            x_labels=x_labels,
            y_labels=layers,
            title=title,
            diverging=False,
            fmt=".1f",
        )

    def render_from_shadow_map(self, shadow_map, max_neurons: int = 50) -> str:
        """Genera da un ShadowMap (onepai.core.shadow_mapper.ShadowMap)."""
        data = {}
        for layer, neurons in shadow_map.layer_activity.items():
            data[layer] = {idx: act.silence_ratio for idx, act in neurons.items()}
        return self.render(data, title=f"Silence Map — {shadow_map.model_name or 'model'}",
                           max_neurons=max_neurons)


class AttentionHeatmap:
    """
    Visualizza una matrice di attenzione (seq_len × seq_len).
    """

    def __init__(self, cell_size: int = 24):
        self.cell_size = cell_size

    def render(
        self,
        attention_matrix: np.ndarray,
        title: str = "Attention Pattern",
        token_labels: Optional[List[str]] = None,
    ) -> str:
        """attention_matrix: (seq_len × seq_len)."""
        mat = np.asarray(attention_matrix, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("attention_matrix deve essere 2D")
        return _matrix_to_svg(
            mat,
            palette=_SEQUENTIAL_PALETTE,
            cell_size=self.cell_size,
            x_labels=token_labels,
            y_labels=token_labels,
            title=title,
            diverging=False,
            fmt=".2f",
        )
