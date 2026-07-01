"""
Timeline visualizer — serie temporali SVG per attivazioni, confidenza e gradienti.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["ActivationTimeline", "ConfidenceTimeline", "GradientTimeline", "line_chart"]

_LINE_COLORS = ["#38e1ff", "#ffb74a", "#3df5a6", "#ff5d72", "#a274ff",
                "#2ee6c8", "#ffd166", "#6cb8ff"]


def _sparkline_svg(
    series: Dict[str, List[Tuple[float, float]]],
    title: str = "",
    width: int = 700,
    height: int = 200,
    y_label: str = "",
    x_label: str = "",
    show_dots: bool = True,
) -> str:
    """
    Genera un SVG con linee multiple.
    series: {name: [(x, y), ...]}
    """
    margin = {"top": 40 if title else 15, "right": 120, "bottom": 40, "left": 55}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    if not series:
        return f'<svg width="{width}" height="{height}"><text x="10" y="30">No data</text></svg>'
    # Range
    all_x = [x for pts in series.values() for x, y in pts]
    all_y = [y for pts in series.values() for x, y in pts]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = (x_max - x_min) + 1e-9
    y_span = (y_max - y_min) + 1e-9

    def px(x: float) -> float:
        return margin["left"] + (x - x_min) / x_span * plot_w

    def py(y: float) -> float:
        return margin["top"] + plot_h - (y - y_min) / y_span * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="ui-monospace, monospace" font-size="11">',
        '<defs>'
        '<filter id="lglow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="2.4" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter></defs>',
        f'<rect width="{width}" height="{height}" fill="none" rx="4"/>',
    ]
    if title:
        lines.append(
            f'<text x="{width//2}" y="22" text-anchor="middle" '
            f'font-size="13" font-weight="500" fill="#38e1ff" '
            f'letter-spacing="0.05em">{title}</text>'
        )
    # Griglia orizzontale
    for i in range(5):
        gy = margin["top"] + i * plot_h / 4
        gy_val = y_max - i * y_span / 4
        lines.append(
            f'<line x1="{margin["left"]}" y1="{gy:.1f}" '
            f'x2="{margin["left"]+plot_w}" y2="{gy:.1f}" '
            f'stroke="rgba(120,165,255,0.12)" stroke-dasharray="3 4"/>'
        )
        lines.append(
            f'<text x="{margin["left"]-5}" y="{gy+4:.1f}" '
            f'text-anchor="end" fill="#7c89a8" font-size="9">{gy_val:.2f}</text>'
        )
    # Assi
    ax_y = margin["top"] + plot_h
    lines.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{ax_y}" stroke="rgba(120,165,255,0.3)" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{margin["left"]}" y1="{ax_y}" '
        f'x2="{margin["left"]+plot_w}" y2="{ax_y}" stroke="rgba(120,165,255,0.3)" stroke-width="1.2"/>'
    )
    # Labels assi
    if y_label:
        lines.append(
            f'<text x="13" y="{margin["top"]+plot_h//2}" text-anchor="middle" '
            f'transform="rotate(-90 13 {margin["top"]+plot_h//2})" fill="#8493b8" font-size="10">'
            f'{y_label}</text>'
        )
    if x_label:
        lines.append(
            f'<text x="{margin["left"]+plot_w//2}" y="{height-5}" '
            f'text-anchor="middle" fill="#8493b8" font-size="10">{x_label}</text>'
        )
    # Linee dati
    for ci, (name, pts) in enumerate(series.items()):
        if not pts:
            continue
        color = _LINE_COLORS[ci % len(_LINE_COLORS)]
        sorted_pts = sorted(pts, key=lambda p: p[0])
        path_d = " ".join(
            f"{'M' if i == 0 else 'L'} {px(x):.1f} {py(y):.1f}"
            for i, (x, y) in enumerate(sorted_pts)
        )
        lines.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" '
            f'opacity="0.95" filter="url(#lglow)"/>'
        )
        if show_dots:
            for x, y in sorted_pts:
                lines.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}" '
                    f'opacity="0.7"><title>{name}: x={x:.2f}, y={y:.4f}</title></circle>'
                )
        # Legenda
        legend_y = margin["top"] + ci * 18
        legend_x = margin["left"] + plot_w + 10
        lines.append(f'<line x1="{legend_x}" y1="{legend_y+6}" x2="{legend_x+18}" y2="{legend_y+6}" stroke="{color}" stroke-width="2"/>')
        short = name[:14] + ("…" if len(name) > 14 else "")
        lines.append(f'<text x="{legend_x+22}" y="{legend_y+10}" fill="#aeb9d6" font-size="9">{short}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def line_chart(
    series: Dict[str, List[Tuple[float, float]]],
    title: str = "",
    width: int = 700,
    height: int = 220,
    x_label: str = "",
    y_label: str = "",
    show_dots: bool = True,
) -> str:
    """
    Line chart generico da coppie (x, y) esplicite.
    series: {nome: [(x, y), ...]}. Usalo quando l'asse x NON è l'indice
    (es. coefficiente di steering, soglia, ecc.).
    """
    return _sparkline_svg(series, title=title, width=width, height=height,
                          x_label=x_label, y_label=y_label, show_dots=show_dots)


class ActivationTimeline:
    """
    Visualizza come le attivazioni di uno o più layer cambiano nel tempo (step).
    """

    def render(
        self,
        layer_activations: Dict[str, List[float]],
        title: str = "Activation Timeline",
        width: int = 700,
        height: int = 220,
    ) -> str:
        """
        layer_activations: {layer_name: [val_step0, val_step1, ...]}.
        """
        series = {
            name: [(float(i), float(v)) for i, v in enumerate(vals)]
            for name, vals in layer_activations.items()
        }
        return _sparkline_svg(series, title=title, width=width, height=height,
                               y_label="activation", x_label="step")

    def render_from_registry(self, registry, max_layers: int = 8) -> str:
        """Da un ShadowRegistry: media dell'activation_value per layer per step."""
        layers = registry.layers()[:max_layers]
        series: Dict[str, List[float]] = {}
        for layer in layers:
            obs = registry.get(layer)
            vals = [o.activation_value for o in obs if o.activation_value is not None]
            if vals:
                series[layer] = [float(v) for v in vals]
        return self.render(series)


class ConfidenceTimeline:
    """
    Visualizza la traiettoria di confidenza del Logit Lens attraverso i layer.
    """

    def render(
        self,
        trajectories: Dict[str, List[Tuple[str, float]]],
        title: str = "Confidence Through Layers",
        width: int = 700,
        height: int = 220,
    ) -> str:
        """
        trajectories: {label: [(layer_name, confidence), ...]}.
        """
        series: Dict[str, List[Tuple[float, float]]] = {}
        for label, traj in trajectories.items():
            series[label] = [(float(i), float(conf)) for i, (_, conf) in enumerate(traj)]
        return _sparkline_svg(series, title=title, width=width, height=height,
                               y_label="confidence", x_label="layer index")

    def render_from_logit_lens(self, lens, positions: Optional[List[int]] = None,
                                step: int = 0) -> str:
        """Da un LogitLens: una linea per posizione token."""
        if positions is None:
            # Trova tutte le posizioni registrate
            all_pos = {p.position for preds in lens._predictions.values() for p in preds}
            positions = list(sorted(all_pos))[:5]  # max 5
        trajectories = {}
        for pos in positions:
            traj = lens.get_confidence_trajectory(pos, step)
            if traj:
                trajectories[f"pos={pos}"] = traj
        return self.render(trajectories)


class GradientTimeline:
    """
    Visualizza l'evoluzione della magnitudine del gradiente per layer.
    """

    def render(
        self,
        gradient_trajectories: Dict[str, List[Tuple[int, float]]],
        title: str = "Gradient Flow",
        width: int = 700,
        height: int = 220,
        log_scale: bool = True,
    ) -> str:
        """
        gradient_trajectories: {layer_name: [(step, magnitude), ...]}.
        """
        series: Dict[str, List[Tuple[float, float]]] = {}
        for name, pts in gradient_trajectories.items():
            if log_scale:
                series[name] = [(float(s), float(np.log10(m + 1e-10)))
                                 for s, m in pts]
            else:
                series[name] = [(float(s), float(m)) for s, m in pts]
        y_lbl = "log₁₀(|grad|)" if log_scale else "|gradient|"
        return _sparkline_svg(series, title=title, width=width, height=height,
                               y_label=y_lbl, x_label="step")

    def render_from_tracer(self, tracer, max_layers: int = 8) -> str:
        """Da un GradientFlowTracer."""
        layers = list({k[0] for k in tracer._records.keys()})[:max_layers]
        trajectories = {
            layer: tracer.get_magnitude_trajectory(layer)
            for layer in layers
        }
        return self.render(trajectories)
