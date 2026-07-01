"""
NeuralGraphRenderer — Grafi SVG dei circuiti neurali e dei percorsi ombra.

Usa un layout topologico semplice (strati orizzontali): non richiede
graphviz né matplotlib — puro SVG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["GraphNode", "GraphEdge", "NeuralGraphRenderer"]

_NODE_COLORS = {
    "active":   "#4a90d9",
    "ghost":    "#c0c0c0",
    "shadow":   "#ff8c42",
    "dead":     "#e04040",
    "critical": "#2ecc71",
    "default":  "#7f8c8d",
}


@dataclass
class GraphNode:
    node_id: str
    label: str
    layer_idx: int
    node_type: str = "default"   # active | ghost | shadow | dead | critical
    value: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    weight: float = 1.0
    edge_type: str = "default"   # attention | gradient | residual | default
    metadata: Dict = field(default_factory=dict)


class NeuralGraphRenderer:
    """
    Costruisce e renderizza grafi neurali come SVG.

    Uso::

        renderer = NeuralGraphRenderer()
        renderer.add_node("emb",  "Embedding",  layer_idx=0, node_type="active")
        renderer.add_node("attn0","Attention 0",layer_idx=1, node_type="shadow")
        renderer.add_node("mlp0", "MLP 0",      layer_idx=1, node_type="active")
        renderer.add_node("out",  "Output",     layer_idx=2, node_type="critical")
        renderer.add_edge("emb",  "attn0", weight=0.8)
        renderer.add_edge("emb",  "mlp0",  weight=0.2)
        renderer.add_edge("attn0","out",   weight=0.5, edge_type="attention")
        renderer.add_edge("mlp0", "out",   weight=0.9, edge_type="gradient")
        svg = renderer.render(title="Shadow Circuit")
    """

    def __init__(self, node_radius: int = 24, h_spacing: int = 130, v_spacing: int = 80):
        self.node_radius = node_radius
        self.h_spacing = h_spacing
        self.v_spacing = v_spacing
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

    # ------------------------------------------------------------------
    # Costruzione
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, label: str, layer_idx: int,
                  node_type: str = "default", value: float = 0.0,
                  metadata: Optional[Dict] = None) -> None:
        self._nodes[node_id] = GraphNode(
            node_id=node_id, label=label, layer_idx=layer_idx,
            node_type=node_type, value=value, metadata=metadata or {},
        )

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0,
                  edge_type: str = "default", metadata: Optional[Dict] = None) -> None:
        self._edges.append(GraphEdge(
            source_id=source_id, target_id=target_id,
            weight=weight, edge_type=edge_type, metadata=metadata or {},
        ))

    def add_from_circuit_shadow(self, circuit_shadow_instance) -> None:
        """Popola il grafo da un CircuitShadow già analizzato."""
        shadows = circuit_shadow_instance.circuit_shadows
        for shadow in shadows:
            for i, layer_id in enumerate(shadow.layer_ids):
                ntype = "shadow" if shadow.activation_frequency < 0.05 else "active"
                self.add_node(f"{shadow.circuit_id}:{layer_id}", layer_id,
                               layer_idx=i, node_type=ntype,
                               value=shadow.activation_frequency)
            for i in range(len(shadow.layer_ids) - 1):
                src = f"{shadow.circuit_id}:{shadow.layer_ids[i]}"
                tgt = f"{shadow.circuit_id}:{shadow.layer_ids[i+1]}"
                self.add_edge(src, tgt, weight=shadow.avg_activation_strength)

    def add_from_silence_report(self, silence_report) -> None:
        """Popola il grafo da un SilenceReport (onepai.core.silence_tracer)."""
        for i, entity in enumerate(silence_report.silent_entities):
            ntype = "dead" if entity.silence_ratio > 0.9 else "shadow"
            self.add_node(entity.name, entity.name, layer_idx=i,
                           node_type=ntype, value=entity.silence_ratio)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _compute_positions(self) -> Dict[str, Tuple[float, float]]:
        """Layout topologico: ogni layer_idx è una colonna, nodi in riga."""
        layers: Dict[int, List[str]] = {}
        for nid, node in self._nodes.items():
            layers.setdefault(node.layer_idx, []).append(nid)
        positions = {}
        margin = 60
        for layer_idx in sorted(layers):
            node_ids = layers[layer_idx]
            x = margin + layer_idx * self.h_spacing
            total_h = (len(node_ids) - 1) * self.v_spacing
            y_start = margin + 40
            for i, nid in enumerate(node_ids):
                positions[nid] = (x, y_start + i * self.v_spacing)
        return positions

    # ------------------------------------------------------------------
    # Rendering SVG
    # ------------------------------------------------------------------

    def render(self, title: str = "Neural Graph", width: Optional[int] = None,
               height: Optional[int] = None) -> str:
        if not self._nodes:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><text x="10" y="30">Empty graph</text></svg>'
        positions = self._compute_positions()
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        svg_w = width or (max(xs) + 120)
        svg_h = height or (max(ys) + 80)
        r = self.node_radius
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
            f'font-family="ui-monospace, monospace" font-size="11">',
            '<defs>'
            '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="3.2" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter></defs>',
            f'<rect width="{svg_w}" height="{svg_h}" fill="none" rx="8"/>',
        ]
        if title:
            lines.append(
                f'<text x="{svg_w//2}" y="22" text-anchor="middle" '
                f'font-size="13" font-weight="500" fill="#38e1ff" '
                f'letter-spacing="0.06em">{title}</text>'
            )
        # Archi (prima, sotto i nodi)
        edge_colors = {
            "attention": "#4a90d9",
            "gradient": "#e67e22",
            "residual": "#27ae60",
            "default": "#aaa",
        }
        for edge in self._edges:
            if edge.source_id not in positions or edge.target_id not in positions:
                continue
            x1, y1 = positions[edge.source_id]
            x2, y2 = positions[edge.target_id]
            color = edge_colors.get(edge.edge_type, "#aaa")
            stroke_w = max(1.0, min(4.0, float(edge.weight) * 3))
            opacity = max(0.2, min(1.0, float(edge.weight)))
            # Bezier curve per i loop
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2 - 20
            lines.append(
                f'<path d="M {x1} {y1} Q {cx} {cy} {x2} {y2}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_w:.1f}" '
                f'opacity="{opacity:.2f}">'
                f'<title>{edge.source_id}→{edge.target_id} w={edge.weight:.3f}</title>'
                f'</path>'
            )
        # Nodi
        for nid, node in self._nodes.items():
            if nid not in positions:
                continue
            x, y = positions[nid]
            color = _NODE_COLORS.get(node.node_type, _NODE_COLORS["default"])
            # Alone esterno glow
            lines.append(
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                f'filter="url(#glow)" opacity="0.35"/>'
            )
            # Cerchio principale
            lines.append(
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                f'stroke="rgba(255,255,255,0.85)" stroke-width="1.5" opacity="0.95">'
                f'<title>{node.label}\ntype: {node.node_type}\nvalue: {node.value:.4f}</title>'
                f'</circle>'
            )
            # Label sotto il nodo
            short_label = node.label[:12] + ("…" if len(node.label) > 12 else "")
            lines.append(
                f'<text x="{x}" y="{y + r + 14}" text-anchor="middle" '
                f'fill="#cdd9f5" font-size="9">{short_label}</text>'
            )
            # Valore dentro il cerchio
            if node.value != 0.0:
                lines.append(
                    f'<text x="{x}" y="{y + 4}" text-anchor="middle" '
                    f'fill="white" font-size="8" font-weight="bold">'
                    f'{node.value:.2f}</text>'
                )
        # Legenda
        legend_x = svg_w - 130
        legend_y = 35
        lines.append(f'<rect x="{legend_x-5}" y="{legend_y-14}" width="125" height="{len(_NODE_COLORS)*16+10}" fill="rgba(10,16,32,0.72)" stroke="rgba(120,165,255,0.25)" rx="6"/>')
        for i, (ntype, color) in enumerate(_NODE_COLORS.items()):
            ly = legend_y + i * 16
            lines.append(f'<circle cx="{legend_x+6}" cy="{ly-3}" r="5" fill="{color}"/>')
            lines.append(f'<text x="{legend_x+16}" y="{ly}" fill="#9fb0d6" font-size="9">{ntype}</text>')
        lines.append('</svg>')
        return "\n".join(lines)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    def __repr__(self) -> str:
        return f"NeuralGraphRenderer(nodes={len(self._nodes)}, edges={len(self._edges)})"
