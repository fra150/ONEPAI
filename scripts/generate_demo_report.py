"""
Genera un report di introspezione ONEPAI demo con dati simulati.

Uso:
    python scripts/generate_demo_report.py
    python scripts/generate_demo_report.py --open      # apre nel browser
    python scripts/generate_demo_report.py -o reports/mio.html
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

# Aggiunge src/ al path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.core.observer import Observation
from onepai.core.registry import ShadowRegistry
from onepai.core.shadow_mapper import ShadowMapper
from onepai.core.silence_tracer import SilenceTracer
from onepai.introspection import (
    AttentionTracer, LogitLens, GradientFlowTracer, ResidualStreamAnalyzer,
)
from onepai.visualization import (
    HTMLReport, ActivationHeatmap, SilenceHeatmap, AttentionHeatmap,
    NeuralGraphRenderer, ActivationTimeline, ConfidenceTimeline, GradientTimeline,
)


def build_demo(seed: int = 42, model_name: str = "demo-transformer-8L") -> HTMLReport:
    np.random.seed(seed)
    layers = ["emb", "attn.0", "mlp.0", "attn.1", "mlp.1", "attn.2", "mlp.2", "head"]

    # ── 1. Registry con dati simulati (mlp.1 = strato quasi morto) ──────
    registry = ShadowRegistry()
    for step in range(30):
        for layer in layers:
            silence_prob = 0.95 if layer == "mlp.1" else 0.1
            val = 0.0 if np.random.rand() < silence_prob else float(np.random.rand())
            registry.add(Observation(name=f"{layer}:{step % 8}",
                                     timestamp=datetime.now(),
                                     data=val, meta={"step": step}))

    all_obs = [o for layer in registry.layers() for o in registry.get(layer)]
    shadow_map = ShadowMapper().create_shadow_map(observations=all_obs, model_name=model_name)
    silence_report = SilenceTracer().trace(shadow_map, dead_neuron_threshold=0.5)

    # ── 2. Attention (head 3 = fantasma in ogni layer) ─────────────────
    at = AttentionTracer(ghost_threshold=0.15)
    for step in range(10):
        for layer in ["attn.0", "attn.1", "attn.2"]:
            for head in range(4):
                if head == 3:
                    w = np.full((8, 8), 0.005)
                else:
                    w = np.random.dirichlet(np.ones(8) * (head + 1), size=8)
                at.record(layer, head, w, step=step)

    # ── 3. Gradient flow (mlp.1 = gradienti morenti) ───────────────────
    gt = GradientFlowTracer()
    for step in range(20):
        for layer in layers:
            mag = 1e-9 if layer == "mlp.1" else np.random.rand() * 0.5 + 0.1
            gt.record(layer, "W", np.random.rand(32) * mag, step=step)

    # ── 4. Logit lens (token 42 emerge gradualmente) ───────────────────
    lens = LogitLens(vocab_size=1000, top_k=5)
    for i, layer in enumerate(layers):
        logits = np.random.randn(1000)
        logits[42] = float(i) * 1.5
        lens.record_logits(layer, position=0, logits=logits, layer_idx=i, step=0)

    # ── 5. Grafo neurale con shadow highlight ──────────────────────────
    gr = NeuralGraphRenderer()
    for i, layer in enumerate(layers):
        ntype = ("dead" if layer == "mlp.1"
                 else "ghost" if "attn.2" in layer
                 else "critical" if layer == "head"
                 else "active")
        sr = shadow_map.layer_activity.get(layer, {})
        val = float(np.mean([a.silence_ratio for a in sr.values()])) if sr else 0.0
        gr.add_node(layer, layer, layer_idx=i // 2, node_type=ntype, value=val)
    for i in range(len(layers) - 1):
        etype = "gradient" if "mlp" in layers[i + 1] else "attention"
        gr.add_edge(layers[i], layers[i + 1], weight=0.7, edge_type=etype)

    # ── 6. Costruzione report ──────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · Deep Introspection Report", model_name=model_name)

    dead, silent = silence_report.dead_neurons, silence_report.silent_layers
    report.add_summary({
        "Layers tracked": len(layers),
        "Dead neurons": len(dead),
        "Silent layers": len(silent),
        "Ghost heads": len(at.identify_ghost_heads()),
    })

    report.add_section("Activation Overview",
                       ActivationHeatmap(22).render_from_registry(registry, 8),
                       icon="◬", note="Media dell'activation_value per layer. mlp.1 è quasi sempre spento.")
    report.add_section("Silence Map",
                       SilenceHeatmap(24).render_from_shadow_map(shadow_map, 10),
                       icon="◍", note="Verde scuro = silenzio totale (silence_ratio → 1.0).")
    report.add_section("Neural Circuit Graph",
                       gr.render("Model circuit — shadow highlight"),
                       icon="⬡", note="Rosso=morto · grigio=fantasma · arancio=ombra · verde=critico.")

    layer_vals = {l: [float(o.data or 0) for o in registry.get(l)[:20]] for l in layers[:6]}
    report.add_section("Activation Timeline",
                       ActivationTimeline().render(layer_vals, "Activation vs step"),
                       icon="◭")
    report.add_section("Logit Lens — Confidence",
                       ConfidenceTimeline().render_from_logit_lens(lens),
                       icon="◎", note="La confidenza del token target cresce strato dopo strato.")

    inf_mat = at.build_influence_matrix(seq_len=8)
    report.add_section("Attention Flow Matrix",
                       AttentionHeatmap(26).render(inf_mat, title="Aggregate attention flow"),
                       icon="◉", collapsed=True)

    gtl = GradientTimeline()
    grad_traj = {l: gt.get_magnitude_trajectory(l) for l in layers}
    report.add_section("Gradient Flow",
                       gtl.render(grad_traj, "Gradient magnitude per layer (log scale)"),
                       icon="▽", note="mlp.1 collassa verso zero: gradienti morenti.")

    report.add_from_attention_summary(at.summarize())
    report.add_from_gradient_report(gt.flow_report())
    report.add_from_silence_report(silence_report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera report demo ONEPAI")
    parser.add_argument("-o", "--output", default="reports/demo_introspection.html")
    parser.add_argument("--open", action="store_true", help="apre il report nel browser")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_demo(seed=args.seed)
    path = report.save(args.output)
    size_kb = path.stat().st_size / 1024
    print(f"Report generato: {path}")
    print(f"Dimensione:      {size_kb:.1f} KB")
    print(f"Sezioni:         {len(report._sections)} + {len(report._tables)} tabelle")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
