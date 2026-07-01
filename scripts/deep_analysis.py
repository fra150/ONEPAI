"""
Analisi profonda L1+L4+L5 su un modello reale, in un unico report.

Combina:
    ModelInstrumentor   (silenzio, attenzione, logit lens reali)
    AbsenceEngine       (dead vs suppressed, token quasi-detti, direzioni abortite)
    InterventionStudy   (ablation sweep + steering sweep)

Uso:
    python scripts/deep_analysis.py --open
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import (
    TinyTransformer, TinyConfig, ModelInstrumentor, AbsenceEngine, InterventionStudy,
)
from onepai.visualization import (
    HTMLReport, SilenceHeatmap, ActivationHeatmap, ActivationTimeline,
    ConfidenceTimeline, NeuralGraphRenderer, line_chart,
)


def build_report(args) -> HTMLReport:
    cfg = TinyConfig(
        n_layers=args.layers, n_heads=args.heads, d_model=args.d_model,
        d_mlp=args.d_model * 2, vocab_size=512, n_ctx=64, seed=11,
        inject_dead_neurons=[(1, 5), (2, 30)],
        inject_suppressed_neurons=[(1, 9), (2, 14)],
        inject_ghost_heads=[(0, 3)],
    )
    model = TinyTransformer(cfg)
    batch = [model.random_tokens(args.seq_len, seed=k) for k in range(args.sequences)]
    print(f"Modello: {model}")

    # --- L1: instrumentazione ---
    inst = ModelInstrumentor(model, silence_threshold=0.05)
    ir = inst.run(batch)
    print("Instrumentor:", ir.summary())

    # --- L4: assenza ---
    absence = AbsenceEngine(model, suppression_pre=1.0).analyze(batch)
    print("Absence    :", absence.summary())

    # --- L5: intervento ---
    ref = batch[0]
    target = model.capture(ref).top_token()       # token che il modello predice
    study = InterventionStudy(model).run_full(ref, target_token=target)
    print("Intervention:", study.summary())

    # ================= Report unificato =================
    report = HTMLReport(title="ONEPAI · Deep Analysis (L1+L4+L5)", model_name=model.model_name)
    report.add_summary({
        "Layers": ir.summary()["layers_tracked"],
        "Dead": absence.summary()["dead"],
        "Suppressed": absence.summary()["suppressed"],
        "Inert heads": ir.summary()["inert_heads"],
        "Crushed tokens": absence.summary()["crushed_tokens"],
    })

    # L1 — silenzio reale
    report.add_section(
        "L1 · Silence Map", SilenceHeatmap(20).render_from_shadow_map(ir.shadow_map, 24),
        icon="◍", note="Neuroni silenziosi su attivazioni MLP reali (verde scuro = morto).")
    report.add_section(
        "L1 · Activation Timeline",
        ActivationTimeline().render(ir.layer_activation_series, "Mean |activation| per layer"),
        icon="◭")
    report.add_section(
        "L1 · Logit Lens (real)", ConfidenceTimeline().render_from_logit_lens(ir.logit_lens),
        icon="◎", note="Confidenza della predizione proiettata a ogni layer reale.")

    # L4 — assenza: dead vs suppressed
    rows = []
    for n in absence.dead_neurons[:40]:
        rows.append({"Neuron": n.name, "Kind": "dead",
                     "Mean pre": f"{n.mean_pre:+.2f}", "|pre|": f"{n.mean_abs_pre:.2f}",
                     "Fired %": f"{n.fired_ratio*100:.1f}"})
    for n in absence.suppressed_neurons[:40]:
        rows.append({"Neuron": n.name, "Kind": "SUPPRESSED",
                     "Mean pre": f"{n.mean_pre:+.2f}", "|pre|": f"{n.mean_abs_pre:.2f}",
                     "Fired %": f"{n.fired_ratio*100:.1f}"})
    report.add_table("L4 · Absence — dead vs suppressed", rows, icon="◌",
                     note="DEAD = niente lo guida (pre≈0). SUPPRESSED = qualcosa lo "
                          "schiaccia (pre forte e negativo, gated dalla GELU).")

    if absence.crushed_tokens:
        crows = [{"Token id": c.token_id, "Peak layer": c.peak_layer,
                  "Peak prob": f"{c.peak_prob:.3f}", "Final prob": f"{c.final_prob:.3f}",
                  "Crush": f"{c.crush:.3f}"} for c in absence.crushed_tokens[:15]]
        report.add_table("L4 · Crushed predictions (quasi-detti)", crows, icon="◑",
                         note="Token che il logit lens vede salire e poi schiacciare "
                              "prima dell'output: ciò che il modello stava per dire.")

    # L5 — intervento: ablation heatmap + head table + steering
    layers = sorted({e.component.rsplit(".", 1)[0] for e in study.ablation_effects})
    kinds = ["attn", "mlp"]
    mat = np.zeros((len(layers), len(kinds)), dtype=np.float32)
    for e in study.ablation_effects:
        base, kind = e.component.rsplit(".", 1)
        if base in layers and kind in kinds:
            mat[layers.index(base), kinds.index(kind)] = e.effect
    report.add_section(
        "L5 · Ablation Sweep",
        ActivationHeatmap(30).render(mat, title=f"Effect on logit[{target}]",
                                     x_labels=kinds, y_labels=layers, diverging=True),
        icon="▽", note="Cosa sottratta. Blu = il componente sosteneva il token; "
                       "rosso = lo deprimeva.")
    if study.head_effects:
        hrows = [{"Head": e.component, "Effect": f"{e.effect:+.3f}"}
                 for e in sorted(study.head_effects, key=lambda e: e.abs_effect, reverse=True)[:12]]
        report.add_table("L5 · Head ablation (ranked)", hrows, icon="◉")
    if study.steering:
        sw = study.steering[0]
        report.add_section(
            "L5 · Steering Sweep",
            line_chart({sw.label: sw.as_series()}, "Target logit vs coefficiente",
                       x_label="coefficient", y_label="target logit"),
            icon="◈", note="Direzione aggiunta. Inietta W_U[target] nel residual stream "
                           "e osserva come risponde il logit del target.")

    report.add_from_attention_summary(ir.attention_tracer.summarize())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Analisi profonda L1+L4+L5")
    ap.add_argument("-o", "--output", default="reports/deep_analysis.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--sequences", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=20)
    args = ap.parse_args()

    report = build_report(args)
    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
