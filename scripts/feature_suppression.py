"""
Il ponte SAE <-> soppressione, su GPT-2 reale.

Non "quale token il modello si trattiene", ma quale CONCETTO: addestra una SAE
sul residual stream, trova le feature che salgono e poi vengono schiacciate,
le interpreta (cosa significano), e trova chi le sopprime.

Uso:
    python scripts/feature_suppression.py --open
    python scripts/feature_suppression.py --prompt "She picked up the" --layer 6
"""

from __future__ import annotations

import argparse
import sys
import warnings
import webbrowser
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")        # console Windows -> UTF-8
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import HFAdapter, HookPoint
from onepai.sae import SparseAutoencoder, collect_activations, FeatureSuppressionTracer
from onepai.visualization import HTMLReport, line_chart


def make_corpus() -> list:
    subj = ["The cat", "A child", "The king", "My friend", "The doctor", "A soldier",
            "The teacher", "The river", "The pilot", "The artist", "Her sister", "The captain"]
    verb = ["saw", "followed", "remembered", "described", "questioned", "painted",
            "carried", "discovered", "ignored", "praised"]
    obj = ["the ancient castle", "a distant melody", "the broken clock",
           "the quiet garden", "a strange light", "the golden key"]
    return [f"{s} {v} {o} that winter." for s in subj for v in verb[:6] for o in obj[:4]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ponte SAE<->soppressione su GPT-2")
    ap.add_argument("-o", "--output", default="reports/feature_suppression.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--d-hidden", type=int, default=1024)
    ap.add_argument("--l1", type=float, default=0.008)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    ad = HFAdapter("gpt2")
    corpus = make_corpus()
    print(f"Addestro SAE su resid_post({args.layer}) ({len(corpus)} frasi)...")
    acts = collect_activations(ad, [ad.encode(s) for s in corpus],
                               HookPoint.resid_post(args.layer), skip_first=1, max_tokens=4000)
    sae = SparseAutoencoder(d_in=ad.d_model, d_hidden=args.d_hidden, l1_coef=args.l1, seed=0)
    sae.fit(acts, epochs=args.epochs, batch_size=256, lr=3e-3)
    print(f"  {acts.shape[0]} attivazioni, varianza spiegata {sae.variance_explained(acts):.2f}")

    tracer = FeatureSuppressionTracer(ad, sae, layer=args.layer)
    toks = ad.encode(args.prompt)
    print(f"Cerco feature soppresse per: {args.prompt!r}")
    suppressed = tracer.find_suppressed_features(toks, top_k=6, crush_min=0.5,
                                                 min_peak=0.8, find_suppressors=False)
    # Attribuisci il colpevole alle prime 2 (costoso)
    for ft in suppressed[:2]:
        ft.suppressors = tracer._find_suppressors(
            toks, ft.feature_idx, -1, ft.peak_layer, ft.final_act, 6)

    for ft in suppressed:
        print("  ", ft.narrate())

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · Feature Suppression (SAE bridge)", model_name="gpt2")
    report.add_summary({
        "Prompt": f'"{args.prompt}"',
        "SAE features": sae.d_hidden,
        "Suppressed features": len(suppressed),
        "Layer": args.layer,
    })

    report.add_section(
        "What this is",
        "<p style='font-size:14px;line-height:1.7;color:var(--text)'>"
        "Una SAE decompone il residual stream in feature sparse. Qui cerchiamo le "
        "feature che il modello <b>costruisce e poi cancella</b> prima dell'output: "
        "non il token trattenuto, ma il <b>concetto</b> trattenuto. Ogni feature è "
        "interpretata decodificandone la direzione attraverso l'unembedding.</p>",
        icon="◈")

    for rank, ft in enumerate(suppressed, 1):
        traj = line_chart(
            {f"feature #{ft.feature_idx}": ft.trajectory},
            title=f"Feature #{ft.feature_idx} activation across layers",
            x_label="layer", y_label="activation")
        meaning = " · ".join(repr(t) for t in ft.top_tokens[:8])
        body = (
            f"<div class='note'>{ft.narrate()}</div>"
            f"<p style='font-family:var(--mono);font-size:13px;color:var(--cyan);"
            f"margin-bottom:12px'>significa → {meaning}</p>"
            f"<div class='svg-wrap'>{traj}</div>"
        )
        report.add_section(
            f"#{rank}  Feature {ft.feature_idx} — peak {ft.peak_act:.1f}@L{ft.peak_layer} → {ft.final_act:.1f}",
            body, icon="◌")
        if ft.suppressors:
            rows = [{"Component": s.component, "Restores feature by": f"{s.restoration:+.2f}"}
                    for s in ft.suppressors if s.restoration > 0][:6]
            if rows:
                report.add_table(f"Who suppresses feature {ft.feature_idx}", rows, icon="▽")

    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
