"""
Sparse Autoencoder su GPT-2 reale — il dizionario delle feature + l'assenza.

Addestra una SAE sulle attivazioni del residual stream di GPT-2 e mostra:
  - quante feature sono VIVE vs MORTE (l'assenza nello spazio delle feature)
  - la sparsità (quante feature attive per token)
  - la "mappa di accensione" delle feature (celle scure = feature mai usate)

Uso:
    python scripts/gpt2_sae.py --open
    python scripts/gpt2_sae.py --layer 6 --d-hidden 1024 --l1 0.008
"""

from __future__ import annotations

import argparse
import sys
import warnings
import webbrowser
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import HFAdapter, HookPoint
from onepai.sae import SparseAutoencoder, collect_activations
from onepai.visualization import HTMLReport, ActivationHeatmap


def make_corpus() -> list:
    subj = ["The cat", "A child", "The king", "My friend", "The doctor",
            "A soldier", "The teacher", "The river", "The machine", "A stranger",
            "The artist", "Her sister", "The captain", "An old man", "The pilot"]
    verb = ["saw", "followed", "remembered", "described", "questioned",
            "painted", "carried", "discovered", "ignored", "praised"]
    obj = ["the ancient castle", "a distant melody", "the broken clock",
           "several old letters", "the quiet garden", "a strange light",
           "the empty street", "three small birds", "the golden key", "a familiar face"]
    return [f"{s} {v} {o} during the long cold winter of that year."
            for s in subj for v in verb for o in obj[:4]]


def main() -> None:
    ap = argparse.ArgumentParser(description="SAE su GPT-2 reale")
    ap.add_argument("-o", "--output", default="reports/gpt2_sae.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--d-hidden", type=int, default=1024)
    ap.add_argument("--l1", type=float, default=0.008)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    ad = HFAdapter("gpt2")
    corpus = make_corpus()
    print(f"Raccolgo attivazioni da resid_post({args.layer}) su {len(corpus)} frasi...")
    acts = collect_activations(ad, [ad.encode(s) for s in corpus],
                               HookPoint.resid_post(args.layer),
                               skip_first=1, max_tokens=args.max_tokens)
    print(f"Attivazioni reali GPT-2: {acts.shape}")

    sae = SparseAutoencoder(d_in=ad.d_model, d_hidden=args.d_hidden,
                            l1_coef=args.l1, seed=0)
    print(f"Addestro SAE {ad.d_model} -> {args.d_hidden} (l1={args.l1}, {args.epochs} epoche)...")
    sae.fit(acts, epochs=args.epochs, batch_size=256, lr=3e-3)
    rep = sae.analyze(acts)
    print("Feature report:", rep.summary())

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · GPT-2 Sparse Features", model_name="gpt2")
    report.add_summary({
        "Features": rep.n_features,
        "Live": rep.n_live,
        "Dead": rep.n_dead,
        "Mean L0": round(rep.mean_l0, 1),
        "Variance expl.": f"{rep.variance_explained:.2f}",
    })

    # Mappa di accensione delle feature: griglia, celle scure = feature morte
    freqs = rep.frequencies
    side = int(np.ceil(np.sqrt(len(freqs))))
    grid = np.zeros(side * side, dtype=np.float32)
    grid[:len(freqs)] = freqs
    grid = grid.reshape(side, side)
    report.add_section(
        "Feature firing map",
        ActivationHeatmap(14).render(grid, title=f"Feature activation frequency (layer {args.layer})",
                                     diverging=False),
        icon="◍", note="Ogni cella è una feature della SAE; il colore è quanto spesso si "
                       "accende. Le celle quasi nere sono feature MORTE: direzioni che il "
                       "dizionario ha imparato ma che GPT-2 non usa qui. L'assenza, nello "
                       "spazio delle feature.")

    # Top feature per frequenza
    rows = [{"Feature": f"#{idx}", "Fires in": f"{freq*100:.1f}% of tokens"}
            for idx, freq in rep.top_features[:15]]
    report.add_table("Most active features", rows, icon="◉",
                     note="Le feature che si attivano più spesso nel residual stream del layer.")

    # Curva di training
    if sae.history:
        from onepai.visualization import line_chart
        series = {"train loss": [(i, v) for i, v in enumerate(sae.history)]}
        report.add_section("Training loss", line_chart(series, "SAE loss per epoch",
                                                       x_label="epoch", y_label="loss"),
                           icon="◭")

    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
