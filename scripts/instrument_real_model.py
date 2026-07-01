"""
Instrumenta un TinyTransformer REALE e genera un report ONEPAI.

A differenza di generate_demo_report.py (dati simulati), qui ogni neurone morto
e ogni testa inerte emergono da un vero forward pass.

Uso:
    python scripts/instrument_real_model.py --open
    python scripts/instrument_real_model.py -o reports/real.html --layers 4
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import TinyTransformer, TinyConfig, ModelInstrumentor


def main() -> None:
    ap = argparse.ArgumentParser(description="Instrumenta un transformer reale")
    ap.add_argument("-o", "--output", default="reports/real_introspection.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--sequences", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=20)
    args = ap.parse_args()

    # Iniettiamo qualche anomalia di ground-truth da far scoprire a ONEPAI
    cfg = TinyConfig(
        n_layers=args.layers, n_heads=args.heads, d_model=args.d_model,
        d_mlp=args.d_model * 2, vocab_size=512, n_ctx=64, seed=11,
        inject_dead_neurons=[(1, 5), (2, 12), (2, 40)],
        inject_ghost_heads=[(0, 3), (3, 1)],
    )
    model = TinyTransformer(cfg)
    print(f"Modello: {model}")
    print(f"Ground truth iniettato: 3 neuroni morti, 2 teste inerti")

    inst = ModelInstrumentor(model, silence_threshold=0.05,
                             dead_neuron_threshold=0.9, ghost_threshold=0.15)
    batch = [model.random_tokens(args.seq_len, seed=k) for k in range(args.sequences)]
    result = inst.run(batch)

    s = result.summary()
    print(f"\nScoperto dal forward reale:")
    print(f"  neuroni morti : {s['dead_neurons']:3d}  -> {result.dead_neuron_names()}")
    print(f"  teste inerti  : {s['inert_heads']:3d}  -> {result.inert_head_names()}")
    print(f"  teste diffuse : {s['diffuse_heads']:3d}")
    print(f"  layer tracciati: {s['layers_tracked']}")

    report = result.to_report(title="ONEPAI · Real Model Introspection")
    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
