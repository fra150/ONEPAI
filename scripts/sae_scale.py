"""
Esperimento B — SAE su corpus reale (wikitext-2) + interpretazione duale.

Affronta due debolezze insieme:
  SCALA:  SAE addestrata su testo Wikipedia vero e diversificato (decine di
          migliaia di token), non su 4k token di frasi-template.
  LENTE:  le feature soppresse vengono interpretate con ENTRAMBE le lenti:
          unembedding (cosa promuove) + max-activating examples (quando si
          accende, su testo reale). Se concordano → più probabilmente
          monosemantiche.

Uso:
    python scripts/sae_scale.py --open
    python scripts/sae_scale.py --max-tokens 30000 --d-hidden 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
import webbrowser
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import HFAdapter, HookPoint
from onepai.sae import (
    SparseAutoencoder, collect_activations, FeatureSuppressionTracer,
    FeatureInterpreter,
)
from onepai.visualization import HTMLReport, line_chart

CORPUS_PATH = _ROOT / "data" / "corpus" / "wikitext2_paragraphs.json"


def load_chunks(ad, max_tokens: int, chunk_len: int = 32) -> list:
    """Tokenizza i paragrafi wikitext in chunk di lunghezza fissa."""
    paragraphs = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks, total = [], 0
    for p in paragraphs:
        ids = ad.encode(p)
        for start in range(0, len(ids) - chunk_len, chunk_len):
            chunks.append(ids[start:start + chunk_len])
            total += chunk_len
            if total >= max_tokens:
                return chunks
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="SAE su corpus wikitext reale")
    ap.add_argument("-o", "--output", default="reports/sae_scale.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--d-hidden", type=int, default=2048)
    ap.add_argument("--l1", type=float, default=0.008)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--index-seqs", type=int, default=400,
                    help="quante sequenze indicizzare per i max-activating examples")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    ad = HFAdapter("gpt2")
    print(f"Tokenizzo wikitext-2 (target {args.max_tokens} token)...")
    chunks = load_chunks(ad, args.max_tokens)
    print(f"  {len(chunks)} chunk da 32 token = {len(chunks)*32} token")

    print(f"Raccolgo attivazioni resid_post({args.layer}) — {len(chunks)} forward...")
    acts = collect_activations(ad, chunks, HookPoint.resid_post(args.layer),
                               skip_first=1)
    print(f"  attivazioni: {acts.shape}")

    print(f"Addestro SAE {ad.d_model} -> {args.d_hidden} (l1={args.l1})...")
    sae = SparseAutoencoder(d_in=ad.d_model, d_hidden=args.d_hidden,
                            l1_coef=args.l1, seed=0)
    sae.fit(acts, epochs=args.epochs, batch_size=256, lr=3e-3)
    rep = sae.analyze(acts)
    print(f"  {rep}")

    print(f"Indicizzo {args.index_seqs} sequenze per max-activating examples...")
    interp = FeatureInterpreter(ad, sae, layer=args.layer, k=6)
    interp.index_corpus(chunks[:args.index_seqs])

    print(f"Feature soppresse per {args.prompt!r}...")
    tracer = FeatureSuppressionTracer(ad, sae, layer=args.layer)
    toks = ad.encode(args.prompt)
    suppressed = tracer.find_suppressed_features(toks, top_k=5, crush_min=0.5,
                                                 min_peak=0.8, find_suppressors=False)

    duals = []
    for ft in suppressed:
        dual = interp.dual_interpretation(ft.feature_idx)
        duals.append((ft, dual))
        promo = ", ".join(repr(t) for t in dual.unembed_tokens[:5])
        print(f"  #{ft.feature_idx}: peak {ft.peak_act:.1f}@L{ft.peak_layer}")
        print(f"     promuove : {promo}")
        for ex in dual.examples[:3]:
            print(f"     si accende: {ex.activation:5.1f}  ...{ex.context[:90]}")

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · SAE at Scale + Dual Interpretation",
                        model_name="gpt2 / wikitext-2")
    report.add_summary({
        "Corpus tokens": acts.shape[0],
        "SAE features": sae.d_hidden,
        "Live": rep.n_live,
        "Dead": rep.n_dead,
        "Mean L0": round(rep.mean_l0, 1),
        "Variance expl.": f"{rep.variance_explained:.2f}",
    })
    report.add_section(
        "Setup",
        "<p style='font-size:13.5px;line-height:1.7'>SAE addestrata su testo "
        "Wikipedia reale (wikitext-2), non su frasi-template. Ogni feature "
        "soppressa è interpretata con due lenti indipendenti: la proiezione "
        "sull'unembedding (cosa <i>scrive</i>) e i max-activating examples sul "
        "corpus (quando <i>si accende</i>). Concordanza → monosemanticità.</p>",
        icon="◈")

    for rank, (ft, dual) in enumerate(duals, 1):
        traj_svg = line_chart({f"feature #{ft.feature_idx}": ft.trajectory},
                              title=f"Activation across layers — prompt: {args.prompt!r}",
                              x_label="layer", y_label="activation")
        promo = " · ".join(repr(t) for t in dual.unembed_tokens[:8])
        ex_html = "".join(
            f"<tr><td style='font-family:var(--mono);font-size:11px'>{e.activation:.1f}</td>"
            f"<td style='font-size:12px'>{e.context}</td></tr>"
            for e in dual.examples[:5])
        body = (
            f"<div class='note'>peak {ft.peak_act:.1f} @ layer {ft.peak_layer} → "
            f"{ft.final_act:.1f} all'output (crush {ft.crush:.1f}) · "
            f"firing rate sul corpus: {dual.firing_rate*100:.1f}%</div>"
            f"<p style='font-family:var(--mono);font-size:12.5px;color:var(--cyan);"
            f"margin:10px 0'>lente unembed → {promo}</p>"
            f"<table><thead><tr><th>act</th><th>max-activating example (wikitext)</th>"
            f"</tr></thead><tbody>{ex_html}</tbody></table>"
            f"<div class='svg-wrap' style='margin-top:14px'>{traj_svg}</div>"
        )
        report.add_section(f"#{rank} · Feature {ft.feature_idx}", body, icon="◌")

    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
