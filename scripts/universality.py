"""
Universalità: ONEPAI come strumento per l'AI in generale, non per un solo modello.

Esegue la stessa identica pipeline di assenza su modelli di ARCHITETTURE DIVERSE
e mostra che i fenomeni (neuroni soppressi, token quasi-detti) compaiono in tutti
— e che alcuni token soppressi sono persino condivisi tra architetture.

Uso:
    python scripts/universality.py --open
    python scripts/universality.py --models gpt2 distilgpt2 EleutherAI/pythia-160m
"""

from __future__ import annotations

import argparse
import sys
import warnings
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import HFAdapter, AbsenceEngine, SuppressionTracer
from onepai.visualization import HTMLReport

CORPUS = [
    "The capital of France is", "She opened the door and",
    "Water boils at a temperature of", "The president announced that",
    "He could not believe what", "The dog ran across the",
    "Once upon a time there was", "My favorite kind of music is",
]


def analyze_model(model_name: str) -> dict:
    ad = HFAdapter(model_name)
    batch = [ad.encode(s) for s in CORPUS]
    rep = AbsenceEngine(ad, suppression_pre=4.0, crush_min=0.03,
                        max_lens_sequences=len(batch)).analyze(batch)
    crushed = [ad.decode([c.token_id]) for c in rep.crushed_tokens[:12]]
    tracer = SuppressionTracer(ad)
    traces = tracer.find_and_trace(ad.encode("The capital of France is"),
                                   top_k=1, crush_min=0.1)
    return {
        "name": model_name,
        "arch": ad.model_type,
        "layers": ad.n_layers,
        "d_model": ad.d_model,
        "suppressed": len(rep.suppressed_neurons),
        "crushed_n": len(rep.crushed_tokens),
        "crushed": crushed,
        "top_suppressed": sorted(rep.suppressed_neurons,
                                 key=lambda x: x.mean_abs_pre, reverse=True)[:4],
        "narrate": traces[0].narrate() if traces else "(nessun token schiacciato sopra soglia)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Universalità di ONEPAI tra architetture")
    ap.add_argument("-o", "--output", default="reports/universality.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "distilgpt2", "EleutherAI/pythia-160m"])
    args = ap.parse_args()

    results = []
    for m in args.models:
        print(f"Analizzo {m} ...")
        try:
            results.append(analyze_model(m))
        except Exception as e:
            print(f"  saltato ({e})")

    # Token soppressi condivisi tra modelli
    token_counts = Counter()
    for r in results:
        for t in set(s.strip() for s in r["crushed"]):
            if t:
                token_counts[t] += 1
    shared = [t for t, c in token_counts.most_common() if c >= 2]

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · Universality across architectures",
                        model_name=f"{len(results)} models")
    report.add_summary({
        "Models": len(results),
        "Architectures": len({r["arch"] for r in results}),
        "Shared crushed tokens": len(shared),
    })

    # Tabella di confronto
    comp = [{"Model": r["name"], "Architecture": r["arch"], "Layers": r["layers"],
             "Suppressed neurons": r["suppressed"], "Crushed tokens": r["crushed_n"]}
            for r in results]
    report.add_table("Absence phenomena across models", comp, icon="◎",
                     note="La STESSA pipeline ONEPAI, su architetture diverse. I fenomeni "
                          "di assenza compaiono ovunque.")

    if shared:
        report.add_section(
            "Universal crushed tokens",
            "<p style='font-family:var(--mono);font-size:15px;color:var(--cyan);"
            "padding:8px 0'>" + " · ".join(repr(t) for t in shared) + "</p>",
            icon="◑", note="Token che vengono 'quasi detti' e poi soppressi in PIÙ "
                           "architetture diverse. L'assenza non è un quirk di un modello: "
                           "è una proprietà generale dei transformer.")

    # Per ogni modello: dettaglio
    for r in results:
        rows = [{"Neuron": n.name, "Mean pre": f"{n.mean_pre:.1f}",
                 "|pre|": f"{n.mean_abs_pre:.1f}"} for n in r["top_suppressed"]]
        body = (f"<div class='note'>{r['narrate']}</div>"
                f"<p style='font-family:var(--mono);font-size:12px;color:var(--muted)'>"
                f"quasi-detti: {' · '.join(repr(t.strip() or t) for t in r['crushed'][:8])}</p>")
        report.add_section(f"{r['name']}  [{r['arch']}]", body, icon="◌")
        if rows:
            report.add_table(f"{r['name']} — top suppressed neurons", rows, icon="◍")

    path = report.save(args.output)
    print(f"\nModelli analizzati: {[r['arch'] for r in results]}")
    print(f"Token soppressi condivisi: {shared}")
    print(f"Report: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
