"""
Esperimento A — Il censore è sempre lo stesso?

TREATMENT: "The capital of {X} is" con target " {X}" — la ripetizione che il
           modello considera e poi sopprime. Un prompt per paese.
CONTROL:   "{Capitale} is the capital of" con target " {X}" — qui il target è
           la risposta GIUSTA: la soppressione NON deve avvenire.

Se il soppressore modale è lo stesso su molti paesi e il crush nei controlli
resta basso, "blocks.11.mlp sopprime la ripetizione" passa da aneddoto a
regolarità misurata.

Uso:
    python scripts/validate_suppressor.py --open
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
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from onepai.model import HFAdapter, SuppressionValidator
from onepai.visualization import HTMLReport

# (paese, capitale) — nomi a token singolo in GPT-2
PAIRS = [
    ("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("Russia", "Moscow"), ("Egypt", "Cairo"), ("England", "London"),
    ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Validazione statistica del soppressore")
    ap.add_argument("-o", "--output", default="reports/validate_suppressor.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    ad = HFAdapter(args.model)
    v = SuppressionValidator(ad, crush_threshold=0.1, max_suppressors=3)

    treatment = [(f"The capital of {c} is", f" {c}") for c, _ in PAIRS]
    control = [(f"{cap} is the capital of", f" {c}") for c, cap in PAIRS[:8]]

    print(f"Treatment: {len(treatment)} prompt | Control: {len(control)} prompt")
    print("Analisi in corso (ogni treatment richiede ~40 forward)...")
    result = v.validate(treatment, control, verbose=True)

    print()
    print(result.narrate())

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · Suppressor Validation", model_name=args.model)
    s = result.summary()
    report.add_summary({
        "Treatment prompts": s["treatment_cases"],
        "Control prompts": s["control_cases"],
        "Top-1 consistency": f"{result.consistency_top1*100:.0f}%",
        "Top-3 consistency": f"{result.consistency_top3*100:.0f}%",
        "Crush treat vs ctrl": f"{s['mean_crush_treatment']:.2f} / {s['mean_crush_control']:.2f}",
    })
    report.add_section(
        "Verdict",
        f"<div class='note' style='font-size:14px'>{result.narrate()}</div>",
        icon="◈",
        note="Treatment: la ripetizione del paese va soppressa. Control: il paese "
             "è la risposta giusta, la soppressione non deve avvenire.")

    trows = []
    for c in result.treatment:
        if c.skipped:
            trows.append({"Prompt": c.prompt, "Target": c.target_str,
                          "Peak": "—", "Crush": "—", "Top suppressor": f"(saltato: {c.skipped})"})
        else:
            trows.append({
                "Prompt": c.prompt, "Target": c.target_str,
                "Peak": f"{c.peak_prob:.2f}@L{c.peak_layer}",
                "Crush": f"{c.crush:.2f}",
                "Top suppressor": f"{c.top_suppressor} (+{c.top_restoration*100:.0f}%)"
                                  if c.top_suppressor else "—",
            })
    report.add_table("Treatment — repetition should be suppressed", trows, icon="▽")

    crows = [{"Prompt": c.prompt, "Target": c.target_str,
              "Final prob": f"{c.final_prob:.2f}", "Crush": f"{c.crush:.2f}",
              "OK (no suppression)": "✓" if c.crush < 0.15 else "✗"}
             for c in result.control if not c.skipped]
    report.add_table("Control — the target IS the answer", crows, icon="◎")

    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
