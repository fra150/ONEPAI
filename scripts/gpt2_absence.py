"""
ONEPAI su GPT-2 REALE: l'interpretabilità dell'assenza + il bisturi.

Mostra, su gpt2 vero:
  - neuroni SOPPRESSI (il modello li spinge attivamente a zero)
  - token QUASI-DETTI (saliti nel logit lens e poi schiacciati prima dell'output)
  - ablation sweep + head ablation (cosa guida la predizione)
  - steering (spingere GPT-2 verso un token scelto)

Uso:
    python scripts/gpt2_absence.py --open
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

from onepai.model import HFAdapter, AbsenceEngine, InterventionStudy, HookPoint
from onepai.visualization import HTMLReport, ActivationHeatmap, line_chart

SENTENCES = [
    "The capital of France is", "She opened the door and",
    "In the beginning there was", "The scientist carefully measured the",
    "He could not believe what", "Once upon a time, a",
    "The stock market crashed after", "Water boils at a temperature of",
    "The president announced that", "My favorite kind of music is",
    "The dog ran across the", "They traveled all the way to",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="ONEPAI su GPT-2 reale")
    ap.add_argument("-o", "--output", default="reports/gpt2_absence.html")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steer-to", default=" Paris")
    args = ap.parse_args()

    ad = HFAdapter(args.model)
    print(ad)

    # ── L4: assenza ──────────────────────────────────────────────────
    batch = [ad.encode(s) for s in SENTENCES]
    eng = AbsenceEngine(ad, fire_threshold=0.05, fired_tolerance=0.02,
                        suppression_pre=4.0, crush_min=0.03, max_lens_sequences=len(batch))
    rep = eng.analyze(batch)
    print("Absence:", rep.summary())

    # ── L5: intervento su una frase ─────────────────────────────────
    ref = ad.encode("The capital of France is")
    pred = int(ad.capture(ref).top_token())
    study = InterventionStudy(ad)
    print(f"GPT-2 predice: {ad.decode([pred])!r}; ablation + head sweep in corso...")
    ablation = study.ablation_sweep(ref, pred)
    heads = study.head_ablation_sweep(ref, pred)
    steer_tok = int(ad.encode(args.steer_to)[0])
    steer = study.steering_sweep(
        ref, HookPoint.resid_post(ad.n_layers - 1),
        study.target_direction(steer_tok), steer_tok,
        coefficients=[-4, -2, 0, 2, 4, 6, 8], label=f"-> {args.steer_to.strip()}")

    # ── Report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · GPT-2 Real Introspection", model_name=args.model)
    report.add_summary({
        "Layers": ad.n_layers,
        "Suppressed neurons": len(rep.suppressed_neurons),
        "Crushed tokens": len(rep.crushed_tokens),
        "Heads tested": len(heads),
    })

    # Neuroni soppressi
    srows = [{"Neuron": n.name, "Mean pre": f"{n.mean_pre:.2f}",
              "|pre|": f"{n.mean_abs_pre:.2f}", "Fired %": f"{n.fired_ratio*100:.1f}"}
             for n in sorted(rep.suppressed_neurons, key=lambda x: x.mean_abs_pre, reverse=True)[:20]]
    report.add_table("Suppressed neurons (gated off)", srows, icon="◌",
                     note="Neuroni con pre-attivazione forte ma NEGATIVA: il modello li "
                          "schiaccia attivamente con la GELU. Non morti — soppressi.")

    # Token quasi-detti (DECODIFICATI)
    crows = [{"Token": repr(ad.decode([c.token_id])), "Peak layer": c.peak_layer,
              "Peak prob": f"{c.peak_prob:.2f}", "Final prob": f"{c.final_prob:.3f}",
              "Crush": f"{c.crush:.2f}"} for c in rep.crushed_tokens[:15]]
    report.add_table("Crushed predictions — what GPT-2 almost said", crows, icon="◑",
                     note="Token che il logit lens vede salire strato dopo strato e poi "
                          "schiacciare prima dell'output. Es: per 'France is' il modello "
                          "considera ripetere 'France' e poi si trattiene.")

    # Ablation heatmap (layer x {attn,mlp})
    layers = sorted({e.component.rsplit(".", 1)[0] for e in ablation},
                    key=lambda s: int(s.split(".")[1]))
    kinds = ["attn", "mlp"]
    mat = np.zeros((len(layers), len(kinds)), dtype=np.float32)
    for e in ablation:
        base, kind = e.component.rsplit(".", 1)
        if base in layers and kind in kinds:
            mat[layers.index(base), kinds.index(kind)] = e.effect
    report.add_section(
        "Ablation sweep", ActivationHeatmap(26).render(
            mat, title=f"Effect on logit[{ad.decode([pred]).strip()!r}]",
            x_labels=kinds, y_labels=[l.replace("blocks.", "L") for l in layers], diverging=True),
        icon="▽", note="Cosa sottratta: azzero ogni componente e misuro l'effetto sul "
                       "token che GPT-2 predice.")

    # Top causal heads
    hrows = [{"Head": e.component.replace("blocks.", "L").replace(".head.", " H"),
              "Effect": f"{e.effect:+.2f}"}
             for e in sorted(heads, key=lambda e: e.abs_effect, reverse=True)[:12]]
    report.add_table("Most causal attention heads", hrows, icon="◉",
                     note="Teste il cui azzeramento sposta di piu' il logit della predizione.")

    # Steering curve
    report.add_section(
        "Steering sweep", line_chart({steer.label: steer.as_series()},
                                     f"logit[{args.steer_to.strip()}] vs coefficiente",
                                     x_label="coefficient", y_label="target logit"),
        icon="◈", note=f"Direzione aggiunta: inietto W_U[{args.steer_to.strip()}] nel "
                       f"residual stream finale e guardo salire il logit del target.")

    path = report.save(args.output)
    print(f"\nReport: {path}  ({path.stat().st_size/1024:.1f} KB)")
    if args.open:
        webbrowser.open(path.as_uri())
        print("Aperto nel browser.")


if __name__ == "__main__":
    main()
