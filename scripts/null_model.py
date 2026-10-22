"""
Esperimento 1 — NULL MODEL: quanto è normale un "pensiero soppresso"?

Il punto di partenza: su Qwen2.5-0.5B il token ' __' arriva al 46% al layer 20,
crolla allo 0.5% e `blocks.23.mlp` lo ripristina del +25% se rimosso. È un
pensiero soppresso o la calibrazione normale dell'ultimo layer?

Questo script misura la distribuzione di base su prompt arbitrari di wikitext:

  1. per N prompt: crush profile completo (un forward), top token, quanti token
     superano la soglia;
  2. per il top token di ogni prompt: chief suppressor con ablation ZERO e MEAN
     (se le due modalità non concordano, il numero era un artefatto);
  3. confronto con il prompt "speciale": in che percentile sta il suo crush?

Uso:
    python scripts/null_model.py --prompts 60 --modes zero,mean --open
    python scripts/null_model.py --prompts 10 --quick      # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from onepai.model import HFAdapter, NullModelStudy, SuppressionTracer
from onepai.model.ablation import AblationController
from onepai.visualization import HTMLReport, line_chart

CORPUS = _ROOT / "data" / "corpus" / "wikitext2_paragraphs.json"
REFERENCE_PROMPT = "The capital of France is"


def load_corpus(min_chars: int = 200) -> list[str]:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus non trovato: {CORPUS}")
    paragraphs = json.loads(CORPUS.read_text(encoding="utf-8"))
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]


def make_windows(ad, texts, n: int, seq_len: int, rng) -> tuple[list[np.ndarray], list[str]]:
    """Estrae n finestre casuali di ~seq_len token dal corpus."""
    batches, labels = [], []
    for _ in range(n):
        text = texts[rng.randint(len(texts))]
        ids = ad.encode(text)
        if ids.size <= seq_len:
            start = 0
        else:
            start = int(rng.randint(0, ids.size - seq_len))
        window = ids[start:start + seq_len]
        if window.size < 4:
            continue
        batches.append(window)
        try:
            label = ad.decode(window).strip().replace("\n", " ")
        except Exception:
            label = ""
        labels.append(label)
    return batches, labels


def build_report(results: dict, ref: dict, model_name: str, args) -> HTMLReport:
    report = HTMLReport(title="ONEPAI · Null Model (è normale un crush?)",
                        model_name=model_name)

    modes = list(results.keys())
    cards = {"Prompts": len(next(iter(results.values())).records),
             "Soglia crush": f"{args.threshold:.2f}"}
    for m, r in results.items():
        cards[f"{m} · top-crush mediano"] = f"{r.median_top_crush:.2f}"
        cards[f"{m} · prompt sopra soglia"] = f"{r.frac_prompts_above*100:.0f}%"
        cards[f"{m} · chief = ultimo MLP"] = f"{r.last_mlp_rate*100:.0f}%"
    report.add_summary(cards)

    report.add_section(
        "Come leggere questo esperimento",
        "<div class='note'>Se su prompt arbitrari il crush è comune e il chief "
        "suppressor è quasi sempre l'MLP dell'ultimo layer, allora il fenomeno "
        "misurato su 'The capital of France is' non è un pensiero soppresso: è "
        "il comportamento normale del modello. Se invece il prompt scelto sta "
        "nel decile alto e il chief cambia con il contenuto, la soppressione è "
        "specifica e vale la pena scavare.</div>",
        icon="◈")

    # Istogrammi
    series = {}
    for m, r in results.items():
        h = r.histogram(bins=12)
        if h:
            series[f"{m}-ablation"] = h["count"]
    if series:
        report.add_section(
            "Distribuzione del top-crush",
            line_chart(series, title="Top-crush per prompt (distribuzione nulla)",
                       x_label="crush (peak - final)", y_label="n prompt"),
            icon="▤",
            note="Il crush del prompt che ti interessa va confrontato con questa curva.")

    # Confronto tra modalità
    if len(modes) > 1:
        a, b = modes[0], modes[1]
        ra, rb = results[a], results[b]
        common = {r.prompt: r for r in rb.records}
        agree, both = 0, 0
        for r in ra.records:
            o = common.get(r.prompt)
            if o is None or not r.chief or not o.chief:
                continue
            both += 1
            agree += int(r.chief == o.chief)
        rate = agree / both if both else 0.0
        report.add_section(
            f"Robustezza: {a} vs {b} ablation",
            f"<div class='note'>Il chief suppressor coincide tra le due modalità "
            f"nel <b>{rate*100:.0f}%</b> dei {both} prompt tracciati. "
            f"Se coincide raramente, i 'colpevoli' trovati con zero-ablation sono "
            f"in larga parte artefatti del fuori-distribuzione.</div>",
            icon="⇄")

    # Prompt speciale
    for m, info in ref.items():
        r = results[m]
        pct = r.percentile_of(info["crush"])
        report.add_section(
            f"Prompt di riferimento ({m}-ablation)",
            f"<div class='note'>"
            f"<b>{REFERENCE_PROMPT!r}</b><br>"
            f"top token {info['token']!r}: {info['peak']*100:.0f}% @L{info['peak_layer']} "
            f"→ {info['final']*100:.1f}% (crush {info['crush']:.2f})<br>"
            f"chief suppressor: <b>{info['chief'] or '—'}</b><br>"
            f"percentile del crush nella distribuzione nulla: <b>{pct:.0f}°</b>"
            f" (mediana {r.median_top_crush:.2f}, p90 {r.p90_top_crush:.2f})"
            f"</div>",
            icon="◎")

    # Tabelle
    for m, r in results.items():
        worst = sorted(r.records, key=lambda x: x.top_crush, reverse=True)[:15]
        report.add_table(f"{m}-ablation · i 15 crush più forti", [w.as_row() for w in worst],
                         icon="▽")
        if r.chief_counts:
            rows = [{"Chief suppressor": k, "Prompt": v,
                     "% tracciati": f"{v/max(len(r.traced),1)*100:.0f}%"}
                    for k, v in r.chief_counts.most_common(10)]
            report.add_table(f"{m}-ablation · chi sopprime", rows, icon="◉")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Null model del crush su prompt arbitrari")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompts", type=int, default=60, help="prompt del corpus da analizzare")
    ap.add_argument("--ref-prompts", type=int, default=40, help="prompt per la baseline mean")
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--modes", default="zero,mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="reports/null_model.html")
    ap.add_argument("--json", default="reports/null_model.json")
    ap.add_argument("--quick", action="store_true", help="smoke test (8 prompt, 1 modalità)")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.prompts = min(args.prompts, 8)
        args.ref_prompts = min(args.ref_prompts, 8)
        args.modes = "zero"

    t_start = time.time()
    print(f"Carico {args.model} ...")
    ad = HFAdapter(args.model)
    print(f"  {ad}")

    rng = np.random.RandomState(args.seed)
    texts = load_corpus()
    batches, labels = make_windows(ad, texts, args.prompts, args.seq_len, rng)
    ref_batches, _ = make_windows(ad, texts, args.ref_prompts, args.seq_len, rng)
    print(f"Corpus: {len(batches)} prompt analizzati, {len(ref_batches)} di riferimento "
          f"({args.seq_len} token ciascuno)")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results, ref_info = {}, {}

    for mode in modes:
        print(f"\n=== ablation: {mode} ===")
        t0 = time.time()
        if mode == "zero":
            tracer = SuppressionTracer(ad)
        else:
            ctrl = AblationController.build(
                ad, mode=mode, reference_batch=ref_batches,
                hook_names=[f"blocks.{i}.mlp.out" for i in range(ad.n_layers)])
            tracer = SuppressionTracer(ad, ablation=mode, controller=ctrl)
        study = NullModelStudy(tracer, crush_threshold=args.threshold, components="mlp")
        result = study.run(batches, prompts=labels, verbose=True)
        results[mode] = result
        print(f"  {result.narrate()}")

        # prompt di riferimento, con lo stesso tracer
        toks = ad.encode(REFERENCE_PROMPT)
        rec = study.analyze(toks, prompt=REFERENCE_PROMPT)
        ref_info[mode] = {
            "prompt": REFERENCE_PROMPT,
            "token": rec.top_token_str,
            "crush": rec.top_crush,
            "peak": rec.top_peak_prob,
            "peak_layer": rec.top_peak_layer,
            "final": rec.top_final_prob,
            "chief": rec.chief,
            "percentile": result.percentile_of(rec.top_crush),
        }
        print(f"  riferimento {REFERENCE_PROMPT!r}: crush {rec.top_crush:.2f} "
              f"({result.percentile_of(rec.top_crush):.0f}° percentile), chief {rec.chief}")
        print(f"  ({time.time()-t0:.0f}s)")

    # ── salvataggio ──────────────────────────────────────────────────
    out_json = _ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": args.model,
        "args": vars(args),
        "results": {m: r.summary() for m, r in results.items()},
        "reference": ref_info,
        "records": {m: [vars(rec) for rec in r.records] for m, r in results.items()},
        "chief_counts": {m: dict(r.chief_counts) for m, r in results.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_report(results, ref_info, args.model, args)
    path = report.save(_ROOT / args.output)
    print(f"\nJSON:   {out_json}")
    print(f"Report: {path}  ({path.stat().st_size/1024:.1f} KB)")
    print(f"Tempo totale: {time.time()-t_start:.0f}s")
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
