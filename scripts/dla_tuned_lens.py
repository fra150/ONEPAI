"""
Esperimento 2 — DLA + TUNED LENS.

Il passo 1 ha mostrato che il crush è comune e che il "chief suppressor" dipende
dalla modalità di ablation: parte del segnale era fuori distribuzione. Qui due
misure indipendenti, entrambe a costo ~zero:

  1. DLA — quanto ogni componente scrive *direttamente* contro W_U[target]
     (un forward). Dice chi contribuisce, non solo chi "se rimosso" ripristina.
  2. TUNED LENS — una mappa affine per layer (ridge) che riporta lo stato
     intermedio nello spazio finale: il logit lens smette di produrre spazzatura
     sui layer bassi (dove il passo 1 aveva picchi a L0-L4 con token come '退').

E poi il confronto che conta: **dove DLA e ablation concordano** c'è un effetto
diretto; **dove discordano** c'è un effetto indiretto.

Uso:
    python scripts/dla_tuned_lens.py --prompts 15 --open
    python scripts/dla_tuned_lens.py --quick
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

from onepai.introspection import DirectLogitAttributor, RidgeTunedLens
from onepai.model import (
    AblationController, HFAdapter, HookPoint, SuppressionTracer,
)
from onepai.visualization import HTMLReport, line_chart

CORPUS = _ROOT / "data" / "corpus" / "wikitext2_paragraphs.json"
REFERENCE_PROMPT = "The capital of France is"


def load_corpus(min_chars: int = 200) -> list[str]:
    paragraphs = json.loads(CORPUS.read_text(encoding="utf-8"))
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]


def make_windows(ad, texts, n, seq_len, rng):
    batches, labels = [], []
    for _ in range(n):
        ids = ad.encode(texts[rng.randint(len(texts))])
        start = 0 if ids.size <= seq_len else int(rng.randint(0, ids.size - seq_len))
        w = ids[start:start + seq_len]
        if w.size < 4:
            continue
        batches.append(w)
        try:
            labels.append(ad.decode(w).strip().replace("\n", " "))
        except Exception:
            labels.append("")
    return batches, labels


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlazione di rango (senza scipy): robusta alle scale diverse."""
    if a.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def top_crushed(tracer, tokens, threshold=0.2):
    """Il token più schiacciato con picco non all'ultimo layer."""
    prof = tracer.crush_profile(tokens)
    n = tracer.model.n_layers
    crush = np.where(prof.peak_layer < n - 1, prof.crush, 0.0)
    tok = int(np.argmax(crush))
    if crush[tok] < threshold:
        return None, prof
    return tok, prof


def main() -> None:
    ap = argparse.ArgumentParser(description="DLA + tuned lens su Qwen")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fit-prompts", type=int, default=60)
    ap.add_argument("--prompts", type=int, default=15, help="prompt per DLA vs ablation")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", default="reports/dla_tuned_lens.html")
    ap.add_argument("--json", default="reports/dla_tuned_lens.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.fit_prompts = min(args.fit_prompts, 20)
        args.prompts = min(args.prompts, 4)

    t0 = time.time()
    print(f"Carico {args.model} ...")
    ad = HFAdapter(args.model)
    print(f"  {ad}")

    rng = np.random.RandomState(args.seed)
    texts = load_corpus()
    fit_batches, _ = make_windows(ad, texts, args.fit_prompts, args.seq_len, rng)
    eval_batches, eval_labels = make_windows(ad, texts, args.prompts, args.seq_len, rng)

    # ── 1. tuned lens ────────────────────────────────────────────────
    print(f"\n[1/3] Tuned lens (ridge, alpha={args.alpha}) su {len(fit_batches)} prompt...")
    lens = RidgeTunedLens(ad, alpha=args.alpha).fit(fit_batches, verbose=False)
    print(f"  fit: {lens.summary()}")

    profile = None
    for tokens in eval_batches[:3]:
        prof = lens.kl_profile(tokens)
        if profile is None:
            profile = {l: {"raw_kl": [], "tuned_kl": [], "raw_top": [], "tuned_top": []}
                       for l in prof}
        for l, row in prof.items():
            profile[l]["raw_kl"].append(row["raw_kl"])
            profile[l]["tuned_kl"].append(row["tuned_kl"])
            profile[l]["raw_top"].append(row["raw_top_match"])
            profile[l]["tuned_top"].append(row["tuned_top_match"])
    kl_table = {l: {
        "raw_kl": float(np.mean(v["raw_kl"])),
        "tuned_kl": float(np.mean(v["tuned_kl"])),
        "raw_top": float(np.mean(v["raw_top"])),
        "tuned_top": float(np.mean(v["tuned_top"])),
    } for l, v in profile.items()}
    first = min(kl_table)
    last = max(kl_table)
    print(f"  KL layer {first}: raw {kl_table[first]['raw_kl']:.2f} → "
          f"tuned {kl_table[first]['tuned_kl']:.2f}")
    print(f"  KL layer {last}: raw {kl_table[last]['raw_kl']:.2f} → "
          f"tuned {kl_table[last]['tuned_kl']:.2f}")

    # ── 2. DLA sul prompt di riferimento ─────────────────────────────
    print("\n[2/3] DLA sul prompt di riferimento...")
    # I tracer usano il tuned lens: i picchi precoci del lens grezzo erano in
    # buona parte rumore (passo 1), quindi il crush va misurato col lens pulito.
    tracer_zero = SuppressionTracer(ad, lens=lens)
    ctrl_mean = AblationController.build(
        ad, mode="mean", reference_batch=fit_batches,
        hook_names=[f"blocks.{i}.mlp.out" for i in range(ad.n_layers)])
    tracer_mean = SuppressionTracer(ad, ablation="mean", controller=ctrl_mean, lens=lens)
    attributor = DirectLogitAttributor(ad)

    ref_tokens = ad.encode(REFERENCE_PROMPT)
    ref_tok, ref_prof = top_crushed(tracer_zero, ref_tokens, threshold=0.05)
    ref_info = {}
    if ref_tok is not None:
        dla = attributor.attribute(ref_tokens, ref_tok)
        ref_info["token"] = ad.decode([ref_tok])
        ref_info["crush"] = float(ref_prof.crush[ref_tok])
        ref_info["peak_layer"] = int(ref_prof.peak_layer[ref_tok])
        ref_info["dla"] = dla
        ref_info["against"] = [(c.component, round(c.value, 3)) for c in dla.against(8)]
        ref_info["for"] = [(c.component, round(c.value, 3)) for c in dla.for_(5)]
        mlp_neg = [c for c in dla.sorted() if c.kind == "mlp" and c.value < 0]
        ref_info["dla_top_mlp"] = mlp_neg[0].component if mlp_neg else ""
        ref_info["trace_zero"] = tracer_zero.trace(
            ref_tokens, ref_tok, max_suppressors=3, components="mlp")
        ref_info["trace_mean"] = tracer_mean.trace(
            ref_tokens, ref_tok, max_suppressors=3, components="mlp")
        print(f"  token {ref_info['token']!r} crush {ref_info['crush']:.2f} "
              f"@L{ref_info['peak_layer']}")
        print(f"  DLA contro: {ref_info['against'][:4]}")
        print(f"  chief zero: {ref_info['trace_zero'].narrate()}")
        print(f"  chief mean: {ref_info['trace_mean'].narrate()}")

    # ── 3. DLA vs ablation su piu' prompt ────────────────────────────
    print(f"\n[3/3] DLA vs ablation su {len(eval_batches)} prompt...")
    rows, match_zero, match_mean, considered = [], 0, 0, 0
    rho_zero, rho_mean = [], []
    for i, tokens in enumerate(eval_batches):
        tok, prof = top_crushed(tracer_zero, tokens, args.threshold)
        if tok is None:
            continue
        peak = int(prof.peak_layer[tok])
        dla = attributor.attribute(tokens, tok)
        # Confronto a parita' di candidati: solo MLP dal picco in poi, come
        # fa lo scan di ablazione. Altrimenti si confrontano insiemi diversi.
        dla_mlp = {c.component: c.value for c in dla.contributions
                   if c.kind == "mlp" and c.layer >= peak}
        dla_top = min(dla_mlp, key=dla_mlp.get) if dla_mlp else ""

        tz = tracer_zero.trace(tokens, tok, max_suppressors=999, components="mlp")
        tm = tracer_mean.trace(tokens, tok, max_suppressors=999, components="mlp")
        z_map = {s.component: s.restoration for s in tz.suppressors}
        m_map = {s.component: s.restoration for s in tm.suppressors}
        z_chief = max(z_map, key=z_map.get) if z_map else ""
        m_chief = max(m_map, key=m_map.get) if m_map else ""

        common = sorted(set(dla_mlp) & set(z_map))
        rz = spearman(np.array([dla_mlp[c] for c in common]),
                      np.array([z_map[c] for c in common])) if common else 0.0
        common_m = sorted(set(dla_mlp) & set(m_map))
        rm = spearman(np.array([dla_mlp[c] for c in common_m]),
                      np.array([m_map[c] for c in common_m])) if common_m else 0.0
        rho_zero.append(rz)
        rho_mean.append(rm)

        considered += 1
        match_zero += int(dla_top == z_chief)
        match_mean += int(dla_top == m_chief)
        rows.append({
            "Prompt": eval_labels[i][:34] + "…",
            "Token": repr(ad.decode([tok])),
            "Crush": f"{prof.crush[tok]:.2f}@L{peak}",
            "DLA (contro)": dla_top or "—",
            "Chief zero": z_chief or "—",
            "Chief mean": m_chief or "—",
            "Z": "✓" if dla_top == z_chief else "✗",
            "M": "✓" if dla_top == m_chief else "✗",
            "ρ Z": f"{rz:+.2f}",
            "ρ M": f"{rm:+.2f}",
        })
        print(f"  [{i+1}] {rows[-1]['Token']} DLA {dla_top} | zero {z_chief} | mean {m_chief} | ρ {rz:+.2f}/{rm:+.2f}")

    rate_zero = match_zero / considered if considered else 0.0
    rate_mean = match_mean / considered if considered else 0.0
    mean_rho_zero = float(np.mean(rho_zero)) if rho_zero else 0.0
    mean_rho_mean = float(np.mean(rho_mean)) if rho_mean else 0.0
    print(f"\nDLA == chief(zero): {rate_zero*100:.0f}% ({match_zero}/{considered})")
    print(f"DLA == chief(mean): {rate_mean*100:.0f}% ({match_mean}/{considered})")
    print(f"Spearman DLA vs ablation: zero {mean_rho_zero:+.2f}, mean {mean_rho_mean:+.2f}")

    # ── report ───────────────────────────────────────────────────────
    report = HTMLReport(title="ONEPAI · DLA + Tuned Lens", model_name=args.model)
    report.add_summary({
        "Layer prima · KL raw": f"{kl_table[first]['raw_kl']:.2f}",
        "Layer prima · KL tuned": f"{kl_table[first]['tuned_kl']:.2f}",
        "Top-1 match (tuned)": f"{kl_table[first]['tuned_top']*100:.0f}%",
        "DLA = chief zero": f"{rate_zero*100:.0f}%",
        "DLA = chief mean": f"{rate_mean*100:.0f}%",
        "Spearman zero": f"{mean_rho_zero:+.2f}",
        "Spearman mean": f"{mean_rho_mean:+.2f}",
        "Prompt confrontati": considered,
    })

    report.add_section(
        "Cosa misurano queste due cose",
        "<div class='note'><b>DLA</b>: quanto ogni componente scrive direttamente "
        "contro il logit del token target (linearizzando l'ultima norma) — un "
        "forward, nessuna ablation. <b>Tuned lens</b>: una mappa affine per layer "
        "che riporta lo stato intermedio nello spazio finale, così il lens non "
        "produce spazzatura sui layer bassi. Se DLA e ablation <b>concordano</b> "
        "l'effetto è diretto; se <b>discordano</b> è indiretto (il componente "
        "agisce tramite altri componenti a valle).</div>",
        icon="◈")

    series = {"raw logit lens": [(l, kl_table[l]["raw_kl"]) for l in sorted(kl_table)],
              "tuned lens (ridge)": [(l, kl_table[l]["tuned_kl"]) for l in sorted(kl_table)]}
    report.add_section(
        "Il tuned lens è meglio del lens grezzo? (e il bug dell'ultimo layer)",
        line_chart(series, title="KL(final || lens) per layer", x_label="layer",
                   y_label="KL"),
        icon="▤",
        note="La curva tuned deve stare sotto la raw e scendere verso zero; "
             "all'ultimo layer coincidono per costruzione. Nota: in transformers 5.x "
             "out.hidden_states[-1] è già normalizzato dalla norma finale, quindi "
             "ONEPAI ora cattura resid_post con hook sui blocchi (prima la norma "
             "veniva applicata due volte sull'ultimo layer).")

    if ref_info:
        tz, tm = ref_info["trace_zero"], ref_info["trace_mean"]
        report.add_section(
            "Prompt di riferimento",
            f"<div class='note'>Token <b>{ref_info['token']!r}</b>, crush "
            f"{ref_info['crush']:.2f}, picco al layer {ref_info['peak_layer']}.<br>"
            f"DLA: scrive più contro → <b>{ref_info['dla_top_mlp'] or '—'}</b><br>"
            f"Ablation zero: <b>{tz.chief_suppressor.component if tz.chief_suppressor else '—'}</b><br>"
            f"Ablation mean: <b>{tm.chief_suppressor.component if tm.chief_suppressor else '—'}</b>"
            f"</div>",
            icon="◎")
        report.add_table(
            "DLA · chi scrive contro il token",
            [{"Componente": c, "Contributo": f"{v:+.3f}"} for c, v in ref_info["against"]],
            icon="▽")
        report.add_table(
            "DLA · chi scrive a favore",
            [{"Componente": c, "Contributo": f"{v:+.3f}"} for c, v in ref_info["for"]],
            icon="△")

        # traiettoria raw vs tuned
        traj = {"raw": [], "tuned": []}
        _, cache = ad.run_with_cache(ref_tokens)
        for l in sorted(lens.fits):
            res = cache[HookPoint.resid_post(l)]
            raw = _softmax(ad.apply_unembed(res)[-1])[ref_tok]
            tuned = _softmax(lens.logits(res, l)[-1])[ref_tok]
            traj["raw"].append((l, float(raw)))
            traj["tuned"].append((l, float(tuned)))
        report.add_section(
            "Traiettoria del token quasi-detto: raw vs tuned",
            line_chart(traj, title=f"P({ref_info['token']!r}) per layer",
                       x_label="layer", y_label="probabilità"),
            icon="◈")

    report.add_table(
        "DLA vs ablation, prompt per prompt", rows, icon="⇄",
        note="Z/M = il componente più negativo secondo DLA coincide con il chief "
             "suppressor dell'ablation zero/mean. ρ = Spearman tra contributo DLA "
             "e restoration dell'ablation sugli stessi MLP: le due misure hanno "
             "POLARITÀ OPPOSTA, quindi ρ < 0 significa accordo (chi scrive contro "
             "è anche chi, rimosso, ripristina). ρ ≈ 0 = l'effetto è indiretto.")

    try:
        import tuned_lens as _belrose  # noqa: F401
        sanity = ("Pacchetto `tuned-lens` di Belrose disponibile: confronto diretto "
                  "possibile in una prossima esecuzione.")
    except Exception:
        sanity = ("Pacchetto `tuned-lens` di Belrose non installato: il confronto "
                  "diretto non è stato eseguito. Unica verifica disponibile qui: "
                  "all'ultimo layer la mappa ridge è ≈ identità e la KL tuned→raw.")
    report.add_section("Sanity check", f"<div class='note'>{sanity}</div>", icon="⚠")

    out_json = _ROOT / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": args.model,
        "args": vars(args),
        "lens_fit": lens.summary(),
        "kl_profile": kl_table,
        "reference": {
            k: v for k, v in ref_info.items() if k in
            ("token", "crush", "peak_layer", "dla_top_mlp", "against", "for")
        },
        "comparison": {"rate_zero": rate_zero, "rate_mean": rate_mean,
                       "spearman_zero": mean_rho_zero,
                       "spearman_mean": mean_rho_mean,
                       "considered": considered, "rows": rows},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    path = report.save(_ROOT / args.output)
    print(f"\nJSON:   {out_json}")
    print(f"Report: {path}  ({path.stat().st_size/1024:.1f} KB)")
    print(f"Tempo totale: {time.time()-t0:.0f}s")
    if args.open:
        webbrowser.open(path.as_uri())


def _softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


if __name__ == "__main__":
    main()
