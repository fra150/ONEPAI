# ONEPAI

[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-198%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-research--prototype-orange.svg)]()

> **A mechanistic interpretability framework for the invisible in AI —
> what a model doesn't say, and why.**

Most interpretability tools explain *how* a model reaches its output. ONEPAI
maps the opposite: neurons that never fire, tokens the model almost predicts
and then crushes, and — going one level deeper — the sparse *concepts* a model
builds up internally and then actively suppresses before they reach the output.

It works on real models (a from-scratch numpy transformer, GPT-2, and
GPT-NeoX/Pythia through a small architecture-agnostic adapter layer), not
synthetic data.

## What it actually does

| Level | Question it answers | Module |
|---|---|---|
| Neuron | Is it *dead* (nothing drives it) or *suppressed* (gated off on purpose)? | `onepai.model.AbsenceEngine` |
| Prediction | Which token did the model almost say, and who crushed it? | `onepai.model.SuppressionTracer` |
| Feature | Which directions in a Sparse Autoencoder's dictionary are alive vs. dead? | `onepai.sae.SparseAutoencoder` |
| Concept | Which *concept* (not token) does the model build and then suppress — and which component is responsible? | `onepai.sae.FeatureSuppressionTracer` |

Plus the machinery that makes the above possible on real models:

- **`onepai.model`** — a `transformer_lens`-style adapter interface
  (`run_with_cache`, `run_with_hooks`, `run_with_interventions`), a pure-numpy
  `TinyTransformer` for fast, dependency-free experiments with injectable
  ground-truth anomalies, and `HFAdapter` for real HuggingFace causal-LMs.
  Architectures are resolved through a small registry (`hf_arch.py`) — GPT-2,
  GPT-Neo, GPT-NeoX, OPT, GPT-J, LLaMA/Mistral — so the same pipeline runs on
  different model families without code changes.
- **`onepai.introspection`** — attention tracing (ghost heads, entropy),
  logit lens, gradient flow, activation patching, token attribution, residual
  stream decomposition, feature attribution (integrated gradients, SmoothGrad).
- **`onepai.sae`** — a from-scratch sparse autoencoder (explicit Adam
  backprop, no ML framework required), activation collection from real models,
  and dual interpretation of features: an unembedding lens (what a feature
  *promotes*) cross-checked against max-activating examples on a real corpus
  (what actually makes it *fire*).
- **`onepai.visualization`** — self-contained dark-themed HTML reports (pure
  SVG + vanilla JS, no build step) with heatmaps, force-style graphs, and
  timelines for everything above.
- **`onepai.core`** — the original shadow-mapping / silence-tracing engine,
  plus an encrypted append-only archive format (`.onepai`, AES-256-GCM).

## What's validated, and how

A claim like "component X suppresses token Y" is only interesting if it holds
up across more than one example. `onepai.model.SuppressionValidator` runs a
suppression hypothesis over N prompts plus control prompts where the
suppression should *not* happen, and reports how consistent the responsible
component is. In one experiment (`scripts/validate_suppressor.py`), a single
late MLP block was the top suppressor across 92% of 12 test prompts — and the
control prompts revealed the hypothesis needed to be broadened (it isn't
anti-repetition specifically, it looks more like a general "last-token
arbiter"). That correction is the point: the tool is built to let the data
push back on the theory, not just illustrate it.

## Installation

```bash
git clone https://github.com/fra150/ONEPAI.git
cd ONEPAI
pip install -e .
```

Optional, for real-model work (`HFAdapter`, GPT-2/Pythia experiments):

```bash
pip install torch transformers
```

## Quickstart

### Instrument a real model and find what it hides

```python
from onepai.model import TinyTransformer, TinyConfig, ModelInstrumentor

# a real (if small) transformer, with a known dead neuron and inert head
cfg = TinyConfig(n_layers=3, n_heads=4, d_model=32, d_mlp=64, vocab_size=256,
                 inject_dead_neurons=[(1, 5)], inject_ghost_heads=[(0, 3)])
model = TinyTransformer(cfg)

instrumentor = ModelInstrumentor(model)
batch = [model.random_tokens(16, seed=k) for k in range(40)]
result = instrumentor.run(batch)

print(result.dead_neuron_names())   # ['blocks.1.mlp.5']  -- recovered, not told
print(result.inert_head_names())    # ['blocks.0.attn_heads.3']
result.to_report().save("report.html")
```

### Trace what GPT-2 almost said

```python
from onepai.model import HFAdapter, SuppressionTracer

model = HFAdapter("gpt2")
tracer = SuppressionTracer(model)
tokens = model.encode("The capital of France is")

for trace in tracer.find_and_trace(tokens, top_k=3):
    print(trace.narrate())
# " France" rises to 62% at layer 9, then collapses to 0%
# -- suppressed by blocks.11.mlp (+7.5% if removed)
```

### Find suppressed concepts, not just suppressed tokens

```python
from onepai.model import HFAdapter, HookPoint
from onepai.sae import SparseAutoencoder, collect_activations, FeatureSuppressionTracer

model = HFAdapter("gpt2")
acts = collect_activations(model, corpus_tokens, HookPoint.resid_post(6))

sae = SparseAutoencoder(d_in=model.d_model, d_hidden=1024, l1_coef=0.008)
sae.fit(acts, epochs=50)

tracer = FeatureSuppressionTracer(model, sae, layer=6)
for feature in tracer.find_suppressed_features(tokens, top_k=5):
    print(feature.narrate())   # what it means, and who suppressed it
```

## Reproducible experiments

Each script writes a self-contained HTML report to `reports/`.

| Script | What it shows |
|---|---|
| `scripts/generate_demo_report.py` | Absence + intervention pipeline on synthetic data |
| `scripts/instrument_real_model.py` | Same pipeline on a real (from-scratch) transformer |
| `scripts/deep_analysis.py` | Combined absence + intervention analysis |
| `scripts/gpt2_absence.py` | Suppressed neurons and crushed tokens on real GPT-2 |
| `scripts/gpt2_sae.py` | Sparse autoencoder trained on real GPT-2 activations |
| `scripts/universality.py` | Same phenomena checked across GPT-2 and Pythia (GPT-NeoX) |
| `scripts/feature_suppression.py` | The SAE↔suppression bridge: suppressed concepts |
| `scripts/validate_suppressor.py` | Statistical validation of a suppressor across N prompts + controls |
| `scripts/sae_scale.py` | SAE trained on real Wikipedia text, dual-lens interpretation |

## Project layout

```
src/onepai/
├── core/            shadow mapping, silence tracing, encrypted archives
├── introspection/   attention/logit-lens/gradient/patching/attribution tools
├── model/           model adapters (TinyTransformer, HFAdapter), absence,
│                    interventions, suppression tracing, validation
├── sae/             sparse autoencoder, activation collection, interpretation,
│                    the feature-suppression bridge
├── visualization/   dark-themed self-contained HTML report generator
├── neural/          circuit shadows, weight ghosts, invisible paths
├── memory/          unsaid vault, undone registry, unthought space
├── utils/, reader/  math/config/logging helpers, query DSL, exporters
└── dashboard/, ui/, interfaces/   FastAPI dashboard and CLI
tests/               198 tests (pytest; GPT-2/Pythia tests auto-skip if
                     torch/transformers or the weights aren't available)
scripts/             the experiments listed above
```

## Testing

```bash
python -m pytest tests/ -v
```

Tests that require real model weights (`tests/test_hf.py`) skip automatically
if `torch`/`transformers` aren't installed or the weights aren't cached
locally — the rest of the suite (183 tests) runs anywhere.

## Honest limitations

- The sparse autoencoders here are proof-of-concept scale (thousands to tens
  of thousands of tokens), not the millions used for production-grade
  dictionaries — feature interpretations should be read as hypotheses, cross-
  checked with the dual-lens method, not settled facts.
- Causal claims ("component X suppresses Y") are per-prompt unless run through
  `SuppressionValidator` across multiple prompts and controls.
- Currently supports causal (decoder-only) language models; encoder or
  encoder-decoder architectures (BERT, T5) are not covered.

## License

MIT — see [LICENSE](LICENSE).

## Author

Francesco Bulla
