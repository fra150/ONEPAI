"""
ModelInstrumentor — versa le catture REALI di un modello nei tracer ONEPAI.

Fa girare un modello (ModelAdapter) su un batch di sequenze e popola:
    - ShadowRegistry / ShadowMap / SilenceReport   (neuroni morti, silenzio)
    - AttentionTracer                              (teste fantasma, entropia)
    - LogitLens                                    (predizioni layer-by-layer reali)
    - ResidualStreamAnalyzer                       (contributi per layer)

Da qui in poi i neuroni morti e le teste fantasma sono VERI: emergono dal
forward pass, non da np.random.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.observer import Observation
from ..core.registry import ShadowRegistry
from ..core.shadow_mapper import ShadowMapper, ShadowMap
from ..core.silence_tracer import SilenceTracer, SilenceReport
from ..introspection import AttentionTracer, LogitLens, ResidualStreamAnalyzer
from .adapter import ModelAdapter, HookPoint

__all__ = ["ModelInstrumentor", "InstrumentationResult"]


@dataclass
class InstrumentationResult:
    """Tutto ciò che l'instrumentor ha raccolto da un modello reale."""
    model_name: str
    registry: ShadowRegistry
    shadow_map: ShadowMap
    silence_report: SilenceReport
    attention_tracer: AttentionTracer
    logit_lens: LogitLens
    residual_analyzer: ResidualStreamAnalyzer
    layer_activation_series: Dict[str, List[float]] = field(default_factory=dict)
    n_sequences: int = 0

    # -- riepilogo --------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "sequences": self.n_sequences,
            "dead_neurons": len(self.dead_neuron_names()),
            "inert_heads": len(self.inert_head_names()),
            "diffuse_heads": len(self.attention_tracer.identify_diffuse_heads()),
            "layers_tracked": len(self.shadow_map.layer_activity),
        }

    def dead_neuron_names(self) -> List[str]:
        """Neuroni MLP morti (silence_ratio ≥ soglia), esclusi gli head."""
        return [e.name for e in self.silence_report.dead_neurons
                if ".mlp." in e.name and ".attn_heads." not in e.name]

    def inert_head_names(self) -> List[str]:
        """Teste di attenzione inerti: contributo z ≈ 0 al residual stream."""
        return [e.name for e in self.silence_report.dead_neurons
                if ".attn_heads." in e.name]

    # -- report visivo ----------------------------------------------------

    def to_report(self, title: str = "ONEPAI · Real Model Introspection"):
        """Costruisce un HTMLReport dai dati reali catturati."""
        from ..visualization import (
            HTMLReport, SilenceHeatmap, NeuralGraphRenderer,
            ActivationTimeline, ConfidenceTimeline,
        )

        report = HTMLReport(title=title, model_name=self.model_name)
        s = self.summary()
        report.add_summary({
            "Layers tracked": s["layers_tracked"],
            "Dead neurons": s["dead_neurons"],
            "Inert heads": s["inert_heads"],
            "Sequences": s["sequences"],
        })

        # Silence map (reale)
        shm = SilenceHeatmap(cell_size=20)
        report.add_section(
            "Silence Map", shm.render_from_shadow_map(self.shadow_map, max_neurons=24),
            icon="◍", note="Neuroni silenziosi su attivazioni MLP reali. Verde scuro = morto.",
        )

        # Grafo del circuito colorato per silenzio
        gr = NeuralGraphRenderer()
        layer_names = list(self.shadow_map.layer_activity.keys())
        for i, lname in enumerate(layer_names):
            neurons = self.shadow_map.layer_activity[lname]
            mean_sil = float(np.mean([a.silence_ratio for a in neurons.values()])) if neurons else 0.0
            ntype = "dead" if mean_sil > 0.7 else "shadow" if mean_sil > 0.3 else "active"
            gr.add_node(lname, lname, layer_idx=i, node_type=ntype, value=mean_sil)
        for i in range(len(layer_names) - 1):
            gr.add_edge(layer_names[i], layer_names[i + 1], weight=0.6)
        if layer_names:
            report.add_section("Layer Silence Graph", gr.render("Per-layer mean silence"),
                               icon="⬡", note="Colore = silenzio medio del layer.")

        # Timeline attivazioni
        if self.layer_activation_series:
            report.add_section(
                "Activation Timeline",
                ActivationTimeline().render(self.layer_activation_series, "Mean |activation| per layer"),
                icon="◭",
            )

        # Logit lens reale
        if self.logit_lens.summary().get("layers", 0) > 0:
            report.add_section(
                "Logit Lens (real)", ConfidenceTimeline().render_from_logit_lens(self.logit_lens),
                icon="◎", note="Confidenza della predizione proiettata a ogni layer reale.",
            )

        report.add_from_attention_summary(self.attention_tracer.summarize())
        report.add_from_silence_report(self.silence_report)
        return report


class ModelInstrumentor:
    """
    Esegue un ModelAdapter su un batch di sequenze e raccoglie le catture reali.

    Esempio::

        model = TinyTransformer(TinyConfig(n_layers=3, n_heads=4, d_model=32,
                                           inject_dead_neurons=[(1, 5)],
                                           inject_ghost_heads=[(0, 3)]))
        inst = ModelInstrumentor(model, silence_threshold=0.05)
        batch = [model.random_tokens(16, seed=k) for k in range(40)]
        result = inst.run(batch)
        print(result.summary())            # dead_neurons/ghost_heads VERI
        result.to_report().save("real.html")
    """

    def __init__(
        self,
        model: ModelAdapter,
        silence_threshold: float = 0.05,
        dead_neuron_threshold: float = 0.9,
        ghost_threshold: float = 0.15,
    ):
        self.model = model
        self.silence_threshold = silence_threshold
        self.dead_neuron_threshold = dead_neuron_threshold
        self.ghost_threshold = ghost_threshold

    def run(
        self,
        token_batch: Sequence[np.ndarray],
        capture_attention: bool = True,
        capture_logit_lens: bool = True,
        capture_residual: bool = True,
        position: int = -1,
    ) -> InstrumentationResult:
        model = self.model
        registry = ShadowRegistry()
        at = AttentionTracer(ghost_threshold=self.ghost_threshold)
        lens = LogitLens(vocab_size=model.vocab_size, top_k=5)
        ra = ResidualStreamAnalyzer(model_dim=model.d_model)
        layer_series: Dict[str, List[float]] = {}
        supports_lens = capture_logit_lens and model.supports_logit_lens()

        for step, tokens in enumerate(token_batch):
            bundle = model.capture(tokens)

            for i in range(model.n_layers):
                # --- MLP: silenzio per neurone (media |post| sulle posizioni) ---
                post = bundle.mlp_post(i)
                if post is not None:
                    per_neuron = np.abs(post).mean(axis=0)        # (d_mlp,)
                    layer_name = f"blocks.{i}.mlp"
                    registry.add(Observation(
                        name=layer_name, timestamp=datetime.now(),
                        data=per_neuron, meta={"step": step, "kind": "mlp_post"},
                    ))
                    layer_series.setdefault(layer_name, []).append(float(per_neuron.mean()))

                # --- Attention: una matrice per testa ---
                if capture_attention:
                    pattern = bundle.attention(i)                 # (n_heads, seq, seq)
                    if pattern is not None:
                        for head in range(pattern.shape[0]):
                            at.record(f"blocks.{i}", head, pattern[head], step=step)
                    # Contributo per-testa z (seq, n_heads, d_head): se ≈0 la testa è inerte
                    z_t = bundle.cache.get(HookPoint.attn_z(i))
                    if z_t is not None:
                        per_head = np.abs(z_t).mean(axis=(0, 2))  # (n_heads,)
                        registry.add(Observation(
                            name=f"blocks.{i}.attn_heads", timestamp=datetime.now(),
                            data=per_head, meta={"step": step, "kind": "attn_z"},
                        ))

                # --- Residual stream: contributi attn/mlp all'ultima posizione ---
                if capture_residual:
                    attn_out = bundle.cache.get(HookPoint.attn_out(i))
                    mlp_out = bundle.cache.get(HookPoint.mlp_out(i))
                    if attn_out is not None:
                        ra.record_delta("attention", i, f"blocks.{i}.attn",
                                        position=0, delta=attn_out[position], step=step)
                    if mlp_out is not None:
                        ra.record_delta("mlp", i, f"blocks.{i}.mlp",
                                        position=0, delta=mlp_out[position], step=step)

                # --- Logit lens reale: proietta resid_post(i) sul vocabolario ---
                if supports_lens:
                    resid = bundle.resid_post(i)
                    if resid is not None:
                        layer_logits = model.apply_unembed(resid)
                        if layer_logits is not None:
                            lens.record_logits(f"blocks.{i}", position=0,
                                               logits=layer_logits[position],
                                               layer_idx=i, step=step)

        # ShadowMap + SilenceReport dai dati reali
        all_obs = [o for layer in registry.layers() for o in registry.get(layer)]
        shadow_map = ShadowMapper(silence_threshold=self.silence_threshold) \
            .create_shadow_map(observations=all_obs, model_name=model.model_name)
        silence_report = SilenceTracer().trace(
            shadow_map, dead_neuron_threshold=self.dead_neuron_threshold)

        return InstrumentationResult(
            model_name=model.model_name,
            registry=registry,
            shadow_map=shadow_map,
            silence_report=silence_report,
            attention_tracer=at,
            logit_lens=lens,
            residual_analyzer=ra,
            layer_activation_series=layer_series,
            n_sequences=len(token_batch),
        )
