"""Tests for onepai.model — real transformer, hooks, interventions, instrumentor."""
import numpy as np
import pytest

from onepai.model import (
    TinyTransformer, TinyConfig, ModelInstrumentor,
    Intervention, CaptureBundle, HookPoint,
)


def _model(**kw):
    cfg = TinyConfig(n_layers=2, n_heads=4, d_model=32, d_mlp=64,
                     vocab_size=128, seed=3, **kw)
    return TinyTransformer(cfg)


# ── TinyTransformer: forward reale ────────────────────────────────────

class TestTinyTransformer:
    def test_deterministic(self):
        m1, m2 = _model(), _model()
        toks = m1.random_tokens(12, seed=5)
        l1, _ = m1.run_with_cache(toks)
        l2, _ = m2.run_with_cache(toks)
        assert np.allclose(l1, l2)

    def test_forward_shapes(self):
        m = _model()
        toks = m.random_tokens(10, seed=1)
        logits, cache = m.run_with_cache(toks)
        assert logits.shape == (10, m.vocab_size)
        assert HookPoint.EMBED in cache
        assert HookPoint.LOGITS in cache
        assert HookPoint.attn_pattern(0) in cache
        assert HookPoint.mlp_post(0) in cache

    def test_attention_is_causal(self):
        m = _model()
        toks = m.random_tokens(8, seed=2)
        _, cache = m.run_with_cache(toks)
        pattern = cache[HookPoint.attn_pattern(0)]  # (n_heads, seq, seq)
        # La riga 0 non puo' attendere posizioni future: pesi futuri ~ 0
        assert pattern[:, 0, 1:].max() < 1e-6
        # Ogni riga somma a 1 (distribuzione)
        assert np.allclose(pattern.sum(axis=-1), 1.0, atol=1e-4)

    def test_dead_neuron_is_actually_dead(self):
        m = _model(inject_dead_neurons=[(1, 7)])
        toks = m.random_tokens(10, seed=4)
        _, cache = m.run_with_cache(toks)
        post = cache[HookPoint.mlp_post(1)]  # (seq, d_mlp)
        assert np.abs(post[:, 7]).max() < 1e-3   # neurone 7 sempre spento

    def test_apply_unembed_shape(self):
        m = _model()
        resid = np.random.randn(6, m.d_model).astype(np.float32)
        logits = m.apply_unembed(resid)
        assert logits.shape == (6, m.vocab_size)
        assert m.supports_logit_lens()

    def test_capture_bundle(self):
        m = _model()
        toks = m.random_tokens(9, seed=6)
        bundle = m.capture(toks)
        assert isinstance(bundle, CaptureBundle)
        assert bundle.n_layers == 2
        assert bundle.attention(0).shape == (4, 9, 9)
        assert bundle.mlp_post(0).shape == (9, 64)


# ── Interventi (L5: bisturi) ──────────────────────────────────────────

class TestInterventions:
    def test_intervention_apply_ablate(self):
        iv = Intervention("blocks.0.mlp.post", kind="ablate", indices=3)
        arr = np.ones((5, 10), dtype=np.float32)
        out = iv.apply(arr)
        assert np.all(out[:, 3] == 0.0)
        assert np.all(out[:, 0] == 1.0)   # gli altri intatti

    def test_intervention_apply_add(self):
        direction = np.ones(8, dtype=np.float32)
        iv = Intervention("blocks.0.resid_post", kind="add",
                          value=direction, coefficient=2.0)
        arr = np.zeros((3, 8), dtype=np.float32)
        out = iv.apply(arr)
        assert np.allclose(out, 2.0)

    def test_intervention_apply_scale(self):
        iv = Intervention("h", kind="scale", coefficient=0.5)
        arr = np.full((4, 4), 4.0, dtype=np.float32)
        assert np.allclose(iv.apply(arr), 2.0)

    def test_ablating_mlp_changes_logits(self):
        m = _model()
        toks = m.random_tokens(10, seed=7)
        base, _ = m.run_with_cache(toks)
        iv = Intervention(HookPoint.mlp_out(0), kind="ablate")
        out = m.run_with_interventions(toks, [iv])
        assert not np.allclose(base, out)   # azzerare l'MLP cambia l'output

    def test_ablating_dead_neuron_has_no_effect(self):
        m = _model(inject_dead_neurons=[(0, 9)])
        toks = m.random_tokens(10, seed=8)
        base, _ = m.run_with_cache(toks)
        # Azzerare un neurone gia' morto non cambia nulla
        iv = Intervention(HookPoint.mlp_post(0), kind="ablate", indices=9)
        out = m.run_with_interventions(toks, [iv])
        assert np.allclose(base, out, atol=1e-5)

    def test_add_direction_changes_logits(self):
        m = _model()
        toks = m.random_tokens(10, seed=9)
        base, _ = m.run_with_cache(toks)
        direction = np.random.RandomState(0).randn(m.d_model).astype(np.float32)
        iv = Intervention(HookPoint.resid_post(1), kind="add",
                          value=direction, coefficient=3.0)
        out = m.run_with_interventions(toks, [iv])
        assert not np.allclose(base, out)

    def test_logit_diff_is_finite(self):
        m = _model()
        toks = m.random_tokens(10, seed=10)
        iv = Intervention(HookPoint.mlp_out(0), kind="ablate")
        diff = m.logit_diff(toks, [iv], target_token=5)
        assert np.isfinite(diff)

    def test_interventions_deterministic(self):
        m = _model()
        toks = m.random_tokens(10, seed=11)
        iv = Intervention(HookPoint.mlp_out(0), kind="scale", coefficient=0.3)
        a = m.run_with_interventions(toks, [iv])
        b = m.run_with_interventions(toks, [iv])
        assert np.allclose(a, b)


# ── Instrumentor: assenza reale ───────────────────────────────────────

class TestInstrumentor:
    @pytest.fixture(scope="class")
    def result(self):
        cfg = TinyConfig(n_layers=3, n_heads=4, d_model=32, d_mlp=64,
                         vocab_size=128, seed=7,
                         inject_dead_neurons=[(1, 5)],
                         inject_ghost_heads=[(0, 3)])
        model = TinyTransformer(cfg)
        inst = ModelInstrumentor(model, silence_threshold=0.05,
                                 dead_neuron_threshold=0.9, ghost_threshold=0.15)
        batch = [model.random_tokens(16, seed=k) for k in range(30)]
        return inst.run(batch)

    def test_detects_injected_dead_neuron(self, result):
        assert "blocks.1.mlp.5" in result.dead_neuron_names()

    def test_detects_injected_inert_head(self, result):
        assert "blocks.0.attn_heads.3" in result.inert_head_names()

    def test_summary_keys(self, result):
        s = result.summary()
        assert s["dead_neurons"] >= 1
        assert s["inert_heads"] >= 1
        assert s["sequences"] == 30

    def test_logit_lens_populated(self, result):
        assert result.logit_lens.summary()["layers"] >= 1

    def test_to_report_html(self, result):
        html = result.to_report().to_html()
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html
        assert result.model_name in html


# ── Interfaccia ModelAdapter ──────────────────────────────────────────

class TestModelAdapter:
    def test_hook_names_count(self):
        m = _model()
        names = m.hook_names()
        # embed + logits + ln_final + 7 hook per layer * 2 layer
        assert HookPoint.EMBED in names
        assert HookPoint.LOGITS in names
        assert len(names) == 3 + 7 * 2

    def test_run_with_interventions_return_cache(self):
        m = _model()
        toks = m.random_tokens(8, seed=12)
        iv = Intervention(HookPoint.mlp_post(0), kind="ablate", indices=2)
        logits, cache = m.run_with_interventions(toks, [iv], return_cache=True)
        assert HookPoint.mlp_post(0) in cache
        assert np.all(cache[HookPoint.mlp_post(0)][:, 2] == 0.0)


# ── AbsenceEngine (L4: l'assenza, dead vs suppressed) ─────────────────

class TestAbsenceEngine:
    @pytest.fixture(scope="class")
    def report(self):
        from onepai.model import AbsenceEngine
        cfg = TinyConfig(n_layers=2, n_heads=4, d_model=32, d_mlp=48,
                         vocab_size=128, seed=5,
                         inject_dead_neurons=[(1, 5)],
                         inject_suppressed_neurons=[(1, 9)])
        model = TinyTransformer(cfg)
        engine = AbsenceEngine(model, suppression_pre=1.0)
        batch = [model.random_tokens(14, seed=k) for k in range(24)]
        return engine.analyze(batch)

    def test_dead_neuron_classified_dead(self, report):
        names = [n.name for n in report.dead_neurons]
        assert "blocks.1.mlp.5" in names

    def test_suppressed_neuron_classified_suppressed(self, report):
        names = [n.name for n in report.suppressed_neurons]
        assert "blocks.1.mlp.9" in names

    def test_suppressed_has_negative_strong_pre(self, report):
        sup = next(n for n in report.suppressed_neurons if n.name == "blocks.1.mlp.9")
        assert sup.mean_pre < 0
        assert sup.mean_abs_pre >= 1.0
        assert sup.suppression_score > 0.5

    def test_dead_has_weak_pre(self, report):
        dead = next(n for n in report.dead_neurons if n.name == "blocks.1.mlp.5")
        assert dead.mean_abs_pre < 1.0   # niente lo guida

    def test_summary_keys(self, report):
        s = report.summary()
        assert s["dead"] >= 1
        assert s["suppressed"] >= 1
        assert "crushed_tokens" in s
        assert "aborted_directions" in s


# ── InterventionStudy (L5: cosa sottratta + direzione aggiunta) ───────

class TestInterventionStudy:
    @pytest.fixture(scope="class")
    def study_model(self):
        cfg = TinyConfig(n_layers=2, n_heads=4, d_model=32, d_mlp=48,
                         vocab_size=128, seed=6, inject_ghost_heads=[(0, 3)])
        return TinyTransformer(cfg)

    def test_ablation_sweep_count(self, study_model):
        from onepai.model import InterventionStudy
        study = InterventionStudy(study_model)
        toks = study_model.random_tokens(12, seed=1)
        effects = study.ablation_sweep(toks, target_token=10)
        assert len(effects) == 2 * study_model.n_layers   # attn + mlp per layer

    def test_head_ablation_count(self, study_model):
        from onepai.model import InterventionStudy
        study = InterventionStudy(study_model)
        toks = study_model.random_tokens(12, seed=2)
        effects = study.head_ablation_sweep(toks, target_token=10)
        assert len(effects) == study_model.n_layers * study_model.n_heads

    def test_inert_head_has_no_ablation_effect(self, study_model):
        from onepai.model import InterventionStudy
        study = InterventionStudy(study_model)
        toks = study_model.random_tokens(12, seed=3)
        effects = study.head_ablation_sweep(toks, target_token=10)
        inert = next(e for e in effects if e.component == "blocks.0.head.3")
        assert inert.abs_effect < 1e-4   # testa gia' inerte: ablarla non cambia nulla

    def test_steering_changes_target_logit(self, study_model):
        from onepai.model import InterventionStudy
        study = InterventionStudy(study_model)
        toks = study_model.random_tokens(12, seed=4)
        direction = study.target_direction(target_token=7)
        sweep = study.steering_sweep(
            toks, HookPoint.resid_post(study_model.n_layers - 1),
            direction, target_token=7, coefficients=[-4, 0, 4])
        # Aggiungere la direzione di unembedding alza il logit del target
        assert sweep.target_logits[-1] > sweep.target_logits[0]

    def test_run_full_and_report(self, study_model):
        from onepai.model import InterventionStudy
        study = InterventionStudy(study_model)
        toks = study_model.random_tokens(12, seed=5)
        result = study.run_full(toks, target_token=10)
        assert result.summary()["components_tested"] > 0
        html = result.to_report().to_html()
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html


# ── SuppressionTracer (collega assenza + intervento) ──────────────────

class TestSuppressionTracer:
    def _model(self):
        return _model(inject_suppressed_neurons=[(1, 9)])

    def test_trajectory_length_and_range(self):
        from onepai.model import SuppressionTracer
        m = self._model()
        tracer = SuppressionTracer(m)
        toks = m.random_tokens(10, seed=1)
        traj = tracer.trajectory(toks, target_token=7)
        assert len(traj) == m.n_layers
        for layer, p in traj:
            assert 0.0 <= p <= 1.0

    def test_trace_structure(self):
        from onepai.model import SuppressionTracer, SuppressionTrace
        m = self._model()
        tracer = SuppressionTracer(m)
        toks = m.random_tokens(10, seed=2)
        tr = tracer.trace(toks, target_token=11, max_suppressors=4)
        assert isinstance(tr, SuppressionTrace)
        assert 0 <= tr.peak_layer < m.n_layers
        assert tr.crush == pytest.approx(tr.peak_prob - tr.final_prob, abs=1e-6)
        # suppressori ordinati per ripristino decrescente
        rest = [s.restoration for s in tr.suppressors]
        assert rest == sorted(rest, reverse=True)

    def test_no_crush_means_no_suppressors(self):
        from onepai.model import SuppressionTracer
        m = self._model()
        tracer = SuppressionTracer(m)
        toks = m.random_tokens(10, seed=3)
        # il token effettivamente predetto sale e basta: niente crush
        pred = m.capture(toks).top_token()
        tr = tracer.trace(toks, pred)
        assert tr.crush < 0.05
        assert tr.suppressors == []

    def test_narrate_returns_string(self):
        from onepai.model import SuppressionTracer
        m = self._model()
        tracer = SuppressionTracer(m)
        toks = m.random_tokens(10, seed=4)
        tr = tracer.trace(toks, target_token=20)
        assert isinstance(tr.narrate(), str)

    def test_find_and_trace_returns_list(self):
        from onepai.model import SuppressionTracer
        m = self._model()
        tracer = SuppressionTracer(m)
        toks = m.random_tokens(12, seed=5)
        traces = tracer.find_and_trace(toks, top_k=2, crush_min=0.01)
        assert isinstance(traces, list)
