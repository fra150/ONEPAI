"""Tests for the null model + proper ablation modes (zero / mean / resample)."""
import numpy as np
import pytest

from onepai.model import (
    AblationController, HookPoint, InterventionStudy, NullModelStudy,
    SuppressionTracer, TinyConfig, TinyTransformer, compute_mean_activations,
)


def _model(**kw):
    cfg = TinyConfig(n_layers=3, n_heads=4, d_model=32, d_mlp=64,
                     vocab_size=128, seed=3, **kw)
    return TinyTransformer(cfg)


def _batch(m, n=4, seq=8):
    return [m.random_tokens(seq, seed=k) for k in range(n)]


class TestAblationController:
    def test_zero_mode_zeroes_everything(self):
        m = _model()
        ctrl = AblationController(m, mode="zero")
        arr = np.ones((5, 32), dtype=np.float32)
        out = ctrl.hook_for(HookPoint.mlp_out(0))(arr, "x")
        assert np.all(out == 0.0)
        assert np.all(arr == 1.0)          # input non mutato

    def test_zero_mode_head_only(self):
        m = _model()
        ctrl = AblationController(m, mode="zero")
        z = np.ones((5, m.n_heads, 8), dtype=np.float32)
        out = ctrl.hook_for(HookPoint.attn_z(0), head=1)(z, "x")
        assert np.all(out[:, 1, :] == 0.0)
        assert np.all(out[:, 0, :] == 1.0)

    def test_mean_mode_broadcasts_baseline(self):
        m = _model()
        base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ctrl = AblationController(m, mode="mean", means={"h": base})
        out = ctrl.hook_for("h")(np.zeros((4, 3), dtype=np.float32), "h")
        assert out.shape == (4, 3)
        assert np.allclose(out, np.tile(base, (4, 1)))

    def test_mean_requires_baseline(self):
        with pytest.raises(ValueError):
            AblationController(_model(), mode="mean")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            AblationController(_model(), mode="average")

    def test_build_mean_from_batch(self):
        m = _model()
        ctrl = AblationController.build(
            m, mode="mean", reference_batch=_batch(m),
            hook_names=[HookPoint.mlp_out(0)])
        manual = compute_mean_activations(
            m, _batch(m), hook_names=[HookPoint.mlp_out(0)])
        assert np.allclose(ctrl.means[HookPoint.mlp_out(0)],
                           manual[HookPoint.mlp_out(0)])

    def test_resample_fits_different_lengths(self):
        m = _model()
        donor = {HookPoint.mlp_out(0): np.ones((6, m.d_model), dtype=np.float32)}
        ctrl = AblationController(m, mode="resample", donor_cache=donor)
        long_target = np.zeros((10, m.d_model), dtype=np.float32)
        out = ctrl.hook_for(HookPoint.mlp_out(0))(long_target, HookPoint.mlp_out(0))
        assert out.shape == (10, m.d_model)
        assert np.allclose(out, 1.0)       # donatore ripetuto fino a 10

        short_target = np.zeros((3, m.d_model), dtype=np.float32)
        out = ctrl.hook_for(HookPoint.mlp_out(0))(short_target, HookPoint.mlp_out(0))
        assert out.shape == (3, m.d_model)

    def test_resample_requires_donor(self):
        with pytest.raises(ValueError):
            AblationController(_model(), mode="resample")


class TestInterventionStudyAblation:
    def test_mean_sweep_runs_and_differs_from_zero(self):
        m = _model()
        toks = m.random_tokens(8, seed=1)
        zero = InterventionStudy(m).ablation_sweep(toks, target_token=7)
        mean = InterventionStudy.with_ablation(
            m, "mean", reference_batch=_batch(m)).ablation_sweep(toks, 7)
        assert len(zero) == len(mean) == 2 * m.n_layers
        # le due modalita' non devono coincidere (altrimenti l'ablation e' inerte)
        assert any(abs(a.effect - b.effect) > 1e-4 for a, b in zip(zero, mean))

    def test_nonzero_mode_without_controller_raises(self):
        m = _model()
        with pytest.raises(ValueError):
            InterventionStudy(m, ablation="mean").ablation_sweep(
                m.random_tokens(6, seed=0), target_token=1)


class TestCrushProfile:
    def test_profile_shapes_and_consistency(self):
        m = _model()
        tracer = SuppressionTracer(m)
        prof = tracer.crush_profile(m.random_tokens(8, seed=2))
        assert prof.crush.shape == (m.vocab_size,)
        assert prof.peak_layer.shape == (m.vocab_size,)
        assert np.all(prof.crush >= 0.0)
        assert np.allclose(prof.crush, np.maximum(prof.peak_prob - prof.final_prob, 0.0))
        assert prof.final_layer == m.n_layers - 1
        assert 0 <= prof.peak_layer.min() and prof.peak_layer.max() <= m.n_layers - 1

    def test_profile_top_excludes_tokens(self):
        m = _model()
        prof = SuppressionTracer(m).crush_profile(m.random_tokens(6, seed=3))
        top = prof.top(k=3, exclude=[prof.top(k=1)[0]])
        assert len(top) == 3

    def test_find_and_trace_mlp_only(self):
        m = _model()
        tracer = SuppressionTracer(m)
        traces = tracer.find_and_trace(m.random_tokens(10, seed=4),
                                       top_k=2, crush_min=0.0, components="mlp")
        for tr in traces:
            assert all("head" not in s.component for s in tr.suppressors)


class TestSuppressionTracerAblationModes:
    def test_mean_tracer_reports_its_mode(self):
        m = _model()
        tracer = SuppressionTracer.with_ablation(m, "mean", reference_batch=_batch(m))
        tr = tracer.trace(m.random_tokens(8, seed=5), target_token=9,
                          max_suppressors=2, components="mlp")
        assert tr.ablation == "mean"
        assert all(s.component.endswith(".mlp") for s in tr.suppressors)

    def test_missing_controller_raises(self):
        with pytest.raises(ValueError):
            SuppressionTracer(_model(), ablation="mean")


class TestNullModelStudy:
    def test_run_aggregates_records(self):
        m = _model()
        tracer = SuppressionTracer(m)
        study = NullModelStudy(tracer, crush_threshold=0.2, components="mlp")
        batches = _batch(m, n=3, seq=8)
        result = study.run(batches, prompts=["a", "b", "c"])

        assert result.n_prompts == 3
        assert result.mode == "zero"
        assert result.n_layers == m.n_layers
        assert len(result.top_crushes) == 3
        assert 0.0 <= result.frac_prompts_above <= 1.0
        assert 0.0 <= result.last_mlp_rate <= 1.0
        for rec in result.records:
            assert rec.n_tokens == 8
            assert rec.top_crush >= 0.0
            assert rec.top_peak_layer < m.n_layers

    def test_traced_chief_always_from_scan(self):
        m = _model()
        study = NullModelStudy(SuppressionTracer(m), crush_threshold=0.1, components="mlp")
        result = study.run(_batch(m, n=2), trace_chief=True)
        for rec in result.traced:
            assert rec.chief.startswith("blocks.")
            assert rec.chief.endswith(".mlp")

    def test_percentile_and_histogram(self):
        m = _model()
        study = NullModelStudy(SuppressionTracer(m), crush_threshold=0.2)
        result = study.run(_batch(m, n=4))
        p = result.percentile_of(result.median_top_crush)
        assert 0.0 <= p <= 100.0
        hist = result.histogram(bins=6)
        assert "count" in hist
        assert len(hist["count"]) == 6

    def test_summary_and_narrate_are_serializable(self):
        import json
        m = _model()
        study = NullModelStudy(SuppressionTracer(m), crush_threshold=0.2)
        result = study.run(_batch(m, n=2))
        assert isinstance(result.narrate(), str)
        json.dumps(result.summary())       # niente numpy types
