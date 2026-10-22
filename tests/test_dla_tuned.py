"""Tests for onepai.introspection — Direct Logit Attribution e tuned lens."""
import numpy as np
import pytest

from onepai.introspection import DirectLogitAttributor, RidgeTunedLens
from onepai.model import HookPoint, TinyConfig, TinyTransformer


def _model(**kw):
    cfg = TinyConfig(n_layers=3, n_heads=4, d_model=32, d_mlp=64,
                     vocab_size=128, seed=5, **kw)
    return TinyTransformer(cfg)


def _batch(m, n=6, seq=8):
    return [m.random_tokens(seq, seed=k) for k in range(n)]


class TestDirectLogitAttribution:
    def test_reconstruction_is_exact(self):
        """La somma dei contributi deve ricostruire il logit (Tiny ha beta=0)."""
        m = _model()
        toks = m.random_tokens(10, seed=1)
        res = DirectLogitAttributor(m).attribute(toks, target_token=7)
        assert res.constant_offset == pytest.approx(0.0, abs=1e-3)
        assert res.warnings == []

    def test_components_partition_the_residual(self):
        m = _model()
        res = DirectLogitAttributor(m).attribute(m.random_tokens(8, seed=2), 3)
        names = [c.component for c in res.contributions]
        assert names.count("embed") == 1
        assert sum(n.endswith(".mlp") for n in names) == m.n_layers
        # con le teste: una voce per testa per layer, e nessun attn aggregato
        assert sum(".head." in n for n in names) == m.n_layers * m.n_heads
        assert not any(n.endswith(".attn") for n in names)

    def test_without_heads_uses_aggregate_attn(self):
        m = _model()
        res = DirectLogitAttributor(m).attribute(
            m.random_tokens(8, seed=3), 5, include_heads=False)
        names = [c.component for c in res.contributions]
        assert sum(n.endswith(".attn") for n in names) == m.n_layers
        assert not any(".head." in n for n in names)
        assert res.constant_offset == pytest.approx(0.0, abs=1e-3)

    def test_against_returns_negative_sorted(self):
        m = _model()
        res = DirectLogitAttributor(m).attribute(m.random_tokens(9, seed=4), 11)
        against = res.against(4)
        assert against, "con vocab 128 almeno un componente deve scrivere contro"
        assert all(c.value < 0 for c in against)
        assert [c.value for c in against] == sorted(c.value for c in against)
        assert all(c.value > 0 for c in res.for_(3))

    def test_position_selects_the_right_token(self):
        m = _model()
        toks = m.random_tokens(9, seed=6)
        early = DirectLogitAttributor(m).attribute(toks, 12, position=0)
        late = DirectLogitAttributor(m).attribute(toks, 12, position=-1)
        assert early.total != pytest.approx(late.total)

    def test_narrate_and_summary(self):
        m = _model()
        res = DirectLogitAttributor(m).attribute(m.random_tokens(6, seed=7), 2)
        assert isinstance(res.narrate(), str)
        assert res.summary()["components"] == len(res.contributions)


class TestRidgeTunedLens:
    def test_fit_covers_all_layers(self):
        m = _model()
        lens = RidgeTunedLens(m).fit(_batch(m))
        assert lens.fitted
        assert len(lens.fits) == m.n_layers
        for l, fit in lens.fits.items():
            assert fit.W.shape == (m.d_model, m.d_model)
            assert fit.b.shape == (m.d_model,)

    def test_last_layer_is_almost_identity(self):
        m = _model()
        lens = RidgeTunedLens(m).fit(_batch(m))
        last = lens.fits[m.n_layers - 1]
        assert np.allclose(last.W, np.eye(m.d_model), atol=0.15)
        assert np.allclose(last.b, 0.0, atol=0.15)

    def test_translate_and_logits_shapes(self):
        m = _model()
        toks = m.random_tokens(7, seed=1)
        _, cache = m.run_with_cache(toks)
        lens = RidgeTunedLens(m).fit(_batch(m))
        resid = cache[HookPoint.resid_post(1)]
        assert lens.translate(resid, 1).shape == resid.shape
        assert lens.logits(resid, 1).shape == (toks.shape[0], m.vocab_size)
        assert np.allclose(lens.probs(resid, 1).sum(axis=-1), 1.0, atol=1e-4)

    def test_unfitted_layer_raises(self):
        m = _model()
        lens = RidgeTunedLens(m).fit(_batch(m), layers=[0])
        with pytest.raises(KeyError):
            lens.translate(np.zeros((1, m.d_model), dtype=np.float32), 2)

    def test_kl_profile_is_complete_and_improves_at_the_end(self):
        m = _model()
        lens = RidgeTunedLens(m).fit(_batch(m))
        prof = lens.kl_profile(m.random_tokens(8, seed=2))
        assert set(prof) == set(range(m.n_layers))
        last = prof[m.n_layers - 1]
        # all'ultimo layer tuned e raw devono coincidere (nessuna traduzione)
        assert last["tuned_kl"] == pytest.approx(last["raw_kl"], abs=1e-3)
        assert last["tuned_kl"] < 1e-2
        for row in prof.values():
            assert row["raw_kl"] >= 0.0 and row["tuned_kl"] >= 0.0
            assert row["tuned_top_match"] in (0.0, 1.0)

    def test_summary(self):
        m = _model()
        lens = RidgeTunedLens(m).fit(_batch(m))
        s = lens.summary()
        assert s["layers_fitted"] == m.n_layers
        assert s["rows"] > 0
