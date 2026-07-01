"""Tests for onepai.sae — Sparse Autoencoder e raccolta attivazioni."""
import numpy as np
import pytest

from onepai.sae import SparseAutoencoder, FeatureReport, collect_activations


@pytest.fixture(scope="module")
def sparse_data():
    """Dati sintetici sparsi: 8 direzioni vere in R^16, 1-3 attive per campione."""
    rng = np.random.RandomState(0)
    d_in, n_true, N = 16, 8, 1200
    D = rng.randn(n_true, d_in).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = np.zeros((N, d_in), dtype=np.float32)
    for i in range(N):
        k = rng.randint(1, 4)
        active = rng.choice(n_true, size=k, replace=False)
        coeffs = rng.uniform(0.5, 2.0, size=k)
        X[i] = (coeffs[:, None] * D[active]).sum(0) + rng.randn(d_in).astype(np.float32) * 0.05
    return X


class TestSparseAutoencoder:
    def test_shapes(self):
        sae = SparseAutoencoder(d_in=16, d_hidden=32)
        x = np.random.randn(10, 16).astype(np.float32)
        f = sae.encode(x)
        assert f.shape == (10, 32)
        assert sae.decode(f).shape == (10, 16)
        assert sae.reconstruct(x).shape == (10, 16)

    def test_encode_is_nonnegative(self):
        sae = SparseAutoencoder(d_in=8, d_hidden=16)
        x = np.random.randn(20, 8).astype(np.float32)
        assert (sae.encode(x) >= 0).all()       # ReLU

    def test_training_reduces_loss(self, sparse_data):
        sae = SparseAutoencoder(d_in=16, d_hidden=32, l1_coef=0.05, seed=1)
        sae.fit(sparse_data, epochs=20, batch_size=128, lr=2e-3)
        assert len(sae.history) == 20
        assert sae.history[-1] < sae.history[0]   # la loss scende

    def test_variance_explained_high(self, sparse_data):
        sae = SparseAutoencoder(d_in=16, d_hidden=32, l1_coef=0.05, seed=1)
        sae.fit(sparse_data, epochs=30, batch_size=128, lr=2e-3)
        assert sae.variance_explained(sparse_data) > 0.8

    def test_is_sparser_than_dense(self, sparse_data):
        sae = SparseAutoencoder(d_in=16, d_hidden=32, l1_coef=0.05, seed=1)
        sae.fit(sparse_data, epochs=30, batch_size=128, lr=2e-3)
        rep = sae.analyze(sparse_data)
        assert rep.mean_l0 < sae.d_hidden        # non tutte le feature attive

    def test_high_l1_creates_dead_features(self, sparse_data):
        sae = SparseAutoencoder(d_in=16, d_hidden=32, l1_coef=0.3, seed=1)
        sae.fit(sparse_data, epochs=30, batch_size=128, lr=2e-3)
        rep = sae.analyze(sparse_data)
        assert rep.n_dead > 0                     # l'assenza nello spazio delle feature

    def test_analyze_report(self, sparse_data):
        sae = SparseAutoencoder(d_in=16, d_hidden=24, l1_coef=0.05, seed=2)
        sae.fit(sparse_data, epochs=15, batch_size=128)
        rep = sae.analyze(sparse_data)
        assert isinstance(rep, FeatureReport)
        s = rep.summary()
        assert s["features"] == 24
        assert s["live"] + s["dead"] == 24
        assert 0.0 <= s["variance_explained"] <= 1.0
        assert len(rep.frequencies) == 24

    def test_feature_direction_shape(self):
        sae = SparseAutoencoder(d_in=12, d_hidden=20)
        assert sae.feature_direction(3).shape == (12,)


class TestCollectActivations:
    def test_collect_from_tiny_transformer(self):
        from onepai.model import TinyTransformer, TinyConfig, HookPoint
        m = TinyTransformer(TinyConfig(n_layers=2, n_heads=4, d_model=32,
                                       d_mlp=64, vocab_size=128, seed=0))
        batch = [m.random_tokens(10, seed=k) for k in range(5)]
        acts = collect_activations(m, batch, HookPoint.resid_post(1))
        assert acts.shape == (5 * 10, 32)       # 5 seq x 10 pos, d_model=32

    def test_collect_respects_max_tokens(self):
        from onepai.model import TinyTransformer, TinyConfig, HookPoint
        m = TinyTransformer(TinyConfig(n_layers=1, n_heads=2, d_model=16,
                                       d_mlp=32, vocab_size=64, seed=0))
        batch = [m.random_tokens(8, seed=k) for k in range(10)]
        acts = collect_activations(m, batch, HookPoint.mlp_post(0), max_tokens=20)
        assert acts.shape[0] == 20

    def test_end_to_end_sae_on_model_activations(self):
        """Pipeline reale: attivazioni di un modello -> SAE addestrata."""
        from onepai.model import TinyTransformer, TinyConfig, HookPoint
        m = TinyTransformer(TinyConfig(n_layers=2, n_heads=4, d_model=32,
                                       d_mlp=64, vocab_size=128, seed=3))
        batch = [m.random_tokens(12, seed=k) for k in range(20)]
        acts = collect_activations(m, batch, HookPoint.resid_post(1))
        sae = SparseAutoencoder(d_in=32, d_hidden=64, l1_coef=0.03, seed=0)
        sae.fit(acts, epochs=15, batch_size=64)
        rep = sae.analyze(acts)
        assert rep.n_features == 64
        assert rep.variance_explained > 0.3      # impara qualcosa di reale


class TestFeatureSuppressionTracer:
    """Il ponte SAE <-> soppressione (su TinyTransformer, veloce)."""

    def _setup(self, layer=1):
        from onepai.model import TinyTransformer, TinyConfig, HookPoint
        from onepai.sae import FeatureSuppressionTracer
        m = TinyTransformer(TinyConfig(n_layers=3, n_heads=4, d_model=32,
                                       d_mlp=64, vocab_size=128, seed=2))
        batch = [m.random_tokens(12, seed=k) for k in range(25)]
        acts = collect_activations(m, batch, HookPoint.resid_post(layer))
        sae = SparseAutoencoder(d_in=32, d_hidden=64, l1_coef=0.02, seed=0)
        sae.fit(acts, epochs=20, batch_size=64)
        return m, sae, FeatureSuppressionTracer(m, sae, layer=layer)

    def test_feature_acts_all_layers_shape(self):
        m, sae, tracer = self._setup()
        toks = m.random_tokens(10, seed=99)
        F = tracer.feature_acts_all_layers(toks)
        assert F.shape == (m.n_layers, sae.d_hidden)

    def test_interpret_feature(self):
        m, sae, tracer = self._setup()
        meaning = tracer.interpret_feature(0, top_k=5)
        assert len(meaning) == 5
        # TinyTransformer non ha decode -> ritorna token id (int)
        assert all(isinstance(t, int) for t in meaning)

    def test_trace_feature_structure(self):
        from onepai.sae import FeatureTrajectory
        m, sae, tracer = self._setup()
        toks = m.random_tokens(10, seed=7)
        ft = tracer.trace_feature(toks, 3, find_suppressors=False)
        assert isinstance(ft, FeatureTrajectory)
        assert len(ft.trajectory) == m.n_layers
        assert ft.crush == pytest.approx(ft.peak_act - ft.final_act, abs=1e-5)
        assert isinstance(ft.narrate(), str)

    def test_trace_feature_with_suppressors(self):
        m, sae, tracer = self._setup()
        toks = m.random_tokens(10, seed=8)
        # forza una feature con un po' di crush cercandone una
        F = tracer.feature_acts_all_layers(toks)
        crush = F.max(0) - F[-1]
        fidx = int(crush.argmax())
        ft = tracer.trace_feature(toks, fidx, find_suppressors=True, max_suppressors=4)
        rest = [s.restoration for s in ft.suppressors]
        assert rest == sorted(rest, reverse=True)   # ordinati per ripristino

    def test_find_suppressed_features_returns_list(self):
        m, sae, tracer = self._setup()
        toks = m.random_tokens(12, seed=5)
        out = tracer.find_suppressed_features(toks, top_k=3, crush_min=0.01,
                                              min_peak=0.0, find_suppressors=False)
        assert isinstance(out, list)
