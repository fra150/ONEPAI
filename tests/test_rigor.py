"""Tests per i moduli di rigore: FeatureInterpreter e SuppressionValidator."""
import numpy as np
import pytest

from onepai.model import TinyTransformer, TinyConfig, HookPoint
from onepai.sae import (
    SparseAutoencoder, collect_activations, FeatureInterpreter,
    DualInterpretation, FeatureExample,
)


def _model(**kw):
    return TinyTransformer(TinyConfig(n_layers=3, n_heads=4, d_model=32,
                                      d_mlp=64, vocab_size=128, seed=2, **kw))


def _sae_for(m, layer=1, batch=None):
    batch = batch or [m.random_tokens(12, seed=k) for k in range(20)]
    acts = collect_activations(m, batch, HookPoint.resid_post(layer))
    sae = SparseAutoencoder(d_in=m.d_model, d_hidden=48, l1_coef=0.02, seed=0)
    sae.fit(acts, epochs=15, batch_size=64)
    return sae, batch


# ── FeatureInterpreter (max-activating examples) ──────────────────────

class TestFeatureInterpreter:
    @pytest.fixture(scope="class")
    def setup(self):
        m = _model()
        sae, batch = _sae_for(m, layer=1)
        interp = FeatureInterpreter(m, sae, layer=1, k=5)
        interp.index_corpus(batch)
        return m, sae, interp

    def test_index_counts_positions(self, setup):
        m, sae, interp = setup
        # 20 sequenze x 12 posizioni, skip_first=1 → 220
        assert interp._n_positions == 20 * 11

    def test_top_examples_sorted_desc(self, setup):
        m, sae, interp = setup
        # trova una feature viva
        live = int(np.argmax(interp._fire_counts))
        exs = interp.top_examples(live)
        assert len(exs) >= 1
        acts = [e.activation for e in exs]
        assert acts == sorted(acts, reverse=True)

    def test_examples_reference_valid_positions(self, setup):
        m, sae, interp = setup
        live = int(np.argmax(interp._fire_counts))
        for e in interp.top_examples(live):
            assert 0 <= e.seq_idx < 20
            assert 0 <= e.position < 12
            assert "[[" in e.context      # token evidenziato

    def test_topk_matches_bruteforce(self, setup):
        """Il merge incrementale deve dare lo stesso max del calcolo diretto."""
        m, sae, interp = setup
        live = int(np.argmax(interp._fire_counts))
        best_direct = -1.0
        for tokens in interp._sequences:
            _, cache = m.run_with_cache(tokens)
            F = sae.encode(cache[HookPoint.resid_post(1)])
            F = F.copy(); F[:1] = 0.0     # stesso skip_first
            best_direct = max(best_direct, float(F[:, live].max()))
        assert interp._top_vals[live, 0] == pytest.approx(best_direct, abs=1e-4)

    def test_firing_rate_range(self, setup):
        m, sae, interp = setup
        for f in range(0, 48, 7):
            assert 0.0 <= interp.firing_rate(f) <= 1.0

    def test_dual_interpretation(self, setup):
        m, sae, interp = setup
        live = int(np.argmax(interp._fire_counts))
        dual = interp.dual_interpretation(live)
        assert isinstance(dual, DualInterpretation)
        assert len(dual.unembed_tokens) > 0
        assert isinstance(dual.narrate(), str)

    def test_dead_feature_has_no_examples(self, setup):
        m, sae, interp = setup
        dead_candidates = np.where(interp._fire_counts == 0)[0]
        if len(dead_candidates) > 0:
            assert interp.top_examples(int(dead_candidates[0])) == []


# ── SuppressionValidator ──────────────────────────────────────────────

class _EncodableTiny(TinyTransformer):
    """TinyTransformer con encode/decode fittizi per testare il validator."""

    def encode(self, text: str) -> np.ndarray:
        words = text.split()
        return np.array([(hash(w) % self.cfg.vocab_size) for w in words],
                        dtype=np.int64)

    def decode(self, ids) -> str:
        return " ".join(f"<{int(i)}>" for i in ids)


class TestSuppressionValidator:
    @pytest.fixture(scope="class")
    def model(self):
        return _EncodableTiny(TinyConfig(n_layers=3, n_heads=4, d_model=32,
                                         d_mlp=64, vocab_size=128, seed=4))

    def test_run_case_structure(self, model):
        from onepai.model import SuppressionValidator
        v = SuppressionValidator(model, crush_threshold=0.0)
        case = v.run_case("uno due tre quattro", "cinque")
        assert case.skipped is None
        assert case.target_token >= 0
        assert case.crush == pytest.approx(case.peak_prob - case.final_prob, abs=1e-6)

    def test_multitoken_target_skipped(self, model):
        from onepai.model import SuppressionValidator
        v = SuppressionValidator(model)
        case = v.run_case("uno due tre", "due parole")
        assert case.skipped is not None

    def test_validate_aggregates(self, model):
        from onepai.model import SuppressionValidator
        v = SuppressionValidator(model, crush_threshold=0.0)
        treatment = [(f"parola{i} test frase", "extra") for i in range(3)]
        control = [("altra frase di controllo", "extra")]
        res = v.validate(treatment, control)
        s = res.summary()
        assert s["treatment_cases"] == 3
        assert s["control_cases"] == 1
        assert 0.0 <= res.consistency_top1 <= 1.0
        assert isinstance(res.narrate(), str)

    def test_consistency_perfect_when_same_prompt(self, model):
        """Prompt identici → stesso soppressore → consistency 1.0 (se c'è crush)."""
        from onepai.model import SuppressionValidator
        v = SuppressionValidator(model, crush_threshold=0.0)
        treatment = [("stessa identica frase qui", "bersaglio")] * 3
        res = v.validate(treatment)
        valid = [c for c in res.treatment if not c.skipped and c.top_suppressor]
        if valid and res.modal_suppressor:
            assert res.consistency_top1 == pytest.approx(1.0)
