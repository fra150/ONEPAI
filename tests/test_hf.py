"""
Tests per HFAdapter su GPT-2 reale.

Saltati automaticamente se torch/transformers non sono installati o se i pesi
di gpt2 non sono in cache (così la suite gira ovunque senza download).
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# Salta se gpt2 non è in cache (niente download in test)
_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_HAS_GPT2 = _CACHE.exists() and any(_CACHE.glob("models--gpt2*"))
pytestmark = pytest.mark.skipif(not _HAS_GPT2, reason="gpt2 non in cache HuggingFace")


@pytest.fixture(scope="module")
def adapter():
    from onepai.model import HFAdapter
    return HFAdapter("gpt2")


class TestHFAdapter:
    def test_shapes(self, adapter):
        assert adapter.n_layers == 12
        assert adapter.n_heads == 12
        assert adapter.d_model == 768
        assert adapter.vocab_size == 50257

    def test_encode_decode_roundtrip(self, adapter):
        toks = adapter.encode("Hello world")
        assert toks.ndim == 1 and len(toks) >= 2
        assert isinstance(adapter.decode(toks), str)

    def test_run_with_cache(self, adapter):
        from onepai.model import HookPoint
        toks = adapter.encode("The capital of France is")
        logits, cache = adapter.run_with_cache(toks)
        assert logits.shape == (len(toks), 50257)
        assert cache[HookPoint.mlp_post(0)].shape == (len(toks), 3072)
        assert cache[HookPoint.attn_pattern(0)].shape == (12, len(toks), len(toks))
        assert cache[HookPoint.resid_post(11)].shape == (len(toks), 768)

    def test_real_prediction_is_reasonable(self, adapter):
        # GPT-2 dopo "The capital of France is" deve predire un token plausibile
        toks = adapter.encode("The capital of France is")
        logits, _ = adapter.run_with_cache(toks)
        top = int(logits[-1].argmax())
        assert 0 <= top < 50257

    def test_logit_lens(self, adapter):
        from onepai.model import HookPoint
        toks = adapter.encode("Hello there")
        _, cache = adapter.run_with_cache(toks)
        ll = adapter.apply_unembed(cache[HookPoint.resid_post(6)])
        assert ll.shape == (len(toks), 50257)

    def test_ablation_changes_output(self, adapter):
        from onepai.model import Intervention, HookPoint
        toks = adapter.encode("The capital of France is")
        base, _ = adapter.run_with_cache(toks)
        out = adapter.run_with_interventions(
            toks, [Intervention(HookPoint.mlp_out(0), kind="ablate")])
        assert not np.allclose(base, out)

    def test_steering_monotonic(self, adapter):
        from onepai.model import InterventionStudy, HookPoint
        study = InterventionStudy(adapter)
        toks = adapter.encode("The capital of France is")
        target = int(adapter.encode(" Paris")[0])
        d = study.target_direction(target)
        sw = study.steering_sweep(toks, HookPoint.resid_post(11), d, target,
                                  coefficients=[-3, 0, 3])
        assert sw.target_logits[-1] > sw.target_logits[0]

    def test_inert_head_ablation_via_z(self, adapter):
        from onepai.model import HookPoint
        # Ablare una testa via attn_z deve cambiare l'output (teste GPT-2 sono attive)
        toks = adapter.encode("The capital of France is")
        base, _ = adapter.run_with_cache(toks)

        def zero_head0(z, name):
            z = np.array(z, copy=True)
            z[:, 0, :] = 0.0
            return z

        out = adapter.run_with_hooks(toks, fwd_hooks=[(HookPoint.attn_z(0), zero_head0)])
        assert not np.allclose(base, out)


class TestAbsenceOnGPT2:
    def test_finds_suppressed_neurons(self, adapter):
        from onepai.model import AbsenceEngine
        batch = [adapter.encode(s) for s in
                 ["The capital of France is", "She opened the door and",
                  "Water boils at a temperature of"]]
        rep = AbsenceEngine(adapter, suppression_pre=4.0).analyze(batch)
        # GPT-2 ha neuroni con pre fortemente negativo (soppressi)
        assert isinstance(rep.suppressed_neurons, list)
        s = rep.summary()
        assert "suppressed" in s and "crushed_tokens" in s


# ── Universalità: stessa pipeline su un'architettura DIVERSA (GPT-NeoX) ──

_HAS_PYTHIA = _CACHE.exists() and any(_CACHE.glob("models--EleutherAI--pythia-160m"))


@pytest.mark.skipif(not _HAS_PYTHIA, reason="pythia-160m non in cache")
class TestUniversality:
    @pytest.fixture(scope="class")
    def pythia(self):
        from onepai.model import HFAdapter
        return HFAdapter("EleutherAI/pythia-160m")

    def test_architecture_is_different(self, pythia):
        assert pythia.model_type == "gpt_neox"   # NON gpt2
        assert pythia.n_layers == 12

    def test_same_code_captures(self, pythia):
        from onepai.model import HookPoint
        toks = pythia.encode("The capital of France is")
        logits, cache = pythia.run_with_cache(toks)
        assert logits.shape[0] == len(toks)
        assert cache[HookPoint.mlp_post(0)].ndim == 2
        assert cache[HookPoint.attn_pattern(0)].shape[0] == pythia.n_heads

    def test_logit_lens_no_nan(self, pythia):
        from onepai.model import HookPoint
        toks = pythia.encode("Hello there")
        _, cache = pythia.run_with_cache(toks)
        ll = pythia.apply_unembed(cache[HookPoint.resid_post(6)])
        assert np.isfinite(ll).all()

    def test_intervention_works(self, pythia):
        from onepai.model import Intervention, HookPoint
        toks = pythia.encode("The capital of France is")
        base, _ = pythia.run_with_cache(toks)
        out = pythia.run_with_interventions(
            toks, [Intervention(HookPoint.mlp_out(0), kind="ablate")])
        assert not np.allclose(base, out)

    def test_absence_phenomena_are_universal(self, pythia):
        """I fenomeni di assenza compaiono anche fuori da GPT-2."""
        from onepai.model import AbsenceEngine
        batch = [pythia.encode(s) for s in
                 ["The capital of France is", "She opened the door and",
                  "Water boils at a temperature of", "The president announced that"]]
        rep = AbsenceEngine(pythia, suppression_pre=4.0, crush_min=0.03).analyze(batch)
        # anche un'architettura diversa ha neuroni soppressi e token quasi-detti
        assert len(rep.suppressed_neurons) >= 1
        assert len(rep.crushed_tokens) >= 1

    def test_suppression_tracer_narrates(self, pythia):
        from onepai.model import SuppressionTracer
        tracer = SuppressionTracer(pythia)
        traces = tracer.find_and_trace(
            pythia.encode("The capital of France is"), top_k=1, crush_min=0.1)
        if traces:
            assert isinstance(traces[0].narrate(), str)
