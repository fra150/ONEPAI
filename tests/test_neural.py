"""Tests for onepai.neural."""
import os
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class TestAbortionDetector:
    def test_instantiation(self):
        from onepai.neural.abortion_detector import AbortionDetector
        assert AbortionDetector() is not None

    def test_aborted_thought_fields(self):
        from datetime import datetime
        from onepai.neural.abortion_detector import AbortedThought
        at = AbortedThought(
            thought_id="t1",
            layer_id="layer3",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            activation_trace=[0.1, 0.2],
            decision_point=5,
            abortion_reason="low_confidence",
            context={},
        )
        assert at.thought_id == "t1"
        assert at.layer_id == "layer3"


class TestCircuitShadow:
    def test_instantiation(self):
        from onepai.neural.circuit_shadow import CircuitShadow
        cs = CircuitShadow(shadow_threshold=0.2)
        assert cs.shadow_threshold == pytest.approx(0.2)

    def test_identify_on_empty(self):
        from onepai.neural.circuit_shadow import CircuitShadow
        assert CircuitShadow().identify_circuit_shadows() == []

    def test_record_dataclass(self):
        from onepai.neural.circuit_shadow import CircuitShadowRecord
        r = CircuitShadowRecord(
            circuit_id="c1", layer_ids=["l1"], neuron_indices=[(0,)],
            activation_frequency=0.05, avg_activation_strength=0.3, context={},
        )
        assert r.circuit_id == "c1"


class TestWeightGhost:
    def test_instantiation(self):
        from onepai.neural.weight_ghost import WeightGhost
        assert WeightGhost(ghost_threshold=0.1).ghost_threshold == pytest.approx(0.1)

    def test_identify_on_empty(self):
        from onepai.neural.weight_ghost import WeightGhost
        assert WeightGhost().identify_ghost_weights() == []

    def test_record_dataclass(self):
        from onepai.neural.weight_ghost import WeightGhostRecord
        r = WeightGhostRecord(
            layer_id="l1", weight_indices=(0, 1), activation_count=2,
            total_opportunities=100, activation_ratio=0.02, magnitude=0.001, context={},
        )
        assert r.activation_ratio == pytest.approx(0.02)


class TestInvisibleNet:
    def test_instantiation(self):
        from onepai.neural.invisible_net import InvisibleNet
        arch = {"layers": [], "connections": []}
        assert InvisibleNet(model_architecture=arch) is not None

    def test_invisible_path_fields(self):
        from onepai.neural.invisible_net import InvisiblePath
        p = InvisiblePath(
            layer_id="layer3",
            neuron_indices=[0, 1, 2],
            potential_activation=0.02,
            context={},
        )
        assert p.layer_id == "layer3"
        assert p.potential_activation == pytest.approx(0.02)
