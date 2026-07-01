"""Tests for onepai.analysis."""
import os
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class TestCalculateVoidEntropy:
    def test_uniform_4(self):
        from onepai.analysis.silence_metrics import calculate_void_entropy
        assert calculate_void_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0, abs=1e-6)

    def test_certain(self):
        from onepai.analysis.silence_metrics import calculate_void_entropy
        assert calculate_void_entropy([1.0]) == pytest.approx(0.0)

    def test_empty(self):
        from onepai.analysis.silence_metrics import calculate_void_entropy
        assert calculate_void_entropy([]) == pytest.approx(0.0)

    def test_zero_prob_skipped(self):
        from onepai.analysis.silence_metrics import calculate_void_entropy
        assert calculate_void_entropy([0.0, 1.0]) == pytest.approx(0.0)


class TestCalculateSilenceEntropy:
    def _frag(self, pats):
        return {"silence_patterns": pats}

    def test_empty(self):
        from onepai.analysis.silence_metrics import calculate_silence_entropy
        assert calculate_silence_entropy([]) == 0.0

    def test_no_key(self):
        from onepai.analysis.silence_metrics import calculate_silence_entropy
        assert calculate_silence_entropy([{"other": "x"}]) == 0.0

    def test_single_pattern_zero_entropy(self):
        from onepai.analysis.silence_metrics import calculate_silence_entropy
        frags = [self._frag(["A"]), self._frag(["A"])]
        assert calculate_silence_entropy(frags) == pytest.approx(0.0)

    def test_two_equal_patterns_one_bit(self):
        from onepai.analysis.silence_metrics import calculate_silence_entropy
        frags = [self._frag(["A"]), self._frag(["B"])]
        assert calculate_silence_entropy(frags) == pytest.approx(1.0, abs=1e-6)


class TestCalculateSilenceRatio:
    def test_empty(self):
        from onepai.analysis.silence_metrics import calculate_silence_ratio
        assert calculate_silence_ratio([], "n1") == 0.0

    def test_returns_float_in_range(self):
        from onepai.analysis.silence_metrics import calculate_silence_ratio
        r = calculate_silence_ratio([{"neuron": "n1"}], "n1")
        assert 0.0 <= r <= 1.0


class TestMeasureVoidDepth:
    def test_empty(self):
        from onepai.analysis.silence_metrics import measure_void_depth
        assert measure_void_depth([], "l0") == 0.0

    def test_no_matching_layer(self):
        from onepai.analysis.silence_metrics import measure_void_depth
        frags = [{"layer_voids": {"other": {"depth_score": 0.8}}}]
        assert measure_void_depth(frags, "l0") == 0.0

    def test_average(self):
        from onepai.analysis.silence_metrics import measure_void_depth
        frags = [
            {"layer_voids": {"l1": {"depth_score": 0.4}}},
            {"layer_voids": {"l1": {"depth_score": 0.6}}},
        ]
        assert measure_void_depth(frags, "l1") == pytest.approx(0.5)


class TestAnalyzeSuppressionPatterns:
    def test_empty(self):
        from onepai.analysis.silence_metrics import analyze_suppression_patterns
        r = analyze_suppression_patterns([])
        assert r["total_suppressions"] == 0
        assert r["suppression_rate"] == 0.0

    def test_counts(self):
        from onepai.analysis.silence_metrics import analyze_suppression_patterns
        frags = [
            {"suppressed_thoughts": [{"type": "A"}, {"type": "A"}], "total_thoughts": 10},
            {"suppressed_thoughts": [{"type": "B"}], "total_thoughts": 5},
        ]
        r = analyze_suppression_patterns(frags)
        assert r["total_suppressions"] == 3
        assert r["suppression_rate"] == pytest.approx(3 / 15)


class TestFindRecurrentAbsences:
    def test_returns_dict(self):
        from onepai.analysis.absence_patterns import find_recurrent_absences
        assert isinstance(find_recurrent_absences([]), dict)


class TestDetectConflictingThoughts:
    def test_empty(self):
        from onepai.analysis.cognitive_dissonance import detect_conflicting_thoughts
        assert detect_conflicting_thoughts([]) == []

    def test_list_returned(self):
        from onepai.analysis.cognitive_dissonance import detect_conflicting_thoughts
        assert isinstance(detect_conflicting_thoughts([{"t": "x"}]), list)


class TestMeasureInternalTension:
    def test_returns_dict(self):
        from onepai.analysis.cognitive_dissonance import measure_internal_tension
        result = measure_internal_tension([])
        assert isinstance(result, dict)
        assert result.get("overall_tension", 0) >= 0

    def test_with_fragments(self):
        from onepai.analysis.cognitive_dissonance import measure_internal_tension
        frags = [{"thoughts": ["go left", "go right"], "total_thoughts": 2}]
        result = measure_internal_tension(frags)
        assert isinstance(result, dict)


class TestDetectDreamSequences:
    def test_empty(self):
        from onepai.analysis.dream_logic import detect_dream_sequences
        assert detect_dream_sequences([]) == []

    def test_list_returned(self):
        from onepai.analysis.dream_logic import detect_dream_sequences
        assert isinstance(detect_dream_sequences([{"thoughts": ["a", "b"]}]), list)


class TestAnalyzeNonLinearPatterns:
    def test_returns_dict(self):
        from onepai.analysis.dream_logic import analyze_non_linear_patterns
        assert isinstance(analyze_non_linear_patterns([]), dict)
