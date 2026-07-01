"""Tests for onepai.introspection and onepai.visualization."""
import pytest
import numpy as np


# ── AttentionTracer ───────────────────────────────────────────────────

class TestAttentionTracer:
    def test_record_and_summarize(self):
        from onepai.introspection import AttentionTracer
        t = AttentionTracer(ghost_threshold=0.1)
        weights = np.array([[0.9, 0.1], [0.3, 0.7]])
        t.record("layer.0", 0, weights)
        summaries = t.summarize()
        assert len(summaries) == 1
        assert summaries[0].max_weight_ever == pytest.approx(0.9, abs=1e-4)

    def test_ghost_head_detected(self):
        from onepai.introspection import AttentionTracer
        # Matrice 20x20 uniforme: dopo normalizzazione max_weight=0.05
        # ghost_score = 1 - 0.05/0.5 = 0.9 > 0.8
        t = AttentionTracer(ghost_threshold=0.5)
        tiny = np.full((20, 20), 0.01)
        t.record("layer.0", 1, tiny)
        ghosts = t.identify_ghost_heads()
        assert len(ghosts) == 1
        assert ghosts[0].ghost_score > 0.8

    def test_no_ghost_for_active_head(self):
        from onepai.introspection import AttentionTracer
        t = AttentionTracer(ghost_threshold=0.1)
        active = np.eye(4) * 0.95
        t.record("layer.0", 0, active)
        ghosts = t.identify_ghost_heads()
        assert len(ghosts) == 0

    def test_entropy_uniform_higher_than_peaked(self):
        from onepai.introspection import AttentionTracer
        t = AttentionTracer()
        n = 8
        uniform = np.full((n, n), 1.0 / n)
        peaked = np.zeros((n, n))
        peaked[:, 0] = 1.0
        t.record("layer.0", 0, uniform)
        t.record("layer.0", 1, peaked)
        ent_uniform = t.compute_head_entropy("layer.0", 0)
        ent_peaked = t.compute_head_entropy("layer.0", 1)
        assert ent_uniform > ent_peaked

    def test_build_influence_matrix_shape(self):
        from onepai.introspection import AttentionTracer
        t = AttentionTracer()
        w = np.random.rand(6, 6).astype(np.float32)
        w = w / w.sum(axis=-1, keepdims=True)
        t.record("l0", 0, w)
        mat = t.build_influence_matrix(seq_len=6)
        assert mat.shape == (6, 6)

    def test_attention_flow_scalar(self):
        from onepai.introspection import AttentionTracer
        t = AttentionTracer()
        w = np.array([[0.8, 0.2], [0.5, 0.5]])
        t.record("l0", 0, w)
        flow = t.get_attention_flow(0, 0)
        assert 0.0 <= flow <= 1.0


# ── LogitLens ─────────────────────────────────────────────────────────

class TestLogitLens:
    def test_record_logits_returns_prediction(self):
        from onepai.introspection import LogitLens
        lens = LogitLens(vocab_size=10, top_k=3)
        logits = np.array([1.0, 5.0, 2.0, 0.1, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0])
        pred = lens.record_logits("layer.0", position=0, logits=logits)
        assert pred.top_token_ids[0] == 1   # token 1 ha logit più alto
        assert pred.confidence > 0.3

    def test_confidence_trajectory_increasing(self):
        from onepai.introspection import LogitLens
        lens = LogitLens(vocab_size=10, top_k=3)
        # Tutti step=0: tutti appartengono allo stesso "forward pass"
        for i, conf_spike in enumerate([0.1, 0.3, 0.6, 0.9]):
            logits = np.zeros(10)
            logits[0] = conf_spike * 20   # token 0 sempre più probabile
            lens.record_logits(f"layer.{i}", position=0, logits=logits,
                               layer_idx=i, step=0)
        traj = lens.get_confidence_trajectory(position=0, step=0)
        confidences = [c for _, c in traj]
        assert len(confidences) == 4
        assert confidences[-1] > confidences[0]

    def test_find_emergence_layer(self):
        from onepai.introspection import LogitLens
        lens = LogitLens(vocab_size=20, top_k=5)
        # Token 7 non emerge nei primi layer — tutti step=0
        for i in range(3):
            logits = np.zeros(20)
            logits[0] = 10.0
            lens.record_logits(f"layer.{i}", position=0, logits=logits,
                               layer_idx=i, step=0)
        # Emerge al layer 3
        logits = np.zeros(20)
        logits[7] = 10.0
        lens.record_logits("layer.3", position=0, logits=logits, layer_idx=3, step=0)
        layer = lens.find_emergence_layer(token_id=7, position=0, min_prob=0.1)
        assert layer == "layer.3"

    def test_summary(self):
        from onepai.introspection import LogitLens
        lens = LogitLens(vocab_size=10)
        lens.record_logits("l0", 0, np.random.rand(10))
        s = lens.summary()
        assert s["layers"] == 1
        assert s["predictions"] == 1


# ── GradientFlowTracer ────────────────────────────────────────────────

class TestGradientFlowTracer:
    def test_record_and_dead_detection(self):
        from onepai.introspection import GradientFlowTracer
        t = GradientFlowTracer()
        tiny = np.full(20, 1e-10)
        for step in range(6):
            t.record("attn.0", "W_Q", tiny, step=step)
        dead = t.identify_dead_parameters(n_steps=5, threshold=1e-8)
        assert any(p == "W_Q" and l == "attn.0" for l, p, _ in dead)

    def test_large_gradient_not_dead(self):
        from onepai.introspection import GradientFlowTracer
        t = GradientFlowTracer()
        large = np.random.rand(50) * 2.0
        for s in range(6):
            t.record("mlp.0", "W_1", large, step=s)
        dead = t.identify_dead_parameters(n_steps=5)
        assert not any(l == "mlp.0" for l, _, _ in dead)

    def test_flow_report_structure(self):
        from onepai.introspection import GradientFlowTracer
        t = GradientFlowTracer()
        t.record("attn.0", "W", np.random.rand(10))
        t.record("mlp.0",  "W", np.random.rand(10))
        report = t.flow_report()
        assert "layers" in report
        assert "attn.0" in report["layers"]

    def test_magnitude_trajectory(self):
        from onepai.introspection import GradientFlowTracer
        t = GradientFlowTracer()
        for s in range(4):
            t.record("l0", "w", np.random.rand(5) * (s + 1), step=s)
        traj = t.get_magnitude_trajectory("l0")
        assert len(traj) == 4
        assert all(isinstance(v, float) for _, v in traj)


# ── ActivationPatcher ─────────────────────────────────────────────────

class TestActivationPatcher:
    def test_patch_returns_result(self):
        from onepai.introspection import ActivationPatcher
        p = ActivationPatcher()
        p.store_run("clean",     {"layer.0": np.ones(8)})
        p.store_run("corrupted", {"layer.0": np.zeros(8)})
        result = p.patch_component("layer.0")
        assert result.component_id == "layer.0"
        assert result.effect_size >= 0.0

    def test_full_causal_trace_normalized(self):
        from onepai.introspection import ActivationPatcher
        p = ActivationPatcher()
        clean = {"l0": np.ones(4), "l1": np.ones(4) * 2}
        corr = {"l0": np.zeros(4), "l1": np.zeros(4)}
        p.store_run("clean", clean)
        p.store_run("corrupted", corr)
        results = p.run_full_causal_trace()
        max_imp = max(r.causal_importance for r in results)
        assert max_imp == pytest.approx(1.0, abs=1e-4)

    def test_critical_path_not_empty(self):
        from onepai.introspection import ActivationPatcher
        p = ActivationPatcher()
        p.store_run("clean",     {"l0": np.array([1.0, 1.0, 1.0])})
        p.store_run("corrupted", {"l0": np.array([0.0, 0.0, 0.0])})
        path = p.find_critical_path(min_importance=0.0)
        assert len(path.components) >= 1


# ── TokenInfluencer ───────────────────────────────────────────────────

class TestTokenInfluencer:
    def test_attention_rollout_sums_to_one(self):
        from onepai.introspection import TokenInfluencer
        inf = TokenInfluencer()
        attn = [np.array([[0.7, 0.3], [0.4, 0.6]]),
                np.array([[0.6, 0.4], [0.2, 0.8]])]
        imap = inf.attention_rollout(attn, target_pos=0)
        total = sum(imap.source_scores.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_gradient_input_shape(self):
        from onepai.introspection import TokenInfluencer
        inf = TokenInfluencer()
        emb = np.random.rand(5, 8).astype(np.float32)
        grd = np.random.rand(5, 8).astype(np.float32)
        imap = inf.gradient_input(emb, grd, target_pos=2)
        assert len(imap.source_scores) == 5

    def test_full_influence_matrix(self):
        from onepai.introspection import TokenInfluencer
        inf = TokenInfluencer()
        attn = [np.random.rand(4, 4).astype(np.float32)]
        # Normalizza
        attn[0] = attn[0] / attn[0].sum(axis=-1, keepdims=True)
        mat = inf.build_full_influence_matrix(attn)
        assert mat.shape == (4, 4)


# ── ResidualStreamAnalyzer ────────────────────────────────────────────

class TestResidualStreamAnalyzer:
    def test_record_and_decompose(self):
        from onepai.introspection import ResidualStreamAnalyzer
        a = ResidualStreamAnalyzer(model_dim=4)
        a.record_delta("attention", 0, "attn.0", position=0, delta=np.ones(4))
        a.record_delta("mlp",       0, "mlp.0",  position=0, delta=np.ones(4) * 2)
        comps = a.decompose_at_position(0)
        assert len(comps) == 2

    def test_layer_type_breakdown(self):
        from onepai.introspection import ResidualStreamAnalyzer
        a = ResidualStreamAnalyzer()
        a.record_delta("attention", 0, "attn.0", 0, np.ones(8))
        a.record_delta("mlp",       0, "mlp.0",  0, np.ones(8) * 3)
        bd = a.layer_type_breakdown(0)
        assert "attention" in bd
        assert "mlp" in bd
        assert bd["mlp"]["total_magnitude"] > bd["attention"]["total_magnitude"]

    def test_interference_pattern_antiparallel(self):
        from onepai.introspection import ResidualStreamAnalyzer
        a = ResidualStreamAnalyzer()
        a.record_delta("attn", 0, "attn.0", 0, np.array([1.0, 0.0, 0.0, 0.0]))
        a.record_delta("attn", 1, "attn.1", 0, np.array([-1.0, 0.0, 0.0, 0.0]))
        interferences = a.detect_interference_patterns(0, cosine_threshold=-0.5)
        assert len(interferences) >= 1
        assert interferences[0][2] < -0.5

    def test_stream_trajectory_grows(self):
        from onepai.introspection import ResidualStreamAnalyzer
        a = ResidualStreamAnalyzer()
        for i in range(4):
            a.record_delta("mlp", i, f"mlp.{i}", 0, np.ones(4) * (i + 1), step=i)
        traj = a.stream_trajectory(0)
        norms = [n for _, n in traj]
        assert norms[-1] > norms[0]


# ── FeatureAttributor ─────────────────────────────────────────────────

class TestFeatureAttributor:
    def test_gradient_input(self):
        from onepai.introspection import FeatureAttributor
        fa = FeatureAttributor()
        g = np.array([0.5, 0.0, -0.3, 1.2])
        x = np.array([1.0, 2.0,  3.0, 0.5])
        attr = fa.gradient_input(g, x)
        assert attr.scores.shape == (4,)
        assert attr.method == "gradient_input"

    def test_perturbation_attribution(self):
        from onepai.introspection import FeatureAttributor
        fa = FeatureAttributor()
        def model(x): return float(x[0] * 2 + x[1])   # feature 0 più importante
        x = np.array([3.0, 1.0, 0.5, 0.1])
        attr = fa.perturbation_attribution(model, x, normalize=True)
        assert attr.scores[0] >= attr.scores[1]    # feature 0 più influente

    def test_integrated_gradients_numerical(self):
        from onepai.introspection import FeatureAttributor
        fa = FeatureAttributor()
        def model(x): return float(np.sum(x ** 2))
        x = np.array([1.0, 2.0])
        bl = np.zeros(2)
        attr = fa.integrated_gradients(model, x, baseline=bl, n_steps=20)
        assert attr.scores.shape == (2,)
        # Per f(x)=x^2 con baseline=0, IG = x^2 (il contributo è proporzionale)
        assert abs(attr.scores[1]) > abs(attr.scores[0])

    def test_top_features(self):
        from onepai.introspection import FeatureAttributor
        fa = FeatureAttributor()
        attr = fa.gradient_input(np.array([0.1, 5.0, 0.2]), np.array([1.0, 1.0, 1.0]))
        top = attr.top(1)
        assert top[0][0] == 1   # feature 1 ha il valore più alto

    def test_consensus_attribution(self):
        from onepai.introspection import FeatureAttributor
        fa = FeatureAttributor()
        fa.gradient_input(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        fa.gradient_input(np.array([0.5, 1.5, 2.5]), np.array([1.0, 1.0, 1.0]))
        consensus = fa.consensus_attribution()
        assert consensus is not None
        assert consensus.method == "consensus"


# ── Visualization — heatmap ───────────────────────────────────────────

class TestHeatmaps:
    def test_activation_heatmap_returns_svg(self):
        from onepai.visualization import ActivationHeatmap
        hm = ActivationHeatmap()
        mat = np.random.rand(3, 4).astype(np.float32)
        svg = hm.render(mat, title="Test")
        assert "<svg" in svg
        assert "Test" in svg

    def test_silence_heatmap_returns_svg(self):
        from onepai.visualization import SilenceHeatmap
        shm = SilenceHeatmap()
        data = {"layer.0": {0: 0.9, 1: 0.1}, "layer.1": {0: 0.5}}
        svg = shm.render(data)
        assert "<svg" in svg

    def test_heatmap_1d_input(self):
        from onepai.visualization import ActivationHeatmap
        hm = ActivationHeatmap()
        vec = np.array([0.1, 0.5, 0.9])
        svg = hm.render(vec)
        assert "<svg" in svg


# ── Visualization — graph_viz ─────────────────────────────────────────

class TestNeuralGraphRenderer:
    def test_empty_graph(self):
        from onepai.visualization import NeuralGraphRenderer
        r = NeuralGraphRenderer()
        svg = r.render()
        assert "Empty graph" in svg

    def test_graph_with_nodes_and_edges(self):
        from onepai.visualization import NeuralGraphRenderer
        r = NeuralGraphRenderer()
        r.add_node("emb", "Embedding", layer_idx=0, node_type="active")
        r.add_node("attn", "Attention", layer_idx=1, node_type="ghost")
        r.add_edge("emb", "attn", weight=0.5)
        svg = r.render(title="Test Graph")
        assert "<svg" in svg
        assert "Test Graph" in svg
        assert "Embedding" in svg or "emb" in svg

    def test_node_types_in_svg(self):
        from onepai.visualization import NeuralGraphRenderer
        r = NeuralGraphRenderer()
        r.add_node("dead_one", "Dead", layer_idx=0, node_type="dead")
        svg = r.render()
        assert "#e04040" in svg   # colore dead


# ── Visualization — timeline ──────────────────────────────────────────

class TestTimelines:
    def test_activation_timeline_svg(self):
        from onepai.visualization import ActivationTimeline
        t = ActivationTimeline()
        data = {"layer.0": [0.1, 0.3, 0.5, 0.2], "layer.1": [0.8, 0.6, 0.4, 0.3]}
        svg = t.render(data, title="Activations")
        assert "<svg" in svg
        assert "Activations" in svg

    def test_confidence_timeline_svg(self):
        from onepai.visualization import ConfidenceTimeline
        ct = ConfidenceTimeline()
        traj = {"pos0": [("l0", 0.2), ("l1", 0.5), ("l2", 0.8)]}
        svg = ct.render(traj)
        assert "<svg" in svg


# ── Visualization — html_report ───────────────────────────────────────

class TestHTMLReport:
    def test_basic_report(self):
        from onepai.visualization import HTMLReport
        r = HTMLReport(title="Test Report")
        r.add_summary({"Layers": 12, "Dead": 3})
        r.add_section("Test Section", "<svg><text>hello</text></svg>", icon="⚡")
        html = r.to_html()
        assert "Test Report" in html
        assert "Test Section" in html
        assert "Layers" in html
        assert "<!DOCTYPE html>" in html

    def test_report_saves_to_file(self, tmp_path):
        from onepai.visualization import HTMLReport
        r = HTMLReport()
        r.add_summary({"x": 1})
        p = r.save(tmp_path / "report.html")
        assert p.exists()
        assert p.stat().st_size > 1000

    def test_report_with_table(self):
        from onepai.visualization import HTMLReport
        r = HTMLReport()
        r.add_table("Dead Params", [{"Layer": "l0", "Param": "W", "Mag": "1e-9"}])
        html = r.to_html()
        assert "<table" in html
        assert "Dead Params" in html


# ── Utils ─────────────────────────────────────────────────────────────

class TestInvisibleMath:
    def test_entropy_uniform(self):
        from onepai.utils import entropy
        p = [0.25, 0.25, 0.25, 0.25]
        assert entropy(p) == pytest.approx(2.0, abs=0.001)

    def test_entropy_certain(self):
        from onepai.utils import entropy
        assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_parallel(self):
        from onepai.utils import cosine_similarity
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        from onepai.utils import cosine_similarity
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_silence_score(self):
        from onepai.utils import silence_score
        vals = [0.0, 0.0, 1.0, 0.0]
        assert silence_score(vals, threshold=1e-6) == pytest.approx(0.75)

    def test_gini_equal(self):
        from onepai.utils import gini_coefficient
        assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=0.01)

    def test_gini_one_dominant(self):
        from onepai.utils import gini_coefficient
        assert gini_coefficient([0.0, 0.0, 100.0]) > 0.6

    def test_normalize_l2(self):
        from onepai.utils import normalize
        v = normalize([3.0, 4.0], method="l2")
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_kl_divergence_identical(self):
        from onepai.utils import kl_divergence
        p = [0.5, 0.5]
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)


class TestShadowUtils:
    def test_compute_shadow_matrix(self):
        from onepai.utils import compute_shadow_matrix
        acts = np.array([[0.0, 1.0], [0.0, 0.0]])
        mat = compute_shadow_matrix(acts, threshold=1e-6)
        assert mat[0, 0] == True
        assert mat[0, 1] == False

    def test_activation_to_shadow_map(self):
        from onepai.utils import activation_to_shadow_map
        m = activation_to_shadow_map({"l0": [0.0, 0.0, 1.0]}, threshold=1e-6)
        assert m["l0"] == pytest.approx(2 / 3, abs=0.01)

    def test_shadow_diff(self):
        from onepai.utils import shadow_diff
        before = {"l0": 0.3, "l1": 0.8}
        after  = {"l0": 0.5, "l1": 0.8}
        diff = shadow_diff(before, after)
        assert diff["l0"] == pytest.approx(0.2, abs=1e-5)
        assert diff["l1"] == pytest.approx(0.0, abs=1e-5)


class TestQuery:
    def setup_method(self):
        from onepai.core.observer import Observation
        from datetime import datetime
        # Observation ha: name, timestamp, data, meta
        # Mettiamo activation_value e silence_ratio in meta
        self.obs = [
            Observation(name="mlp.0:0", timestamp=datetime.now(), data=None,
                        meta={"activation_value": 0.5, "silence_ratio": 0.1}),
            Observation(name="mlp.0:1", timestamp=datetime.now(), data=None,
                        meta={"activation_value": 0.001, "silence_ratio": 0.9}),
            Observation(name="attn.0:0", timestamp=datetime.now(), data=None,
                        meta={"activation_value": 0.8, "silence_ratio": 0.05}),
        ]

    def test_where_layer_filter(self):
        from onepai.reader import Query
        result = Query().where_layer("mlp.0").execute_on(self.obs)
        assert result.total_matched == 2

    def test_where_silence_filter(self):
        from onepai.reader import Query
        result = Query().where_silence(min_ratio=0.5).execute_on(self.obs)
        assert result.total_matched == 1

    def test_limit(self):
        from onepai.reader import Query
        result = Query().limit(1).execute_on(self.obs)
        assert len(result.observations) == 1

    def test_order_by_activation(self):
        from onepai.reader import Query
        result = (Query()
                  .order_by("activation_value", descending=True)
                  .execute_on(self.obs))
        # Primo risultato deve avere activation_value maggiore del secondo
        vals = [o.meta["activation_value"] for o in result]
        assert vals[0] >= vals[1]


class TestExporter:
    def setup_method(self):
        from onepai.core.observer import Observation
        from datetime import datetime
        self.obs = [
            Observation(name="l0:0", timestamp=datetime.now(),
                        data=0.5, meta={"activation_value": 0.5})
        ]

    def test_to_json(self):
        from onepai.reader import Exporter
        import json
        text = Exporter(self.obs).to_json()
        data = json.loads(text)
        assert isinstance(data, list)
        assert data[0]["name"] == "l0:0"

    def test_to_csv_contains_header(self):
        from onepai.reader import Exporter
        text = Exporter(self.obs).to_csv()
        assert "name" in text

    def test_to_markdown_contains_pipe(self):
        from onepai.reader import Exporter
        text = Exporter(self.obs).to_markdown()
        assert "|" in text

    def test_to_html_table(self):
        from onepai.reader import Exporter
        html = Exporter(self.obs).to_html_table()
        assert "<table" in html
