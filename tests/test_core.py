"""Tests for onepai.core."""
import os
from datetime import datetime
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class TestObservation:
    def test_fields(self):
        from onepai.core.observer import Observation
        obs = Observation(name="l1:0", timestamp=datetime.utcnow(), data=[1.0], meta={"x": 1})
        assert obs.name == "l1:0"
        assert obs.meta == {"x": 1}

    def test_data_none(self):
        from onepai.core.observer import Observation
        obs = Observation(name="l1:0", timestamp=datetime.utcnow(), data=None)
        assert obs.data is None


class TestObserver:
    def test_initially_empty(self):
        from onepai.core.observer import Observer
        assert Observer().get_records() == []

    def test_clear(self):
        from onepai.core.observer import Observer, Observation
        o = Observer()
        o._records.append(Observation(name="x", timestamp=datetime.utcnow(), data=None))
        o.clear()
        assert o.get_records() == []

    def test_context_manager(self):
        from onepai.core.observer import Observer
        with Observer():
            pass


class TestShadowRegistry:
    def test_add_and_get(self, sample_observation):
        from onepai.core.registry import ShadowRegistry
        r = ShadowRegistry()
        r.add(sample_observation)
        assert len(r.get("layer1")) == 1

    def test_get_unknown_layer(self):
        from onepai.core.registry import ShadowRegistry
        assert ShadowRegistry().get("nope") == []

    def test_summary(self, populated_registry):
        s = populated_registry.summary()
        assert s["layer1"] == 1

    def test_len(self, populated_registry):
        assert len(populated_registry) == 1

    def test_clear(self, populated_registry):
        populated_registry.clear()
        assert len(populated_registry) == 0


class TestNeuronActivity:
    def test_silence_ratio_zero_obs(self):
        from onepai.core.shadow_mapper import NeuronActivity
        assert NeuronActivity().silence_ratio == 0.0

    def test_silence_ratio(self):
        from onepai.core.shadow_mapper import NeuronActivity
        a = NeuronActivity(silence_count=3, total_observations=10)
        assert a.silence_ratio == pytest.approx(0.3)


class TestShadowMapper:
    def _obs(self, name, data):
        from onepai.core.observer import Observation
        return Observation(name=name, timestamp=datetime.utcnow(), data=data)

    def test_empty(self):
        from onepai.core.shadow_mapper import ShadowMapper
        smap = ShadowMapper().create_shadow_map(observations=[])
        assert smap.layers == []

    def test_none_data_is_silence(self):
        from onepai.core.shadow_mapper import ShadowMapper
        smap = ShadowMapper().create_shadow_map([self._obs("l:0", None)])
        assert smap.layer_activity["l:0"][0].silence_count == 1

    def test_nonzero_not_silent(self):
        import numpy as np
        from onepai.core.shadow_mapper import ShadowMapper
        obs = self._obs("mlp:0", np.array([0.5, 0.0, -0.8]))
        smap = ShadowMapper(silence_threshold=1e-6).create_shadow_map([obs])
        assert smap.layer_activity["mlp:0"][0].silence_count == 0
        assert smap.layer_activity["mlp:0"][1].silence_count == 1

    def test_model_name(self):
        from onepai.core.shadow_mapper import ShadowMapper
        smap = ShadowMapper().create_shadow_map(observations=[], model_name="gpt2")
        assert smap.model_name == "gpt2"

    def test_from_registry(self, populated_registry):
        from onepai.core.shadow_mapper import ShadowMapper
        smap = ShadowMapper(registry=populated_registry).create_shadow_map()
        assert smap.observation_count == 1


class TestSilenceTracer:
    def _map(self, silence, total):
        from onepai.core.shadow_mapper import ShadowMap, NeuronActivity
        sm = ShadowMap(model_name="m", observation_count=total)
        act = sm.neuron("l0", 0)
        act.silence_count = silence
        act.total_observations = total
        return sm

    def test_dead_neuron(self):
        from onepai.core.silence_tracer import SilenceTracer
        report = SilenceTracer().trace(self._map(100, 100), dead_neuron_threshold=0.99)
        assert len(report.dead_neurons) == 1

    def test_active_neuron_not_flagged(self):
        from onepai.core.silence_tracer import SilenceTracer
        report = SilenceTracer().trace(self._map(5, 100), dead_neuron_threshold=0.99)
        assert len(report.dead_neurons) == 0

    def test_invalid_threshold(self):
        from onepai.core.silence_tracer import SilenceTracer, ShadowMap
        with pytest.raises(ValueError):
            SilenceTracer().trace(ShadowMap(), dead_neuron_threshold=2.0)


class TestArchive:
    def test_roundtrip(self, tmp_archive):
        from onepai.core.archive import Archive
        from onepai.core.observer import Observation
        a = Archive(tmp_archive)
        a.append([Observation(name="l1:0", timestamp=datetime.utcnow(), data=None)])
        assert len(list(a.iter_records())) == 1

    def test_bad_magic(self, tmp_archive):
        from onepai.core.archive import Archive
        tmp_archive.write_bytes(b"GARBAGE!")
        a = Archive.__new__(Archive)
        a.path = tmp_archive
        with pytest.raises(IOError):
            list(a.iter_records())

    def test_treasure_archive_list(self, tmp_archive):
        from onepai.core.archive import TreasureArchive
        from onepai.core.observer import Observation
        ta = TreasureArchive(tmp_archive)
        ta.append([Observation(name="shadow_1", timestamp=datetime.utcnow(), data=None)])
        assert "shadow_1" in ta.list_treasures()

    def test_treasure_archive_search(self, tmp_archive):
        from onepai.core.archive import TreasureArchive
        from onepai.core.observer import Observation
        ta = TreasureArchive(tmp_archive)
        ta.append([
            Observation(name="hidden_42", timestamp=datetime.utcnow(), data=None),
            Observation(name="output_token", timestamp=datetime.utcnow(), data=None),
        ])
        assert len(ta.search_treasures("hidden")) == 1


class TestCrypto:
    def test_generate_key(self):
        from onepai.core.crypto import generate_key
        k = generate_key()
        assert isinstance(k, bytes) and len(k) == 44

    def test_encrypt_decrypt(self):
        from onepai.core.crypto import generate_key, encrypt_bytes, decrypt_bytes
        key = generate_key()
        plain = b"hello shadow"
        assert decrypt_bytes(encrypt_bytes(plain, key), key) == plain

    def test_treasure_cipher(self):
        from onepai.core.treasure_cipher import TreasureCipher
        c = TreasureCipher(password="pw")
        data = b"secret"
        assert c.decrypt(c.encrypt(data)) == data
