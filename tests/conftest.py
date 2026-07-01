"""Shared pytest fixtures for the ONEPAI test suite."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture()
def tmp_archive(tmp_path):
    return tmp_path / "test.onepai"


@pytest.fixture()
def sample_observation():
    from datetime import datetime
    from onepai.core.observer import Observation
    return Observation(name="layer1:0", timestamp=datetime.utcnow(), data=[0.0, 1.0, 0.5])


@pytest.fixture()
def populated_registry(sample_observation):
    from onepai.core.registry import ShadowRegistry
    r = ShadowRegistry()
    r.add(sample_observation)
    return r
