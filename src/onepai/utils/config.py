"""Configurazione centralizzata del framework ONEPAI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["ONEPAIConfig", "load_config", "save_config", "get_default_config"]

_ENV_PREFIX = "ONEPAI_"
_DEFAULT_CONFIG_PATH = Path.home() / ".onepai" / "config.json"


@dataclass
class ONEPAIConfig:
    # Soglie core
    silence_threshold: float = 1e-6
    ghost_threshold: float = 0.1
    shadow_threshold: float = 0.05
    dead_gradient_threshold: float = 1e-7

    # Archivio
    archive_path: str = "data/treasures/default.onepai"
    encrypt_archives: bool = False
    compression: str = "zlib"            # "zlib" | "lzma" | "none"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    # Introspezione
    introspection_top_k: int = 5         # top-k per LogitLens
    gradient_n_steps: int = 5            # ultimi N step per dead-param check
    attention_seq_len: int = 128         # seq_len max per attention rollout

    # Percorsi dati
    data_dir: str = "data"
    graphs_dir: str = "graphs"
    reports_dir: str = "reports"

    def as_dict(self) -> dict:
        return asdict(self)

    def override_from_env(self) -> "ONEPAIConfig":
        """
        Sovrascrive i campi con variabili d'ambiente ONEPAI_<CAMPO_UPPERCASE>.
        Es. ONEPAI_SILENCE_THRESHOLD=1e-5
        """
        for key in self.as_dict():
            env_key = _ENV_PREFIX + key.upper()
            val = os.environ.get(env_key)
            if val is not None:
                current = getattr(self, key)
                try:
                    if isinstance(current, bool):
                        setattr(self, key, val.lower() in ("1", "true", "yes"))
                    elif isinstance(current, int):
                        setattr(self, key, int(val))
                    elif isinstance(current, float):
                        setattr(self, key, float(val))
                    else:
                        setattr(self, key, val)
                except (ValueError, TypeError):
                    pass
        return self


def get_default_config() -> ONEPAIConfig:
    """Ritorna la configurazione di default con eventuali override da env."""
    return ONEPAIConfig().override_from_env()


def load_config(path: Optional[Path | str] = None) -> ONEPAIConfig:
    """Carica configurazione da file JSON, poi applica override da env."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    cfg = ONEPAIConfig()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except (json.JSONDecodeError, OSError):
            pass
    return cfg.override_from_env()


def save_config(cfg: ONEPAIConfig, path: Optional[Path | str] = None) -> Path:
    """Salva la configurazione su disco."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.as_dict(), indent=2), encoding="utf-8")
    return p
