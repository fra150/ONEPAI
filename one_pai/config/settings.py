"""
ONEPAI Settings Configuration
=============================

Configurazioni centrali per il sistema ONEPAI.
Gestisce tutti i parametri per mappare l'invisibile dell'AI.

Author: Francesco Bulla (Brainverse)
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class OnepaiSettings:
    """
    Configurazione principale di ONEPAI - Il Tesoro dell'AI
    
    Parametri per controllare:
    - Sensibilità di mappatura delle ombre
    - Soglie di rilevamento del silenzio  
    - Profondità di analisi del vuoto
    - Cifratura dei tesori (.onepai files)
    - Modalità quantistica
    """
    
    # === CORE PARAMETERS ===
    shadow_sensitivity: float = 0.7        # Sensibilità mappatura ombre (0.0-1.0)
    silence_threshold: float = 0.3          # Soglia rilevamento silenzi
    void_detection_depth: int = 5           # Profondità analisi vuoto
    treasure_encryption: bool = True        # Cifratura file .onepai
    quantum_mode: bool = False              # Modalità quantistica
    
    # === NEURAL PARAMETERS ===
    weight_ghost_threshold: float = 0.1     # Soglia pesi fantasma
    abortion_sensitivity: float = 0.5       # Sensibilità pensieri abortiti
    circuit_shadow_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    
    # === MEMORY PARAMETERS ===
    unsaid_vault_capacity: int = 10000      # Capacità archivio non-detto
    undone_registry_size: int = 5000        # Dimensione registro non-agito
    unthought_space_depth: int = 3          # Profondità spazio non-pensato
    
    # === ANALYSIS PARAMETERS ===
    absence_pattern_window: int = 100       # Finestra pattern assenze
    silence_metrics_interval: float = 0.1   # Intervallo metriche silenzio
    void_statistics_samples: int = 1000     # Campioni statistiche vuoto
    
    # === FILE PATHS ===
    treasures_dir: str = "data/treasures"   # Directory file .onepai
    shadows_dir: str = "data/shadows"       # Directory mappe ombre
    voids_dir: str = "data/voids"          # Directory spazi vuoti
    silences_dir: str = "data/silences"     # Directory silenzi
    
    # === CRYPTO PARAMETERS ===
    cipher_algorithm: str = "AES-256-GCM"   # Algoritmo cifratura
    key_derivation: str = "PBKDF2"          # Derivazione chiave
    salt_length: int = 32                   # Lunghezza salt
    
    # === API PARAMETERS ===
    api_host: str = "localhost"             # Host API
    api_port: int = 8888                    # Porta API
    websocket_enabled: bool = True          # WebSocket per real-time
    
    # === DASHBOARD PARAMETERS ===
    dashboard_theme: str = "shadow"         # Tema UI (shadow/void/silence)
    real_time_updates: bool = True          # Aggiornamenti real-time
    max_visualization_points: int = 5000    # Max punti visualizzazione
    
    def __post_init__(self):
        """Validazione e creazione directory dopo inizializzazione"""
        self._validate_parameters()
        self._create_directories()
    
    def _validate_parameters(self):
        """Valida i parametri di configurazione"""
        if not 0.0 <= self.shadow_sensitivity <= 1.0:
            raise ValueError("shadow_sensitivity deve essere tra 0.0 e 1.0")
        
        if not 0.0 <= self.silence_threshold <= 1.0:
            raise ValueError("silence_threshold deve essere tra 0.0 e 1.0")
            
        if self.void_detection_depth < 1:
            raise ValueError("void_detection_depth deve essere >= 1")
            
        if self.api_port < 1024 or self.api_port > 65535:
            raise ValueError("api_port deve essere tra 1024 e 65535")
    
    def _create_directories(self):
        """Crea le directory necessarie se non esistono"""
        dirs = [
            self.treasures_dir,
            self.shadows_dir, 
            self.voids_dir,
            self.silences_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'OnepaiSettings':
        """
        Carica configurazione da file YAML
        
        Args:
            config_path: Percorso file configurazione
            
        Returns:
            OnepaiSettings: Istanza configurazione
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"File configurazione non trovato: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """
        Salva configurazione su file YAML
        
        Args:
            config_path: Percorso file di destinazione
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte dataclass in dizionario
        config_dict = {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_treasure_path(self, treasure_name: str) -> Path:
        """Ottiene percorso completo per un file tesoro"""
        return Path(self.treasures_dir) / f"{treasure_name}.onepai"
    
    def get_shadow_path(self, shadow_name: str) -> Path:
        """Ottiene percorso completo per una mappa ombra"""
        return Path(self.shadows_dir) / f"{shadow_name}.shadow"
    
    def get_void_path(self, void_name: str) -> Path:
        """Ottiene percorso completo per uno spazio vuoto"""
        return Path(self.voids_dir) / f"{void_name}.void"
    
    def get_silence_path(self, silence_name: str) -> Path:
        """Ottiene percorso completo per un silenzio"""
        return Path(self.silences_dir) / f"{silence_name}.silence"


# Istanza globale settings
_settings_instance: Optional[OnepaiSettings] = None

def get_settings() -> OnepaiSettings:
    """
    Ottiene istanza globale delle impostazioni ONEPAI
    
    Returns:
        OnepaiSettings: Configurazione globale
    """
    global _settings_instance
    
    if _settings_instance is None:
        # Cerca file configurazione in ordine di priorità
        config_paths = [
            "onepai.yaml",
            "config/onepai.yaml", 
            os.getenv("ONEPAI_CONFIG", ""),
        ]
        
        for config_path in config_paths:
            if config_path and Path(config_path).exists():
                _settings_instance = OnepaiSettings.from_file(config_path)
                break
        else:
            # Usa configurazione predefinita
            _settings_instance = OnepaiSettings()
    
    return _settings_instance

def set_settings(settings: OnepaiSettings):
    """
    Imposta istanza globale delle impostazioni
    
    Args:
        settings: Nuova configurazione
    """
    global _settings_instance
    _settings_instance = settings