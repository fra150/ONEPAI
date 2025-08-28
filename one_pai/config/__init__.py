"""
ONEPAI Configuration Package
============================

Configurazione centrale per il sistema ONEPAI - Il Tesoro dell'AI
Gestisce settings, logging e configurazioni globali per mappare l'invisibile.

Author: Francesco Bulla (Brainverse)
"""

from .settings import OnepaiSettings, get_settings
from .logging_config import setup_logging, get_logger

__version__ = "0.1.0"
__author__ = "Francesco Bulla"
__description__ = "Configuration package per ONEPAI - L'intelligenza dell'invisibile"

# Configurazione globale predefinita
DEFAULT_CONFIG = {
    "shadow_sensitivity": 0.7,
    "silence_threshold": 0.3,
    "void_detection_depth": 5,
    "treasure_encryption": True,
    "quantum_mode": False
}

def initialize_onepai():
    """
    Inizializza la configurazione globale di ONEPAI
    Prepara il sistema per mappare l'invisibile.
    """
    settings = get_settings()
    logger = setup_logging()
    
    logger.info("🔥 ONEPAI Sistema Inizializzato")
    logger.info(f"📊 Shadow Sensitivity: {settings.shadow_sensitivity}")
    logger.info(f"🔇 Silence Threshold: {settings.silence_threshold}")
    logger.info(f"🕳️ Void Detection Depth: {settings.void_detection_depth}")
    logger.info(f"🔐 Treasure Encryption: {'ENABLED' if settings.treasure_encryption else 'DISABLED'}")
    
    return settings, logger

__all__ = [
    "OnepaiSettings",
    "get_settings", 
    "setup_logging",
    "get_logger",
    "initialize_onepai",
    "DEFAULT_CONFIG"
]