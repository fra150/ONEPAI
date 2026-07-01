"""ONEPAI User Interface Package

Questo package contiene le interfacce utente per ONEPAI, inclusi:
- CLI (Command Line Interface) per l'interazione da terminale
- Dashboard per l'interfaccia web
- Utilities per la visualizzazione dei dati

Author: ONEPAI Team
Version: 1.0.0
"""

from .cli import OnepaiCLI
from .dashboard import DashboardUI

__version__ = "1.0.0"
__author__ = "ONEPAI Team"
__description__ = "User Interface components for ONEPAI framework"

__all__ = [
    # Interfacce principali
    'OnepaiCLI',
    'DashboardUI',
    
    # Metadati
    '__version__',
    '__author__',
    '__description__'
]