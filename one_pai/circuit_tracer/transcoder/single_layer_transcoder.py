"""ONEPAI Scripts Package

Questo package contiene script utili per l'utilizzo di ONEPAI:
- Script di avvio per dashboard e servizi
- Utilities per la gestione degli archivi
- Tool per l'analisi e la mappatura
- Script di build e deployment

Author: ONEPAI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ONEPAI Team"
__description__ = "Utility scripts for ONEPAI framework"

# Lista degli script disponibili
AVAILABLE_SCRIPTS = [
    'build_onepai.py',
    'create_treasure.py', 
    'decode_silence.py',
    'manage_archive.py',
    'map_shadows.py',
    'run_analysis.py',
    'run_dashboard.py'
]

__all__ = [
    'AVAILABLE_SCRIPTS',
    '__version__',
    '__author__',
    '__description__'
]