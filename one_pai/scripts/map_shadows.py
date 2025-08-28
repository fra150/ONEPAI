#!/usr/bin/env python3
"""
Script di avvio rapido per l'interfaccia dashboard di ONEPAI.
"""

import sys
import os
from pathlib import Path

# Aggiungi il percorso src al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from onepai.dashboard.server import start_dashboard

def main():
    """Avvia il dashboard ONEPAI."""
    print("🚀 Avvio del dashboard ONEPAI...")
    
    # Imposta le configurazioni di default
    host = "127.0.0.1"
    port = 8000
    
    try:
        start_dashboard(host=host, port=port)
    except KeyboardInterrupt:
        print("\n⚡ Dashboard interrotto dall'utente")
    except Exception as e:
        print(f"❌ Errore durante l'avvio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()