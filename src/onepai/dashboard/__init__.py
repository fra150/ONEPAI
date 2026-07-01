"""ONEPAI Dashboard module."""

__version__ = "1.0.0"
__author__ = "ONEPAI Team"

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "ONEPAI Dashboard",
    "description": "Shadow-Scope Interface - Monitoring Neural Network Activations",
    "version": __version__,
    "endpoints": {
        "health": "/api/health",
        "observations": "/api/observations",
        "websocket": "ws://localhost:8000/ws/stream",
    },
}

__all__ = [
    "DASHBOARD_CONFIG",
    "start_dashboard",
    "__version__",
    "__author__",
]


def start_dashboard(host: str = "127.0.0.1", port: int = 8000):
    """Avvia il server della dashboard (import pigro: richiede fastapi/uvicorn)."""
    from .server import start_dashboard as _start

    return _start(host=host, port=port)
