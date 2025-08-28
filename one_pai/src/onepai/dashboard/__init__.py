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
        "websocket": "ws://localhost:8000/ws/stream"
    }
}

# Export main components
__all__ = [
    "DASHBOARD_CONFIG",
    "__version__",
    "__author__"
]