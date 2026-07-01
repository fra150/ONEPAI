"""
API REST per l'accesso allo Scrigno (.onepai).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def create_api_server(archive_path: str | Path | None = None):
    """Crea un'app FastAPI che espone il contenuto di un archivio .onepai.

    Args:
        archive_path: Percorso del file .onepai da servire. Se None usa il
            percorso di default di :class:`~onepai.core.archive.TreasureArchive`.

    Returns:
        Un'istanza FastAPI pronta da servire con uvicorn.
    """
    from fastapi import FastAPI

    from ..core.archive import TreasureArchive

    app = FastAPI(title="ONEPAI Scrigno API", description="Accesso allo Scrigno")
    path = Path(archive_path) if archive_path else TreasureArchive.DEFAULT_PATH

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "healthy", "service": "ONEPAI Scrigno API"}

    @app.get("/api/treasures")
    async def treasures(limit: int = 100, query: Optional[str] = None) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        archive = TreasureArchive(path)
        records = (
            archive.search_treasures(query) if query else list(archive.iter_records())
        )
        return [obs.__dict__ for obs in records[:limit]]

    return app
