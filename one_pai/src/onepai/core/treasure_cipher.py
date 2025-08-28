"""
Server FastAPI per l'interfaccia dashboard "Shadow-Scope".
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from ..core import Archive

app = FastAPI(title="ONEPAI Dashboard", description="Shadow-Scope Interface")

# Configurazione
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
ARCHIVE_PATH = DATA_DIR / "treasures" / "default.onepai"

# Monta i file statici
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "ui" / "dist"), name="static")

@app.get("/")
async def root():
    """Serve la pagina principale."""
    html_path = Path(__file__).parent / "ui" / "dist" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>ONEPAI Dashboard</h1><p>Frontend non ancora configurato</p>")

@app.get("/api/health")
async def health_check():
    """Endpoint di health check."""
    return {"status": "healthy", "service": "ONEPAI Dashboard"}

@app.get("/api/observations")
async def get_observations(limit: int = 100) -> List[Dict[str, Any]]:
    """Recupera le osservazioni dall'archivio."""
    try:
        archive = Archive(ARCHIVE_PATH)
        observations = []
        
        for i, obs in enumerate(archive.iter_records()):
            if i >= limit:
                break
            observations.append(obs.__dict__)
            
        return observations
    except Exception as e:
        return [{"error": str(e)}]

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket per lo streaming in tempo reale."""
    await websocket.accept()
    try:
        while True:
            # Invia aggiornamenti ogni 5 secondi
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": str(Path().stat().st_mtime)
            })
    except WebSocketDisconnect:
        pass

def start_dashboard(host: str = "127.0.0.1", port: int = 8000):
    """Funzione di avvio per il dashboard."""
    # Crea directory se non esistono
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🌐 Dashboard disponibile su: http://{host}:{port}")
    print(f"📊 API disponibile su: http://{host}:{port}/api")
    
    uvicorn.run(
        "onepai.dashboard.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_dashboard()