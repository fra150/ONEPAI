# -----------------------------------------------------------------------------
# File: circuit_tracer/frontend/local_server.py
# -----------------------------------------------------------------------------
"""
Server locale per la visualizzazione di grafi del circuit tracer.

- Server HTTP multithread
- API /api/graphs  -> lista grafi (file .json nella cartella)
- API /api/graph/<name> -> contenuto JSON del grafo <name>.json
- Pagina /         -> index minimale
- /health          -> healthcheck
- CORS completo + OPTIONS preflight
"""

from __future__ import annotations

import json
import re
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse

# Pattern sicuro per i nomi dei grafi (evita traversal)
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class GraphServerHandler(SimpleHTTPRequestHandler):
    """Handler personalizzato per servire grafi e API."""

    # Con Python 3.7+ SimpleHTTPRequestHandler accetta 'directory'
    # Se vuoi servire static anche da una cartella assets, passa directory nel partial()
    def __init__(self, *args, graph_dir: Optional[str] = None, **kwargs):
        self.graph_dir = Path(graph_dir or "./graphs")
        super().__init__(*args, **kwargs)

    # --------------------- util ---------------------
    def _send_json(self, status: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: int, text: str, content_type="text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _list_graphs(self):
        graphs = []
        if self.graph_dir.exists():
            for p in sorted(self.graph_dir.glob("*.json")):
                graphs.append(
                    {
                        "name": p.stem,
                        "filename": p.name,
                        "size": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime),
                    }
                )
        return graphs

    # --------------------- CORS ---------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---------------------- GET ---------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            return self._send_json(200, {"ok": True, "uptime": True})

        if path == "/api/graphs":
            try:
                return self._send_json(200, self._list_graphs())
            except Exception as e:
                return self._send_json(500, {"error": f"Errore nel caricamento dei grafi: {e}"})

        if path.startswith("/api/graph/"):
            # /api/graph/<name>
            name = path.split("/")[-1]
            if not SAFE_NAME.match(name):
                return self._send_json(400, {"error": "Nome grafo non valido"})
            f = (self.graph_dir / f"{name}.json").resolve()
            # Evita che il path esca dalla cartella
            try:
                f.relative_to(self.graph_dir.resolve())
            except Exception:
                return self._send_json(400, {"error": "Accesso non consentito"})
            if not f.exists():
                return self._send_json(404, {"error": f"Grafo {name} non trovato"})
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return self._send_json(200, data)
            except Exception as e:
                return self._send_json(500, {"error": f"Errore nel caricamento del grafo: {e}"})

        if path == "/" or path == "/index.html":
            return self._send_text(200, INDEX_HTML)

        # fallback a file statici (se servono)
        return super().do_GET()


INDEX_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Circuit Tracer - Visualizzatore Grafi</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:0; padding:24px; background:#0b1020; color:#e9eef7; }
  .container { max-width: 1100px; margin:0 auto; }
  h1 { text-align:center; margin-bottom: 12px; }
  .status,.graphs-list { background: rgba(255,255,255,.06); padding:16px; border-radius:12px; margin-bottom:16px; }
  .graph-item { background: rgba(255,255,255,.06); padding:12px; border-radius:10px; margin:10px 0; border-left: 4px solid #30d158; }
  .btn { background:#30d158; color:#041018; padding:10px 14px; border:none; border-radius:8px; cursor:pointer; }
  .btn:hover { filter: brightness(1.05); }
  code { background: rgba(255,255,255,.08); padding:2px 6px; border-radius:6px; }
  .muted { opacity: .8; }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Circuit Tracer</h1>
  <div class="status">
    <h2>✅ Server Attivo</h2>
    <p class="muted">Visualizzatore di grafi neurali in esecuzione</p>
  </div>

  <div class="graphs-list">
    <h3>📊 Grafi Disponibili</h3>
    <div id="graphs-container"><div class="muted">🔄 Caricamento grafi...</div></div>
    <button class="btn" onclick="loadGraphs()">🔄 Aggiorna Lista</button>
  </div>

  <div class="status">
    <h3>💡 Come Usare</h3>
    <p>1. Genera grafi con:<br/>
      <code>python -m circuit_tracer attribute --prompt "testo" --graph_file_dir ./graphs --graph_output_path ./graphs/out.pt --slug my-model</code></p>
    <p>2. I grafi appariranno qui automaticamente</p>
    <p>3. Clicca su un grafo per visualizzarne i dettagli</p>
  </div>
</div>

<script>
async function loadGraphs() {
  try {
    const res = await fetch('/api/graphs');
    const graphs = await res.json();
    const el = document.getElementById('graphs-container');
    if (!Array.isArray(graphs) || graphs.length === 0) {
      el.innerHTML = '<div class="muted">📁 Nessun grafo trovato. Generane uno con il comando sopra.</div>';
      return;
    }
    el.innerHTML = graphs.map(g => `
      <div class="graph-item">
        <h4>📈 ${g.name}</h4>
        <p>📄 ${g.filename} — ${(g.size/1024).toFixed(2)} KB</p>
        <button class="btn" onclick="viewGraph('${g.name}')">👁️ Visualizza</button>
      </div>`).join('');
  } catch (e) {
    document.getElementById('graphs-container').innerHTML = '<div class="muted">❌ Errore nel caricamento</div>';
  }
}

async function viewGraph(name) {
  try {
    const res = await fetch('/api/graph/' + encodeURIComponent(name));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const w = window.open('', '_blank');
    w.document.write('<pre style="font-family:monospace;white-space:pre-wrap;padding:16px;">' +
                     JSON.stringify(data, null, 2) + '</pre>');
    w.document.title = 'Grafo: ' + name;
  } catch (e) {
    alert('Errore nel caricamento: ' + e.message);
  }
}

loadGraphs();
setInterval(loadGraphs, 5000);
</script>
</body>
</html>
"""

def serve(graph_file_dir: str, port: int = 8001, open_browser: bool = True, static_dir: Optional[str] = None) -> None:
    """
    Avvia il server di visualizzazione grafi.

    Args:
        graph_file_dir: cartella che contiene i file .json dei grafi
        port: porta del server HTTP
        open_browser: se True apre il browser automaticamente
        static_dir: se fornita, verrà usata come directory per file statici di default
    """
    root = Path(graph_file_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Se vuoi servire static anche da una cartella (es. assets), passa directory=static_dir qui.
    Handler = partial(
        GraphServerHandler,
        graph_dir=str(root),
        directory=(static_dir if static_dir else None)
    )

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"🚀 Circuit Tracer Server avviato su http://localhost:{port}")
    print(f"📁 Directory grafi: {root.resolve()}")
    print(f"🔗 Apri http://localhost:{port} nel browser")
    print("⏹️  Ctrl+C per fermare")

    if open_browser:
        def _open():
            time.sleep(1)
            try:
                webbrowser.open(f"http://localhost:{port}", new=2)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server fermato")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve("./graphs", 8001)
