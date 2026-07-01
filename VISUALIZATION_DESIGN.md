# ONEPAI — Visualization Design

> Documento di riferimento per costruire la UI grafica del framework.
> Generato automaticamente — aggiorna liberamente.

---

## 1. Filosofia

L'invisibile deve diventare visibile. ONEPAI non mostra cosa fa il modello;
mostra **cosa NON fa**: i neuroni silenziosi, le teste che non guardano mai,
i pesi che esistono ma non contano, i pensieri abortiti prima di emergere.

Il design deve riflettere questa dualità: **luce/ombra**, **segnale/silenzio**.

---

## 2. Componenti già pronti (Python)

| Classe | Modulo | Output |
|--------|--------|--------|
| `ActivationHeatmap` | `visualization.heatmap` | SVG inline |
| `SilenceHeatmap` | `visualization.heatmap` | SVG inline (verde) |
| `AttentionHeatmap` | `visualization.heatmap` | SVG inline |
| `NeuralGraphRenderer` | `visualization.graph_viz` | SVG inline |
| `ActivationTimeline` | `visualization.timeline_viz` | SVG inline |
| `ConfidenceTimeline` | `visualization.timeline_viz` | SVG inline |
| `GradientTimeline` | `visualization.timeline_viz` | SVG inline |
| `HTMLReport` | `visualization.html_report` | File HTML autocontenuto |

Genera un report completo in 3 righe:

```python
from onepai.visualization import HTMLReport
report = HTMLReport.from_pipeline(registry, shadow_map, silence_report)
report.save("onepai_report.html")
```

---

## 3. Struttura del report HTML

```
┌─────────────────────────────────────────────────────┐
│ ◈ ONEPAI                     [sidebar, 230px]       │
│  ⚡ Activation Overview                             │
│  🔇 Silence Map                                     │
│  👁  Attention Heads                                │
│  📉 Gradient Flow                                   │
│  🔬 Logit Lens                                      │
│  ≡  Silent Entities (table)                         │
│  💀 Dead Parameters (table)                         │
└─────────────────────────────────────────────────────┘

┌─── Main area ───────────────────────────────────────┐
│  ONEPAI Pipeline Report                             │
│                                                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │
│  │Layers  │ │Dead    │ │Ghost   │ │Health  │      │
│  │  12    │ │neurons │ │heads   │ │  OK    │      │
│  │        │ │   3    │ │   2    │ │        │      │
│  └────────┘ └────────┘ └────────┘ └────────┘      │
│                                                     │
│  ▼ Activation Overview                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ [SVG heatmap: layer × activation]           │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ▼ Silence Map                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ [SVG grid: layer × neuron, colore=silence]  │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ▶ Attention Flow Matrix  [collapsed]               │
│  ▼ Gradient Timeline                                │
│  ┌─────────────────────────────────────────────┐   │
│  │ [SVG line chart: step vs |gradient|]        │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 4. Frontend React (da costruire)

Se vuoi un'interfaccia interattiva più ricca, il punto di partenza è:

```
frontend/
├── package.json
├── vite.config.ts
└── src/
    ├── App.tsx
    ├── components/
    │   ├── Sidebar.tsx          — navigazione sezioni
    │   ├── SummaryCards.tsx     — metriche chiave
    │   ├── HeatmapCanvas.tsx    — canvas 2D via WebGL (D3 / regl)
    │   ├── NetworkGraph.tsx     — force-directed con D3-force
    │   ├── TimelineChart.tsx    — recharts / visx
    │   ├── AttentionGrid.tsx    — SVG attention patterns
    │   └── TokenInfluence.tsx   — barre di attribution
    ├── hooks/
    │   ├── useWebSocket.ts      — WebSocket /ws/stream
    │   └── useArchive.ts        — fetch /api/observations
    └── api/
        └── client.ts            — FastAPI client
```

### Librerie consigliate

| Categoria | Libreria | Perché |
|-----------|----------|--------|
| Heatmap interattiva | `d3-selection` + canvas | Performante su matrici >1000 celle |
| Grafo neurale | `d3-force` | Già nel grafo del circuito tracer |
| Timeline | `recharts` o `visx` | Semplice, TypeScript-first |
| Stato globale | `zustand` | Leggero, no boilerplate |
| Stile | `tailwindcss` | Utility-first, no setup |

### Wireframe componenti chiave

#### HeatmapCanvas
```
Props: matrix: number[][], labels?: string[], colorScale: "diverging" | "silence"
- Usa canvas 2D per performance
- Tooltip on hover (posizione assoluta)
- Zoom/pan con wheel + drag
- Export PNG via canvas.toDataURL()
```

#### NetworkGraph
```
Props: nodes: GraphNode[], edges: GraphEdge[]
- Force simulation: charge=-300, link distance=80
- Colore nodo: active=blue, ghost=gray, shadow=orange, dead=red, critical=green
- Click su nodo → pannello dettaglio laterale
- Drag nodi per riorganizzare
```

#### AttentionGrid
```
Props: heads: AttentionHead[], tokens: string[]
- Grid di (n_layers × n_heads) miniature
- Click miniatura → espande in full-screen
- Slider per step temporale
```

---

## 5. API WebSocket (live streaming)

Il server già espone `ws://localhost:8000/ws/stream`.

Schema messaggi dal server:

```json
{
  "type": "heartbeat" | "observation" | "shadow_update" | "alert",
  "ts": "2026-06-12T14:30:00",
  "payload": { ... }
}
```

Payload per `observation`:
```json
{
  "name": "attn.0:42",
  "activation_value": 0.003,
  "silence_ratio": 0.87,
  "layer": "attn.0",
  "neuron_idx": 42
}
```

Payload per `alert`:
```json
{
  "level": "warning" | "critical",
  "message": "Dead neuron detected: mlp.3:17",
  "entity": "mlp.3:17"
}
```

---

## 6. Palette colori

| Token | HEX | Uso |
|-------|-----|-----|
| `--active` | `#4a90d9` | Neuroni attivi, flusso principale |
| `--ghost` | `#c0c0c0` | Teste fantasma, pesi inutilizzati |
| `--shadow` | `#ff8c42` | Circuiti ombra, attenzione marginale |
| `--dead` | `#e04040` | Neuroni morti, parametri zero |
| `--critical` | `#2ecc71` | Path causale critica |
| `--bg` | `#f0f2f5` | Sfondo pagina |
| `--surface` | `#ffffff` | Card / pannelli |
| `--sidebar` | `#1a1a2e` | Sidebar scura |
| `--text` | `#1a1a2e` | Testo principale |
| `--muted` | `#888888` | Testo secondario |

---

## 7. Prossimi step suggeriti

1. `npm create vite@latest frontend -- --template react-ts` nella root del progetto
2. Copia la palette da questa sezione in `frontend/src/theme.ts`
3. Implementa `HeatmapCanvas` usando il canvas API — è il componente più usato
4. Connetti il WebSocket per live streaming del silence score
5. Aggiungi export PNG ai componenti (per inserire nel report HTML)

---

*Documento generato da ONEPAI v0.1 — 2026-06-12*
