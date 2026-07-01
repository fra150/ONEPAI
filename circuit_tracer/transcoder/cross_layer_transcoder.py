# ONEPAI - Il Tesoro dell'AI 🧠✨

**L'intelligenza dell'invisibile. Un framework avanzato per mappare, analizzare e comprendere le dinamiche nascoste delle reti neurali.**

---

[![Stato Sviluppo](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/brainverse/onepai)
[![Licenza](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dashboard](https://img.shields.io/badge/dashboard-web-green.svg)](http://127.0.0.1:8000)

## 🎯 La Missione

Le reti neurali sono spesso viste come "scatole nere". ONEPAI (pronunciato "One-Pie", come in *One Piece of the Puzzle*) è un framework rivoluzionario dedicato a illuminare queste scatole nere attraverso un approccio poetico e scientifico.

Non si limita a osservare le attivazioni; **mappa le "ombre" cognitive**, **traccia i "silenzi" neurali**, **archivia i "tesori" di conoscenza** e **analizza i "vuoti" dell'incomprensione** - tutti i pattern nascosti che definiscono il vero comportamento di un modello.

### 🔍 Obiettivi Principali

- **🗺️ Shadow Mapping**: Mappare l'attività neuronale nascosta e le dinamiche cognitive
- **🤫 Silence Tracing**: Identificare neuroni morti, pensieri soppressi e decisioni non espresse
- **💎 Treasure Archiving**: Archiviare pattern neurali significativi in vault sicuri e cifrati
- **🕳️ Void Analysis**: Comprendere ciò che la rete *non* sa o *non* può fare
- **🧭 Cognitive Navigation**: Fornire strumenti intuitivi per esplorare l'intelligenza artificiale

## ⭐ Caratteristiche Innovative

### 🔬 Core Engine
- **Observer Avanzato**: Monitoraggio in tempo reale delle attivazioni neurali
- **Shadow Mapper**: Generazione di mappe cognitive dettagliate
- **Silence Tracer**: Rilevamento di neuroni inattivi e pattern soppressi
- **Treasure Cipher**: Sistema di cifratura AES-256-GCM per la sicurezza dei dati
- **Void Analyzer**: Analisi delle lacune cognitive e dei limiti del modello

### 🎨 Interfacce Utente
- **Dashboard Web**: Interfaccia "Shadow-Scope" moderna e interattiva
- **CLI Avanzata**: Strumenti da riga di comando per analisi approfondite
- **API REST**: Endpoints per integrazione con sistemi esterni
- **WebSocket**: Streaming in tempo reale dei dati di analisi

### 🗄️ Sistema di Archiviazione
- **Archive Sicuro**: Sistema append-only con compressione `zstd`
- **Registry SQLite**: Indicizzazione rapida delle tracce cognitive
- **Checksum Blake3**: Integrità garantita dei dati neurali
- **Vault Cifrati**: Protezione avanzata per dati sensibili

## 🚀 Installazione

### Prerequisiti
- Python 3.9 o superiore
- pip (gestore pacchetti Python)

### Installazione Rapida

```bash
# Clona il repository
git clone https://github.com/brainverse/onepai.git
cd onepai

# Installa le dipendenze
pip install -r requirements.txt

# Installa in modalità sviluppo
pip install -e .
```

### Avvio Rapido del Dashboard

```bash
# Avvia il dashboard web
python scripts/run_dashboard.py

# Il dashboard sarà disponibile su: http://127.0.0.1:8000
```

## 📁 Struttura del Progetto

```
onepai/
├── 📂 src/onepai/           # Codice sorgente principale
│   ├── 📂 core/             # Motore centrale
│   │   ├── main.py          # Orchestratore principale
│   │   ├── observer.py      # Sistema di osservazione
│   │   ├── shadow_mapper.py # Mappatura delle ombre
│   │   ├── silence_tracer.py# Tracciamento dei silenzi
│   │   ├── treasure_cipher.py# Cifratura dei tesori
│   │   ├── void_analyzer.py # Analisi dei vuoti
│   │   └── archive.py       # Sistema di archiviazione
│   ├── 📂 neural/           # Componenti neurali
│   │   ├── circuit_shadow.py# Ombre dei circuiti
│   │   ├── weight_ghost.py  # Fantasmi dei pesi
│   │   ├── invisible_net.py # Reti invisibili
│   │   └── abortion_detector.py# Rilevatore di aborti
│   ├── 📂 analysis/         # Moduli di analisi
│   │   ├── cognitive_dissonance.py# Dissonanza cognitiva
│   │   ├── dream_logic.py   # Logica dei sogni
│   │   ├── silence_metrics.py# Metriche del silenzio
│   │   └── absence_patterns.py# Pattern di assenza
│   ├── 📂 memory/           # Sistema di memoria
│   │   ├── unsaid_vault.py  # Vault del non detto
│   │   ├── unthought_space.py# Spazio del non pensato
│   │   └── undone_registry.py# Registro del non fatto
│   ├── 📂 dashboard/        # Interfaccia web
│   │   └── server.py        # Server FastAPI
│   └── 📂 ui/               # Interfacce utente
│       ├── cli.py           # Interfaccia a riga di comando
│       └── dashboard.py     # Dashboard web
├── 📂 scripts/              # Script di utilità
│   ├── run_dashboard.py     # Avvio dashboard
│   ├── map_shadows.py       # Mappatura ombre
│   ├── create_treasure.py   # Creazione tesori
│   ├── decode_silence.py    # Decodifica silenzi
│   ├── manage_archive.py    # Gestione archivi
│   └── run_analysis.py      # Analisi completa
├── 📂 data/                 # Directory dati
│   ├── 📂 shadows/          # Mappe delle ombre
│   ├── 📂 silences/         # Tracce dei silenzi
│   ├── 📂 treasures/        # Tesori archiviati
│   └── 📂 voids/            # Analisi dei vuoti
├── 📂 docs/                 # Documentazione
└── 📂 config/               # Configurazioni
    ├── settings.py          # Impostazioni globali
    └── logging_config.py    # Configurazione logging
```

## 💡 Esempi di Utilizzo

### 🎮 Dashboard Web (Modo più Semplice)

```bash
# Avvia il dashboard interattivo
python scripts/run_dashboard.py

# Apri il browser su: http://127.0.0.1:8000
# Esplora l'interfaccia "Shadow-Scope" per:
# - Visualizzare mappe cognitive in tempo reale
# - Analizzare pattern di silenzio
# - Gestire archivi di tesori
# - Monitorare la salute del sistema
```

### 🔬 Analisi Programmatica

```python
import torch
import torch.nn as nn
from onepai.core.main import ONEPAI
from onepai.neural import CircuitShadow, WeightGhost

# 1. Definisci il tuo modello
class CognitiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(128, 256)
        self.attention = nn.MultiheadAttention(256, 8)
        self.output = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        h = torch.relu(self.hidden(x))
        a, _ = self.attention(h, h, h)
        return self.output(self.dropout(a))

# 2. Inizializza ONEPAI con configurazione avanzata
onepai = ONEPAI(
    db_path="cognitive_analysis.db",
    vault_path="neural_treasures.vault",
    password="secure_cognitive_key_2024"
)

# 3. Configura l'osservazione cognitiva
model = CognitiveNet()
circuit_shadow = CircuitShadow(model.attention)
weight_ghost = WeightGhost(model.hidden)

# 4. Esegui analisi cognitiva
with onepai.observer.cognitive_context():
    for epoch in range(10):
        # Simula training/inference
        batch = torch.randn(32, 128)
        output = model(batch)
        
        # Cattura ombre cognitive
        shadows = circuit_shadow.capture_shadows()
        ghosts = weight_ghost.detect_phantoms()
        
        # Archivia tesori neurali
        if shadows.significance > 0.8:
            onepai.archive.store_treasure({
                'type': 'cognitive_pattern',
                'data': shadows,
                'epoch': epoch,
                'significance': shadows.significance
            })

# 5. Analizza i risultati
silence_map = onepai.silence_tracer.generate_map()
void_analysis = onepai.void_analyzer.analyze_gaps()
cognitive_summary = onepai.generate_report()

print(f"🧠 Neuroni attivi: {silence_map.active_neurons}")
print(f"🤫 Neuroni silenziosi: {silence_map.silent_neurons}")
print(f"🕳️ Vuoti cognitivi rilevati: {len(void_analysis.gaps)}")
print(f"💎 Tesori archiviati: {len(onepai.archive.list_treasures())}")
```

### 🛠️ Script da Riga di Comando

```bash
# Mappa le ombre cognitive di un modello
python scripts/map_shadows.py --model path/to/model.pth --output shadows.json --verbose

# Crea un tesoro cognitivo
python scripts/create_treasure.py --type insight --name "attention_pattern" --data analysis.json

# Decodifica pattern di silenzio
python scripts/decode_silence.py --model path/to/model.pth --threshold 0.01 --format yaml

# Gestisci archivi
python scripts/manage_archive.py --action backup --output backup_$(date +%Y%m%d).tar.gz

# Analisi completa
python scripts/run_analysis.py --model path/to/model.pth --comprehensive --save-report
```

## 🎯 Script Disponibili

| Script | Descrizione | Uso Principale |
|--------|-------------|----------------|
| `run_dashboard.py` | 🌐 Avvia dashboard web | Interfaccia grafica interattiva |
| `map_shadows.py` | 🗺️ Mappa ombre cognitive | Analisi pattern nascosti |
| `create_treasure.py` | 💎 Crea tesori neurali | Archiviazione insight |
| `decode_silence.py` | 🤫 Analizza silenzi | Rilevamento neuroni morti |
| `manage_archive.py` | 📦 Gestisce archivi | Backup e manutenzione |
| `run_analysis.py` | 🔬 Analisi completa | Workflow di analisi completo |
| `build_onepai.py` | 🏗️ Build del progetto | Compilazione e setup |

## Contribuire

Le contribuzioni sono benvenute! Si prega di leggere `CONTRIBUTING.md` per i dettagli.

## Licenza

Questo progetto è rilasciato sotto la licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.