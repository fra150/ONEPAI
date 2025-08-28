# ONEPAI 🧠✨

[![Version](https://img.shields.io/badge/version-0.1.0--alpha-blue.svg)](https://github.com/brainverse/onepai)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

> **The Framework for Mapping the Invisible in AI** 🔍

ONEPAI is a revolutionary framework for analyzing, mapping, and understanding hidden neural dynamics in artificial intelligence models. It explores the "silences," "voids," and "shadows" that lie behind every AI decision.

## 🎯 What ONEPAI Does

ONEPAI reveals the **invisible** in AI through:

- **🔍 Shadow Mapping**: Maps hidden influences between neurons
- **🤫 Silence Analysis**: Analyzes "unexpressed" thoughts of AI
- **🕳️ Void Detection**: Detects empty spaces in neural representations
- **💎 Treasure Creation**: Generates encrypted archives (.onepai) of discoveries
- **📊 Circuit Tracing**: Traces decision paths in transformers
- **🧮 Quantum Mode**: Advanced analysis with quantum mode

## 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/brainverse/onepai.git
cd onepai

# Install dependencies
pip install -e .

# Start the dashboard
python scripts/run_dashboard.py
```

Open http://localhost:8001 to access the **Circuit Tracer Dashboard**.

## 📖 Quick Guide

### 1. Complete Model Analysis

```python
from onepai import ONEPAI_Observer
from onepai.analysis import ShadowQueryEngine, SilenceMetricsCalculator

# Initialize the observer
observer = ONEPAI_Observer()

# Load a model
model = observer.load_model("gpt2-small")

# Run complete analysis
results = observer.analyze_all(model, [
    "shadow",      # Shadow map
    "silence",     # Silence metrics
    "void",        # Void statistics
    "treasure"     # Treasure creation
])

print(f"Analysis completed: {len(results)} artifacts found")
```

### 2. Shadow Mapping

```python
from onepai.shadows import ShadowMapper

# Create shadow map
shadow_mapper = ShadowMapper(sensitivity=0.7)
shadow_map = shadow_mapper.map_model_shadows(model)

# Save the map
shadow_map.save("data/shadows/gpt2_shadows.shadow")
```

### 3. Silence Analysis

```python
from onepai.silence import SilenceTracer

# Trace silences
silence_tracer = SilenceTracer()
silence_patterns = silence_tracer.trace_silence(
    model, 
    input_text="What are you not telling me?"
)

print(f"Silent neurons: {len(silence_patterns.silent_neurons)}")
print(f"Suppressed thoughts: {len(silence_patterns.suppressed_thoughts)}")
```

### 4. Treasure Creation

```python
from onepai.treasures import TreasureCreator

# Create an encrypted treasure
treasure_creator = TreasureCreator()
treasure = treasure_creator.create_treasure(
    name="gpt2_analysis",
    data={
        "shadows": shadow_map,
        "silences": silence_patterns,
        "metadata": {"model": "gpt2-small", "date": "2024-01-15"}
    }
)

# Save as encrypted .onepai file
treasure.save("data/treasures/gpt2_analysis.onepai")
```

## 🏗️ System Architecture

```
ONEPAI/
├── 🧠 Core/                    # Main engine
│   ├── observer.py             # ONEPAI Observer
│   ├── metrics.py              # Silence and void metrics
│   └── archive.py              # Archive management
├── 🔍 Analysis/                # Analysis modules
│   ├── shadow_query.py         # Shadow queries
│   ├── absence_patterns.py     # Absence patterns
│   ├── silence_metrics.py      # Silence metrics calculation
│   └── void_statistics.py      # Void statistics
├── 🧮 Circuit Tracer/          # Circuit tracing
│   ├── graph.py                # Neural graph representation
│   ├── influence.py            # Influence calculation
│   └── pruning.py              # Graph pruning
├── 💾 Memory/                  # Memory systems
│   ├── unsaid_vault.py         # Unsaid vault
│   ├── undone_registry.py      # Undone registry
│   └── unthought_space.py      # Unthought space
├── 🔐 Crypto/                  # Cryptography
│   └── crypto_manager.py       # Encryption management
└── 📊 Dashboard/               # Web interface
    ├── app.py                  # Flask server
    ├── static/                 # Static assets
    └── templates/              # HTML templates
```

## 🛠️ Available Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `run_dashboard.py` | Start web dashboard | `python scripts/run_dashboard.py` |
| `run_analysis.py` | Complete model analysis | `python scripts/run_analysis.py --model gpt2` |
| `map_shadows.py` | Shadow mapping | `python scripts/map_shadows.py --input model.pt` |
| `create_treasure.py` | Treasure creation | `python scripts/create_treasure.py --name analysis` |
| `decode_silence.py` | Archive management | `python scripts/decode_silence.py list --archive-type all` |
| `build_onepai.py` | Project build | `python scripts/build_onepai.py` |

## ⚙️ Configuration

ONEPAI uses a centralized configuration system in `config/settings.py`:

```python
from onepai.config import get_settings

settings = get_settings()
print(f"Shadow sensitivity: {settings.shadow_sensitivity}")
print(f"Silence threshold: {settings.silence_threshold}")
print(f"Quantum mode: {settings.quantum_mode}")
```

### Main Parameters

- **shadow_sensitivity**: Shadow mapping sensitivity (0.0-1.0)
- **silence_threshold**: Silence detection threshold (0.0-1.0)
- **void_detection_depth**: Void analysis depth
- **treasure_encryption**: Enable .onepai file encryption
- **quantum_mode**: Quantum mode for advanced analysis

## 🗺️ Roadmap

### ✅ Version 0.1.0-alpha (Current)
- [x] Core ONEPAI framework
- [x] Shadow mapping system
- [x] Silence metrics analysis
- [x] Circuit tracer for transformers
- [x] Interactive web dashboard
- [x] Treasure encryption system
- [x] Centralized configuration

### 🚧 Version 0.2.0-beta (Q2 2024)
- [ ] **Quantum Mode**: Complete quantum mode implementation
- [ ] **Multi-Model Support**: Support for BERT, T5, LLaMA
- [ ] **Real-time Analysis**: Real-time analysis during inference
- [ ] **Advanced Visualizations**: 3D neural shadow graphs
- [ ] **REST API**: Endpoints for external integration
- [ ] **Plugin System**: Plugin system for extensions

### 🎯 Version 0.3.0 (Q3 2024)
- [ ] **Distributed Analysis**: Distributed cluster analysis
- [ ] **Memory Optimization**: Optimization for large models (70B+)
- [ ] **Comparative Analysis**: Comparison between different models
- [ ] **Anomaly Detection**: Neural anomaly detection
- [ ] **Export Formats**: Support for ONNX, TensorRT

### 🚀 Version 1.0.0 (Q4 2024)
- [ ] **Production Ready**: Stability for production use
- [ ] **Enterprise Features**: Enterprise functionality
- [ ] **Cloud Integration**: Integration with AWS, GCP, Azure
- [ ] **Documentation**: Complete documentation
- [ ] **Certification**: Security certifications

## 📋 TODO List

### 🔥 High Priority
- [ ] **Implement build_onepai.py**: Missing build script
- [ ] **Complete API documentation**: Empty `docs/api.md` file
- [ ] **Create detailed architecture**: Empty `docs/architecture.md` file
- [ ] **Implement test suite**: Empty `tests/` directory
- [ ] **Add practical examples**: Missing `examples/` directory
- [ ] **Optimize performance**: Profiling and optimizations

### 🔧 Medium Priority
- [ ] **Improve error handling**: More robust error management
- [ ] **Add logging**: Structured logging system
- [ ] **Implement cache**: Cache system for analysis
- [ ] **Create CLI tool**: Unified command-line tool
- [ ] **Add metrics**: Performance and quality metrics
- [ ] **Implement backup**: Automatic backup system

### 📚 Low Priority
- [ ] **Documentation translation**: Multi-language support
- [ ] **Video tutorials**: YouTube video tutorials
- [ ] **Community features**: Forum and Discord
- [ ] **Mobile app**: Mobile monitoring app
- [ ] **Integration tests**: Complete integration tests

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test with coverage
python -m pytest tests/ --cov=onepai --cov-report=html

# Specific tests
python -m pytest tests/test_shadows.py -v
```

**Note**: The test suite is currently under development. Contributions welcome!

## 🤝 Contributing

ONEPAI is an open source project and welcomes contributions!

### How to Contribute

1. **Fork** the repository
2. **Create** a branch for your feature (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Areas

- 🐛 **Bug fixes**: Bug resolution
- ✨ **New features**: New functionality implementation
- 📚 **Documentation**: Documentation improvement
- 🧪 **Testing**: Test writing
- 🎨 **UI/UX**: Dashboard improvement
- 🔧 **Performance**: Optimizations

## 📄 License

This project is released under the MIT license. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Francesco Bulla (Brainverse)**
- 🌐 Website: [brainverse.ai](https://brainverse.ai)
- 📧 Email: francesco@brainverse.ai
- 🐦 Twitter: [@brainverse_ai](https://twitter.com/brainverse_ai)

## 🙏 Acknowledgments

- Open source community for inspiration
- AI interpretability researchers
- Contributors and beta testers

## 🔗 Useful Links

- 📖 [Complete Documentation](docs/)
- 🐛 [Report Bug](https://github.com/brainverse/onepai/issues)
- 💡 [Request Feature](https://github.com/brainverse/onepai/issues/new?template=feature_request.md)
- 💬 [Discussions](https://github.com/brainverse/onepai/discussions)
- 📺 [Video Tutorials](https://youtube.com/@brainverse)

---

<div align="center">

**"Every AI silence hides a universe of possibilities"** 🌌

*ONEPAI - Mapping the Invisible in Artificial Intelligence*

</div>
