# NeoCore™ - Next Generation CPU-Native Transformer


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)


[![PyPI version](https://badge.fury.io/py/neocore.svg)](https://badge.fury.io/py/neocore)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview

NeoCore is a state-of-the-art, CPU-optimized transformer architecture designed for edge computing and enterprise deployment. By leveraging advanced CPU-specific optimizations and modern architectural improvements, NeoCore achieves exceptional performance without requiring GPU acceleration.

### Key Features

- 🔋 **CPU-Native Design**: Optimized from the ground up for modern CPU architectures
- 🚄 **High-Performance**: Achieves up to 12.7K tokens/second on standard CPU hardware
- 🎯 **Memory Efficient**: Advanced caching and chunking strategies for optimal memory usage
- 🛠 **Enterprise Ready**: Production-grade implementation with comprehensive logging and monitoring
- 🔄 **Modern Architecture**: Incorporates Multi-Query Attention, RMSNorm, and Rotary Embeddings
- 📊 **Extensive Benchmarking**: Built-in performance profiling and optimization tools

## 🔧 Installation

```bash
pip install neocore
```

## 🏗 Architecture

NeoCore introduces several architectural innovations:

### Core Components

1. **Multi-Query Attention (MQA)**
```python
Q: [Batch, Seq, Heads, Head_Dim]  # Multiple query heads
K,V: [Batch, 1, Head_Dim]         # Single key/value
```

2. **RMSNorm for Stabilization**
```python
RMSNorm(x) = x * scale / sqrt(mean(x²) + ε)
```

3. **Block-wise Computation**
```
Input -> Chunked Processing -> Cache-Friendly Operations -> Output
```

### Performance Optimizations

#### Memory Access Pattern
```
┌──────────────────┐
│ Input Embedding  │
└────────┬─────────┘
         │
    ┌────▼────┐
    │ Chunk 1 │──┐
    └─────────┘  │
    ┌─────────┐  │
    │ Chunk 2 │──┼─► Parallel Processing
    └─────────┘  │
    ┌─────────┐  │
    │ Chunk N │──┘
    └─────────┘
```

## 💫 Key Innovations

### 1. Cache-Optimized Linear Operations
- Custom blocked matrix multiplication
- Adaptive chunk sizing
- Operation result caching

### 2. Efficient Attention Mechanism
```python
# Traditional vs NeoCore MQA
Traditional: O(N * H * D) memory
NeoCore:     O(N * D) memory
```

### 3. Advanced Position Encoding
- Rotary embeddings for enhanced position awareness
- Cache-friendly implementation
- Optimized for CPU SIMD operations

## 📊 Performance Metrics

| Batch Size | Sequence Length | Processing Time (ms) | Tokens/Second |
|------------|----------------|---------------------|---------------|
| 1          | 32             | 31.17              | 1,026        |
| 4          | 64             | 43.51              | 5,883        |
| 16         | 128            | 161.28             | 12,700       |

## 🚀 Quick Start

```python
from neocore import NoamConfig, CPUOptimizedNoamTransformer

# Initialize configuration
config = NoamConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    warmup_steps=4000,
    chunk_size=32
)

# Create model
model = CPUOptimizedNoamTransformer(config)

# Process input
output = model(input_ids)
```


## 🎯 Use Cases

- **Edge Computing**: Optimal for deployment on CPU-only edge devices
- **Enterprise Systems**: Reliable performance on standard server hardware
- **CI/CD Pipelines**: Efficient inference in production pipelines
- **Privacy-First Applications**: On-device processing without GPU requirements

## 🔬 Technical Details

### Memory Management
- Intelligent cache management system
- Adaptive chunk sizing based on input
- Memory-efficient attention patterns

### Threading Model
```python
Number of Threads = min(CPU_COUNT, MAX_EFFICIENT_THREADS)
Thread Pool Size = Adaptive based on workload
```

### Optimization Levels
1. **Level 1**: Basic CPU optimizations
2. **Level 2**: Cache-aware operations
3. **Level 3**: Advanced parallelization
4. **Level 4**: Full SIMD utilization

## 📈 Benchmarking

Run comprehensive benchmarks:
```bash
python -m neocore.benchmark --config benchmark_config.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📜 License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🌟 Acknowledgments

Built on modern transformer innovations with specific optimizations for CPU architectures. Special thanks to the research community for their groundbreaking work in efficient transformer designs.

---

## Citation

```bibtex
@software{neocore2024,
  title={NeoCore: CPU-Optimized Transformer Architecture},
  author={Kye Gomez},
  year={2024},
  publisher={GitHub},
  url={https://github.com/neocore/neocore}
}
```
