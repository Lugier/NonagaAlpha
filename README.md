# NonagaAlpha 🧠🚀

**NonagaAlpha** is a state-of-the-art **AlphaZero-inspired AI engine** for the board game **Nonaga**. It is meticulously optimized for high-performance training on massive hardware setups, specifically targeting the **NVIDIA RTX 3090** and high-core-count **AMD Threadripper** processors.

## 🌟 Key Features

- **AlphaZero Core:** Implements the AlphaZero paradigm with a Deep Convolutional Neural Network (CNN) and Monte Carlo Tree Search (MCTS).
- **High-Performance Parallelization:** Massive self-play throughput using optimized multiprocessing (forkserver) and CUDA.
- **Hardware Optimized:**
  - **Lazy Tree Expansion:** Reduces CPU/GPU overhead by deferred state calculation.
  - **CUDA Latency Fix:** Optimized GPU-to-CPU data transfers by batching and offloading non-tensor operations to CPU efficiently.
- **Complete Rule Engine:** Full implementation of Nonaga's dynamic 19-disc board, including compound moves, edge-disc relocation, and connectivity win detection.
- **Interactive CLI:** Comprehensive command-line interface for human-vs-AI play, self-play arenas, and parallel training.
- **Native Performance:** Optimized for both **Apple Silicon (MPS)** and **NVIDIA (CUDA)** backends.

---

## 🚀 Performance Benchmarks (RTX 3090 + Threadripper)

| Feature | Performance |
| :--- | :--- |
| **GPU Utilization** | ~90% during parallel self-play |
| **Parallel Workers** | Supports 32+ simultaneous game instances |
| **Search Speed** | Optimized MCTS with thousands of simulations per second |
| **Model Architecture** | 10 Residual Blocks with 128 channels |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch (CUDA supported for training)
- Rich (for advanced CLI progress tracking)

### Clone & Install
```bash
git clone https://github.com/Lugier/NonagaAlpha.git
cd NonagaAlpha
pip install torch rich pytest
```

---

## 🎮 How to Use

### Parallel AlphaZero Training
To start the massive parallel training pipeline on a high-end setup (e.g., RunPod):
```bash
python3 -m nonaga.cli az-train-parallel --workers 32 --games 64 --sims 100
```

### Human vs. AI
Test your skills against the trained neural network:
```bash
python3 -m nonaga.cli human --human red --depth 3 --time-limit 1.0
```

### Self-Play Arena
Benchmark different models against each other:
```bash
python3 -m nonaga.cli arena --games 20 --depth 3 --time-limit 0.5
```

---

## 🧪 Testing
The engine is rigorously tested with regression tests for the rule engine and search logic:
```bash
pytest tests/
```

---

## 🧠 Project Architecture

- `nonaga/nn.py`: The Deep CNN architecture optimized for game state evaluation.
- `nonaga/mcts.py`: Monte Carlo Tree Search implementation with lazy expansion.
- `nonaga/train_nn_parallel.py`: The master training pipeline for high-throughput self-play.
- `nonaga/geometry.py`: Hexadecimal coordinate system and graph-theoretic win detection.
- `nonaga/rules.py`: The core game rule engine.

---

## 📜 Implementation Notes
NonagaAlpha handles the complex "edge-disc sliding" rule using a strong geometric approximation (half-plane contact test) which is regression-tested for correctness. The Connectivity Win is detected by solving for 3-node connected subgraphs in the occupied hex-grid.

---

## 🤝 Contribution
Contributions to the search algorithm or neural architecture are welcome! Feel free to open an issue or submit a pull request.

**Developed with ❤️ and High-End Hardware for Nonaga Enthusiasts.**
