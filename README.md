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

## 🏠 Web Interface
NonagaAlpha comes with a built-in **Web UI** for interactive play.
- **Port:** Default is 8888.
- **Backend:** FastAPI (Python).
- **Frontend:** Vanilla JS / HTML5.
- **Features:** Real-time move visualization, legal move highlighting, and AI selection.

To launch the web interface:
```bash
python3 -m nonaga.cli web --port 8888
```

---

## ☁️ RunPod Deployment
The engine is pre-configured for **RunPod** instances (Ubuntu 22.04 + CUDA).
1. **Connect:** Use the provided SSH credentials.
2. **Setup:** Run `scripts/runpod_setup.sh` to install dependencies.
3. **Start Game:** Use the `nonaga_web` tmux session to keep the server running 24/7.
4. **Access:** Click the "Connect" button in RunPod and select "HTTP Service [Port 8888]".

---

## 📦 Large File Storage (Git LFS)
This repository contains over **4.4GB** of trained model weights and replay buffers. 
**Note:** You must have [Git LFS](https://git-lfs.github.com/) installed to download the `.pt` files correctly.

```bash
# Install Git LFS
git lfs install
# Pull the large weight files
git lfs pull
```

---

## 🧠 Project Architecture
See [architecture_overview.md](architecture_overview.md) for a deep dive into the Neural Network and MCTS logic.

---

## 📜 Implementation Notes
NonagaAlpha handles the complex "edge-disc sliding" rule using a strong geometric approximation (half-plane contact test). The AlphaZero model was trained over **40 iterations** on a 3090 GPU, producing a highly competitive agent.

---

## 🤝 Contribution
Contributions to the search algorithm or neural architecture are welcome! Feel free to open an issue or submit a pull request.

**Developed with ❤️ and High-End Hardware for Nonaga Enthusiasts.**
