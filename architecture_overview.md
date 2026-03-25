# Nonaga Engine Architecture Overview

This document provides a high-level explanation of how the Nonaga AI engine works.

## 1. Core Principles
Nonaga is a **zero-sum, perfect information game**. This means:
- No hidden information (like cards in hand).
- No randomness (like dice).
- A gain for one player is a loss for the other.

## 2. Component Design

### 🔷 Geometry (`geometry.py`)
Nonaga uses a **hexagonal grid**. To make calculations easy, we use **Axial Coordinates** (q, r). 
- Every disc has a unique (q, r) address.
- Neighbors and straight lines are calculated using simple vector additions.
- **Compound Move Logic**: The engine must handle both the piece sliding and the disc relocation. 

### 📜 Rules & State (`rules.py`, `state.py`)
- **Immutability**: Every game state is immutable. Making a move creates a *new* state. This is crucial for search efficiency and undoing moves.
- **Move Generation**: 
  - First, it finds all legal "Piece Slides".
  - For each slide, it finds all legal "Tile Relocations".
  - A single turn is a combination of both.
- **Slidability**: To remove a disc, the code checks if it can "physically" slide out without bumping into neighbors. This is done via a geometric test on the neighboring directions.

### 🧠 The Neural Network ("The Brain") (`nn.py`)
For the advanced AlphaZero mode, the engine uses a **Deep Residual CNN**:
- **Architecture**: 10 Residual Blocks with 128 channels each.
- **Heads**:
  - **Policy Head**: Predicts the probability distribution over all possible moves (approx. 336 dimensions for Nonaga).
  - **Value Head**: Benchmarks the current state between -1 (Loss) and +1 (Win).
- **State Encoding**: The 19-disc hex grid is mapped to a 4-plane tensor representing piece positions and movement constraints.

### 🔍 Search Engines

#### A. Alpha-Beta Pruning (`search.py`)
- Classical minimax search with iterative deepening.
- Optimized for quick, heuristic-based Play.
- **Current Limit**: Set to 30.0 seconds for deep analysis.

#### B. AlphaZero MCTS (`mcts.py`)
- **Simulations**: Default 400 per move for high-quality play.
- **Logic**: Uses the neural network to guide the tree search via the PUCT (Predictor Upper Confidence Bound applied to Trees) formula.
- **Efficiency**: Keeps the best model cached in GPU VRAM for sub-second response times.

## 3. Parallel Training Pipeline (`train_nn_parallel.py`)
The engine features a massive parallelization suite:
1. **Self-Play Workers**: Multiple processes generate games simultaneously.
2. **Replay Buffer**: Stores millions of state-action-value triplets.
3. **Training Loop**: Continuously optimizes the `NonagaNet` using the latest self-play data via Stochastic Gradient Descent (SGD) with momentum.

## 4. Web Interface (`web.py`)
A modern FastAPI backend serves the engine through a visual board.
- **Static Assets**: HTML5/CSS3/Vanilla JS frontend.
- **API**: High-concurrency endpoints for legal moves and AI decision-making.

## 5. The "Nonaga Loop"
1. **Input**: Current board state.
2. **Search**: AI simulates thousands of possible future moves.
3. **Score**: Each terminal state is scored by the evaluation function.
4. **Decision**: AI chooses the move that leads to the best guaranteed score, assuming the opponent also plays perfectly.
