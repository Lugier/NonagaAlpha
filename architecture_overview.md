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

### 🧠 The "Brain" (`eval.py`)
Since the AI cannot see until the end of every game (it's too complex), it uses a **Heuristic Evaluation Function**. It gives a score to any position:
- **Connectivity**: Higher score if your 3 pieces are close together (ready to form a win).
- **Mobility**: Higher score if your pieces have many sliding options.
- **Disruption**: Penalizes the opponent for having good connectivity.

### 🔍 The Search Engine (`search.py`)
The AI thinks ahead using **Alpha-Beta Pruning** (a highly optimized version of Minimax).
- **Iterative Deepening**: It search 1 ply deep, then 2, then 3... until time runs out.
- **Transposition Tables**: It remembers positions it has already seen (and their symmetries!) so it doesn't have to re-calculate them.
- **Killer Heuristic**: It tries "strong" moves (that caused cutoffs in other branches) first to save time.

## 3. The "Nonaga Loop"
1. **Input**: Current board state.
2. **Search**: AI simulates thousands of possible future moves.
3. **Score**: Each terminal state is scored by the evaluation function.
4. **Decision**: AI chooses the move that leads to the best guaranteed score, assuming the opponent also plays perfectly.
