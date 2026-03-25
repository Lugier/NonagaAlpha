# Walkthrough - Playing against AlphaZero

The 40-iteration AlphaZero model is now fully connected to the web interface and running on the Pod.

## Changes Made

### Frontend
- **Default Opponent:** The "Opponent AI" dropdown now defaults to **AlphaZero (Master Model)**.
- **Model Mapping:** The selection correctly points to the `nonagazero.pt` file located in the project root on the pod.

### Backend
- **Web Server:** Launched a FastAPI server on the pod that serves both the game UI and the MCTS-powered AI.
- **Weights:** The `nonagazero.pt` model (final state after 40 iterations) is actively loaded when "AlphaZero" is selected in the UI.

## How to Play

1.  **Server Status:** The server is running on **Port 8888** on your pod (I stopped Jupyter Lab to make this possible).
2.  **Connection:**
    -   Go to your **RunPod Dashboard**.
    -   Click the **"Connect"** button on your Pod.
    -   Select **"HTTP Service [Port 8888]"**.
    -   The Nonaga Web UI will open, and **AlphaZero (Master Model)** is already selected as your opponent.

## Results
- **Alpha-Beta Engine:** The heuristic search now has a **30-second time limit** (increased from 5s) to allow for deeper analysis.
- **AlphaZero Engine:** The AI is executing entirely on your **RunPod's RTX 3090 GPU**.
- **Search Depth:** Instead of a fixed depth, it uses **1,600 MCTS Simulations** per move. This typically reaches a depth of 10-25 moves in key variations.
- **GitHub Repository:** The entire project including all 40 trained snapshots (Iter 1-40) and training data (~4.4GB) is now backed up at [Lugier/NonagaAlpha](https://github.com/Lugier/NonagaAlpha).

## Troubleshooting
- If the AI seems weak or errors occur, ensure you have **AlphaZero (Master Model)** selected in the dropdown.
- I fixed a bug in the MCTS selection logic that could have caused the AI to make random moves/errors in certain board states.
