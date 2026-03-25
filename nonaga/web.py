import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import RandomAgent, SearchAgent, SearchConfig
from .mcts import MCTSAgent, MCTSConfig
from .nn import NonagaNet, get_device
from .rules import apply_move, is_terminal, legal_moves, winner
from .state import BLACK, RED, GameState

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nonaga AI Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StateData(BaseModel):
    discs: List[Tuple[int, int]]
    red: List[Tuple[int, int]]
    black: List[Tuple[int, int]]
    forbidden_disc: Optional[Tuple[int, int]]
    side_to_move: int  # 1 for Red, 2 for Black


class AIConfigRequest(BaseModel):
    state: StateData
    ai_type: str  # "random", "heuristic", "alphazero"
    strength: str  # e.g., "3" for depth 3, or "nonagazero.pt"


def parse_state(data: StateData) -> GameState:
    forbidden = tuple(data.forbidden_disc) if data.forbidden_disc else None
    return GameState(
        discs=frozenset(tuple(x) for x in data.discs),
        red=frozenset(tuple(x) for x in data.red),
        black=frozenset(tuple(x) for x in data.black),
        forbidden_disc=forbidden,
        side_to_move=int(data.side_to_move),
    )


@app.post("/api/legal_moves")
def get_legal_moves(state_data: StateData):
    """Returns all legal moves for the current human player to highlight valid UI actions."""
    state = parse_state(state_data)
    moves = legal_moves(state)
    won = winner(state)
    
    res = []
    for mv in moves:
        res.append({
            "slide": {
                "start": mv.slide.start,
                "end": mv.slide.end
            },
            "tile": {
                "remove_from": mv.tile.remove_from,
                "place_to": mv.tile.place_to
            }
        })
    return {"legal_moves": res, "winner": won}


MODEL_CACHE = {}

@app.post("/api/ai_move")
def get_ai_move(req: AIConfigRequest):
    """Calculates the best move using the selected AI engine."""
    state = parse_state(req.state)
    if not legal_moves(state):
        raise HTTPException(status_code=400, detail="No legal moves available in this state.")
        
    try:
        if req.ai_type == "random":
            agent = RandomAgent()
            mv = agent.choose_move(state)
            
        elif req.ai_type == "heuristic":
            depth = int(req.strength)
            config = SearchConfig(max_depth=depth, time_limit=30.0)
            agent = SearchAgent(config=config)
            mv = agent.choose_move(state)
            
        elif req.ai_type == "alphazero":
            model_path = req.strength
            device = get_device()
            
            if model_path not in MODEL_CACHE:
                print(f"Loading model from {model_path} onto {device}...")
                net = NonagaNet(num_res_blocks=10, num_channels=128).to(device)
                if os.path.exists(model_path):
                    import torch
                    net.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    print(f"Warning: AlphaZero model {model_path} not found.")
                net.eval()
                MODEL_CACHE[model_path] = net
            else:
                net = MODEL_CACHE[model_path]
            
            config = MCTSConfig(num_simulations=400, temperature=0.0)
            agent = MCTSAgent(net, config)
            mv = agent.choose_move(state)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown AI type: {req.ai_type}")
            
    except Exception as e:
        import traceback
        with open("/root/nonaga_ai/error.log", "a") as f:
            f.write(traceback.format_exc())
            f.write("\n")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    return {
        "slide": {
            "start": mv.slide.start,
            "end": mv.slide.end
        },
        "tile": {
            "remove_from": mv.tile.remove_from,
            "place_to": mv.tile.place_to
        }
    }


# Mount the static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
