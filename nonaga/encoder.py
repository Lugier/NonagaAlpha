from __future__ import annotations

import torch
import torch.nn.functional as F

from .state import RED, BLACK, GameState
from .moves import CompoundMove


def encode_state(state: GameState) -> torch.Tensor:
    """
    Returns a (5, 19, 19) float32 tensor representing the canonical board state.
    Coord mappings: (q, r) are strictly >= 0 due to state.normalized().
    The maximum dimension theoretically for a 19-node graph is 18.
    """
    t = torch.zeros((5, 19, 19), dtype=torch.float32)
    
    for q, r in state.discs:
        if 0 <= q < 19 and 0 <= r < 19:
            t[0, q, r] = 1.0
            
    for q, r in state.red:
        if 0 <= q < 19 and 0 <= r < 19:
            t[1, q, r] = 1.0
            
    for q, r in state.black:
        if 0 <= q < 19 and 0 <= r < 19:
            t[2, q, r] = 1.0
            
    if state.forbidden_disc is not None:
        q, r = state.forbidden_disc
        if 0 <= q < 19 and 0 <= r < 19:
            t[3, q, r] = 1.0
            
    if state.side_to_move == RED:
        t[4, :, :] = 1.0
    else:
        t[4, :, :] = -1.0
        
    return t


def extract_move_probabilities(
    pi_tensor: torch.Tensor, legal_moves: list[CompoundMove]
) -> tuple[torch.Tensor, list[CompoundMove]]:
    """
    pi_tensor: (4, 19, 19)
    Extracts logit for each move and converts them to a probability distribution over the legal moves.
    logit(move) = pi[0, sq, sr] + pi[1, eq, er] + pi[2, rq, rr] + pi[3, pq, pr]
    """
    if not legal_moves:
        return torch.tensor([]), []
        
    logits = []
    pi_cpu = pi_tensor.cpu()
    
    for mv in legal_moves:
        sq, sr = mv.slide.start
        eq, er = mv.slide.end
        rq, rr = mv.tile.remove_from
        pq, pr = mv.tile.place_to
        
        # Safely constrain within 19x19 bounding if some freak anomaly happens
        sq, sr = max(0, min(18, sq)), max(0, min(18, sr))
        eq, er = max(0, min(18, eq)), max(0, min(18, er))
        rq, rr = max(0, min(18, rq)), max(0, min(18, rr))
        pq, pr = max(0, min(18, pq)), max(0, min(18, pr))
        
        val = (
            pi_cpu[0, sq, sr]
            + pi_cpu[1, eq, er]
            + pi_cpu[2, rq, rr]
            + pi_cpu[3, pq, pr]
        )
        logits.append(val)
        
    logits_tensor = torch.stack(logits)
    probs = F.softmax(logits_tensor, dim=0)
    
    return probs, legal_moves
