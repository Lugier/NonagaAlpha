from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .agents import Agent
from .encoder import encode_state, extract_move_probabilities
from .moves import CompoundMove
from .nn import NonagaNet, get_device
from .rules import apply_move, legal_moves, winner
from .state import GameState


class MCTSNode:
    def __init__(self, state: GameState | None, parent: MCTSNode | None = None, move: CompoundMove | None = None, prior_p: float = 0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: dict[CompoundMove, MCTSNode] = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_p = prior_p
        
        self.is_expanded = False
        
    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def expand(self, move_probs: list[tuple[CompoundMove, float]]) -> None:
        for mv, prob in move_probs:
            if mv not in self.children:
                self.children[mv] = MCTSNode(state=None, parent=self, move=mv, prior_p=prob)
        self.is_expanded = True
        
    def select_child(self, c_puct: float) -> tuple[CompoundMove, MCTSNode]:
        best_score = -math.inf
        best_child = None
        best_move = None
        
        sqrt_total_visits = math.sqrt(self.visit_count)
        for move, child in self.children.items():
            u = c_puct * child.prior_p * sqrt_total_visits / (1 + child.visit_count)
            # The value is from the child's perspective, so for the parent it is negated.
            score = -child.q_value + u 
            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
                
        assert best_child is not None and best_move is not None
        return best_move, best_child

    def backpropagate(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            # The parent is from the perspective of the other player, so value flips.
            self.parent.backpropagate(-value)


@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 1.25
    temperature: float = 1.0


class MCTSAgent(Agent):
    name = "mcts"
    
    def __init__(self, net: NonagaNet, config: MCTSConfig | None = None):
        self.net = net
        self.config = config or MCTSConfig()
        # Use the device the network is already on
        self.device = next(net.parameters()).device
        self.net.eval()
        
    @torch.no_grad()
    def get_action_prob(self, state: GameState, temp: float = 1.0) -> tuple[CompoundMove, list[float]]:
        root = MCTSNode(state)
        
        for _ in range(self.config.num_simulations):
            node = root
            # 1. Selection
            while node.is_expanded and node.children:
                _, node = node.select_child(self.config.c_puct)
                if node.state is None:
                    node.state = apply_move(node.parent.state, node.move)
                
            # 2. Evaluation / Check terminal
            won = winner(node.state)
            if won is not None:
                # If current player already won (rare/impossible since win ends turn before), val=1
                # Usually won != state.side_to_move, which means current player just lost, val=-1
                val = 1.0 if won == node.state.side_to_move else -1.0
                node.backpropagate(val)
                continue
                
            l_moves = legal_moves(node.state)
            if not l_moves:
                node.backpropagate(0.0)  # Draw by no legal moves
                continue
                
            # 3. Expansion using Neural Network
            t = encode_state(node.state).unsqueeze(0).to(self.device)
            pi_logits, val_tensor = self.net(t)
            
            probs, moves = extract_move_probabilities(pi_logits[0], l_moves)
            prob_list = probs.tolist()
            move_probs = list(zip(moves, prob_list))
            
            node.expand(move_probs)
            
            # Backpropagate the network's value estimation
            v = val_tensor.item()
            node.backpropagate(v)
            
        # 4. Action Selection
        if not root.children:
            l_moves = legal_moves(state)
            if not l_moves:
                # Should not happen if web.py checked, but for safety:
                return None, [0.0] 
            mv = random.choice(l_moves)
            return mv, [1.0 / len(l_moves)] * len(l_moves)
            
        visits = [child.visit_count for _, child in root.children.items()]
        moves = list(root.children.keys())
        
        if temp == 0:
            best_idx = visits.index(max(visits))
            # Just returning a dummy prob array where the chosen move is 1.0
            probs_out = [1.0 if i == best_idx else 0.0 for i in range(len(moves))]
            return moves[best_idx], probs_out
            
        v_temp = [v ** (1.0 / temp) for v in visits]
        total = sum(v_temp)
        probs_out = [v / total for v in v_temp]
        
        idx = random.choices(range(len(moves)), weights=probs_out, k=1)[0]
        return moves[idx], probs_out
        
    def choose_move(self, state: GameState) -> CompoundMove:
        mv, _ = self.get_action_prob(state, temp=0.0) # Greedy choice for actual gameplay
        return mv
