import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.progress import track

from .nn import NonagaNet, get_device
from .mcts import MCTSAgent, MCTSConfig
from .state import GameState
from .rules import apply_move, is_terminal, winner, legal_moves
from .encoder import encode_state

console = Console()

class ReplayBuffer(Dataset):
    def __init__(self) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[idx]
        
    def add(self, state_tensor: torch.Tensor, policy_tensor: torch.Tensor, value: torch.Tensor) -> None:
        self.examples.append((state_tensor, policy_tensor, value))


def generate_selfplay_data(net: NonagaNet, num_games: int, mcts_sims: int) -> ReplayBuffer:
    net.eval()
    buffer = ReplayBuffer()
    config = MCTSConfig(num_simulations=mcts_sims, temperature=1.0)
    agent = MCTSAgent(net, config)
    
    for g in track(range(num_games), description="Self-Play Games"):
        state = GameState.initial()
        history = []
        
        while not is_terminal(state, max_ply=150):
            l_moves = legal_moves(state)
            if not l_moves:
                break
                
            move, probs_list = agent.get_action_prob(state, temp=1.0)
            
            s_t = encode_state(state)
            p_t = torch.zeros((4, 19, 19), dtype=torch.float32)
            
            for mv, prob in zip(l_moves, probs_list):
                sq, sr = max(0, min(18, mv.slide.start[0])), max(0, min(18, mv.slide.start[1]))
                eq, er = max(0, min(18, mv.slide.end[0])), max(0, min(18, mv.slide.end[1]))
                rq, rr = max(0, min(18, mv.tile.remove_from[0])), max(0, min(18, mv.tile.remove_from[1]))
                pq, pr = max(0, min(18, mv.tile.place_to[0])), max(0, min(18, mv.tile.place_to[1]))
                
                p_t[0, sq, sr] += prob
                p_t[1, eq, er] += prob
                p_t[2, rq, rr] += prob
                p_t[3, pq, pr] += prob
                
            history.append((s_t, p_t, state.side_to_move))
            state = apply_move(state, move)
            
        won = winner(state)
        for s_t, p_t, color in history:
            if won is None:
                val = 0.0
            else:
                val = 1.0 if won == color else -1.0
            buffer.add(s_t, p_t, torch.tensor([val], dtype=torch.float32))
            
    return buffer


def train_epoch(net: NonagaNet, optimizer: optim.Optimizer, buffer: ReplayBuffer, batch_size: int) -> float:
    device = get_device()
    net.train()
    loader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    
    total_loss = 0.0
    for states, policies, values in loader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        optimizer.zero_grad()
        pi_logits, val_pred = net(states)
        
        v_loss = F.mse_loss(val_pred, values)
        
        pi_logits_flat = pi_logits.view(-1, 4, 361)
        policies_flat = policies.view(-1, 4, 361)
        
        log_probs = F.log_softmax(pi_logits_flat, dim=2)
        p_loss = -torch.sum(policies_flat * log_probs, dim=2).mean()
        
        loss = v_loss + p_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def train_pipeline(
    iterations: int = 10,
    games_per_iter: int = 20,
    mcts_sims: int = 200,
    epochs: int = 5,
    batch_size: int = 64,
    save_path: str = "nonagazero.pt"
) -> None:
    device = get_device()
    console.print(f"[bold green]Initializing Neural Network on {device}[/bold green]")
    # Using 10 blocks, 128 channels optimized for RTX 3090 RunPod as requested
    net = NonagaNet(num_res_blocks=10, num_channels=128).to(device)
    
    if os.path.exists(save_path):
        console.print(f"Loading existing checkpoint from {save_path}")
        net.load_state_dict(torch.load(save_path, map_location=device))
        
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for i in range(iterations):
        console.print(f"\n[bold magenta]--- AlphaZero Iteration {i+1}/{iterations} ---[/bold magenta]")
        
        buffer = generate_selfplay_data(net, games_per_iter, mcts_sims)
        console.print(f"Generated [bold yellow]{len(buffer)}[/bold yellow] state transitions.")
        
        for e in range(epochs):
            loss = train_epoch(net, optimizer, buffer, batch_size)
            console.print(f"  Epoch {e+1}/{epochs} - Loss: [bold cyan]{loss:.4f}[/bold cyan]")
            
        torch.save(net.state_dict(), save_path)
        console.print(f"Checkpoint saved to [bold white]{save_path}[/bold white]")
