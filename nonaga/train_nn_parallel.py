import torch
import torch.nn.functional as F
import multiprocessing as mp
import random
import os
import time
import sys
import warnings
from contextlib import contextmanager

# Suppress PyTorch warnings for a cleaner UI
warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text

from .state import GameState, RED, BLACK
from .rules import initial_state, apply_move, is_terminal, winner, legal_moves
from .mcts import MCTSAgent, MCTSConfig
from .nn import NonagaNet, get_device
from .encoder import encode_state

console = Console()

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

def run_single_game(model_path, mcts_sims, device_type):
    """Worker function for a single parallel game with full policy encoding."""
    torch.set_num_threads(1)
    device = torch.device(device_type)
    net = NonagaNet(num_res_blocks=10, num_channels=128).to(device)
    
    # Load weights safely
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
        net.load_state_dict(sd)
    except Exception:
        sd = torch.load(model_path, map_location=device, weights_only=False)
        net.load_state_dict(sd)
    net.eval()
    
    agent = MCTSAgent(net, MCTSConfig(num_simulations=mcts_sims))
    state = GameState.initial()
    history = []
    
    with torch.no_grad():
        while not is_terminal(state, max_ply=150):
            l_moves = legal_moves(state)
            if not l_moves: break
            
            # Get move and entire probability distribution from MCTS
            move, probs_list = agent.get_action_prob(state, temp=1.0)
            
            s_t = encode_state(state)
            p_t = torch.zeros((4, 19, 19), dtype=torch.float32)
            
            # Correctly map probabilities into the 4-channel policy tensor
            for mv, prob in zip(l_moves, probs_list):
                sq, sr = max(0, min(18, mv.slide.start[0])), max(0, min(18, mv.slide.start[1]))
                eq, er = max(0, min(18, mv.slide.end[0])), max(0, min(18, mv.slide.end[1]))
                rq, rr = max(0, min(18, mv.tile.remove_from[0])), max(0, min(18, mv.tile.remove_from[1]))
                pq, pr = max(0, min(18, mv.tile.place_to[0])), max(0, min(18, mv.tile.place_to[1]))
                
                p_t[0, sq, sr] += prob
                p_t[1, eq, er] += prob
                p_t[2, rq, rr] += prob
                p_t[3, pq, pr] += prob
                
            history.append((s_t.numpy(), p_t.numpy(), state.side_to_move))
            state = apply_move(state, move)
        
    won = winner(state)
    results = []
    for s_np, p_np, color in history:
        if won is None:
            val = 0.0
        else:
            val = 1.0 if won == color else -1.0
        results.append((s_np, p_np, val))
    return results

class Dashboard:
    def __init__(self, iterations, games_per_iter, workers):
        self.iterations = iterations
        self.games_per_iter = games_per_iter
        self.workers = workers
        self.current_iter = 0
        self.finished_games = 0
        self.total_buffer = 0
        self.last_loss = 0.0
        self.last_v_loss = 0.0
        self.last_p_loss = 0.0
        self.logs = ["Training Initialized"]
        self.start_time = time.time()
        
    def add_log(self, msg):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.logs) > 10:
            self.logs.pop(0)

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="logs", ratio=1)
        )
        
        # Header
        layout["header"].update(Panel(Text(f"🧠 NONAGA ALPHA-ZERO TRAINING DASHBOARD", style="bold white on blue", justify="center"), border_style="blue"))
        
        # Stats Table
        stats_table = Table(box=None, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold yellow")
        stats_table.add_row("Master Iteration", f"{self.current_iter}/{self.iterations}")
        stats_table.add_row("Self-Play Progress", f"{self.finished_games}/{self.games_per_iter} Games")
        stats_table.add_row("Transitions In Buffer", f"{self.total_buffer}")
        stats_table.add_row("Policy-Value Loss", f"P: {self.last_p_loss:.4f} V: {self.last_v_loss:.4f}")
        elapsed = int(time.time() - self.start_time)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        stats_table.add_row("Active Runtime", f"[bold white]{time_str}[/bold white]")
        
        layout["progress"].update(Panel(stats_table, title="[bold green]📊 Statistics[/bold green]", border_style="green"))
        
        # Event logs
        log_text = Text("\n".join(self.logs), style="dim white")
        layout["logs"].update(Panel(log_text, title="[bold magenta]📜 Event Log[/bold magenta]", border_style="magenta"))
        
        # Footer
        footer_text = Text(f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", style="bold cyan", justify="center")
        layout["footer"].update(Panel(footer_text, border_style="cyan"))
        
        return layout


def _plain_status_line(dash: Dashboard) -> str:
    return (
        f"[nonaga] iter {dash.current_iter}/{dash.iterations} | "
        f"games {dash.finished_games}/{dash.games_per_iter} | "
        f"buffer {dash.total_buffer} | "
        f"loss {dash.last_loss:.4f} (p {dash.last_p_loss:.4f} v {dash.last_v_loss:.4f}) | "
        f"{int(time.time() - dash.start_time)}s elapsed"
    )


@contextmanager
def _training_ui(dash: Dashboard, *, plain_log: bool):
    if plain_log:
        console.print(
            "[bold cyan]Plain log mode[/bold cyan] — status lines go to stdout (good for tee / RunPod log view)."
        )
        yield
    else:
        with Live(dash, refresh_per_second=2, console=console) as _live:
            yield


def train_pipeline_parallel(
    iterations=40,
    games_per_iter=128,
    mcts_sims=150,
    epochs=10,
    batch_size=1024,
    save_path="nonagazero.pt",
    num_workers=4,
    *,
    plain_log: bool | None = None,
):
    device = get_device()
    net = NonagaNet(num_res_blocks=10, num_channels=128).to(device)
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path, map_location=device))
    
    buffer = ReplayBuffer(200000)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    if plain_log is None:
        plain_log = (not sys.stdout.isatty()) or os.environ.get("NONAGA_PLAIN_LOG", "").strip() in ("1", "true", "yes")

    dash = Dashboard(iterations, games_per_iter, num_workers)
    
    # Load existing buffer if available to resume progress
    BUFFER_PATH = "nonaga_buffer.pt"
    if os.path.exists(BUFFER_PATH):
        dash.add_log("Resuming from saved buffer...")
        try:
            buffer.buffer = torch.load(BUFFER_PATH, weights_only=False)
            dash.total_buffer = len(buffer.buffer)
        except Exception:
            dash.add_log("Buffer load failed.")

    log_every_games = max(1, min(10, games_per_iter // 10 or 1))

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        with _training_ui(dash, plain_log=plain_log):
            for i in range(iterations):
                dash.current_iter = i + 1
                dash.finished_games = 0
                dash.add_log(f"Iteration {i+1} Start")
                if plain_log:
                    print(
                        f"[nonaga] === iteration {i + 1}/{iterations} start | "
                        f"self-play {games_per_iter} games | {num_workers} workers ===",
                        flush=True,
                    )

                torch.save(net.state_dict(), save_path)
                all_results = []

                results_async = [pool.apply_async(run_single_game, (save_path, mcts_sims, "cpu")) for _ in range(games_per_iter)]

                iter_start_time = time.time()
                while len(results_async) > 0:
                    time.sleep(0.5 if plain_log else 1.0)
                    to_remove = []
                    for r in results_async:
                        if r.ready():
                            try:
                                res = r.get()
                                all_results.extend(res)
                                dash.finished_games += 1
                                if plain_log and (
                                    dash.finished_games % log_every_games == 0
                                    or dash.finished_games == games_per_iter
                                ):
                                    print(_plain_status_line(dash), flush=True)
                            except Exception as e:
                                dash.add_log(f"Worker Error: {str(e)}")
                            to_remove.append(r)

                    for r in to_remove:
                        results_async.remove(r)
                    
                    # Safety threshold: if 95% games are done and 25 mins passed, skip the slow ones
                    if (time.time() - iter_start_time > 1500 and len(results_async) <= 4):
                        dash.add_log(f"Skipping {len(results_async)} slow games.")
                        break

                buffer.add(all_results)
                dash.total_buffer = len(buffer.buffer)
                dash.add_log(f"Buffer now {len(buffer.buffer)} samples.")
                
                # Save buffer to disk to prevent data loss on pod restart
                torch.save(buffer.buffer, BUFFER_PATH)

                net.train()
                for epoch in range(epochs):
                    samples = buffer.sample(batch_size)
                    if not samples:
                        continue

                    s_batch = torch.stack([torch.from_numpy(s) for s, p, v in samples]).to(device)
                    p_batch = torch.stack([torch.from_numpy(p) for s, p, v in samples]).to(device)
                    v_batch = torch.tensor([v for s, p, v in samples], dtype=torch.float32).unsqueeze(1).to(device)

                    optimizer.zero_grad()
                    pi_logits, v_pred = net(s_batch)
                    
                    # Multi-channel Cross Entropy Loss for Policy
                    pi_logits_flat = pi_logits.view(-1, 4, 361)
                    target_policy_flat = p_batch.view(-1, 4, 361)
                    log_probs = F.log_softmax(pi_logits_flat, dim=2)
                    p_loss = -torch.sum(target_policy_flat * log_probs, dim=2).mean()
                    
                    # MSE for Value
                    v_loss = F.mse_loss(v_pred, v_batch)
                    loss = v_loss + p_loss
                    loss.backward()
                    optimizer.step()

                    dash.last_loss = loss.item()
                    dash.last_v_loss = v_loss.item()
                    dash.last_p_loss = p_loss.item()
                    if plain_log and (epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0):
                        print(
                            f"[nonaga] epoch {epoch + 1}/{epochs} loss={loss.item():.4f} "
                            f"v={v_loss.item():.4f} p={p_loss.item():.4f}",
                            flush=True,
                        )

                torch.save(net.state_dict(), save_path)
                snapshot_path = save_path.replace(".pt", f"_iter_{i+1}.pt")
                torch.save(net.state_dict(), snapshot_path)
                dash.add_log(f"Iteration {i+1} saved.")
                if plain_log:
                    print(
                        f"[nonaga] iteration {i + 1} done | saved {save_path} + {snapshot_path}",
                        flush=True,
                    )

if __name__ == "__main__":
    train_pipeline_parallel(
        iterations=40, 
        games_per_iter=128, 
        mcts_sims=150, 
        epochs=10, 
        batch_size=1024, 
        num_workers=32
    )
