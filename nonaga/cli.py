from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .agents import GreedyAgent, RandomAgent, SearchAgent
from .eval import EvalWeights
from .learn import LearnConfig, tune
from .rules import apply_move, legal_moves, winner
from .search import SearchConfig
from .selfplay import arena, play_game
from .state import BLACK, RED, Color, GameState

# Custom theme for Nonaga
NONAGA_THEME = Theme({
    "red_token": "bold red",
    "black_token": "bold dodger_blue1",  # Blue is easier to see on dark terminals than black.
    "empty_disc": "dim cyan",
    "info": "italic cyan",
    "winner_red": "bold red on white",
    "winner_black": "bold dodger_blue1 on white",
    "move_short": "bold yellow",
})

console = Console(theme=NONAGA_THEME)


def render_board(state: GameState) -> Panel:
    discs = state.discs
    qs = [q for q, _ in discs]
    rs = [r for _, r in discs]
    min_q, max_q = min(qs), max(qs)
    min_r, max_r = min(rs), max(rs)
    occupied = {c: "red" for c in state.red}
    occupied.update({c: "black" for c in state.black})
    
    board_text = Text()
    for r in range(max_r, min_r - 1, -1):
        row_cells = [(q, r) for q in range(min_q, max_q + 1) if (q, r) in discs]
        if not row_cells:
            continue
        
        # Add padding for hexagonal alignment
        board_text.append("  " * (max_r - r))
        
        for i, c in enumerate(row_cells):
            token_type = occupied.get(c)
            if token_type == "red":
                board_text.append("●", style="red_token")
            elif token_type == "black":
                board_text.append("●", style="black_token")
            else:
                board_text.append("·", style="empty_disc")
            
            if i < len(row_cells) - 1:
                board_text.append(" ")
        board_text.append("\n")

    side_str = "RED" if state.side_to_move == RED else "BLACK"
    side_style = "red_token" if state.side_to_move == RED else "black_token"
    
    status_text = Text.assemble(
        ("Turn: ", "info"), (side_str, side_style),
        ("  Ply: ", "info"), (str(state.ply), "bold yellow"),
        ("  Forbidden: ", "info"), (str(state.forbidden_disc or "None"), "dim blue")
    )
    
    return Panel(
        board_text,
        title="[bold white]Nonaga Board[/bold white]",
        subtitle=status_text,
        border_style="bright_blue",
        expand=False
    )


def load_weights(path: str | None) -> EvalWeights:
    if path is None:
        return EvalWeights()
    return EvalWeights.from_json(path)


def make_search_agent(args: argparse.Namespace, weights: EvalWeights | None = None) -> SearchAgent:
    cfg = SearchConfig(
        max_depth=args.depth,
        time_limit=args.time_limit,
        max_branching=args.max_branching,
        max_ply=args.max_ply,
    )
    return SearchAgent(config=cfg, weights=weights or load_weights(args.weights))


def cmd_show(args: argparse.Namespace) -> None:
    state = GameState.initial(RED)
    console.print(render_board(state))
    console.print(f"Legal moves from start: [bold green]{len(legal_moves(state))}[/bold green]")


def cmd_list_moves(args: argparse.Namespace) -> None:
    state = GameState.initial(RED)
    moves = legal_moves(state)
    console.print(render_board(state))
    
    table = Table(title="Legal Moves", title_style="bold magenta")
    table.add_column("Index", justify="right", style="cyan", no_wrap=True)
    table.add_column("Piece", style="magenta")
    table.add_column("Slide", style="move_short")
    table.add_column("Tile Relocation", style="info")
    
    for i, mv in enumerate(moves):
        table.add_row(
            str(i),
            f"#{mv.slide.piece_index}",
            f"{mv.slide.start} -> {mv.slide.end}",
            f"{mv.tile.remove_from} -> {mv.tile.place_to}"
        )
    console.print(table)


def cmd_human(args: argparse.Namespace) -> None:
    state = GameState.initial(RED)
    ai_color = BLACK if args.human == "red" else RED
    ai = make_search_agent(args)
    
    console.print(Panel("[bold yellow]Nonaga: Human vs AI[/bold yellow]", border_style="yellow"))
    
    while True:
        console.print(render_board(state))
        won = winner(state)
        if won is not None:
            winner_str = "RED" if won == RED else "BLACK"
            style = "winner_red" if won == RED else "winner_black"
            console.print(Panel(Text(f" {winner_str} WINS! ", style=style), expand=False))
            return
            
        moves = legal_moves(state)
        if not moves:
            winner_col = BLACK if state.side_to_move == RED else RED
            winner_str = "BLACK" if winner_col == BLACK else "RED"
            console.print(f"[bold red]No legal moves.[/bold red] Winner: [bold green]{winner_str}[/bold green]")
            return
            
        if state.side_to_move == ai_color:
            with console.status("[bold blue]AI is thinking...[/bold blue]"):
                mv = ai.choose_move(state)
            console.print(Text.assemble(("AI plays: ", "info"), (mv.short(), "move_short")))
        else:
            # Short table for selection
            table = Table(header_style="bold cyan", box=None)
            table.add_column("Index", justify="right")
            table.add_column("Move")
            for i, mv in enumerate(moves):
                table.add_row(str(i), mv.short())
            console.print(table)
            
            try:
                idx = int(console.input("[bold green]Enter move index: [/bold green]"))
                mv = moves[idx]
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection, try again.[/bold red]")
                continue
        state = apply_move(state, mv)


def cmd_selfplay(args: argparse.Namespace) -> None:
    red = make_search_agent(args)
    black = make_search_agent(args)
    console.print("[bold cyan]Starting self-play session...[/bold cyan]")
    result = play_game(red, black, max_ply=args.max_ply)
    
    console.print(render_board(result.final_state))
    console.print(f"Termination: [bold]{result.termination}[/bold]")
    if result.winner:
        w_str = "RED" if result.winner == RED else "BLACK"
        console.print(f"Winner: [bold green]{w_str}[/bold green]")
    console.print(f"Total Plies: [bold yellow]{result.plies}[/bold yellow]")


def cmd_arena(args: argparse.Namespace) -> None:
    strong_weights = load_weights(args.weights)
    baseline_weights = load_weights(args.baseline_weights) if args.baseline_weights else EvalWeights()
    strong_cfg = SearchConfig(max_depth=args.depth, time_limit=args.time_limit, max_branching=args.max_branching)
    base_cfg = SearchConfig(max_depth=args.baseline_depth, time_limit=args.baseline_time_limit, max_branching=args.max_branching)
    
    console.print(Panel("[bold magenta]Nonaga Arena Benchmarking[/bold magenta]", border_style="magenta"))
    
    with console.status("[bold blue]Running battle...[/bold blue]"):
        summary = arena(
            agent_a_factory=lambda: SearchAgent(config=strong_cfg, weights=strong_weights),
            agent_b_factory=lambda: SearchAgent(config=base_cfg, weights=baseline_weights),
            games=args.games,
            max_ply=args.max_ply,
        )
    
    table = Table(title="Arena Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold yellow")
    table.add_row("Total Games", str(summary.games))
    table.add_row("Agent A Wins", str(summary.a_wins))
    table.add_row("Agent B Wins", str(summary.b_wins))
    table.add_row("Draws", str(summary.draws))
    table.add_row("Win Rate A", f"{summary.a_wins / summary.games:.1%}")
    console.print(table)


def cmd_learn(args: argparse.Namespace) -> None:
    cfg = LearnConfig(
        generations=args.generations,
        candidates_per_generation=args.candidates,
        arena_games=args.games,
        mutation_scale=args.mutation_scale,
        seed=args.seed,
        output_path=args.output,
        max_depth=args.depth,
        time_limit=args.time_limit,
        max_branching=args.max_branching,
    )
    initial = load_weights(args.weights) if args.weights else EvalWeights()
    
    console.print(Panel("[bold green]Tuning Evaluation Weights via Self-Play[/bold green]", border_style="green"))
    result = tune(initial=initial, config=cfg)
    
    console.print(f"[bold green]Tuning complete![/bold green] Results saved to [bold white]{args.output}[/bold white]")
    console.print(json.dumps({"final_weights": result.weights.__dict__, "generations": len(result.history)}, indent=2))


def cmd_az_train_parallel(args: argparse.Namespace) -> None:
    from .train_nn_parallel import train_pipeline_parallel
    from .nn import get_device
    from rich.console import Console
    
    device = get_device()
    console = Console()
    console.print(f"[bold cyan]🔥 STARTING PARALLEL MASTER-TRAINING ON {device.type.upper()} 🔥[/bold cyan]")
    console.print(f"Workers: {args.workers} | Sims: {args.sims} | Games: {args.games}")

    plain: bool | None
    if getattr(args, "rich_dashboard", False):
        plain = False
    elif getattr(args, "plain_log", False):
        plain = True
    else:
        plain = None
    if plain is None and os.environ.get("NONAGA_PLAIN_LOG", "").strip() in ("1", "true", "yes"):
        plain = True
    if plain is None and not sys.stdout.isatty():
        plain = True

    train_pipeline_parallel(
        iterations=args.iterations,
        games_per_iter=args.games,
        mcts_sims=args.sims,
        epochs=args.epochs,
        batch_size=args.batch,
        save_path=args.model,
        num_workers=args.workers,
        plain_log=plain,
    )


def cmd_az_train(args: argparse.Namespace) -> None:
    from .train_nn import train_pipeline
    from .nn import get_device
    from rich.console import Console
    
    device = get_device()
    console = Console()
    console.print(f"[bold cyan]Running AlphaZero Training directly on {device.type.upper()} with user-defined settings.[/bold cyan]")

    iterations = args.iterations
    games_per_iter = args.games
    mcts_sims = args.sims
    epochs = args.epochs
    batch_size = args.batch

    train_pipeline(
        iterations=iterations,
        games_per_iter=games_per_iter,
        mcts_sims=mcts_sims,
        epochs=epochs,
        batch_size=batch_size,
        save_path=args.model
    )


def cmd_az_play(args: argparse.Namespace) -> None:
    import os
    import torch
    from .mcts import MCTSAgent, MCTSConfig
    from .nn import NonagaNet, get_device
    
    device = get_device()
    console.print(f"[bold green]Loading AlphaZero Model on {device}...[/bold green]")
    net = NonagaNet(num_res_blocks=10, num_channels=128).to(device)
    if os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model, map_location=device))
    else:
        console.print(f"[bold yellow]Warning: Model {args.model} not found. Using untrained network.[/bold yellow]")
        
    config = MCTSConfig(num_simulations=args.sims, temperature=0.0)
    ai = MCTSAgent(net, config)
    
    state = GameState.initial(RED)
    ai_color = BLACK if args.human == "red" else RED
    
    console.print(Panel("[bold cyan]Nonaga: Human vs AlphaZero[/bold cyan]", border_style="cyan"))
    
    while True:
        console.print(render_board(state))
        won = winner(state)
        if won is not None:
            winner_str = "RED" if won == RED else "BLACK"
            style = "winner_red" if won == RED else "winner_black"
            console.print(Panel(Text(f" {winner_str} WINS! ", style=style), expand=False))
            return
            
        moves = legal_moves(state)
        if not moves:
            winner_col = BLACK if state.side_to_move == RED else RED
            winner_str = "BLACK" if winner_col == BLACK else "RED"
            console.print(f"[bold red]No legal moves.[/bold red] Winner: [bold green]{winner_str}[/bold green]")
            return
            
        if state.side_to_move == ai_color:
            with console.status(f"[bold blue]AlphaZero is thinking ({args.sims} sims)...[/bold blue]"):
                mv = ai.choose_move(state)
            console.print(Text.assemble(("AlphaZero plays: ", "info"), (mv.short(), "move_short")))
        else:
            table = Table(header_style="bold cyan", box=None)
            table.add_column("Index", justify="right")
            table.add_column("Move")
            for i, mv in enumerate(moves):
                table.add_row(str(i), mv.short())
            console.print(table)
            
            try:
                idx = int(console.input("[bold green]Enter move index: [/bold green]"))
                mv = moves[idx]
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection, try again.[/bold red]")
                continue
        state = apply_move(state, mv)


def cmd_web(args: argparse.Namespace) -> None:
    import uvicorn
    from rich.console import Console
    Console().print(f"[bold green]Starting Nonaga Web UI on http://localhost:{args.port}[/bold green]")
    uvicorn.run("nonaga.web:app", host="0.0.0.0", port=args.port, reload=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nonaga engine and AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("moves")
    s.set_defaults(func=cmd_list_moves)

    s = sub.add_parser("human")
    s.add_argument("--human", choices=["red", "black"], default="red")
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--time-limit", type=float, default=1.0)
    s.add_argument("--max-branching", type=int, default=80)
    s.add_argument("--max-ply", type=int, default=160)
    s.add_argument("--weights")
    s.set_defaults(func=cmd_human)

    s = sub.add_parser("selfplay")
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--time-limit", type=float, default=0.5)
    s.add_argument("--max-branching", type=int, default=80)
    s.add_argument("--max-ply", type=int, default=160)
    s.add_argument("--weights")
    s.set_defaults(func=cmd_selfplay)

    s = sub.add_parser("arena")
    s.add_argument("--games", type=int, default=10)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--time-limit", type=float, default=0.5)
    s.add_argument("--baseline-depth", type=int, default=2)
    s.add_argument("--baseline-time-limit", type=float, default=0.2)
    s.add_argument("--max-branching", type=int, default=80)
    s.add_argument("--max-ply", type=int, default=160)
    s.add_argument("--weights")
    s.add_argument("--baseline-weights")
    s.set_defaults(func=cmd_arena)

    s = sub.add_parser("learn")
    s.add_argument("--generations", type=int, default=8)
    s.add_argument("--candidates", type=int, default=4)
    s.add_argument("--games", type=int, default=8)
    s.add_argument("--mutation-scale", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--output", default="best_weights.json")
    s.add_argument("--weights")
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--time-limit", type=float, default=0.2)
    s.add_argument("--max-branching", type=int, default=60)
    s.set_defaults(func=cmd_learn)
    
    s = sub.add_parser("az-train", help="Train AlphaZero NN using Self-Play")
    s.add_argument("--iterations", type=int, default=10)
    s.add_argument("--games", type=int, default=20)
    s.add_argument("--sims", type=int, default=200)
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--model", type=str, default="nonagazero.pt")
    s.set_defaults(func=cmd_az_train)
    
    s = sub.add_parser("az-train-parallel", help="Parallel AlphaZero training (Threadripper + 3090)")
    s.add_argument("--iterations", type=int, default=10)
    s.add_argument("--games", type=int, default=100)
    s.add_argument("--sims", type=int, default=200)
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--workers", type=int, default=mp.cpu_count())
    s.add_argument("--model", type=str, default="nonagazero.pt")
    s.add_argument(
        "--plain-log",
        action="store_true",
        help="Line-based progress (use in RunPod Logs / tee / nohup); else auto if stdout is not a TTY",
    )
    s.add_argument(
        "--rich-dashboard",
        action="store_true",
        help="Force animated Rich dashboard (TTY)",
    )
    s.set_defaults(func=cmd_az_train_parallel)

    s = sub.add_parser("az-play", help="Play against the trained AlphaZero agent")
    s.add_argument("--human", choices=["red", "black"], default="red")
    s.add_argument("--sims", type=int, default=400)
    s.add_argument("--model", type=str, default="nonagazero.pt")
    s.set_defaults(func=cmd_az_play)
    
    s = sub.add_parser("web", help="Start the interactive Web UI")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_web)
    
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
