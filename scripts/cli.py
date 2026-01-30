#!/usr/bin/env python3
"""RLBot Deep RL - Command Line Interface."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.panel import Panel
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "typer", "rich", "-q"])
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.panel import Panel

import torch
import yaml

app = typer.Typer(
    name="rlbot",
    help="RLBot Deep RL - Train Rocket League bots with PPO",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

CONFIG_DIR = Path(__file__).parent.parent / "configs"
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "checkpoints"


def load_config(name: str = "base") -> dict:
    """Load a YAML config file."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        raise typer.Exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def list_checkpoints() -> List[dict]:
    """List available checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for f in sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt")):
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            checkpoints.append({
                "path": f,
                "name": f.name,
                "step": data.get("step", 0),
                "size_mb": f.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(f.stat().st_mtime),
            })
        except Exception:
            pass

    latest = CHECKPOINT_DIR / "latest.pt"
    if latest.exists():
        try:
            data = torch.load(latest, map_location="cpu", weights_only=False)
            checkpoints.append({
                "path": latest,
                "name": "latest.pt",
                "step": data.get("step", 0),
                "size_mb": latest.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(latest.stat().st_mtime),
                "is_latest": True,
            })
        except Exception:
            pass

    return checkpoints


# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def interactive_menu():
    """Show interactive menu when no command given."""
    console.print(Panel.fit(
        "[bold]RLBot Deep RL[/bold]\nTrain Rocket League bots with PPO",
        border_style="blue"
    ))

    # Show status
    cps = list_checkpoints()
    if cps:
        latest = next((c for c in cps if c.get('is_latest')), None)
        if latest:
            console.print(f"Latest checkpoint: [cyan]{latest['name']}[/cyan] (step {latest['step']:,})")
    else:
        console.print("No checkpoints found")

    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  [cyan]1[/cyan] Train        - Start or resume training")
    console.print("  [cyan]2[/cyan] Evaluate     - Test a trained model")
    console.print("  [cyan]3[/cyan] Watch        - Watch bot play (no RL needed)")
    console.print("  [cyan]4[/cyan] Checkpoints  - List saved checkpoints")
    console.print("  [cyan]5[/cyan] Config       - View/edit configuration")
    console.print("  [cyan]6[/cyan] System Info  - Show system status")
    console.print("  [cyan]7[/cyan] Clean        - Remove old checkpoints")
    console.print("  [cyan]q[/cyan] Quit")
    console.print()

    choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5", "6", "7", "q"], default="1")

    if choice == "1":
        interactive_train()
    elif choice == "2":
        interactive_eval()
    elif choice == "3":
        interactive_watch()
    elif choice == "4":
        do_list(20)
    elif choice == "5":
        interactive_config()
    elif choice == "6":
        do_info()
    elif choice == "7":
        interactive_clean()
    elif choice == "q":
        return


def interactive_train():
    """Interactive training setup."""
    console.print("\n[bold]Training Setup[/bold]\n")

    # Check for existing checkpoint
    cps = list_checkpoints()
    resume_from = None

    if cps:
        latest = next((c for c in cps if c.get('is_latest')), None)
        if latest:
            if Confirm.ask(f"Resume from {latest['name']} (step {latest['step']:,})?", default=True):
                resume_from = latest['name']

    # Quick config
    workers = IntPrompt.ask("Workers", default=64)

    use_wandb = Confirm.ask("Enable W&B logging?", default=True)
    wandb_project = None
    if use_wandb:
        wandb_project = Prompt.ask("W&B project", default="rlbot-competitive")

    console.print()
    do_train(
        workers=workers,
        steps=10_000_000_000,
        resume=resume_from,
        lr=1e-4,
        batch_size=100_000,
        device="auto",
        wandb=wandb_project,
        no_curriculum=False,
        config="base",
    )


def interactive_eval():
    """Interactive evaluation setup."""
    console.print("\n[bold]Evaluation Setup[/bold]\n")

    cps = list_checkpoints()
    if not cps:
        console.print("[red]No checkpoints found[/red]")
        return

    # Show available checkpoints
    console.print("Available checkpoints:")
    for i, cp in enumerate(cps[:5], 1):
        console.print(f"  {i}. {cp['name']} (step {cp['step']:,})")

    checkpoint = Prompt.ask("Checkpoint", default="latest.pt")
    episodes = IntPrompt.ask("Episodes", default=100)

    console.print()
    do_eval(checkpoint, episodes, True, "cpu", False)


def interactive_watch():
    """Interactive watch setup."""
    console.print("\n[bold]Watch Bot Play[/bold]\n")

    cps = list_checkpoints()
    if not cps:
        console.print("[red]No checkpoints found. Train first![/red]")
        return

    checkpoint = Prompt.ask("Checkpoint", default="latest.pt")
    episodes = IntPrompt.ask("Episodes", default=5)
    speed = float(Prompt.ask("Speed (1.0 = real-time)", default="1.0"))

    import subprocess
    cmd = [sys.executable, str(Path(__file__).parent / "watch.py")]
    cmd.extend(["--checkpoint", checkpoint])
    cmd.extend(["--episodes", str(episodes)])
    cmd.extend(["--speed", str(speed)])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\nStopped")
    except subprocess.CalledProcessError:
        console.print("[red]Watch failed[/red]")


def interactive_config():
    """Interactive config viewer."""
    console.print("\n[bold]Configuration[/bold]\n")

    action = Prompt.ask("Action", choices=["view", "edit"], default="view")

    if action == "edit":
        do_config("base", None, edit=True)
    else:
        section = Prompt.ask("Section (or 'all')", default="all")
        if section == "all":
            section = None
        do_config("base", section, edit=False)


def interactive_clean():
    """Interactive cleanup."""
    console.print("\n[bold]Cleanup[/bold]\n")

    keep = IntPrompt.ask("Checkpoints to keep", default=5)
    do_clean(keep, yes=False)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def do_train(workers, steps, resume, lr, batch_size, device, wandb, no_curriculum, config):
    """Execute training."""
    overrides = [
        f"training.n_workers={workers}",
        f"training.total_steps={steps}",
        f"ppo.learning_rate={lr}",
        f"ppo.batch_size={batch_size}",
        f"device={device}",
    ]

    if wandb:
        overrides.append(f"training.wandb_project={wandb}")
    if no_curriculum:
        overrides.append("curriculum.enabled=false")

    console.print(f"\nTraining Configuration:")
    console.print(f"  Workers:    {workers}")
    console.print(f"  Steps:      {steps:,}")
    console.print(f"  LR:         {lr}")
    console.print(f"  Batch:      {batch_size:,}")
    console.print(f"  Device:     {device}")
    console.print(f"  Curriculum: {'disabled' if no_curriculum else 'enabled'}")
    if resume:
        console.print(f"  Resume:     {resume}")
    console.print()

    cmd = [sys.executable, str(Path(__file__).parent / "train.py")]
    cmd.extend(overrides)

    env = os.environ.copy()
    if resume:
        checkpoint_path = CHECKPOINT_DIR / resume if not Path(resume).is_absolute() else Path(resume)
        if not checkpoint_path.exists():
            console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
            raise typer.Exit(1)
        env["RLBOT_RESUME_CHECKPOINT"] = str(checkpoint_path)

    import subprocess
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        console.print("\nTraining interrupted")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Training failed (exit {e.returncode})[/red]")
        raise typer.Exit(1)


def do_eval(checkpoint, episodes, deterministic, device, verbose):
    """Execute evaluation."""
    checkpoint_path = CHECKPOINT_DIR / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)

    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        raise typer.Exit(1)

    console.print(f"Evaluating: {checkpoint_path.name}")
    console.print(f"Episodes: {episodes}, Device: {device}\n")

    from rlbot_agent.deployment.inference import InferenceEngine
    from rlbot_agent.environment import create_environment
    import numpy as np

    engine = InferenceEngine.from_checkpoint(str(checkpoint_path), device=device)
    env = create_environment()

    rewards, lengths = [], []

    for ep in range(episodes):
        obs = env.reset()
        total_reward, steps = 0.0, 0
        done = False

        while not done:
            action, _ = engine.get_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)

        if verbose:
            console.print(f"  Episode {ep+1}: reward={total_reward:.2f}, length={steps}")

    env.close()

    console.print(f"\nResults ({episodes} episodes):")
    console.print(f"  Mean reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    console.print(f"  Min/Max:      {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    console.print(f"  Mean length:  {np.mean(lengths):.1f}")


def do_list(limit):
    """List checkpoints."""
    cps = list_checkpoints()

    if not cps:
        console.print("No checkpoints found")
        console.print(f"Directory: {CHECKPOINT_DIR}")
        return

    cps.sort(key=lambda x: x.get("step", 0), reverse=True)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Step", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Modified")

    for cp in cps[:limit]:
        name = cp['name']
        if cp.get('is_latest'):
            name = f"[bold]{name}[/bold]"

        table.add_row(
            name,
            f"{cp['step']:,}",
            f"{cp['size_mb']:.1f} MB",
            cp['modified'].strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)

    if len(cps) > limit:
        console.print(f"\n({len(cps) - limit} more checkpoints)")


def do_config(name, section, edit):
    """View or edit config."""
    config_path = CONFIG_DIR / f"{name}.yaml"

    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        console.print(f"\nAvailable: {', '.join(p.stem for p in CONFIG_DIR.glob('*.yaml'))}")
        raise typer.Exit(1)

    if edit:
        editor = os.environ.get("EDITOR", "vim")
        os.system(f"{editor} {config_path}")
        return

    with open(config_path) as f:
        content = f.read()

    if section:
        cfg = yaml.safe_load(content)
        if section not in cfg:
            console.print(f"[red]Section '{section}' not found[/red]")
            console.print(f"Available: {', '.join(cfg.keys())}")
            raise typer.Exit(1)
        content = yaml.dump({section: cfg[section]}, default_flow_style=False, sort_keys=False)

    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


def do_info():
    """Show system info."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Python", "[green]OK[/green]", sys.version.split()[0])

    cuda = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU"
    table.add_row("PyTorch", "[green]OK[/green]", f"{torch.__version__} ({cuda})")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        table.add_row("GPU", "[green]OK[/green]", f"{name} ({mem:.0f}GB)")
    else:
        table.add_row("GPU", "[yellow]--[/yellow]", "Not available")

    try:
        import rlgym
        table.add_row("RLGym", "[green]OK[/green]", "Installed")
    except ImportError:
        table.add_row("RLGym", "[yellow]--[/yellow]", "Not installed")

    n = len(list_checkpoints())
    table.add_row("Checkpoints", str(n), str(CHECKPOINT_DIR))

    console.print(table)


def do_clean(keep, yes):
    """Clean old checkpoints."""
    cps = [cp for cp in list_checkpoints() if not cp.get('is_latest')]
    cps.sort(key=lambda x: x['step'], reverse=True)

    to_delete = cps[keep:]

    if not to_delete:
        console.print("Nothing to clean")
        return

    total_mb = sum(cp['size_mb'] for cp in to_delete)
    console.print(f"Will delete {len(to_delete)} checkpoints ({total_mb:.1f} MB)")

    if not yes:
        if not Confirm.ask("Continue?"):
            return

    for cp in to_delete:
        cp['path'].unlink()
        console.print(f"  Deleted {cp['name']}")

    console.print(f"\nFreed {total_mb:.1f} MB")


# ============================================================================
# CLI COMMANDS (for direct invocation)
# ============================================================================

@app.callback()
def callback(ctx: typer.Context):
    """RLBot Deep RL - Train Rocket League bots with PPO."""
    if ctx.invoked_subcommand is None:
        interactive_menu()


@app.command()
def train(
    workers: int = typer.Option(64, "--workers", "-w", help="Number of parallel workers"),
    steps: int = typer.Option(10_000_000_000, "--steps", "-s", help="Total training steps"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Checkpoint to resume from"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(100_000, "--batch-size", "-b", help="Batch size"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
    wandb: Optional[str] = typer.Option(None, "--wandb", help="W&B project name"),
    no_curriculum: bool = typer.Option(False, "--no-curriculum", help="Disable curriculum"),
    config: str = typer.Option("base", "--config", "-c", help="Config file"),
):
    """Start training."""
    do_train(workers, steps, resume, lr, batch_size, device, wandb, no_curriculum, config)


@app.command()
def eval(
    checkpoint: str = typer.Argument("latest.pt", help="Checkpoint to evaluate"),
    episodes: int = typer.Option(100, "--episodes", "-n", help="Number of episodes"),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic", help="Policy mode"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show each episode"),
):
    """Evaluate a trained model."""
    do_eval(checkpoint, episodes, deterministic, device, verbose)


@app.command("list")
def list_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="Max to show"),
):
    """List saved checkpoints."""
    do_list(limit)


@app.command()
def config(
    name: str = typer.Argument("base", help="Config name"),
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show only this section"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open in editor"),
):
    """View or edit configuration."""
    do_config(name, section, edit)


@app.command()
def info():
    """Show system information."""
    do_info()


@app.command()
def clean(
    keep: int = typer.Option(5, "--keep", "-k", help="Checkpoints to keep"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove old checkpoints."""
    do_clean(keep, yes)


@app.command()
def watch(
    checkpoint: Optional[str] = typer.Argument(None, help="Checkpoint to watch (uses latest if not specified)"),
    episodes: int = typer.Option(5, "--episodes", "-n", help="Number of episodes to watch"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Playback speed (1.0 = real-time)"),
    stochastic: bool = typer.Option(False, "--stochastic", help="Use stochastic actions"),
):
    """Watch a trained bot play with visualization (no Rocket League needed)."""
    import subprocess

    cmd = [sys.executable, str(Path(__file__).parent / "watch.py")]

    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    cmd.extend(["--episodes", str(episodes)])
    cmd.extend(["--speed", str(speed)])
    if stochastic:
        cmd.append("--stochastic")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\nStopped")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Watch failed (exit {e.returncode})[/red]")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
