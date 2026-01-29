#!/usr/bin/env python3
"""RLBot entry point."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agent import CompetitiveAgent


def main():
    """Main entry point for RLBot."""
    # Import RLBot framework
    try:
        from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
        from rlbot.utils.structures.game_data_struct import GameTickPacket
    except ImportError:
        print("RLBot not installed. Please install rlbot package.")
        return

    # Create and run agent
    agent = CompetitiveAgent()

    print(f"CompetitiveBot initialized")
    print(f"  Model: Actor-Attention-Critic")
    print(f"  Actions: 1944")
    print(f"  Tick skip: 8")

    # The actual bot loop is handled by RLBot framework
    # This script just sets up the agent


if __name__ == "__main__":
    main()
