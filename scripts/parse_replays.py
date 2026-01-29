#!/usr/bin/env python3
"""Script for parsing Rocket League replay files."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlbot_agent.replay_analysis import ReplayParser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse Rocket League replay files")

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .replay files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/replays",
        help="Output directory for parsed data",
    )
    parser.add_argument(
        "--tick-skip",
        type=int,
        default=8,
        help="Number of ticks to skip between samples",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Rocket League Replay Parser")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create parser
    parser = ReplayParser(tick_skip=args.tick_skip)

    # Parse replays
    print(f"\nParsing replays from: {args.input_dir}")
    parsed_data = parser.parse_directory(args.input_dir)

    if not parsed_data:
        print("No replays parsed successfully")
        return

    # Combine all data
    all_states = []
    all_actions = []

    for replay in parsed_data:
        all_states.extend(replay['states'])
        all_actions.extend(replay['actions'])

    print(f"\nTotal samples: {len(all_states)}")

    # Save to npz
    output_path = output_dir / "replay_data.npz"
    np.savez(
        output_path,
        states=np.array(all_states, dtype=object),
        actions=np.array(all_actions, dtype=object),
    )

    print(f"Saved parsed data to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Replays parsed: {len(parsed_data)}")
    print(f"  Total states: {len(all_states)}")
    print(f"  Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
