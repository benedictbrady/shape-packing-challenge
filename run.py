"""CLI entry point: python run.py"""

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Semicircle Packing Challenge")
    parser.add_argument(
        "solution", nargs="?", default="solution.json",
        help="path to JSON solution file (default: solution.json)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="show a matplotlib plot of the packing",
    )
    parser.add_argument(
        "--save-plot", type=str, metavar="FILE",
        help="save plot to file (png, svg, pdf)",
    )
    args = parser.parse_args()

    from semicircle_packing.scoring import validate_and_score, print_report
    from semicircle_packing.geometry import Semicircle

    semicircles = _load_solution(args.solution)

    start = time.time()
    result = validate_and_score(semicircles)
    elapsed = time.time() - start

    print_report(result)
    print(f"  Validation time: {elapsed:.3f}s")
    print()

    if args.visualize or args.save_plot:
        from semicircle_packing.visualization import plot_packing
        plot_packing(semicircles, result.mec, save_path=args.save_plot)


def _load_solution(path: str) -> list:
    """Load a solution from a JSON file: [{"x": ..., "y": ..., "theta": ...}, ...]"""
    from semicircle_packing.geometry import Semicircle

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of {x, y, theta} objects")

    semicircles = []
    for i, item in enumerate(data):
        for key in ("x", "y", "theta"):
            if key not in item:
                raise ValueError(f"Item {i} missing required field '{key}'")
        semicircles.append(Semicircle(x=float(item["x"]), y=float(item["y"]), theta=float(item["theta"])))

    return semicircles


if __name__ == "__main__":
    main()
