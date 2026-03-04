# Semicircle Packing Challenge

Pack **15 unit semicircles** (radius = 1) into the smallest possible enclosing circle.

Your **score** is the radius of the enclosing circle. Lower is better.

The baseline scores 3.50. The area lower bound is ~2.74 (you can't beat it, but you probably can't reach it either). How close can you get?

![Baseline packing](baseline.png)

## Getting Started

```bash
uv sync
uv run python run.py
```

This scores `solution.json` — the file you edit. Open it and tweak the coordinates.

## Format

`solution.json` is a JSON array of 15 semicircle placements:

```json
[
  { "x": -2.0, "y": 0.0, "theta": 0.0 },
  { "x": -2.0, "y": 0.0, "theta": 3.141593 },
  ...
]
```

- **(x, y)** — center of the full disk
- **theta** — angle (radians) the curved part extends toward. The flat edge passes through (x, y) perpendicular to theta.

This is the same format the [Optimization Arena](https://arena.lol/packing) accepts — paste it directly to submit.

## Other Commands

```bash
uv run python run.py my_solution.json         # score a different file
uv run python run.py --save-plot packing.png   # save a visualization
uv run python run.py --visualize               # open plot in a window
uv run pytest                                  # run tests
```
