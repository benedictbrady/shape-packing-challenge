"""Scoring: minimum enclosing circle (Welzl), validation, reporting."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .config import N
from .geometry import (
    Semicircle,
    semicircle_boundary_points,
    semicircles_overlap,
    semicircle_contained_in_circle,
    farthest_boundary_point_from,
)


# ---------------------------------------------------------------------------
# Minimum enclosing circle (Welzl's algorithm)
# ---------------------------------------------------------------------------

def _circle_from_1(p):
    return (p[0], p[1], 0.0)


def _circle_from_2(p1, p2):
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    r = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) / 2
    return (cx, cy, r)


def _circle_from_3(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        # Degenerate: pick the largest pair
        c1 = _circle_from_2(p1, p2)
        c2 = _circle_from_2(p1, p3)
        c3 = _circle_from_2(p2, p3)
        return max([c1, c2, c3], key=lambda c: c[2])
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy, r)


def _in_circle(c, p, eps=1e-10):
    return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + eps


def _make_circle(boundary):
    if len(boundary) == 0:
        return (0, 0, 0)
    if len(boundary) == 1:
        return _circle_from_1(boundary[0])
    if len(boundary) == 2:
        return _circle_from_2(boundary[0], boundary[1])
    return _circle_from_3(boundary[0], boundary[1], boundary[2])


def minimum_enclosing_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Compute the minimum enclosing circle for an (n,2) array of points.

    Iterative Welzl algorithm. Returns (cx, cy, radius).
    """
    pts = [(float(points[i, 0]), float(points[i, 1])) for i in range(len(points))]
    random.seed(42)
    random.shuffle(pts)

    n = len(pts)
    c = _make_circle([])
    for i in range(n):
        if not _in_circle(c, pts[i]):
            c = _make_circle([pts[i]])
            for j in range(i):
                if not _in_circle(c, pts[j]):
                    c = _make_circle([pts[i], pts[j]])
                    for k in range(j):
                        if not _in_circle(c, pts[k]):
                            c = _make_circle([pts[i], pts[j], pts[k]])
    return c


def compute_mec(semicircles: list[Semicircle]) -> tuple[float, float, float]:
    """Compute the MEC enclosing all semicircles.

    Uses sampled boundary points for the initial Welzl solve, then iteratively
    refines by analytically finding the true farthest boundary point on each
    semicircle from the current MEC center. Converges to machine precision.
    """
    from .config import MEC_BOUNDARY_POINTS
    all_pts = np.vstack([semicircle_boundary_points(sc, MEC_BOUNDARY_POINTS) for sc in semicircles])
    cx, cy, cr = minimum_enclosing_circle(all_pts)

    for _ in range(20):
        new_pts = []
        for sc in semicircles:
            fx, fy = farthest_boundary_point_from(sc, cx, cy)
            dist = math.hypot(fx - cx, fy - cy)
            if dist > cr + 1e-12:
                new_pts.append([fx, fy])

        if not new_pts:
            break

        all_pts = np.vstack([all_pts, np.array(new_pts)])
        cx, cy, cr = minimum_enclosing_circle(all_pts)

    return cx, cy, cr


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    score: float | None
    mec: tuple[float, float, float] | None
    errors: list[str] = field(default_factory=list)


def validate_and_score(semicircles: list[Semicircle]) -> ValidationResult:
    """Validate a packing and return the score (enclosing circle radius)."""
    errors: list[str] = []

    # Check count
    if len(semicircles) != N:
        errors.append(f"Expected {N} semicircles, got {len(semicircles)}")
        return ValidationResult(valid=False, score=None, mec=None, errors=errors)

    # Check overlaps
    for i in range(len(semicircles)):
        for j in range(i + 1, len(semicircles)):
            if semicircles_overlap(semicircles[i], semicircles[j]):
                errors.append(f"Semicircles {i} and {j} overlap")

    if errors:
        return ValidationResult(valid=False, score=None, mec=None, errors=errors)

    # Compute MEC
    cx, cy, cr = compute_mec(semicircles)

    # Check containment
    for i, sc in enumerate(semicircles):
        if not semicircle_contained_in_circle(sc, cx, cy, cr):
            errors.append(f"Semicircle {i} not fully contained in enclosing circle")

    if errors:
        return ValidationResult(valid=False, score=None, mec=(cx, cy, cr), errors=errors)

    return ValidationResult(valid=True, score=cr, mec=(cx, cy, cr))


def print_report(result: ValidationResult) -> None:
    """Print a human-readable validation report."""
    print()
    print("  Semicircle Packing Challenge")
    print(f"  {N} unit semicircles into smallest enclosing circle")
    print()

    if result.valid:
        print(f"  Status:  VALID")
        print(f"  Score:   {result.score:.6f}  (enclosing circle radius)")
        if result.mec:
            cx, cy, cr = result.mec
            print(f"  MEC:     center=({cx:.4f}, {cy:.4f})  radius={cr:.6f}")
    else:
        print(f"  Status:  INVALID")
        for err in result.errors:
            print(f"    - {err}")

    print()
