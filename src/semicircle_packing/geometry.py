"""Semicircle geometry: representation, Shapely polygon, overlap & containment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, Point

from .config import POLYGON_ARC_POINTS, OVERLAP_TOL, MEC_BOUNDARY_POINTS


@dataclass(frozen=True)
class Semicircle:
    """A unit semicircle.

    (x, y) is the center of the full disk.
    theta is the angle (radians) the curved part extends toward.
    The flat edge passes through (x, y) perpendicular to theta.
    """
    x: float
    y: float
    theta: float


def semicircle_polygon(sc: Semicircle, n_arc: int = POLYGON_ARC_POINTS) -> Polygon:
    """Build a Shapely Polygon for the given semicircle."""
    from .config import RADIUS

    # Arc from theta - pi/2 to theta + pi/2
    angles = np.linspace(sc.theta - math.pi / 2, sc.theta + math.pi / 2, n_arc)
    arc_x = sc.x + RADIUS * np.cos(angles)
    arc_y = sc.y + RADIUS * np.sin(angles)

    # Close with the flat edge (first and last arc points connect through center line)
    coords = list(zip(arc_x, arc_y))
    return Polygon(coords)


def semicircle_boundary_points(sc: Semicircle, n: int = MEC_BOUNDARY_POINTS) -> np.ndarray:
    """Sample n points along the boundary of a semicircle (arc + flat edge).

    Returns an (n, 2) array.
    """
    from .config import RADIUS

    # Half the points on the arc, half on the flat edge
    n_arc = n // 2
    n_flat = n - n_arc

    # Arc points
    angles = np.linspace(sc.theta - math.pi / 2, sc.theta + math.pi / 2, n_arc)
    arc = np.column_stack([sc.x + RADIUS * np.cos(angles),
                           sc.y + RADIUS * np.sin(angles)])

    # Flat edge endpoints
    perp = sc.theta + math.pi / 2
    ex1 = sc.x + RADIUS * math.cos(perp), sc.y + RADIUS * math.sin(perp)
    ex2 = sc.x - RADIUS * math.cos(perp), sc.y - RADIUS * math.sin(perp)

    t = np.linspace(0, 1, n_flat).reshape(-1, 1)
    flat = np.array(ex1) * (1 - t) + np.array(ex2) * t

    return np.vstack([arc, flat])


def semicircles_overlap(a: Semicircle, b: Semicircle) -> bool:
    """Return True if two semicircles have overlapping interior area."""
    pa = semicircle_polygon(a)
    pb = semicircle_polygon(b)
    return pa.intersection(pb).area > OVERLAP_TOL


def farthest_boundary_point_from(sc: Semicircle, qx: float, qy: float) -> tuple[float, float]:
    """Find the point on the semicircle boundary farthest from (qx, qy).

    Analytical computation — no sampling needed.
    On the arc, distance from (qx, qy) is maximized at the angle pointing
    from query toward the semicircle center, clamped to the arc range.
    On the flat edge (a line segment), the farthest point is always an endpoint.
    """
    from .config import RADIUS

    dx = sc.x - qx
    dy = sc.y - qy

    candidates: list[tuple[float, float]] = []

    # Arc: maximize dx*cos(a) + dy*sin(a) over a in [theta-pi/2, theta+pi/2]
    optimal_angle = math.atan2(dy, dx)
    # Check if optimal_angle falls within the arc range
    diff = math.atan2(math.sin(optimal_angle - sc.theta), math.cos(optimal_angle - sc.theta))
    if -math.pi / 2 <= diff <= math.pi / 2:
        candidates.append((sc.x + RADIUS * math.cos(optimal_angle),
                           sc.y + RADIUS * math.sin(optimal_angle)))

    # Arc endpoints (which are also flat edge endpoints)
    a1 = sc.theta - math.pi / 2
    a2 = sc.theta + math.pi / 2
    candidates.append((sc.x + RADIUS * math.cos(a1), sc.y + RADIUS * math.sin(a1)))
    candidates.append((sc.x + RADIUS * math.cos(a2), sc.y + RADIUS * math.sin(a2)))

    return max(candidates, key=lambda p: (p[0] - qx) ** 2 + (p[1] - qy) ** 2)


def semicircle_contained_in_circle(sc: Semicircle, cx: float, cy: float, cr: float) -> bool:
    """Check whether a semicircle is fully contained in a circle (cx, cy, cr).

    Uses analytical farthest-point computation instead of sampling.
    """
    from .config import CONTAINMENT_TOL
    fx, fy = farthest_boundary_point_from(sc, cx, cy)
    dist = math.hypot(fx - cx, fy - cy)
    return dist <= cr + CONTAINMENT_TOL
