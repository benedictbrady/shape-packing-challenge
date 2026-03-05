"""Semicircle geometry: representation, Shapely polygon, overlap & containment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, Point

from .config import POLYGON_ARC_POINTS, MEC_BOUNDARY_POINTS


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


def _point_strictly_inside_semicircle(px: float, py: float, sc: Semicircle) -> bool:
    """Check if (px, py) is strictly inside the semicircle interior."""
    from .config import RADIUS
    dist_sq = (px - sc.x) ** 2 + (py - sc.y) ** 2
    if dist_sq >= RADIUS ** 2 - 1e-10:
        return False
    dot = (px - sc.x) * math.cos(sc.theta) + (py - sc.y) * math.sin(sc.theta)
    return dot > 1e-10


def _angle_on_arc(angle: float, sc: Semicircle, tol: float = 1e-10) -> bool:
    """Check if angle falls within the semicircle's arc range [theta-pi/2, theta+pi/2]."""
    diff = math.atan2(math.sin(angle - sc.theta), math.cos(angle - sc.theta))
    return -math.pi / 2 - tol <= diff <= math.pi / 2 + tol


def _arc_arc_intersections(a: Semicircle, b: Semicircle) -> list[tuple[float, float]]:
    """Find intersection points of two semicircular arcs (both radius R)."""
    from .config import RADIUS
    dx = b.x - a.x
    dy = b.y - a.y
    d = math.hypot(dx, dy)

    if d > 2 * RADIUS - 1e-12 or d < 1e-12:
        return []

    h_sq = RADIUS ** 2 - (d / 2) ** 2
    if h_sq < 1e-20:
        return []
    h = math.sqrt(h_sq)

    mx = (a.x + b.x) / 2
    my = (a.y + b.y) / 2
    px, py = -dy / d, dx / d

    points = []
    for sign in [1, -1]:
        ix = mx + sign * h * px
        iy = my + sign * h * py
        angle_a = math.atan2(iy - a.y, ix - a.x)
        angle_b = math.atan2(iy - b.y, ix - b.x)
        if _angle_on_arc(angle_a, a) and _angle_on_arc(angle_b, b):
            points.append((ix, iy))

    return points


def _arc_segment_intersections(
    arc_sc: Semicircle, seg_p1: tuple[float, float], seg_p2: tuple[float, float]
) -> list[tuple[float, float]]:
    """Find intersections of a semicircular arc with a line segment."""
    from .config import RADIUS
    sx, sy = seg_p1
    ex, ey = seg_p2
    ddx, ddy = ex - sx, ey - sy
    fx, fy = sx - arc_sc.x, sy - arc_sc.y

    a_coeff = ddx ** 2 + ddy ** 2
    if a_coeff < 1e-14:
        return []
    b_coeff = 2 * (fx * ddx + fy * ddy)
    c_coeff = fx ** 2 + fy ** 2 - RADIUS ** 2

    disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    points = []
    for sign in [-1, 1]:
        t = (-b_coeff + sign * sqrt_disc) / (2 * a_coeff)
        if -1e-10 <= t <= 1 + 1e-10:
            ix = sx + t * ddx
            iy = sy + t * ddy
            angle = math.atan2(iy - arc_sc.y, ix - arc_sc.x)
            if _angle_on_arc(angle, arc_sc):
                points.append((ix, iy))

    return points


def _segment_segment_intersection(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> list[tuple[float, float]]:
    """Find intersection of two line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-14:
        return []

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if -1e-10 <= t <= 1 + 1e-10 and -1e-10 <= u <= 1 + 1e-10:
        return [(x1 + t * (x2 - x1), y1 + t * (y2 - y1))]

    return []


def _semicircle_endpoints(sc: Semicircle) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two flat-edge endpoints of a semicircle."""
    from .config import RADIUS
    a1 = sc.theta - math.pi / 2
    a2 = sc.theta + math.pi / 2
    return (
        (sc.x + RADIUS * math.cos(a1), sc.y + RADIUS * math.sin(a1)),
        (sc.x + RADIUS * math.cos(a2), sc.y + RADIUS * math.sin(a2)),
    )


def semicircles_overlap(a: Semicircle, b: Semicircle) -> bool:
    """Analytical overlap check: True if two semicircles share positive interior area.

    A semicircle is the intersection of a disk and a half-plane, so overlap is
    checked via: (1) interior point containment, and (2) boundary crossings
    (arc-arc, arc-segment, segment-segment). No polygon approximation.
    """
    from .config import RADIUS

    # Quick reject: centers too far apart for any overlap
    if math.hypot(a.x - b.x, a.y - b.y) > 2 * RADIUS:
        return False

    # (1) Check if interior points of one semicircle are inside the other.
    #     Sample at multiple angles across each semicircle's interior to avoid
    #     missing overlaps that don't align with the theta direction.
    for s1, s2 in [(a, b), (b, a)]:
        for angle_offset in [-math.pi / 3, -math.pi / 6, 0, math.pi / 6, math.pi / 3]:
            angle = s1.theta + angle_offset
            for frac in [0.3, 0.6, 0.9]:
                px = s1.x + frac * RADIUS * math.cos(angle)
                py = s1.y + frac * RADIUS * math.sin(angle)
                if _point_strictly_inside_semicircle(px, py, s2):
                    return True

    # (2) Check boundary crossings
    a_e1, a_e2 = _semicircle_endpoints(a)
    b_e1, b_e2 = _semicircle_endpoints(b)

    all_crossings: list[tuple[float, float]] = []
    all_crossings.extend(_arc_arc_intersections(a, b))
    all_crossings.extend(_arc_segment_intersections(a, b_e1, b_e2))
    all_crossings.extend(_arc_segment_intersections(b, a_e1, a_e2))
    all_crossings.extend(_segment_segment_intersection(a_e1, a_e2, b_e1, b_e2))

    if len(all_crossings) >= 2:
        # Two convex shapes with 2+ boundary crossings overlap. Verify by
        # finding a point strictly inside both. Try: the straight-line midpoint
        # of crossing pairs, then nudge toward each semicircle's interior
        # (handles cases where the midpoint lands on a half-plane boundary).
        for i in range(len(all_crossings)):
            for j in range(i + 1, len(all_crossings)):
                mid_x = (all_crossings[i][0] + all_crossings[j][0]) / 2
                mid_y = (all_crossings[i][1] + all_crossings[j][1]) / 2

                candidates = [(mid_x, mid_y)]
                for sc in [a, b]:
                    # Nudge midpoint toward sc's interior (along theta direction)
                    nx = mid_x + 0.1 * RADIUS * math.cos(sc.theta)
                    ny = mid_y + 0.1 * RADIUS * math.sin(sc.theta)
                    candidates.append((nx, ny))
                    # Nudge toward sc's center
                    nx = mid_x + 0.1 * (sc.x - mid_x)
                    ny = mid_y + 0.1 * (sc.y - mid_y)
                    candidates.append((nx, ny))

                for cx, cy in candidates:
                    if (_point_strictly_inside_semicircle(cx, cy, a)
                            and _point_strictly_inside_semicircle(cx, cy, b)):
                        return True

    return False


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
