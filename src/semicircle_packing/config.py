"""Constants for the semicircle packing challenge."""

N = 15
RADIUS = 1.0

# Number of points on the arc when building the Shapely polygon
POLYGON_ARC_POINTS = 256

# Number of boundary sample points per semicircle for initial MEC sampling
# (analytical refinement then converges to machine precision)
MEC_BOUNDARY_POINTS = 128

# Overlap tolerance: intersection area must be below this to be "non-overlapping"
OVERLAP_TOL = 1e-6

# Containment tolerance: semicircle boundary must be within this distance of MEC
# (now uses analytical farthest-point, so can be much tighter)
CONTAINMENT_TOL = 1e-9
