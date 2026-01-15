from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional

@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in detector coordinates (cm)."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def contains(self, x, y, z):
        return (self.xmin <= x) & (x <= self.xmax) & (self.ymin <= y) & (y <= self.ymax) & (self.zmin <= z) & (z <= self.zmax)

def module_boxes_cm() -> Dict[int, Box]:
    """Module boundaries from the DUNE ND-LAr 2x2 Geometry Description technote (Table 1).

    Coordinate system:
      - right-handed
      - z: downstream (beam direction approx +z)
      - y: up
      - x: horizontal (parallel/antiparallel to drift)

    Returns
    -------
    dict: module_id -> Box(xmin,xmax,ymin,ymax,zmin,zmax) in cm
    """
    y = (-61.85, 61.85)
    boxes = {
        0: Box(  3.07, 63.93, y[0], y[1],  2.68, 64.32),
        1: Box(  3.07, 63.93, y[0], y[1], -64.32, -2.68),
        2: Box(-63.93, -3.07, y[0], y[1],  2.68, 64.32),
        3: Box(-63.93, -3.07, y[0], y[1], -64.32, -2.68),
    }
    return boxes

def module_centers_cm() -> Dict[int, Tuple[float, float, float]]:
    """Approx module centers (from technote text: centers at ±33.5 cm in x and z; y ~ 0)."""
    # Using the sign convention consistent with Table 1 bounds.
    # For x: modules 0/1 are +x, 2/3 are -x
    # For z: modules 0/2 are +z, 1/3 are -z
    return {
        0: ( 33.5, 0.0,  33.5),
        1: ( 33.5, 0.0, -33.5),
        2: (-33.5, 0.0,  33.5),
        3: (-33.5, 0.0, -33.5),
    }

def module_id_from_xyz_cm(x: float, y: float, z: float) -> Optional[int]:
    """Return module id if point lies inside one of the module boxes, else None."""
    for mid, box in module_boxes_cm().items():
        if (box.xmin <= x <= box.xmax) and (box.ymin <= y <= box.ymax) and (box.zmin <= z <= box.zmax):
            return mid
    return None
