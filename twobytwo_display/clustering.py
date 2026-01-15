from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from sklearn.cluster import DBSCAN
except ImportError as e:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from e


@dataclass
class ClusterSummary:
    label: int
    n_hits: int
    centroid: np.ndarray          # (3,)
    direction: np.ndarray         # (3,) unit vector (PCA main axis)
    theta_z_rad: float
    total_Q: float
    extent_rms_cm: float
    extent_max_cm: float


def _pca_line_direction(X: np.ndarray) -> np.ndarray:
    """
    X: (N,3) centered coordinates.
    Returns: unit vector of largest-variance direction.
    """
    # SVD on centered coords: X = U S Vt; principal axis is Vt[0]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    # normalize (should already be)
    v = v / np.linalg.norm(v)
    return v


def _angle_to_z(v: np.ndarray) -> float:
    # angle to +z axis; use abs so up/down gives same theta
    vz = float(np.clip(abs(v[2]), 0.0, 1.0))
    return float(np.arccos(vz))


def dbscan_clusters(
    hits,
    *,
    eps_cm: float = 1.5,
    min_samples: int = 10,
    use_charge_weight: bool = False,
    q_field: str = "Q",
    mask: Optional[np.ndarray] = None,
) -> List[ClusterSummary]:
    """
    Cluster hits in (x,y,z) using DBSCAN and compute PCA line + angle to z.
    `hits` is your calib_prompt_hits structured array with fields x,y,z, Q.
    """
    if len(hits) == 0:
        return []

    if mask is None:
        mask = np.ones(len(hits), dtype=bool)
    hits = hits[mask]
    if len(hits) == 0:
        return []

    xyz = np.vstack([hits["x"], hits["y"], hits["z"]]).T.astype(np.float32)

    labels = DBSCAN(eps=eps_cm, min_samples=min_samples).fit_predict(xyz)

    out: List[ClusterSummary] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue  # noise

        m = labels == lab
        pts = xyz[m]
        n = int(pts.shape[0])
        if n < 2:
            continue

        # centroid
        if use_charge_weight and q_field in hits.dtype.names:
            w = hits[q_field][m].astype(float)
            w = np.clip(w, 0.0, None)
            if w.sum() > 0:
                centroid = np.average(pts, axis=0, weights=w)
            else:
                centroid = pts.mean(axis=0)
        else:
            centroid = pts.mean(axis=0)

        centered = pts - centroid[None, :]
        v = _pca_line_direction(centered)
        theta = _angle_to_z(v)

        # extents
        r = np.linalg.norm(centered, axis=1)
        extent_rms = float(np.sqrt(np.mean(r**2)))
        extent_max = float(np.max(r))

        total_Q = float(np.sum(hits[q_field][m])) if q_field in hits.dtype.names else float("nan")

        out.append(ClusterSummary(
            label=int(lab),
            n_hits=n,
            centroid=centroid.astype(float),
            direction=v.astype(float),
            theta_z_rad=theta,
            total_Q=total_Q,
            extent_rms_cm=extent_rms,
            extent_max_cm=extent_max,
        ))

    return out
