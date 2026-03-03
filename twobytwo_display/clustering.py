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
    centroid: np.ndarray
    total_Q: float
    extent_rms_cm: float
    extent_max_cm: float


def dbscan_clusters(
    hits,
    *,
    eps_cm: float = 1.5,
    min_samples: int = 10,
    use_charge_weight: bool = False,
    q_field: str = "Q",
    mask: Optional[np.ndarray] = None,
    debug: bool = False,
) -> List[ClusterSummary]:

    if len(hits) == 0:
        if debug: print("[DBSCAN] hits is empty (len=0)")
        return []

    if mask is None:
        mask = np.ones(len(hits), dtype=bool)

    if debug:
        print(f"[DBSCAN] total hits: {len(hits)}  mask kept: {int(mask.sum())}")

    hits = hits[mask]
    if len(hits) == 0:
        if debug: print("[DBSCAN] after mask: 0 hits")
        return []

    # Check fields exist
    for f in ["x", "y", "z"]:
        if f not in hits.dtype.names:
            raise ValueError(f"[DBSCAN] hits is missing field '{f}'. Available: {hits.dtype.names}")

    xyz = np.vstack([hits["x"], hits["y"], hits["z"]]).T.astype(np.float32)

    # Basic sanity checks
    finite = np.isfinite(xyz).all(axis=1)
    if debug:
        print(f"[DBSCAN] finite xyz: {int(finite.sum())}/{len(xyz)}")
        if finite.sum() < len(xyz):
            bad = np.where(~finite)[0][:5]
            print("[DBSCAN] example bad rows:", xyz[bad])

    xyz = xyz[finite]
    if len(xyz) == 0:
        if debug: print("[DBSCAN] all xyz were non-finite after filtering")
        return []

    if debug:
        mins = xyz.min(axis=0); maxs = xyz.max(axis=0)
        spans = maxs - mins
        print(f"[DBSCAN] xyz min: {mins}  max: {maxs}  span: {spans}")
        # Typical nearest-neighbor scale hint
        if len(xyz) >= 2:
            # cheap-ish: sample a subset if huge
            M = min(len(xyz), 2000)
            samp = xyz[np.random.choice(len(xyz), size=M, replace=False)]
            # compute nearest neighbor distance approx
            # (O(M^2) but M<=2000 => ok-ish; if too slow drop to 800)
            d2 = ((samp[:, None, :] - samp[None, :, :])**2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(d2.min(axis=1))
            print(f"[DBSCAN] approx NN dist: median={np.median(nn):.3g}, p90={np.percentile(nn,90):.3g}")

    labels = DBSCAN(eps=eps_cm, min_samples=min_samples).fit_predict(xyz)

    if debug:
        n_noise = int(np.sum(labels == -1))
        labs = sorted(set(labels))
        print(f"[DBSCAN] labels: {labs[:10]}{'...' if len(labs)>10 else ''}")
        print(f"[DBSCAN] noise points: {n_noise}/{len(labels)}")
        # cluster sizes
        for lab in sorted(set(labels)):
            if lab == -1: 
                continue
            print(f"[DBSCAN] cluster {lab}: n={int(np.sum(labels==lab))}")

    out: List[ClusterSummary] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue  # noise

        m = labels == lab
        pts = xyz[m]
        n = int(pts.shape[0])
        if n < 1:
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

        # centroid computed above...
        centered = pts - centroid[None, :]
        
        # extents
        r = np.linalg.norm(centered, axis=1)
        extent_rms = float(np.sqrt(np.mean(r**2)))
        extent_max = float(np.max(r))

        total_Q = float(np.sum(hits[q_field][m])) if q_field in hits.dtype.names else float("nan")

        out.append(ClusterSummary(
            label=int(lab),
            n_hits=n,
            centroid=centroid.astype(float),
            total_Q=total_Q,
            extent_rms_cm=extent_rms,
            extent_max_cm=extent_max,
        ))

    return out

def angle_to_z_of_centroid_line(clusters) -> float | None:
    """
    Fit a line to cluster centroids (PCA) and return angle to +z in radians.
    Returns None if fewer than 2 clusters.
    """
    if clusters is None or len(clusters) < 2:
        return None

    P = np.array([c.centroid for c in clusters], dtype=float)  # (K,3)
    x0 = P.mean(axis=0)
    X = P - x0[None, :]

    # PCA via SVD
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    v = v / np.linalg.norm(v)

    # angle to z (same convention you used elsewhere: up/down treated same)
    vz = float(np.clip(abs(v[2]), 0.0, 1.0))
    return float(np.arccos(vz))