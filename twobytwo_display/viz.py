from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .geometry import module_boxes_cm


def _point_to_segment_distance(P, A, B):
    AB = B - A
    AP = P - A
    denom = float(np.dot(AB, AB))
    if denom <= 0:
        return np.linalg.norm(P - A, axis=1)
    t = (AP @ AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB[None, :]
    return np.linalg.norm(P - proj, axis=1)


def muon_region_labels(hits, muon_track, r_core=5.0, r_near=25.0):
    if muon_track is None or len(hits) == 0:
        return np.full(len(hits), 2, dtype=np.int8)
    A = np.array([muon_track["x_start"], muon_track["y_start"], muon_track["z_start"]], dtype=float)
    B = np.array([muon_track["x_end"], muon_track["y_end"], muon_track["z_end"]], dtype=float)
    P = np.vstack([hits["x"], hits["y"], hits["z"]]).T.astype(float)
    d = _point_to_segment_distance(P, A, B)
    lab = np.full(len(hits), 2, dtype=np.int8)
    lab[d <= r_near] = 1
    lab[d <= r_core] = 0
    return lab


def _safe_log10(x, eps=1e-12):
    return np.log10(np.clip(np.asarray(x), eps, None))


def color_array(hits, mode: str, muon_track=None, r_core=5.0, r_near=25.0):
    if mode == "Q":
        c = np.where(np.isfinite(hits["Q"].astype(float)), hits["Q"].astype(float), 0.0)
        return _safe_log10(c), "log10(Q)"
    if mode == "t_drift":
        return hits["t_drift"].astype(float), "t_drift"
    if mode == "ts_pps":
        c = hits["ts_pps"].astype(float)
        return c - np.nanmin(c), "ts_pps - min"
    if mode == "muon_region":
        return muon_region_labels(hits, muon_track, r_core=r_core, r_near=r_near), "muon region"
    raise ValueError("mode must be one of: Q, t_drift, ts_pps, muon_region")


def _sample_hits(hits, max_hits: int):
    if len(hits) > max_hits:
        idx = np.random.choice(len(hits), size=max_hits, replace=False)
        return hits[idx]
    return hits


def _hover_customdata(hits):
    module = hits["iogroup"].astype(int) if "iogroup" in (hits.dtype.names or ()) else np.full(len(hits), -1)
    return np.stack([
        hits["Q"].astype(float),
        hits["t_drift"].astype(float) if "t_drift" in (hits.dtype.names or ()) else np.full(len(hits), np.nan),
        hits["ts_pps"].astype(float) if "ts_pps" in (hits.dtype.names or ()) else np.full(len(hits), np.nan),
        module,
    ], axis=1)


def make_plotly_2d_projections(hits, color_mode="Q", max_hits=40000, point_size=3, muon_track=None):
    if len(hits) == 0:
        fig = go.Figure()
        fig.update_layout(title="No hits in event")
        return fig

    hits = _sample_hits(hits, max_hits)
    c, clabel = color_array(hits, color_mode, muon_track=muon_track)
    customdata = _hover_customdata(hits)
    hover = "x=%{x:.2f}<br>y=%{y:.2f}<br>Q=%{customdata[0]:.3g}<br>t_drift=%{customdata[1]:.3g}<br>ts_pps=%{customdata[2]:.3g}<br>module=%{customdata[3]}<extra></extra>"

    fig = make_subplots(rows=2, cols=2, subplot_titles=("XY", "XZ", "YZ", "t_drift vs Q"))
    projections = [
        (hits["x"], hits["y"], 1, 1, "x [cm]", "y [cm]"),
        (hits["x"], hits["z"], 1, 2, "x [cm]", "z [cm]"),
        (hits["y"], hits["z"], 2, 1, "y [cm]", "z [cm]"),
        (hits["t_drift"] if "t_drift" in (hits.dtype.names or ()) else np.arange(len(hits)), hits["Q"], 2, 2, "t_drift", "Q"),
    ]

    for i, (xx, yy, r, col, xl, yl) in enumerate(projections):
        fig.add_trace(
            go.Scattergl(
                x=xx,
                y=yy,
                mode="markers",
                marker=dict(size=point_size, color=c, colorscale="Viridis", showscale=(i == 0), colorbar=dict(title=clabel)),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            ),
            row=r,
            col=col,
        )
        fig.update_xaxes(title_text=xl, row=r, col=col)
        fig.update_yaxes(title_text=yl, row=r, col=col)

    fig.update_layout(height=780, margin=dict(l=10, r=10, t=40, b=10), title="2D projections")
    return fig


def make_plotly_3d(
    hits,
    color_mode="Q",
    max_hits=40000,
    point_size=2,
    show_boxes=True,
    muon_track=None,
    r_core=5.0,
    r_near=25.0,
    clusters=None,
    mc_segments=None,
    mc_vertices=None,
    mc_max_segments=3000,
    mc_only_muons=False,
    mc_label="MC truth segments",
):
    if len(hits) == 0:
        fig = go.Figure()
        fig.update_layout(title="No hits in event")
        return fig

    hits = _sample_hits(hits, max_hits)
    c, clabel = color_array(hits, color_mode, muon_track=muon_track, r_core=r_core, r_near=r_near)

    fig = go.Figure()
    if color_mode == "muon_region":
        labels = c.astype(int)
        for lab, name, col in [(0, f"core (≤ {r_core:g} cm)", "red"), (1, f"near (≤ {r_near:g} cm)", "orange"), (2, "far", "gray")]:
            m = labels == lab
            fig.add_trace(go.Scatter3d(x=hits["z"][m], y=hits["x"][m], z=hits["y"][m], mode="markers", marker=dict(size=point_size, color=col), name=name))
    else:
        customdata = np.stack([hits["Q"].astype(float)], axis=1)
        fig.add_trace(go.Scatter3d(
            x=hits["z"], y=hits["x"], z=hits["y"], mode="markers", name="hits",
            marker=dict(size=point_size, color=c, colorscale="Viridis", showscale=True, colorbar=dict(title=clabel)),
        ))

    if show_boxes:
        for _, b in module_boxes_cm().items():
            corners = np.array([[b.xmin, b.ymin, b.zmin], [b.xmax, b.ymin, b.zmin], [b.xmax, b.ymax, b.zmin], [b.xmin, b.ymax, b.zmin],
                                [b.xmin, b.ymin, b.zmax], [b.xmax, b.ymin, b.zmax], [b.xmax, b.ymax, b.zmax], [b.xmin, b.ymax, b.zmax]], dtype=float)
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            ex, ey, ez = [], [], []
            for i, j in edges:
                ex += [corners[i, 2], corners[j, 2], None]
                ey += [corners[i, 0], corners[j, 0], None]
                ez += [corners[i, 1], corners[j, 1], None]
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2), showlegend=False, opacity=0.5))

    if muon_track is not None:
        A = np.array([muon_track["x_start"], muon_track["y_start"], muon_track["z_start"]], dtype=float)
        B = np.array([muon_track["x_end"], muon_track["y_end"], muon_track["z_end"]], dtype=float)
        fig.add_trace(go.Scatter3d(x=[A[2], B[2]], y=[A[0], B[0]], z=[A[1], B[1]], mode="lines", line=dict(width=6), name="rock muon track"))

    if clusters:
        for i, csum in enumerate(clusters):
            cen = np.asarray(csum.centroid, dtype=float)
            fig.add_trace(go.Scatter3d(x=[cen[2]], y=[cen[0]], z=[cen[1]], mode="markers", marker=dict(size=7), name=f"cluster {i}" if i == 0 else "cluster", showlegend=(i == 0)))

    if mc_segments is not None and len(mc_segments):
        segs = mc_segments
        if mc_only_muons and "pdg_id" in (segs.dtype.names or ()):
            segs = segs[np.abs(segs["pdg_id"].astype(int)) == 13]
        if mc_max_segments and len(segs) > mc_max_segments:
            segs = segs[np.random.choice(len(segs), size=mc_max_segments, replace=False)]

        xline, yline, zline = [], [], []
        for s in segs:
            xline += [float(s["z_start"]), float(s["z_end"]), None]
            yline += [float(s["x_start"]), float(s["x_end"]), None]
            zline += [float(s["y_start"]), float(s["y_end"]), None]
        fig.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode="lines", line=dict(width=4), name=mc_label, opacity=0.7))

    if mc_vertices is not None and len(mc_vertices) > 0 and "vertex" in (mc_vertices.dtype.names or ()):
        vx = mc_vertices["vertex"][:, 0].astype(float)
        vy = mc_vertices["vertex"][:, 1].astype(float)
        vz = mc_vertices["vertex"][:, 2].astype(float)
        hovertext = []
        for row in mc_vertices:
            eid = int(row["event_id"]) if "event_id" in (mc_vertices.dtype.names or ()) else -1
            iid = int(row["interaction_id"]) if "interaction_id" in (mc_vertices.dtype.names or ()) else -1
            vid = int(row["vertex_id"]) if "vertex_id" in (mc_vertices.dtype.names or ()) else -1
            hovertext.append(f"event_id={eid}<br>interaction_id={iid}<br>vertex_id={vid}")
        fig.add_trace(go.Scatter3d(x=vz, y=vx, z=vy, mode="markers", marker=dict(size=6, symbol="diamond"), name="MC vertices", text=hovertext, hoverinfo="text"))

    fig.update_layout(scene=dict(xaxis_title="z [cm]", yaxis_title="x [cm]", zaxis_title="y [cm]", aspectmode="data"), margin=dict(l=0, r=0, t=35, b=0), title="3D view (interactive)")
    return fig
