from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import plotly.graph_objects as go

from .geometry import module_boxes_cm


# ----------------------------
# Geometry / distance helpers
# ----------------------------
def _point_to_segment_distance(P, A, B):
    """
    Distance from points P (N,3) to line segment AB (3,) in 3D.
    """
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
    """
    Return integer labels per hit:
      0 = core (<= r_core)
      1 = near (r_core < d <= r_near)
      2 = far  (> r_near)
    """
    if muon_track is None or len(hits) == 0:
        return np.full(len(hits), 2, dtype=np.int8)  # all far

    A = np.array([muon_track["x_start"], muon_track["y_start"], muon_track["z_start"]], dtype=float)
    B = np.array([muon_track["x_end"],   muon_track["y_end"],   muon_track["z_end"]], dtype=float)
    P = np.vstack([hits["x"], hits["y"], hits["z"]]).T.astype(float)

    d = _point_to_segment_distance(P, A, B)
    lab = np.full(len(hits), 2, dtype=np.int8)
    lab[d <= r_near] = 1
    lab[d <= r_core] = 0
    return lab


# ----------------------------
# Coloring helpers
# ----------------------------
def _safe_log10(x, eps=1e-12):
    x = np.asarray(x)
    return np.log10(np.clip(x, eps, None))


def color_array(hits, mode: str, muon_track=None, r_core=5.0, r_near=25.0):
    """
    Returns (c, clabel)
      - For continuous modes: c is float array, use colormap + colorbar
      - For muon_region: c is int labels {0,1,2}, treat as categorical
    """
    if mode == "Q":
        c = hits["Q"].astype(float)
        c = np.where(np.isfinite(c), c, 0.0)
        return _safe_log10(c), "log10(Q)"
    if mode == "t_drift":
        c = hits["t_drift"].astype(float)
        return c, "t_drift"
    if mode == "ts_pps":
        c = hits["ts_pps"].astype(float)
        c = c - np.nanmin(c)
        return c, "ts_pps - min"
    if mode == "muon_region":
        lab = muon_region_labels(hits, muon_track, r_core=r_core, r_near=r_near)
        return lab, f"muon region (core≤{r_core:g} cm, near≤{r_near:g} cm)"
    raise ValueError("mode must be one of: Q, t_drift, ts_pps, muon_region")


# ----------------------------
# Matplotlib projections helpers
# ----------------------------
def _draw_boxes_zx(ax, boxes, **kwargs):
    for mid, b in boxes.items():
        ax.plot([b.zmin, b.zmax, b.zmax, b.zmin, b.zmin],
                [b.xmin, b.xmin, b.xmax, b.xmax, b.xmin], **kwargs)
        ax.text((b.zmin + b.zmax) / 2, (b.xmin + b.xmax) / 2, str(mid),
                ha="center", va="center")


def _draw_boxes_zy(ax, boxes, y_extent_cm=(-61.85, 61.85), **kwargs):
    # NOTE: In ZY projection, modules that differ only in X overlap.
    y0, y1 = y_extent_cm
    slabs = {}
    for mid, b in boxes.items():
        slabs.setdefault((b.zmin, b.zmax), []).append(mid)
    for (zmin, zmax), mids in slabs.items():
        ax.plot([zmin, zmax, zmax, zmin, zmin],
                [y0,   y0,   y1,   y1,   y0], **kwargs)
        ax.text((zmin + zmax) / 2, (y0 + y1) / 2, f"modules {sorted(mids)}",
                ha="center", va="center")


def _scatter_muon_regions_2d(ax, z, v, labels, point_size, r_core, r_near):
    """
    Make 3 scatters so Matplotlib legend can show 3 entries.
    Returns list of handles for a legend.
    """
    labels = labels.astype(int)

    m_far  = labels == 2
    m_near = labels == 1
    m_core = labels == 0

    # Plot far first, then near, then core on top (so core is visible)
    ax.scatter(z[m_far],  v[m_far],  s=point_size, color="gray",   label="far")
    ax.scatter(z[m_near], v[m_near], s=point_size, color="orange", label=f"near (≤ {r_near:g} cm)")
    ax.scatter(z[m_core], v[m_core], s=point_size, color="red",    label=f"core (≤ {r_core:g} cm)")

    # Use proxy artists for a consistent legend (works even if a region has 0 points)
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor="red",    markeredgecolor="red",
               label=f"core (≤ {r_core:g} cm)"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor="orange", markeredgecolor="orange",
               label=f"near (≤ {r_near:g} cm)"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, markerfacecolor="gray",   markeredgecolor="gray",
               label="far"),
    ]
    return handles


# ----------------------------
# Public: Matplotlib figure
# ----------------------------
def make_matplotlib_figure(
    hits,
    color_mode="Q",
    max_hits=40000,
    point_size=2,
    show_boxes=True,
    muon_track=None,
    r_core=5.0,
    r_near=25.0
):
    """
    Return a matplotlib Figure with:
      - 3D (static) view
      - ZX projection
      - ZY projection
      - Colorbar (continuous modes) OR legend (muon_region)
    """
    if len(hits) == 0:
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle("No hits in event")
        return fig

    if len(hits) > max_hits:
        idx = np.random.choice(len(hits), size=max_hits, replace=False)
        hits = hits[idx]

    x = hits["x"].astype(float)
    y = hits["y"].astype(float)
    z = hits["z"].astype(float)

    c, clabel = color_array(hits, color_mode, muon_track=muon_track, r_core=r_core, r_near=r_near)

    fig = plt.figure(figsize=(17, 5))
    gs = GridSpec(nrows=1, ncols=4, width_ratios=[1.2, 1.0, 1.0, 0.22], wspace=0.25)

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_zx = fig.add_subplot(gs[0, 1])
    ax_zy = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    # --- 3D (static) ---
    if color_mode == "muon_region":
        # show categorical colors directly
        colors = np.array(["red", "orange", "gray"], dtype=object)
        c3 = colors[c.astype(int)]
        ax3.scatter(z, x, y, s=point_size, c=c3)
    else:
        ax3.scatter(z, x, y, s=point_size, c=c)

    ax3.set_xlabel("z [cm]")
    ax3.set_ylabel("x [cm]")
    ax3.set_zlabel("y [cm]")
    ax3.set_title("3D (static)")

    # --- ZX + ZY ---
    if color_mode == "muon_region":
        labels = c
        handles = _scatter_muon_regions_2d(ax_zx, z, x, labels, point_size, r_core, r_near)
        _scatter_muon_regions_2d(ax_zy, z, y, labels, point_size, r_core, r_near)

        # Put legend in the cax column (so it never covers data)
        cax.axis("off")
        cax.legend(handles=handles, loc="upper left", frameon=True, title="Muon region")

    else:
        sc_zx = ax_zx.scatter(z, x, s=point_size, c=c)
        sc_zy = ax_zy.scatter(z, y, s=point_size, c=c)

        cb = fig.colorbar(sc_zy, cax=cax)
        cb.set_label(clabel)

    ax_zx.set_xlabel("z [cm]")
    ax_zx.set_ylabel("x [cm]")
    ax_zx.set_title("ZX")
    ax_zx.set_aspect("equal", adjustable="box")

    ax_zy.set_xlabel("z [cm]")
    ax_zy.set_ylabel("y [cm]")
    ax_zy.set_title("ZY")

    # --- module overlays ---
    if show_boxes:
        boxes = module_boxes_cm()
        _draw_boxes_zx(ax_zx, boxes, color="k", linewidth=1)
        _draw_boxes_zy(ax_zy, boxes, color="k", linewidth=1)

    # --- muon line overlay ---
    if muon_track is not None:
        A = np.array([muon_track["x_start"], muon_track["y_start"], muon_track["z_start"]], dtype=float)
        B = np.array([muon_track["x_end"],   muon_track["y_end"],   muon_track["z_end"]], dtype=float)

        ax3.plot([A[2], B[2]], [A[0], B[0]], [A[1], B[1]], linewidth=3, label="rock muon track")
        ax_zx.plot([A[2], B[2]], [A[0], B[0]], linewidth=3, label="rock muon track")
        ax_zy.plot([A[2], B[2]], [A[1], B[1]], linewidth=3, label="rock muon track")

        # If we are in categorical mode, we can add the track to the legend area too
        if color_mode == "muon_region":
            track_handle = Line2D([0], [0], color="blue", linewidth=3, label="rock muon track")
            # Add it to the legend already in cax
            leg = cax.get_legend()
            if leg is not None:
                handles = leg.legend_handles + [track_handle]
                labels = [h.get_label() for h in handles]
                cax.clear()
                cax.axis("off")
                cax.legend(handles=handles, labels=labels, loc="upper left", frameon=True, title="Muon region")
        else:
            # Continuous mode: add legend on a plot axis (optional)
            # (keeps behavior minimal; comment out if you don’t want it)
            ax_zx.legend(loc="upper right", frameon=True)

    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.05, right=0.98)
    return fig


# ----------------------------
# Public: Plotly 3D (interactive)
# ----------------------------
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

    if len(hits) > max_hits:
        idx = np.random.choice(len(hits), size=max_hits, replace=False)
        hits = hits[idx]

    x = hits["x"].astype(float)
    y = hits["y"].astype(float)
    z = hits["z"].astype(float)

    c, clabel = color_array(hits, color_mode, muon_track=muon_track, r_core=r_core, r_near=r_near)

    fig = go.Figure()

    if color_mode == "muon_region":
        labels = c.astype(int)

        for lab, name, col in [(0, f"core (≤ {r_core:g} cm)", "red"),
                               (1, f"near (≤ {r_near:g} cm)", "orange"),
                               (2, "far", "gray")]:
            m = labels == lab
            fig.add_trace(go.Scatter3d(
                x=z[m], y=x[m], z=y[m],  # (z,x,y)
                mode="markers",
                marker=dict(size=point_size, color=col),
                name=name,
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=z, y=x, z=y,  # (z,x,y)
            mode="markers",
            marker=dict(
                size=point_size,
                color=c,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=clabel,
                    len=0.65,          # <-- 65% of figure height
                    y=0.45,            # <-- slightly lower than center
                    yanchor="top",
                    thickness=18,      # <-- slimmer bar
                ),
            ),
            name="hits",
        ))

    if show_boxes:
        boxes = module_boxes_cm()
        for mid, b in boxes.items():
            corners = np.array([
                [b.xmin, b.ymin, b.zmin],
                [b.xmax, b.ymin, b.zmin],
                [b.xmax, b.ymax, b.zmin],
                [b.xmin, b.ymax, b.zmin],
                [b.xmin, b.ymin, b.zmax],
                [b.xmax, b.ymin, b.zmax],
                [b.xmax, b.ymax, b.zmax],
                [b.xmin, b.ymax, b.zmax],
            ], dtype=float)

            edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            ex, ey, ez = [], [], []
            for i, j in edges:
                ex += [corners[i, 2], corners[j, 2], None]  # z
                ey += [corners[i, 0], corners[j, 0], None]  # x
                ez += [corners[i, 1], corners[j, 1], None]  # y

            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(width=2),
                name=f"module {mid}",
                showlegend=False,
                opacity=0.6,
            ))

    if muon_track is not None:
        A = np.array([muon_track["x_start"], muon_track["y_start"], muon_track["z_start"]], dtype=float)
        B = np.array([muon_track["x_end"],   muon_track["y_end"],   muon_track["z_end"]], dtype=float)

        fig.add_trace(go.Scatter3d(
            x=[A[2], B[2]], y=[A[0], B[0]], z=[A[1], B[1]],
            mode="lines",
            line=dict(width=6),
            name="rock muon track",
        ))

    # ----------------------------
    # Cluster overlays (optional)
    #   - centroid markers always
    #   - ONE fitted line through centroids if >= 2 clusters (scenario 2)
    # ----------------------------
    if clusters:
        # --- centroid markers ---
        for ci, csum in enumerate(clusters):
            cen = np.asarray(csum.centroid, dtype=float)

            fig.add_trace(go.Scatter3d(
                x=[cen[2]], y=[cen[0]], z=[cen[1]],  # (z,x,y)
                mode="markers",
                marker=dict(size=7),
                name=f"cluster {ci} centroid",
                showlegend=True if ci == 0 else False,  # avoid legend spam
                hoverinfo="text",
                text=[(
                    f"cluster {ci} centroid<br>"
                    f"nhits={csum.n_hits}<br>"
                    f"sumQ={csum.total_Q:.2g}<br>"
                    f"rms={csum.extent_rms_cm:.2f} cm<br>"
                    f"max={csum.extent_max_cm:.2f} cm"
                )],
            ))

        # --- fitted centroid line (only if >= 2 clusters) ---
        if len(clusters) >= 2:
            P = np.array([c.centroid for c in clusters], dtype=float)  # (K,3)

            # Best-fit line direction via PCA on centroids (works for K=2 too)
            x0 = P.mean(axis=0)
            X = P - x0[None, :]
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            v = vt[0]
            v = v / np.linalg.norm(v)

            # Choose a segment that spans the centroids along the fitted direction
            t = X @ v
            t0, t1 = float(np.min(t)), float(np.max(t))
            pad = 10.0  # cm padding on both ends (tune)
            a = x0 + (t0 - pad) * v
            b = x0 + (t1 + pad) * v

            # Angle to z (treat up/down as same)
            vz = float(np.clip(abs(v[2]), 0.0, 1.0))
            theta_deg = float(np.degrees(np.arccos(vz)))

            fig.add_trace(go.Scatter3d(
                x=[a[2], b[2]], y=[a[0], b[0]], z=[a[1], b[1]],  # (z,x,y)
                mode="lines",
                line=dict(width=7),
                name="centroid-line fit",
                showlegend=True,
                hoverinfo="text",
                text=[f"centroid-line fit<br>n_clusters={len(clusters)}<br>theta_z={theta_deg:.1f}°"] * 2,
            ))

    # ----------------------------
    # MC truth overlay (segments + optional vertices)
    # ----------------------------
    if mc_segments is not None:
        segs = mc_segments

        if mc_only_muons and "pdg_id" in segs.dtype.names:
            segs = segs[np.abs(segs["pdg_id"].astype(int)) == 13]

        # downsample if too many segments (keeps UI responsive)
        if mc_max_segments and len(segs) > mc_max_segments:
            idx = np.random.choice(len(segs), size=mc_max_segments, replace=False)
            segs = segs[idx]

        if len(segs) > 0:
            # Plotly line segments: use None separators to break lines
            xline, yline, zline = [], [], []
            hover = []
            for s in segs:
                # stored as (x_start,y_start,z_start) and (x_end,y_end,z_end)
                A = (float(s["x_start"]), float(s["y_start"]), float(s["z_start"]))
                B = (float(s["x_end"]),   float(s["y_end"]),   float(s["z_end"]))
                pdg = int(s["pdg_id"]) if "pdg_id" in segs.dtype.names else 0
                dE  = float(s["dE"]) if "dE" in segs.dtype.names else float("nan")
                xline += [A[2], B[2], None]  # z -> plotly x
                yline += [A[0], B[0], None]  # x -> plotly y
                zline += [A[1], B[1], None]  # y -> plotly z
                hover += [f"pdg={pdg}<br>dE={dE:.3g}", f"pdg={pdg}<br>dE={dE:.3g}", None]

            fig.add_trace(go.Scatter3d(
                x=xline, y=yline, z=zline,
                mode="lines",
                line=dict(width=4),
                name=mc_label,
                opacity=0.7,
                hoverinfo="text",
                text=hover,
            ))

    if mc_vertices is not None and len(mc_vertices) > 0:
        # Vertex is stored as a (4,) array in field 'vertex': (x,y,z,t) usually
        vx = mc_vertices["vertex"][:, 0].astype(float)
        vy = mc_vertices["vertex"][:, 1].astype(float)
        vz = mc_vertices["vertex"][:, 2].astype(float)

        fig.add_trace(go.Scatter3d(
            x=vz, y=vx, z=vy,  # (z,x,y)
            mode="markers",
            marker=dict(size=6, symbol="diamond"),
            name="MC vertices",
            opacity=0.9,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="z [cm]",
            yaxis_title="x [cm]",
            zaxis_title="y [cm]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        title="3D view (interactive)",
        legend=dict(itemsizing="constant"),
    )
    return fig
