"""Panel event display UI.

Run:
  panel serve -m twobytwo_display.app_panel --show --args --h5 /path/to/file.FLOW.hdf5
"""

from __future__ import annotations

import argparse
import bisect
import os

import numpy as np
import panel as pn
import plotly.graph_objects as go

from twobytwo_display.clustering import angle_to_z_of_centroid_line, dbscan_clusters
from twobytwo_display.io import FlowFile
from twobytwo_display.viz import (
    make_plotly_2d_projections,
    make_plotly_3d,
    make_plotly_analysis,
    muon_region_labels,
)

pn.extension("plotly")


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--h5", type=str, default=os.environ.get("TWOBYTWO_H5", ""))
    parser.add_argument("--max_hits", type=int, default=40000)
    return parser.parse_known_args()[0]


class AppState:
    def __init__(self, path: str, max_hits: int):
        self.path = path
        self.max_hits = max_hits
        self.flow = None
        self.muon_indices = []

    def open(self, path: str):
        self.close()
        self.path = path
        if not path:
            return
        self.flow = FlowFile.open(path)
        self.muon_indices = self.flow.muon_event_indices()

    def close(self):
        if self.flow is not None:
            self.flow.close()
            self.flow = None
        self.muon_indices = []


ARGS = _parse_args()
state = AppState(ARGS.h5, ARGS.max_hits)
if state.path:
    state.open(state.path)

# ---------- widgets ----------
file_input = pn.widgets.TextInput(name="HDF5 path", value=state.path)
open_btn = pn.widgets.Button(name="Open", button_type="primary")
status = pn.pane.Markdown("", height=160)

# navigation
event_slider = pn.widgets.IntSlider(name="Event", start=0, end=0, value=0)
event_input = pn.widgets.IntInput(name="Jump to event", value=0, step=1)
prev_btn = pn.widgets.Button(name="◀ Prev", width=90)
next_btn = pn.widgets.Button(name="Next ▶", button_type="primary", width=90)
muon_only = pn.widgets.Toggle(name="Muon-only scan", value=False)

# display
hit_type = pn.widgets.Select(name="Hit type", options=["prompt", "final"], value="prompt")
color_mode = pn.widgets.Select(name="Color mode", options=["Q", "t_drift", "ts_pps", "muon_region"], value="Q")
max_hits = pn.widgets.IntInput(name="Max hits", value=ARGS.max_hits, step=1000, start=1000)
point_size = pn.widgets.IntSlider(name="Point size", start=1, end=10, value=2)
show_boxes = pn.widgets.Checkbox(name="Show geometry boxes", value=True)

# truth
show_truth = pn.widgets.Checkbox(name="Enable truth overlay", value=False)
truth_mode = pn.widgets.Select(name="Truth mode", options=["backtrack", "window"], value="backtrack")
truth_event = pn.widgets.Select(name="Window truth event_id", options=["auto"], value="auto")
show_vertices = pn.widgets.Checkbox(name="Show truth vertices", value=True)
show_all_window_vertices = pn.widgets.Checkbox(name="Window mode: all vertices in window", value=False)
mc_only_muons = pn.widgets.Checkbox(name="Truth only muons (|pdg|=13)", value=False)
mc_max_segments = pn.widgets.IntInput(name="Draw max truth segments", value=3000, step=500, start=0)
mc_topk = pn.widgets.IntInput(name="Backtrack top-K segments", value=2000, step=200, start=0)
mc_minw = pn.widgets.FloatInput(name="Backtrack min weight", value=0.0, step=0.01)

# muon
show_muon = pn.widgets.Checkbox(name="Show rock muon track", value=False)

# clustering
show_clusters = pn.widgets.Checkbox(name="Enable DBSCAN clusters", value=False)
db_eps = pn.widgets.FloatInput(name="DBSCAN eps [cm]", value=1.5, step=0.1)
db_min = pn.widgets.IntInput(name="DBSCAN min_samples", value=10, step=1)
cluster_min_hits = pn.widgets.IntInput(name="Keep nhits ≥", value=20, step=1)
cluster_max_extent = pn.widgets.FloatInput(name="Keep max extent ≤ [cm]", value=8.0, step=0.5)
clusters_info = pn.pane.Markdown("", height=180)

view3d = pn.pane.Plotly(sizing_mode="stretch_both")
view2d = pn.pane.Plotly(sizing_mode="stretch_both")
view_analysis = pn.pane.Plotly(sizing_mode="stretch_both")


def _set_status(message: str, ok: bool = True):
    icon = "✅" if ok else "❌"
    status.object = f"### Status {icon}\n{message}"


def _sync_slider_bounds():
    if state.flow is None:
        event_slider.start = event_slider.end = event_slider.value = 0
        event_input.value = 0
        return
    end = max(0, state.flow.n_events() - 1)
    event_slider.start, event_slider.end = 0, end
    event_slider.value = min(event_slider.value, end)
    event_input.value = int(event_slider.value)


def _goto_event(ev: int):
    event_slider.value = max(event_slider.start, min(event_slider.end, int(ev)))


def _next_event(_=None):
    if state.flow is None:
        return
    if muon_only.value and state.muon_indices:
        pos = bisect.bisect_right(state.muon_indices, int(event_slider.value))
        _goto_event(state.muon_indices[pos % len(state.muon_indices)])
    else:
        _goto_event(int(event_slider.value) + 1)


def _prev_event(_=None):
    if state.flow is None:
        return
    if muon_only.value and state.muon_indices:
        pos = bisect.bisect_left(state.muon_indices, int(event_slider.value)) - 1
        _goto_event(state.muon_indices[pos % len(state.muon_indices)])
    else:
        _goto_event(int(event_slider.value) - 1)


def _selected_truth_event_id():
    if truth_event.value == "auto":
        return None
    try:
        return int(truth_event.value)
    except Exception:
        return None


def _refresh_control_visibility():
    truth_enabled = bool(show_truth.value)
    truth_mode.visible = truth_enabled
    truth_event.visible = truth_enabled and truth_mode.value == "window"
    show_all_window_vertices.visible = truth_enabled and truth_mode.value == "window"
    show_vertices.visible = truth_enabled
    mc_only_muons.visible = truth_enabled
    mc_max_segments.visible = truth_enabled
    mc_topk.visible = truth_enabled and truth_mode.value == "backtrack"
    mc_minw.visible = truth_enabled and truth_mode.value == "backtrack"

    db_eps.visible = bool(show_clusters.value)
    db_min.visible = bool(show_clusters.value)
    cluster_min_hits.visible = bool(show_clusters.value)
    cluster_max_extent.visible = bool(show_clusters.value)
    clusters_info.visible = bool(show_clusters.value)


def _compute_clusters(hits, muon_track):
    if not show_clusters.value:
        clusters_info.object = ""
        return None
    labels = muon_region_labels(hits, muon_track, r_core=5.0, r_near=25.0)
    mask_far = labels == 2
    clusters = dbscan_clusters(hits, eps_cm=float(db_eps.value), min_samples=int(db_min.value), mask=mask_far)
    clusters = [c for c in clusters if c.n_hits >= int(cluster_min_hits.value) and c.extent_max_cm <= float(cluster_max_extent.value)]
    if not clusters:
        clusters_info.object = "**Clusters kept:** 0"
        return []
    lines = [f"- cluster {i}: nhits={c.n_hits}, sumQ={c.total_Q:.2g}, max={c.extent_max_cm:.2f} cm" for i, c in enumerate(clusters)]
    theta_line = angle_to_z_of_centroid_line(clusters)
    suffix = "\n\n**Centroid-line fit:** need ≥2 clusters"
    if theta_line is not None:
        suffix = f"\n\n**Centroid-line angle to z:** {theta_line * 180 / np.pi:.1f}°"
    clusters_info.object = "**Clusters kept:**\n" + "\n".join(lines) + suffix
    return clusters


def _analysis_markdown(hits, clusters, truth_info):
    lines = [
        f"### Event analysis",
        f"- hits: **{len(hits)}**",
        f"- total Q: **{np.nansum(hits['Q'].astype(float)):.4g}**" if len(hits) else "- total Q: **0**",
    ]
    if clusters is None:
        lines.append("- clusters: *(disabled)*")
    else:
        lines.append(f"- clusters kept: **{len(clusters)}**")
        if len(clusters):
            for i, c in enumerate(clusters[:8]):
                lines.append(f"  - cluster {i}: nhits={c.n_hits}, sumQ={c.total_Q:.3g}, max_extent={c.extent_max_cm:.2f} cm")
    if truth_info is None:
        lines.append("- truth: *(disabled)*")
    else:
        lines.append(f"- truth mode: **{truth_info.get('selection','?')}**")
        lines.append(f"- truth segments: **{truth_info.get('chosen_n_segments', 0)}**")
        if truth_info.get("multi", False):
            lines.append("- window has multiple truth event_ids")
    return "\n".join(lines)


def _refresh_views(*_):
    _refresh_control_visibility()
    if state.flow is None:
        view3d.object = None
        view2d.object = None
        return

    ev = int(event_slider.value)
    event_input.value = ev
    hits = state.flow.get_event_hits(ev, hit_type=hit_type.value)
    muon_track = state.flow.get_muon_track_for_event(ev) if show_muon.value else None

    clusters = _compute_clusters(hits, muon_track)

    truth_segments, truth_vertices, truth_info = None, None, None
    if show_truth.value:
        truth_segments, truth_info = state.flow.get_truth_overlay(
            ev,
            mode=truth_mode.value,
            hit_type=hit_type.value,
            top_k_segments=int(mc_topk.value),
            min_weight=float(mc_minw.value),
            truth_event_id=_selected_truth_event_id(),
            mc_only_muons=bool(mc_only_muons.value),
        )
        if show_vertices.value:
            truth_vertices = state.flow.get_truth_vertices(
                ev,
                mode=truth_mode.value,
                hit_type=hit_type.value,
                top_k_segments=int(mc_topk.value),
                min_weight=float(mc_minw.value),
                truth_event_id=_selected_truth_event_id(),
                include_all_window_vertices=bool(show_all_window_vertices.value),
            )

        if truth_mode.value == "window":
            ids = truth_info.get("truth_event_ids", []) if truth_info else []
            truth_event.options = ["auto"] + [str(v) for v in ids]
            if truth_event.value not in truth_event.options:
                truth_event.value = "auto"
        else:
            truth_event.options = ["auto"]
            truth_event.value = "auto"

    truth_summary = "truth overlay off"
    if truth_info is not None:
        if truth_info.get("missing", True):
            truth_summary = f"{truth_mode.value}: missing"
        else:
            truth_summary = f"{truth_mode.value}: segs={truth_info.get('chosen_n_segments', 0)}"
            if truth_info.get("multi", False):
                truth_summary += " (multi-window)"

    fig3d = viz.make_plotly_3d(
        hits,
        color_mode=color_mode.value,
        max_hits=int(max_hits.value),
        point_size=int(point_size.value),
        show_boxes=bool(show_boxes.value),
        muon_track=muon_track,
        clusters=clusters,
        mc_segments=truth_segments,
        mc_vertices=truth_vertices,
        mc_max_segments=int(mc_max_segments.value),
        mc_only_muons=bool(mc_only_muons.value),
        mc_label=f"MC segments ({truth_mode.value})",
    )
    fig2d = make_plotly_2d_projections(
        hits,
        color_mode=color_mode.value,
        max_hits=int(max_hits.value),
        point_size=max(2, int(point_size.value)),
        muon_track=muon_track,
    )
    fig_analysis = make_plotly_analysis(hits, clusters=clusters)

    view3d.object = fig3d
    view2d.object = fig2d
    view_analysis.object = pn.Column(
        pn.pane.Markdown(_analysis_markdown(hits, clusters, truth_info), sizing_mode="stretch_width"),
        pn.pane.Plotly(fig_analysis, min_height=450, sizing_mode="stretch_width"),
    )
    _set_status(
        f"**File:** `{os.path.basename(state.path)}`  \n"
        f"**Event:** `{ev}`  \n"
        f"**Hit type:** `{hit_type.value}`  \n"
        f"**N hits:** `{len(hits)}`  \n"
        f"**Truth:** {truth_summary}",
        ok=True,
    )


def _open_file(_=None):
    path = file_input.value.strip()
    if not path:
        state.close()
        _sync_slider_bounds()
        _set_status("No file path provided.", ok=False)
        return
    if not os.path.exists(path):
        _set_status(f"File not found: `{path}`", ok=False)
        return
    try:
        state.open(path)
        _sync_slider_bounds()
        _refresh_views()
    except Exception as exc:
        _set_status(f"Failed to open file: `{exc}`", ok=False)


# ---------- callbacks ----------
open_btn.on_click(_open_file)
next_btn.on_click(_next_event)
prev_btn.on_click(_prev_event)

event_slider.param.watch(lambda e: _goto_event(e.new), "value")
event_input.param.watch(lambda e: _goto_event(e.new), "value")

for w in [
    event_slider,
    hit_type,
    color_mode,
    max_hits,
    point_size,
    show_boxes,
    show_muon,
    show_truth,
    truth_mode,
    truth_event,
    show_vertices,
    show_all_window_vertices,
    mc_only_muons,
    mc_max_segments,
    mc_topk,
    mc_minw,
    show_clusters,
    db_eps,
    db_min,
    cluster_min_hits,
    cluster_max_extent,
]:
    w.param.watch(_refresh_views, "value")

_sync_slider_bounds()
_refresh_control_visibility()
if state.flow is not None:
    _refresh_views()
else:
    _set_status("Provide an HDF5 path and click **Open**.")

navigation_card = pn.Card(pn.Row(prev_btn, next_btn), muon_only, event_slider, event_input, title="Navigation", collapsed=False)
display_card = pn.Card(hit_type, color_mode, max_hits, point_size, show_boxes, title="Display options", collapsed=False)
truth_card = pn.Card(
    show_truth,
    truth_mode,
    truth_event,
    show_vertices,
    show_all_window_vertices,
    mc_only_muons,
    mc_max_segments,
    mc_topk,
    mc_minw,
    title="Truth overlay",
    collapsed=False,
)
cluster_card = pn.Card(show_clusters, db_eps, db_min, cluster_min_hits, cluster_max_extent, clusters_info, title="Clustering", collapsed=True)
muon_card = pn.Card(show_muon, title="Muon track", collapsed=True)

sidebar = pn.Column(
    pn.Row(file_input, open_btn),
    status,
    navigation_card,
    display_card,
    truth_card,
    cluster_card,
    muon_card,
    width=420,
    sizing_mode="stretch_height",
)

main_tabs = pn.Tabs(("3D", view3d), ("2D", view2d), ("Analysis", view_analysis), dynamic=True)
layout = pn.Row(sidebar, main_tabs, sizing_mode="stretch_both")
layout.servable(title="2x2 Event Display")
