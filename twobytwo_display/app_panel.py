"""Panel web app for browsing 2x2 ndlar_flow HDF5 files.

Run:
  panel serve -m twobytwo_display.app_panel --show --args --h5 path/to/file.FLOW.hdf5

Notes:
  - 3D is Plotly (always interactive in browser).
  - 2D projections are Matplotlib (fast).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import panel as pn

from twobytwo_display.io import FlowFile
from twobytwo_display.viz import make_plotly_3d, make_matplotlib_figure
from twobytwo_display.viz import muon_region_labels
from twobytwo_display.clustering import dbscan_clusters
from twobytwo_display.clustering import angle_to_z_of_centroid_line



pn.extension("plotly")

muon_indices = []
muon_pos = 0

def _parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--h5", type=str, default=os.environ.get("TWOBYTWO_H5", ""), help="Path to FLOW HDF5 file")
    p.add_argument("--max_hits", type=int, default=40000)
    return p.parse_known_args()[0]

_ARGS = _parse_args()

class AppState:
    def __init__(self, path: str, max_hits: int):
        self.path = path
        self.max_hits = max_hits
        self.flow = None

    def open(self, path: str):
        if self.flow is not None:
            try: self.flow.close()
            except Exception: pass
        self.path = path
        if not path:
            self.flow = None
            return
        self.flow = FlowFile.open(path)

    def close(self):
        if self.flow is not None:
            self.flow.close()
            self.flow = None

state = AppState(_ARGS.h5, _ARGS.max_hits)
if state.path:
    state.open(state.path)

# Widgets
file_input = pn.widgets.TextInput(name="HDF5 path", value=state.path, placeholder="/path/to/file.FLOW.hdf5")
open_btn = pn.widgets.Button(name="Open", button_type="primary")
status = pn.pane.Markdown("")

color_mode = pn.widgets.Select(name="Color", options=["Q", "t_drift", "ts_pps", "muon_region"], value="Q")
max_hits = pn.widgets.IntInput(name="Max hits (downsample)", value=state.max_hits, step=1000, start=1000, width=140)
point_size = pn.widgets.IntSlider(name="Point size", start=1, end=8, value=2)
show_boxes = pn.widgets.Checkbox(name="Show module boxes", value=True)
show_muon = pn.widgets.Checkbox(name="Overlay rock muon track (if available)", value=False)

event_slider = pn.widgets.IntSlider(name="Event", start=0, end=0, value=0, step=1)
event_input = pn.widgets.IntInput(name="Event #", value=0, step=1, width=140)

prev_btn = pn.widgets.Button(name="◀ Prev", button_type="primary", width=90)
next_btn = pn.widgets.Button(name="Next ▶", button_type="primary", width=90)
muon_only = pn.widgets.Toggle(name="Muon-only scan",button_type="primary", value=False)

show_clusters = pn.widgets.Checkbox(name="Show clusters (DBSCAN)", value=False)
db_eps = pn.widgets.FloatInput(name="DBSCAN eps [cm]", value=1.5, step=0.1, width=140)
db_min = pn.widgets.IntInput(name="DBSCAN min_samples", value=10, step=1, width=140)

cluster_min_hits = pn.widgets.IntInput(name="Keep clusters with nhits ≥", value=20, step=1, width=180)
cluster_max_extent = pn.widgets.FloatInput(name="Keep clusters with max extent ≤ [cm]", value=8.0, step=0.5, width=220)

show_mc = pn.widgets.Checkbox(name="Overlay MC truth segments", value=False)
mc_max_segments = pn.widgets.IntInput(name="MC max segments", value=3000, step=500, start=0, width=140)
mc_only_muons = pn.widgets.Checkbox(name="MC only muons (|pdg|=13)", value=False)
mc_truth_event = pn.widgets.Select(name="MC truth event_id", options=["auto"], value="auto", width=220)
mc_info = pn.pane.Markdown("", height=80, sizing_mode="stretch_width")
mc_select_mode = pn.widgets.Select(
    name="MC selection",
    options=["backtrack (recommended)", "window (legacy)"],
    value="backtrack (recommended)",
    width=220,
)

mc_hit_type = pn.widgets.Select(
    name="Backtrack hit type",
    options=["prompt", "final"],
    value="prompt",
    width=160,
)

mc_topk = pn.widgets.IntInput(
    name="Backtrack top-K segments",
    value=2000,
    step=200,
    start=0,
    width=200,
)

mc_minw = pn.widgets.FloatInput(
    name="Backtrack min weight",
    value=0.0,
    step=0.01,
    width=160,
)

clusters_info = pn.pane.Markdown("", height=220, sizing_mode="stretch_width")
clusters_box = pn.Column(clusters_info, height=240, scroll=True)


view3d = pn.pane.Plotly(height=700,width=900)
view2d = pn.pane.Matplotlib(height=520, tight=False)

def _set_status(msg: str, ok: bool = True):
    status.object = f"**Status:** {'✅' if ok else '❌'} {msg}"

def _sync_slider():
    if state.flow is None:
        event_slider.start = 0
        event_slider.end = 0
        event_slider.value = 0
        return
    n = state.flow.n_events()
    event_slider.start = 0
    event_slider.end = max(0, n-1)
    event_slider.value = min(event_slider.value, event_slider.end)

# --- sync slider <-> input (avoid infinite loop) ---
_syncing = {"flag": False}

def _sync_event_widgets(value):
    if _syncing["flag"]:
        return
    _syncing["flag"] = True
    try:
        v = int(value)
        v = max(event_slider.start, min(event_slider.end, v))
        event_slider.value = v
        event_input.value = v
    finally:
        _syncing["flag"] = False

event_slider.param.watch(lambda e: _sync_event_widgets(e.new), "value")
event_input.param.watch(lambda e: _sync_event_widgets(e.new), "value")


def _goto_event(ev):
    ev = int(ev)
    ev = max(event_slider.start, min(event_slider.end, ev))
    event_slider.value = ev  # triggers refresh

def _next_clicked(_):
    global muon_pos
    if state.flow is None:
        return

    if muon_only.value and muon_indices:
        # move to next muon event >= current+1
        cur = int(event_slider.value)
        # find insertion point
        import bisect
        muon_pos = bisect.bisect_right(muon_indices, cur)
        if muon_pos >= len(muon_indices):
            muon_pos = 0  # wrap
        _goto_event(muon_indices[muon_pos])
    else:
        _goto_event(int(event_slider.value) + 1)

def _prev_clicked(_):
    global muon_pos
    if state.flow is None:
        return

    if muon_only.value and muon_indices:
        cur = int(event_slider.value)
        import bisect
        muon_pos = bisect.bisect_left(muon_indices, cur) - 1
        if muon_pos < 0:
            muon_pos = len(muon_indices) - 1  # wrap
        _goto_event(muon_indices[muon_pos])
    else:
        _goto_event(int(event_slider.value) - 1)

next_btn.on_click(_next_clicked)
prev_btn.on_click(_prev_clicked)


def _refresh_views(*_):
    if state.flow is None:
        view3d.object = None
        view2d.object = None
        return

    ev = int(event_slider.value)
    hits = state.flow.get_event_hits(ev)

    muon_track = None
    if bool(show_muon.value):
        muon_track = state.flow.get_muon_track_for_event(ev)

    clusters = None
    if bool(show_clusters.value):
        # You can choose whether to require a muon track for clustering; I recommend yes
        # if muon_track is None:
        if False:
            clusters = []
            clusters_info.object = "**Clusters:** (no muon track in this event)"
        else:
            labs = muon_region_labels(hits, muon_track, r_core=5.0, r_near=25.0)
            mask_far = (labs == 2)   # keep only hits outside the near/core region
            clusters_all = dbscan_clusters(
                hits,
                eps_cm=float(db_eps.value),
                min_samples=int(db_min.value),
                mask=mask_far,
            )

            # filter “isolated-ish” clusters
            clusters = []
            for c in clusters_all:
                if c.n_hits < int(cluster_min_hits.value):
                    continue
                if c.extent_max_cm > float(cluster_max_extent.value):
                    continue
                clusters.append(c)

            # print summary
            if clusters:
                lines = [
                    f"- cluster {i}: nhits={c.n_hits}, sumQ={c.total_Q:.2g}, max={c.extent_max_cm:.2f} cm"
                    for i, c in enumerate(clusters)
                ]
                clusters_info.object = "**Clusters kept:**\n" + "\n".join(lines)
            else:
                clusters_info.object = "**Clusters kept:** 0"
            
            theta_line = angle_to_z_of_centroid_line(clusters)

            if theta_line is None:
                clusters_info.object += "\n\n**Centroid-line fit:** need ≥2 clusters"
            else:
                deg = theta_line * 180/np.pi
                clusters_info.object += f"\n\n**Centroid-line angle to z:** {deg:.1f}°  (n_clusters={len(clusters)})"

    mc_segments = None
    mc_vertices = None
    mc_overlay_info = None

    if bool(show_mc.value):
        if mc_select_mode.value.startswith("backtrack"):
            mc_segments, mc_vertices, mc_overlay_info = state.flow.get_mc_overlay_for_charge_event_backtrack(
                ev,
                hit_type=str(mc_hit_type.value),
                top_k_segments=int(mc_topk.value),
                min_weight=float(mc_minw.value),
                mc_only_muons=bool(mc_only_muons.value),
            )

            if mc_overlay_info is not None and not mc_overlay_info.get("missing", True):
                mc_info.object = (
                    f"**MC (backtrack):** hit_type=`{mc_overlay_info['hit_type']}`  \n"
                    f"- hits: `{mc_overlay_info['n_hits']}`  \n"
                    f"- bt rows used: `{mc_overlay_info['n_bt_rows']}`  \n"
                    f"- unique segs: `{mc_overlay_info['n_unique_segments']}`  \n"
                    f"- drawn segs: `{mc_overlay_info['chosen_n_segments']}`"
                )
            else:
                mc_info.object = "**MC (backtrack):** *(none for this event)*"

            # In backtrack mode, don’t use the truth_event dropdown
            mc_truth_event.options = ["auto"]
            mc_truth_event.value = "auto"

        else:
            # legacy window-based mode (your existing selector)
            chosen = None
            if mc_truth_event.value != "auto":
                try:
                    chosen = int(mc_truth_event.value)
                except Exception:
                    chosen = None

            mc_segments, mc_vertices, mc_overlay_info = state.flow.get_mc_overlay_for_charge_event(
                ev, select="dominant", truth_event_id=chosen
            )

            if mc_overlay_info is not None and (not mc_overlay_info["missing"]):
                ids = mc_overlay_info["truth_event_ids"]
                opts = ["auto"] + [str(x) for x in ids]
                mc_truth_event.options = opts
                if mc_truth_event.value not in opts:
                    mc_truth_event.value = "auto"

                chosen_id = mc_overlay_info["chosen_event_id"]
                if mc_overlay_info["multi"]:
                    mc_info.object = (
                        f"**MC (window):** ⚠ multi event_id in charge window  \n"
                        f"- available: `{ids}`  \n"
                        f"- chosen: `{chosen_id}`"
                    )
                else:
                    mc_info.object = f"**MC (window):** event_id `{chosen_id}`"
            else:
                mc_truth_event.options = ["auto"]
                mc_truth_event.value = "auto"
                mc_info.object = "**MC (window):** *(none for this event)*"
    else:
        mc_info.object = ""

    mc_label = "MC truth segments"
    if mc_overlay_info is not None and not mc_overlay_info.get("missing", True):
        if mc_overlay_info.get("selection") == "backtrack":
            mc_label = f"MC segments (backtrack, {mc_overlay_info.get('hit_type','?')})"
        else:
            chosen = mc_overlay_info.get("chosen_event_id", None)
            if chosen is not None:
                mc_label = f"MC segments (event_id {chosen})"
            if mc_overlay_info.get("multi", False):
                mc_label += " [multi]"

    fig3d = make_plotly_3d(
        hits,
        color_mode=str(color_mode.value),
        max_hits=int(max_hits.value),
        point_size=int(point_size.value),
        show_boxes=bool(show_boxes.value),
        muon_track=muon_track,
        clusters=clusters,
        mc_segments=mc_segments,
        mc_vertices=mc_vertices,
        mc_max_segments=int(mc_max_segments.value),
        mc_only_muons=bool(mc_only_muons.value),
        mc_label=mc_label,
    )
    fig2d = make_matplotlib_figure(
        hits,
        color_mode=str(color_mode.value),
        max_hits=int(max_hits.value),
        point_size=int(point_size.value),
        show_boxes=bool(show_boxes.value),
        muon_track=muon_track
    )

    view3d.object = fig3d
    view2d.object = fig2d
    import matplotlib.pyplot as plt
    plt.close(fig2d)

    # event quick summary
    try:
        nh = len(hits)
        _set_status(f"Opened {os.path.basename(state.path)} | event {ev} | nhits={nh}")
    except Exception:
        _set_status("Loaded event", ok=True)

def _open_clicked(_):
    path = file_input.value.strip()
    if not path:
        state.close()
        _sync_slider()
        _set_status("No file path provided.", ok=False)
        return
    if not os.path.exists(path):
        _set_status(f"File not found: {path}", ok=False)
        return
    try:
        state.open(path)
        _sync_slider()
        _refresh_views()
        global muon_indices, muon_pos
        muon_indices = state.flow.muon_event_indices() if state.flow is not None else []
        muon_pos = 0
    except Exception as e:
        _set_status(f"Failed to open file: {e}", ok=False)

open_btn.on_click(_open_clicked)

# reactive callbacks
for w in [event_slider, color_mode, max_hits, point_size, show_boxes, show_muon,
          show_clusters, db_eps, db_min, cluster_min_hits, cluster_max_extent, show_mc, mc_max_segments, mc_only_muons, mc_truth_event, mc_select_mode, mc_hit_type, mc_topk, mc_minw]:
    w.param.watch(lambda *_: _refresh_views(), 'value')

# initial
_sync_slider()
if state.flow is not None:
    _refresh_views()
else:
    _set_status("Provide an HDF5 path and click Open.", ok=True)

controls = pn.Card(
    pn.Row(file_input, open_btn),
    status,
    pn.layout.Divider(),
    pn.Row(event_slider, event_input),
    pn.Row(color_mode, max_hits),
    pn.Row(point_size),
    pn.Row(show_boxes, show_muon),
    pn.layout.Divider(),
    pn.Row(pn.Spacer(), prev_btn, muon_only, next_btn, pn.Spacer()),
    pn.layout.Divider(),
    pn.Row(show_mc),
    pn.Row(mc_select_mode),
    pn.Row(mc_max_segments, mc_only_muons),
    pn.Row(mc_hit_type, mc_topk),
    pn.Row(mc_minw),
    pn.Row(mc_truth_event),   # only meaningful in legacy mode
    mc_info,   
    pn.layout.Divider(),
    pn.Row(show_clusters),
    pn.Row(db_eps, db_min),
    pn.Row(cluster_min_hits, cluster_max_extent),
    clusters_box,
    title="Controls",
    collapsed=False
)

layout = pn.Row(
    pn.Column(controls, width=500),
    pn.Column(
        pn.Tabs(
            ("3D", view3d),
            ("2D", view2d),
            active=0
        ),
        sizing_mode="stretch_width"
    ),
    sizing_mode="stretch_width"
)

# Panel looks for a variable named 'servable'
layout.servable(title="2x2 Event Display")
    