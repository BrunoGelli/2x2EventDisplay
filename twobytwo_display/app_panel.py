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
from twobytwo_display.clustering import dbscan_clusters

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
muon_only = pn.widgets.Checkbox(name="Muon-only scan", value=False)

show_clusters = pn.widgets.Checkbox(name="Show clusters (DBSCAN)", value=False)
db_eps = pn.widgets.FloatInput(name="DBSCAN eps [cm]", value=1.5, step=0.1, width=140)
db_min = pn.widgets.IntInput(name="DBSCAN min_samples", value=10, step=1, width=140)

cluster_min_hits = pn.widgets.IntInput(name="Keep clusters with nhits ≥", value=20, step=1, width=180)
cluster_max_extent = pn.widgets.FloatInput(name="Keep clusters with max extent ≤ [cm]", value=8.0, step=0.5, width=220)

clusters_info = pn.pane.Markdown("")


view3d = pn.pane.Plotly(height=520)
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
        if muon_track is None:
            clusters = []
            clusters_info.object = "**Clusters:** (no muon track in this event)"
        else:
            clusters_all = dbscan_clusters(
                hits,
                eps_cm=float(db_eps.value),
                min_samples=int(db_min.value),
                muon_track=None,   # clustering is spatial only; we don't need the track here
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
                lines = [f"- cluster {i}: nhits={c.n_hits}, sumQ={c.total_Q:.2g}, "
                         f"max={c.extent_max_cm:.2f} cm, theta_z={c.theta_z_rad*180/np.pi:.1f}°"
                         for i, c in enumerate(clusters)]
                clusters_info.object = "**Clusters kept:**\n" + "\n".join(lines)
            else:
                clusters_info.object = "**Clusters kept:** 0"
    else:
        clusters_info.object = ""


    fig3d = make_plotly_3d(
        hits,
        color_mode=str(color_mode.value),
        max_hits=int(max_hits.value),
        point_size=int(point_size.value),
        show_boxes=bool(show_boxes.value),
        muon_track=muon_track,
        clusters=clusters
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
          show_clusters, db_eps, db_min, cluster_min_hits, cluster_max_extent]:
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
    pn.Row(prev_btn, next_btn, muon_only),
    pn.Row(show_clusters),
    pn.Row(db_eps, db_min),
    pn.Row(cluster_min_hits, cluster_max_extent),
    clusters_info,
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
    