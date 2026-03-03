# 2x2 ND-LAr MiniRun6 Event Display

Browser-based event display for DUNE ND-LAr 2×2 `ndlar_flow` HDF5 (`.FLOW.hdf5`) files.

## What this app does

- Interactive **3D hit display** (Plotly)
- **2D projections**: XY, XZ, YZ + charge histogram panel
- Optional overlays:
  - module geometry boxes
  - rock muon track (`analysis/rock_muon_tracks`)
  - MC truth segments + MC truth vertices
  - clustering (DBSCAN-based)
- Event-by-event navigation with quick status summary

## Truth overlay model (important)

Two truth selection modes are available:

1. **`backtrack` (recommended)**
2. **`window` (legacy/debug)**

The recommended mode is hit-backtracking based:

```text
charge event -> hits -> hit_backtrack -> (segment_ids, fraction) -> segments
```

This avoids fragile `charge/events.id == mc_truth/interactions.event_id` assumptions and is consistent with ND-LAr trigger/window event building.

See [`docs/truth_overlay_selection.md`](docs/truth_overlay_selection.md) for details.

## Installation

```bash
cd /workspace/2x2EventDisplay
pip install -e .
```

## Run

```bash
panel serve -m twobytwo_display.app_panel --show --args --h5 /path/to/file.FLOW.hdf5
```

or with environment variable:

```bash
export TWOBYTWO_H5=/path/to/file.FLOW.hdf5
panel serve -m twobytwo_display.app_panel --show
```

If `-m` is unavailable in your Panel version, use:

```bash
PYTHONPATH=/workspace/2x2EventDisplay panel serve twobytwo_display/app_panel.py --show
```

## UI layout

Left sidebar (collapsible cards):

- **Navigation**: event controls, prev/next, muon-only scan
- **Display options**: hit type, color mode, max hits, point size, geometry
- **Truth overlay**: mode-specific controls (backtrack/window)
- **Clustering**: DBSCAN options (shown when enabled)
- **Muon track**: toggle

Main area tabs:

- **3D**: interactive event view + overlays
- **2D**: XY/XZ/YZ + charge histogram
- **Analysis**: summary markdown + extra distributions

## Notes on geometry

Module boundaries are hard-coded (cm) from 2×2 geometry technote Table 1:

- Module 0: x ∈ [3.07, 63.93], y ∈ [-61.85, 61.85], z ∈ [2.68, 64.32]
- Module 1: x ∈ [3.07, 63.93], y ∈ [-61.85, 61.85], z ∈ [-64.32, -2.68]
- Module 2: x ∈ [-63.93, -3.07], y ∈ [-61.85, 61.85], z ∈ [2.68, 64.32]
- Module 3: x ∈ [-63.93, -3.07], y ∈ [-61.85, 61.85], z ∈ [-64.32, -2.68]
