# twobytwo-display

Browser-based event display for DUNE ND-LAr 2×2 `ndlar_flow` HDF5 (FLOW) files.

**Features**
- Panel web app (runs in your browser)
- 3D interactive view (Plotly)
- ZX / ZY projections (Matplotlib)
- Optional module boundary overlay (from the 2×2 geometry technote, Table 1)
- Optional overlay of `analysis/rock_muon_tracks` line (if present)

## Install (editable / dev mode)

```bash
cd twobytwo_event_display
pip install -e .
```

## Run the web app

```bash
panel serve -m twobytwo_display.app_panel --show --args --h5 /path/to/packet-XXXX.FLOW.hdf5
```

Or set an env var:

```bash
export TWOBYTWO_H5=/path/to/file.FLOW.hdf5
panel serve -m twobytwo_display.app_panel --show
```

## Notes on geometry

The module boundaries are hard-coded from the geometry technote (Table 1) in cm:
- Module 0: x ∈ [3.07, 63.93], y ∈ [-61.85, 61.85], z ∈ [2.68, 64.32]
- Module 1: x ∈ [3.07, 63.93], y ∈ [-61.85, 61.85], z ∈ [-64.32, -2.68]
- Module 2: x ∈ [-63.93, -3.07], y ∈ [-61.85, 61.85], z ∈ [2.68, 64.32]
- Module 3: x ∈ [-63.93, -3.07], y ∈ [-61.85, 61.85], z ∈ [-64.32, -2.68]

(See the technote for definitions and coordinate conventions.)
