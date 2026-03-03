# Truth overlay selection: recommended workflow

This display supports two truth-selection modes:

1. **`backtrack` (recommended, production-consistent)**
2. **`window` (legacy/debug fallback)**

## Why `backtrack` is recommended

`charge/events.id` is pipeline-local (copied from `charge/raw_events.id`) and is **not** a generator truth key.
A charge event can validly include multiple truth interactions because ND-LAr event building is trigger/window based.

To keep overlays visually aligned with displayed hits, use:

```text
charge event → selected hits → hit_backtrack rows → (segment_ids, fraction) → segments
```

This is implemented in `FlowFile.get_truth_overlay(..., mode="backtrack")`.

### Backtrack tuning knobs

- `top_k_segments`: keep only the K highest total-weight segments after per-hit accumulation.
- `min_weight`: discard low-contribution segments.

These controls reduce clutter while preserving dominant truth contributors.

## Legacy `window` mode

`window` mode maps charge event → raw event → interaction window and optionally selects a single `event_id`.
This is retained for debugging and comparisons, but can misalign visually in multi-truth windows.

## Vertices

`FlowFile.get_truth_vertices(...)` mirrors the same mode:

- `window`: vertices from selected interaction(s), or all interactions in the window.
- `backtrack`: interactions inferred from selected backtracked segments, then plotted as vertex markers.
