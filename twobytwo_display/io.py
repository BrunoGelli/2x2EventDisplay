from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class FlowFile:
    """High-level accessor for ndlar_flow HDF5 files.

    The public API intentionally keeps truth selection narrow:
      * :meth:`get_event_hits`
      * :meth:`get_truth_overlay`
      * :meth:`get_truth_vertices`

    This keeps selection policies in one place and lets UI code remain thin.
    """

    h5: h5py.File
    _segment_id_to_row: Optional[Dict[int, int]] = field(default=None, init=False)

    @classmethod
    def open(cls, path: str, mode: str = "r") -> "FlowFile":
        return cls(h5py.File(path, mode))

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass

    # ---------- datasets ----------
    @property
    def events(self):
        return self.h5["charge/events/data"]

    @property
    def mc_segments(self):
        return self.h5.get("mc_truth/segments/data", None)

    @property
    def mc_interactions(self):
        return self.h5.get("mc_truth/interactions/data", None)

    @property
    def rock_muon_tracks(self):
        return self.h5.get("analysis/rock_muon_tracks/data", None)

    # ---------- basic event helpers ----------
    def n_events(self) -> int:
        return len(self.events)

    def event_id(self, event_index: int) -> int:
        ev = self.events
        if "id" in (ev.dtype.names or ()):
            return int(ev[event_index]["id"])
        return int(event_index)

    def get_event_hits(self, event_index: int, hit_type: str = "prompt"):
        hits_ds, rr_ds = self._get_hits_and_event_ref(hit_type)
        if hits_ds is None or rr_ds is None:
            return np.array([], dtype=np.float32)
        rr = rr_ds[int(event_index)]
        return hits_ds[int(rr["start"]): int(rr["stop"])]

    # ---------- public truth API ----------
    def get_truth_overlay(
        self,
        event_index: int,
        *,
        mode: str = "backtrack",
        hit_type: str = "prompt",
        top_k_segments: int = 2000,
        min_weight: float = 0.0,
        truth_event_id: Optional[int] = None,
        select: str = "dominant",
        mc_only_muons: bool = False,
    ):
        """Return `(segments, info)` for one charge event.

        `mode='backtrack'` is production-consistent and recommended.
        `mode='window'` preserves legacy trigger-window selection for debugging.
        """
        mode = str(mode).lower()
        if mode == "backtrack":
            segs, info = self._truth_overlay_backtrack(
                event_index,
                hit_type=hit_type,
                top_k_segments=top_k_segments,
                min_weight=min_weight,
                mc_only_muons=mc_only_muons,
            )
            return segs, info
        if mode == "window":
            segs, _, info = self._truth_overlay_window(
                event_index,
                select=select,
                truth_event_id=truth_event_id,
            )
            if mc_only_muons and segs is not None and len(segs) and "pdg_id" in (segs.dtype.names or ()):
                segs = segs[np.abs(segs["pdg_id"].astype(int)) == 13]
                info["chosen_n_segments"] = int(len(segs))
            return segs, info
        raise ValueError("mode must be 'backtrack' or 'window'")

    def get_truth_vertices(
        self,
        event_index: int,
        *,
        mode: str = "backtrack",
        hit_type: str = "prompt",
        top_k_segments: int = 2000,
        min_weight: float = 0.0,
        truth_event_id: Optional[int] = None,
        include_all_window_vertices: bool = False,
    ):
        """Return truth interactions/vertices aligned to selected truth content."""
        inter = self.mc_interactions
        if inter is None:
            return None

        mode = str(mode).lower()
        if mode == "window":
            _, chosen_interactions, info = self._truth_overlay_window(
                event_index,
                select="dominant",
                truth_event_id=truth_event_id,
            )
            if info.get("missing", True):
                return inter[:0]
            if include_all_window_vertices:
                i0, i1 = self.get_mc_interaction_rowrange_for_event(event_index)
                return inter[i0:i1]
            return chosen_interactions

        segments, info = self._truth_overlay_backtrack(
            event_index,
            hit_type=hit_type,
            top_k_segments=top_k_segments,
            min_weight=min_weight,
            mc_only_muons=False,
        )
        if info.get("missing", True) or segments is None or len(segments) == 0:
            return inter[:0]

        inter_ids = self._interaction_ids_from_segments(segments)
        if inter_ids.size == 0 or "interaction_id" not in (inter.dtype.names or ()):
            return inter[:0]

        mask = np.isin(inter["interaction_id"].astype(np.int64), inter_ids)
        return inter[mask]

    # ---------- compatibility wrappers ----------
    def get_mc_overlay_for_charge_event(self, event_index: int, *, select: str = "dominant", truth_event_id: Optional[int] = None):
        segs, interactions, info = self._truth_overlay_window(event_index, select=select, truth_event_id=truth_event_id)
        return segs, interactions, info

    def get_mc_overlay_for_charge_event_backtrack(
        self,
        event_index: int,
        *,
        hit_type: str = "prompt",
        top_k_segments: int = 2000,
        min_weight: float = 0.0,
        mc_only_muons: bool = False,
    ):
        segs, info = self._truth_overlay_backtrack(
            event_index,
            hit_type=hit_type,
            top_k_segments=top_k_segments,
            min_weight=min_weight,
            mc_only_muons=mc_only_muons,
        )
        return segs, None, info

    # ---------- muon helpers ----------
    def find_muon_track_index_for_event(self, event_index: int) -> Optional[int]:
        tr = self.rock_muon_tracks
        if tr is None:
            return None
        ev_id = self.event_id(event_index)
        m = np.where(tr["event_id"] == ev_id)[0]
        return int(m[0]) if len(m) else None

    def get_muon_track_for_event(self, event_index: int):
        idx = self.find_muon_track_index_for_event(event_index)
        return None if idx is None else self.rock_muon_tracks[idx]

    def muon_event_indices(self) -> List[int]:
        tr = self.rock_muon_tracks
        if tr is None:
            return []
        ev = self.events
        if "id" not in (ev.dtype.names or ()):
            cand = tr["event_id"].astype(int)
            cand = cand[(cand >= 0) & (cand < len(ev))]
            return sorted(set(map(int, cand)))
        ids = ev["id"].astype(tr["event_id"].dtype, copy=False)
        id_to_index = {int(v): i for i, v in enumerate(ids)}
        return sorted({id_to_index[int(eid)] for eid in tr["event_id"] if int(eid) in id_to_index})

    # ---------- internals ----------
    def _get_hits_and_event_ref(self, hit_type: str):
        hit_type = str(hit_type).lower()
        if hit_type == "prompt":
            return (
                self.h5.get("charge/calib_prompt_hits/data", None),
                self.h5.get("charge/events/ref/charge/calib_prompt_hits/ref_region", None),
            )
        if hit_type == "final":
            return (
                self.h5.get("charge/calib_final_hits/data", None),
                self.h5.get("charge/events/ref/charge/calib_final_hits/ref_region", None),
            )
        raise ValueError("hit_type must be 'prompt' or 'final'")

    def _raw_event_index_for_charge_event(self, event_index: int) -> int:
        try:
            return int(self.h5["charge/events/ref/charge/raw_events/ref"][event_index][1])
        except Exception:
            pass
        try:
            ref = self.h5["charge/raw_events/ref/charge/events/ref"][:]
            m = np.where(ref[:, 1].astype(np.int64) == int(event_index))[0]
            if len(m):
                return int(m[0])
        except Exception:
            pass
        return int(event_index)

    def get_mc_interaction_rowrange_for_event(self, event_index: int) -> Tuple[int, int]:
        raw_idx = self._raw_event_index_for_charge_event(event_index)
        rr_ds = self.h5.get("charge/raw_events/ref/mc_truth/interactions/ref_region", None)
        if rr_ds is None:
            return (0, 0)
        rr = rr_ds[int(raw_idx)]
        return int(rr["start"]), int(rr["stop"])

    def _truth_overlay_window(self, event_index: int, *, select: str, truth_event_id: Optional[int]):
        seg = self.mc_segments
        inter = self.mc_interactions
        base_info = {
            "selection": "window",
            "missing": True,
            "multi": False,
            "truth_event_ids": [],
            "chosen_event_id": None,
            "n_interactions": 0,
            "chosen_n_segments": 0,
        }
        if seg is None or inter is None:
            return None, None, base_info

        i0, i1 = self.get_mc_interaction_rowrange_for_event(event_index)
        if i1 <= i0:
            return seg[:0], inter[i0:i1], base_info
        inter_slice = inter[i0:i1]
        if len(inter_slice) == 0 or "event_id" not in (inter_slice.dtype.names or ()):
            segs_all = self._segments_for_global_interaction_rows(range(i0, i1))
            info = dict(base_info, missing=False, n_interactions=int(len(inter_slice)), chosen_n_segments=int(len(segs_all)))
            return segs_all, inter_slice, info

        ids = inter_slice["event_id"].astype(np.int64)
        uniq_ids = sorted(set(map(int, ids.tolist())))
        multi = len(uniq_ids) > 1
        chosen = int(truth_event_id) if truth_event_id is not None and int(truth_event_id) in uniq_ids else None

        if chosen is None:
            if not multi:
                chosen = uniq_ids[0]
            elif select == "dominant":
                chosen = self._dominant_truth_event_id_for_interactions(i0, i1, uniq_ids)
            else:
                chosen = uniq_ids[0]

        m = ids == chosen
        chosen_inter = inter_slice[m]
        chosen_rows = (np.where(m)[0] + i0).tolist()
        chosen_segs = self._segments_for_global_interaction_rows(chosen_rows)
        info = dict(
            base_info,
            missing=False,
            multi=multi,
            truth_event_ids=uniq_ids,
            chosen_event_id=int(chosen),
            n_interactions=int(len(chosen_inter)),
            chosen_n_segments=int(len(chosen_segs)),
        )
        return chosen_segs, chosen_inter, info

    def _truth_overlay_backtrack(
        self,
        event_index: int,
        *,
        hit_type: str,
        top_k_segments: int,
        min_weight: float,
        mc_only_muons: bool,
    ):
        seg = self.mc_segments
        info = {
            "selection": "backtrack",
            "hit_type": hit_type,
            "missing": True,
            "n_hits": 0,
            "n_bt_rows": 0,
            "n_unique_segments": 0,
            "chosen_n_segments": 0,
        }
        if seg is None:
            return None, info

        hit_type = str(hit_type).lower()
        if hit_type == "prompt":
            rr = self.h5.get("charge/events/ref/charge/calib_prompt_hits/ref_region", None)
            bt = self.h5.get("mc_truth/calib_prompt_hit_backtrack/data", None)
            ref = self.h5.get("charge/calib_prompt_hits/ref/mc_truth/calib_prompt_hit_backtrack/ref", None)
        elif hit_type == "final":
            rr = self.h5.get("charge/events/ref/charge/calib_final_hits/ref_region", None)
            bt = self.h5.get("mc_truth/calib_final_hit_backtrack/data", None)
            ref = self.h5.get("charge/calib_final_hits/ref/mc_truth/calib_final_hit_backtrack/ref", None)
        else:
            raise ValueError("hit_type must be 'prompt' or 'final'")

        if rr is None or bt is None or ref is None:
            return seg[:0], info

        r = rr[int(event_index)]
        h0, h1 = int(r["start"]), int(r["stop"])
        info["n_hits"] = max(0, h1 - h0)
        if info["n_hits"] <= 0:
            return seg[:0], info

        bt_rows = ref[h0:h1, 1].astype(np.int64, copy=False)
        bt_rows = bt_rows[(bt_rows >= 0) & (bt_rows < len(bt))]
        if bt_rows.size == 0:
            return seg[:0], info

        bt_names = bt.dtype.names or ()
        if "segment_ids" not in bt_names or "fraction" not in bt_names:
            return seg[:0], info

        w_by_seg: Dict[int, float] = defaultdict(float)
        bt_used = 0
        for idx in bt_rows:
            row = bt[int(idx)]
            seg_ids = np.asarray(row["segment_ids"], dtype=np.int64)
            fracs = np.asarray(row["fraction"], dtype=float)
            if seg_ids.size == 0 or fracs.size == 0:
                continue
            bt_used += 1
            for sid, w in zip(seg_ids[: min(len(seg_ids), len(fracs))], fracs[: min(len(seg_ids), len(fracs))]):
                if np.isfinite(w) and w > 0:
                    w_by_seg[int(sid)] += float(w)

        info["n_bt_rows"] = int(bt_used)
        info["n_unique_segments"] = int(len(w_by_seg))
        if not w_by_seg:
            return seg[:0], info

        items = [(sid, w) for sid, w in w_by_seg.items() if w >= float(min_weight)]
        items.sort(key=lambda t: -t[1])
        if top_k_segments and len(items) > int(top_k_segments):
            items = items[: int(top_k_segments)]

        seg_ids = np.asarray([sid for sid, _ in items], dtype=np.int64)
        selected = self._segments_from_segment_ids(seg_ids)
        if mc_only_muons and selected is not None and len(selected) and "pdg_id" in (selected.dtype.names or ()):
            selected = selected[np.abs(selected["pdg_id"].astype(int)) == 13]

        info["missing"] = False
        info["chosen_n_segments"] = int(len(selected))
        return selected, info

    def _segments_for_global_interaction_rows(self, global_rows: Iterable[int]) -> np.ndarray:
        seg = self.mc_segments
        if seg is None:
            return None
        seg_rr = self.h5.get("mc_truth/interactions/ref/mc_truth/segments/ref_region", None)
        if seg_rr is None:
            return seg[:0]

        pieces = []
        for ii in global_rows:
            r = seg_rr[int(ii)]
            s0, s1 = int(r["start"]), int(r["stop"])
            if s1 > s0:
                pieces.append(seg[s0:s1])
        return np.concatenate(pieces) if pieces else seg[:0]

    def _dominant_truth_event_id_for_interactions(self, i0: int, i1: int, uniq_ids: List[int]) -> int:
        inter = self.mc_interactions
        seg_rr = self.h5.get("mc_truth/interactions/ref/mc_truth/segments/ref_region", None)
        if inter is None or seg_rr is None:
            return uniq_ids[0]
        counts = {eid: 0 for eid in uniq_ids}
        for ii in range(i0, i1):
            eid = int(inter[ii]["event_id"])
            rr = seg_rr[ii]
            counts[eid] += max(0, int(rr["stop"]) - int(rr["start"]))
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _ensure_segment_id_index(self):
        if self._segment_id_to_row is not None:
            return
        seg = self.mc_segments
        if seg is None:
            self._segment_id_to_row = {}
            return
        names = seg.dtype.names or ()
        if "segment_id" in names:
            self._segment_id_to_row = {int(sid): i for i, sid in enumerate(seg["segment_id"])}
        else:
            self._segment_id_to_row = {}

    def _segments_from_segment_ids(self, segment_ids: np.ndarray) -> np.ndarray:
        seg = self.mc_segments
        if seg is None:
            return None
        ids = np.asarray(segment_ids, dtype=np.int64)
        if ids.size == 0:
            return seg[:0]

        self._ensure_segment_id_index()
        names = seg.dtype.names or ()
        if "segment_id" in names and self._segment_id_to_row:
            rows = [self._segment_id_to_row.get(int(sid), None) for sid in ids]
            rows = np.asarray([r for r in rows if r is not None], dtype=np.int64)
        else:
            rows = ids[(ids >= 0) & (ids < len(seg))]

        if rows.size == 0:
            return seg[:0]

        # h5py fancy indexing requires increasing order.
        order = np.argsort(rows)
        rows_sorted = rows[order]
        seg_sorted = seg[rows_sorted]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return seg_sorted[inv]

    def _interaction_ids_from_segments(self, segments: np.ndarray) -> np.ndarray:
        if segments is None or len(segments) == 0:
            return np.array([], dtype=np.int64)
        names = segments.dtype.names or ()
        if "interaction_id" in names:
            return np.unique(segments["interaction_id"].astype(np.int64))
        if "vertex_id" in names:
            inter = self.mc_interactions
            if inter is None or "vertex_id" not in (inter.dtype.names or ()) or "interaction_id" not in (inter.dtype.names or ()):
                return np.array([], dtype=np.int64)
            vtx_ids = np.unique(segments["vertex_id"].astype(np.int64))
            m = np.isin(inter["vertex_id"].astype(np.int64), vtx_ids)
            return np.unique(inter["interaction_id"][m].astype(np.int64))
        return np.array([], dtype=np.int64)
