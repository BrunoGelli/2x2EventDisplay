from __future__ import annotations

import h5py
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class FlowFile:

    def __post_init__(self):
        # Lazy cache: segment_id -> row index in mc_truth/segments/data
        self._segment_id_to_row = None

    """Thin wrapper around an ndlar_flow-style HDF5 file."""
    h5: h5py.File

    @classmethod
    def open(cls, path: str, mode: str = "r") -> "FlowFile":
        return cls(h5py.File(path, mode))

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass

    # ---- mc truth (optional) ----
    @property
    def mc_segments(self):
        return self.h5.get("mc_truth/segments/data", None)

    @property
    def mc_interactions(self):
        return self.h5.get("mc_truth/interactions/data", None)

    # ---- core datasets ----
    @property
    def events(self):
        return self.h5["charge/events/data"]

    @property
    def hits(self):
        return self.h5["charge/calib_prompt_hits/data"]

    @property
    def hits_ref_region(self):
        return self.h5["charge/events/ref/charge/calib_prompt_hits/ref_region"]

    # ---- muon objects (optional) ----
    @property
    def rock_muon_tracks(self):
        return self.h5.get("analysis/rock_muon_tracks/data", None)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------
    def n_events(self) -> int:
        return len(self.events)

    def event_id(self, event_index: int) -> int:
        ev = self.events
        if "id" in ev.dtype.names:
            return int(ev[event_index]["id"])
        return int(event_index)

    def get_event_hits(self, event_index: int):
        rr = self.hits_ref_region[event_index]
        start = int(rr["start"])
        stop = int(rr["stop"])
        return self.hits[start:stop]

    # ------------------------------------------------------------------
    # Muon helpers
    # ------------------------------------------------------------------
    def find_muon_track_index_for_event(self, event_index: int) -> Optional[int]:
        tr = self.rock_muon_tracks
        if tr is None:
            return None
        ev_id = self.event_id(event_index)
        m = np.where(tr["event_id"] == ev_id)[0]
        if len(m):
            return int(m[0])
        return None

    def get_muon_track_for_event(self, event_index: int):
        idx = self.find_muon_track_index_for_event(event_index)
        if idx is None:
            return None
        return self.rock_muon_tracks[idx]

    def muon_event_indices(self):
        """
        Return sorted list of event_index that have at least one rock_muon_track.
        Uses track.event_id -> match charge/events/id.
        """
        tr = self.rock_muon_tracks
        if tr is None:
            return []

        ev = self.events
        if "id" not in ev.dtype.names:
            cand = tr["event_id"].astype(int)
            cand = cand[(cand >= 0) & (cand < len(ev))]
            return sorted(set(map(int, cand)))

        ids = ev["id"].astype(tr["event_id"].dtype, copy=False)
        id_to_index = {int(v): i for i, v in enumerate(ids)}
        out = []
        for eid in tr["event_id"]:
            i = id_to_index.get(int(eid))
            if i is not None:
                out.append(i)
        return sorted(set(out))

    # ------------------------------------------------------------------
    # MC truth mapping (robust)
    # ------------------------------------------------------------------
    def _raw_event_index_for_charge_event(self, event_index: int) -> int:
        """
        Return the charge/raw_events row index corresponding to a given charge/events row.
        Different productions may store the reference in either direction, so try both.
        """
        # Preferred: charge/events -> charge/raw_events
        try:
            ref = self.h5["charge/events/ref/charge/raw_events/ref"][event_index]
            return int(ref[1])
        except Exception:
            pass

        # Alternate: charge/raw_events -> charge/events (search)
        try:
            ref = self.h5["charge/raw_events/ref/charge/events/ref"][:]  # (n_raw, 2)
            raw_matches = np.where(ref[:, 1].astype(np.int64) == int(event_index))[0]
            if len(raw_matches):
                return int(raw_matches[0])
        except Exception:
            pass

        # Fallback: assume 1:1 ordering
        return int(event_index)

    def get_mc_interaction_rowrange_for_event(self, event_index: int) -> Tuple[int, int]:
        """
        Returns [start, stop) row indices into mc_truth/interactions/data for this charge event.
        """
        raw_idx = self._raw_event_index_for_charge_event(event_index)

        rr_ds = self.h5.get("charge/raw_events/ref/mc_truth/interactions/ref_region", None)
        if rr_ds is None:
            return (0, 0)

        rr = rr_ds[raw_idx]
        return int(rr["start"]), int(rr["stop"])

    def get_mc_interactions_for_event(self, event_index: int):
        inter = self.mc_interactions
        if inter is None:
            return None
        i0, i1 = self.get_mc_interaction_rowrange_for_event(event_index)
        if i1 <= i0:
            return inter[:0]
        return inter[i0:i1]

    # ------------------------------------------------------------------
    # NEW: MC overlay selection (handles missing + multi truth)
    # ------------------------------------------------------------------
    def get_mc_overlay_for_charge_event(
        self,
        event_index: int,
        *,
        select: str = "dominant",
        truth_event_id: Optional[int] = None,
    ):
        """
        Return (segments, interactions, info) for the truth overlay corresponding to a charge event.

        - If there is no truth: returns empty segments/interactions and info["missing"]=True.
        - If multiple truth event_id are present: chooses one by policy (default: dominant by segment count),
          unless truth_event_id is explicitly provided.

        info keys:
          missing: bool
          multi: bool
          truth_event_ids: List[int]
          chosen_event_id: Optional[int]
          n_segments: int
          n_interactions: int
        """
        seg = self.mc_segments
        inter = self.mc_interactions
        if seg is None or inter is None:
            info = {
                "missing": True,
                "multi": False,
                "truth_event_ids": [],
                "chosen_event_id": None,
                "n_segments": 0,
                "n_interactions": 0,
            }
            return None, None, info

        i0, i1 = self.get_mc_interaction_rowrange_for_event(event_index)
        if i1 <= i0:
            info = {
                "missing": True,
                "multi": False,
                "truth_event_ids": [],
                "chosen_event_id": None,
                "n_segments": 0,
                "n_interactions": 0,
            }
            return seg[:0], inter[i0:i1], info

        inter_slice = inter[i0:i1]
        if len(inter_slice) == 0:
            info = {
                "missing": True,
                "multi": False,
                "truth_event_ids": [],
                "chosen_event_id": None,
                "n_segments": 0,
                "n_interactions": 0,
            }
            return seg[:0], inter_slice, info

        if "event_id" not in inter_slice.dtype.names:
            # can't disambiguate; return all
            segs_all = self._segments_for_global_interaction_rows(range(i0, i1))
            info = {
                "missing": False,
                "multi": False,
                "truth_event_ids": [],
                "chosen_event_id": None,
                "n_segments": int(len(segs_all)),
                "n_interactions": int(len(inter_slice)),
            }
            return segs_all, inter_slice, info

        ids = inter_slice["event_id"].astype(np.int64)
        uniq_ids = sorted(set(map(int, ids.tolist())))
        multi = len(uniq_ids) > 1

        # Choose truth_event_id
        chosen: Optional[int] = None
        if truth_event_id is not None:
            chosen = int(truth_event_id)
            if chosen not in uniq_ids:
                # invalid override -> fall back
                chosen = None

        if chosen is None:
            if not multi:
                chosen = uniq_ids[0]
            else:
                if select == "dominant":
                    chosen = self._dominant_truth_event_id_for_interactions(i0, i1, uniq_ids)
                else:
                    # fallback: first id
                    chosen = uniq_ids[0]

        # Filter interactions to chosen id
        m = ids == chosen
        chosen_inter = inter_slice[m]

        # Gather segments only for the chosen interactions (need GLOBAL indices)
        chosen_global_rows = (np.where(m)[0] + i0).tolist()
        chosen_segs = self._segments_for_global_interaction_rows(chosen_global_rows)

        info = {
            "missing": False,
            "multi": multi,
            "truth_event_ids": uniq_ids,
            "chosen_event_id": chosen,
            "n_segments": int(len(chosen_segs)),
            "n_interactions": int(len(chosen_inter)),
        }
        return chosen_segs, chosen_inter, info

    def _segments_for_global_interaction_rows(self, global_rows) -> np.ndarray:
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
        """
        Choose truth event_id with the largest number of segments in this charge window.
        """
        inter = self.mc_interactions
        if inter is None:
            return uniq_ids[0]

        seg_rr = self.h5.get("mc_truth/interactions/ref/mc_truth/segments/ref_region", None)
        if seg_rr is None:
            return uniq_ids[0]

        counts = {eid: 0 for eid in uniq_ids}
        for ii in range(i0, i1):
            eid = int(inter[ii]["event_id"])
            r = seg_rr[ii]
            n = int(r["stop"]) - int(r["start"])
            if n > 0:
                counts[eid] += n

        # pick max; stable tie-breaker = smaller id
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        # ------------------------------------------------------------------
    # Recommended: MC overlay selection via hit backtracking
    # ------------------------------------------------------------------
    def _ensure_segment_id_index(self):
        """Build segment_id -> row index for mc_truth/segments/data (lazy)."""
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
            # Common case: segment_id == row index
            self._segment_id_to_row = {}

    def _segments_from_segment_ids(self, segment_ids: np.ndarray) -> np.ndarray:
        seg = self.mc_segments
        if seg is None:
            return None
        segment_ids = np.asarray(segment_ids).astype(np.int64, copy=False)
        if segment_ids.size == 0:
            return seg[:0]

        self._ensure_segment_id_index()
        names = seg.dtype.names or ()

        if "segment_id" in names and self._segment_id_to_row:
            rows = [self._segment_id_to_row.get(int(sid), None) for sid in segment_ids]
            rows = [r for r in rows if r is not None]
            if not rows:
                return seg[:0]

            rows = np.asarray(rows, dtype=np.int64)

            # h5py requires increasing order for fancy indexing:
            order = np.argsort(rows)
            rows_sorted = rows[order]
            seg_sorted = seg[rows_sorted]          # <- safe HDF5 read
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            return seg_sorted[inv]                # <- restore original (weight) order

        # fallback: treat segment_id as row index
        rows = segment_ids[(segment_ids >= 0) & (segment_ids < len(seg))]
        if rows.size == 0:
            return seg[:0]

        order = np.argsort(rows)
        rows_sorted = rows[order]
        seg_sorted = seg[rows_sorted]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return seg_sorted[inv]

    def _get_hit_backtrack_table(self, hit_type: str):
        """
        Return (hits_dataset, event->hit ref_region, backtrack_data, hit->backtrack ref).
        hit_type: 'prompt' or 'final'
        """
        hit_type = str(hit_type).lower()
        if hit_type not in ("prompt", "final"):
            raise ValueError("hit_type must be 'prompt' or 'final'")

        if hit_type == "prompt":
            hits = self.h5.get("charge/calib_prompt_hits/data", None)
            rr   = self.h5.get("charge/events/ref/charge/calib_prompt_hits/ref_region", None)
            bt   = self.h5.get("mc_truth/calib_prompt_hit_backtrack/data", None)
            ref  = self.h5.get("charge/calib_prompt_hits/ref/mc_truth/calib_prompt_hit_backtrack/ref", None)
        else:
            hits = self.h5.get("charge/calib_final_hits/data", None)
            rr   = self.h5.get("charge/events/ref/charge/calib_final_hits/ref_region", None)
            bt   = self.h5.get("mc_truth/calib_final_hit_backtrack/data", None)
            ref  = self.h5.get("charge/calib_final_hits/ref/mc_truth/calib_final_hit_backtrack/ref", None)

        return hits, rr, bt, ref

    def get_mc_overlay_for_charge_event_backtrack(
        self,
        event_index: int,
        *,
        hit_type: str = "prompt",
        top_k_segments: int = 2000,
        min_weight: float = 0.0,
        mc_only_muons: bool = False,
    ):
        """
        Production-consistent overlay:
          charge event -> hits -> hit_backtrack -> (segment_id, fraction) -> segments.

        Returns (segments, vertices=None, info).
        """
        seg = self.mc_segments
        if seg is None:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": 0, "n_bt_rows": 0, "n_unique_segments": 0, "chosen_n_segments": 0}
            return None, None, info

        hits, rr, bt, ref = self._get_hit_backtrack_table(hit_type)
        if hits is None or rr is None or bt is None or ref is None:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": 0, "n_bt_rows": 0, "n_unique_segments": 0, "chosen_n_segments": 0}
            return seg[:0], None, info

        r = rr[event_index]
        h0, h1 = int(r["start"]), int(r["stop"])
        n_hits = max(0, h1 - h0)
        if n_hits == 0:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": 0, "n_bt_rows": 0, "n_unique_segments": 0, "chosen_n_segments": 0}
            return seg[:0], None, info

        # hit -> backtrack row (usually 1:1; ref[:,1] is target row)
        bt_rows = ref[h0:h1, 1].astype(np.int64, copy=False)
        bt_rows = bt_rows[(bt_rows >= 0) & (bt_rows < len(bt))]
        if bt_rows.size == 0:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": int(n_hits), "n_bt_rows": 0, "n_unique_segments": 0, "chosen_n_segments": 0}
            return seg[:0], None, info

        bt_names = bt.dtype.names or ()
        if "segment_ids" not in bt_names or "fraction" not in bt_names:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": int(n_hits), "n_bt_rows": int(len(bt_rows)), "n_unique_segments": 0, "chosen_n_segments": 0}
            return seg[:0], None, info

        w_by_seg = defaultdict(float)
        n_bt_used = 0

        for idx in bt_rows:
            row = bt[int(idx)]
            seg_ids = np.asarray(row["segment_ids"], dtype=np.int64)
            fracs   = np.asarray(row["fraction"], dtype=float)
            if seg_ids.size == 0 or fracs.size == 0:
                continue
            n_bt_used += 1
            m = min(seg_ids.size, fracs.size)
            for sid, w in zip(seg_ids[:m], fracs[:m]):
                if np.isfinite(w) and w > 0:
                    w_by_seg[int(sid)] += float(w)

        if not w_by_seg:
            info = {"missing": True, "selection": "backtrack", "hit_type": hit_type,
                    "n_hits": int(n_hits), "n_bt_rows": int(n_bt_used), "n_unique_segments": 0, "chosen_n_segments": 0}
            return seg[:0], None, info

        items = [(sid, w) for sid, w in w_by_seg.items() if w >= float(min_weight)]
        items.sort(key=lambda t: -t[1])

        if top_k_segments and len(items) > int(top_k_segments):
            items = items[:int(top_k_segments)]

        chosen_seg_ids = np.array([sid for sid, _ in items], dtype=np.int64)
        chosen_segs = self._segments_from_segment_ids(chosen_seg_ids)

        if mc_only_muons and chosen_segs is not None and len(chosen_segs) > 0 and "pdg_id" in chosen_segs.dtype.names:
            chosen_segs = chosen_segs[np.abs(chosen_segs["pdg_id"].astype(int)) == 13]

        info = {"missing": False, "selection": "backtrack", "hit_type": hit_type,
                "n_hits": int(n_hits), "n_bt_rows": int(n_bt_used),
                "n_unique_segments": int(len(w_by_seg)), "chosen_n_segments": int(len(chosen_segs))}
        return chosen_segs, None, info