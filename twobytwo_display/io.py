from __future__ import annotations

import h5py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class FlowFile:
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

    def get_event_hits(self, event_index: int):
        rr = self.hits_ref_region[event_index]
        start = int(rr["start"])
        stop  = int(rr["stop"])
        return self.hits[start:stop]

    def n_events(self) -> int:
        return len(self.events)

    def event_id(self, event_index: int) -> int:
        ev = self.events
        if "id" in ev.dtype.names:
            return int(ev[event_index]["id"])
        return int(event_index)

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
            # fallback: if event_id is already an index
            cand = tr["event_id"].astype(int)
            cand = cand[(cand >= 0) & (cand < len(ev))]
            return sorted(set(map(int, cand)))

        # map event_id values to event indices
        ids = ev["id"].astype(tr["event_id"].dtype, copy=False)
        id_to_index = {int(v): i for i, v in enumerate(ids)}
        out = []
        for eid in tr["event_id"]:
            i = id_to_index.get(int(eid))
            if i is not None:
                out.append(i)
        return sorted(set(out))
