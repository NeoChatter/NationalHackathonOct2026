"""
Experience memory: the place database learns today's appearance while driving.

If the run's conditions differ from the recorded database (different weather / time of day), retrieval against the
original database degrades. Whenever the tracker is SURE where it is, the current descriptor is stored at the
estimated pose. From the second lap on, retrieval compares the run against itself, and the memory keeps following
the conditions as they change during the run. This is a generic domain-adaptation mechanism, not specific to any
condition, and it uses no evaluation data.

Causal and GT-free: only past images and the tracker's own estimate are used.

Poisoning is the risk (a wrong pose stored would attract later matches), so a frame is only stored when
  - the tracker is TRACKING with >= mem_min_streak consecutive accepted retrieval updates,
  - its position uncertainty is <= mem_max_pos_std,
  - the along-track filter independently agrees (|s_filter - s_ekf| <= mem_sf_agree_m, enough peak mass),
  - this part of the track was not stored in the last mem_revisit_s seconds (one entry per pass).
Memory is bounded: mem_per_bin entries per mem_bin_m of track; a full bin replaces its oldest entry, so stale
light is forgotten. The night database itself is never modified.
"""
import math

import numpy as np
import torch

DEFAULT_MEM = dict(
    mem_bin_m=2.0,
    mem_per_bin=2,
    mem_max=5000,
    mem_min_streak=40,       # ~4 s of unbroken confident tracking before a frame may be stored
    mem_max_pos_std=0.6,
    mem_sf_agree_m=2.0,
    mem_sf_mass=0.6,
    mem_revisit_s=20.0,
    mem_sim_hi=0.90,         # only store when the BEST database match is below this: conditions differ from the DB,
                             # so a new descriptor adds information. On matched conditions (sim ~ 0.99) store nothing,
                             # which stops matched-condition drift from poisoning the database.
)


class ExperienceMemory:
    def __init__(self, db_emb: torch.Tensor, track, cfg=None):
        self.c = {**DEFAULT_MEM, **(cfg or {})}
        N, D = db_emb.shape
        self.n0 = N
        self.buf = torch.empty((N + int(self.c['mem_max']), D), dtype=db_emb.dtype, device=db_emb.device)
        self.buf[:N] = db_emb
        self.n = N
        self.track = track
        self.nb = max(int(math.ceil(track.length / self.c['mem_bin_m'])), 1)
        self.bins = [[] for _ in range(self.nb)]
        self.last_t = np.full(self.nb, -np.inf)
        self.n_added = 0
        self.n_replaced = 0

    @property
    def emb(self) -> torch.Tensor:
        """Night database + memory, aligned with the tracker's database arrays."""
        return self.buf[:self.n]

    @property
    def size(self) -> int:
        return self.n - self.n0

    def maybe_add(self, emb_vec: torch.Tensor, tracker, t_s: float, top1_sim: float) -> int:
        """0 = not stored, 1 = new entry, 2 = replaced the oldest entry of this track bin.
        top1_sim: best similarity to the current database this frame. Storing is skipped when it is high,
        i.e. the database already matches the conditions and a new descriptor would add nothing (and could only
        poison the map with the tracker's own along-track drift)."""
        if top1_sim >= self.c['mem_sim_hi'] or not tracker.memory_ok(self.c):
            return 0
        s, _, _ = self.track.project1(tracker.x[0], tracker.x[1], tracker.x[2])
        b = int(s // self.c['mem_bin_m']) % self.nb
        if t_s - self.last_t[b] < self.c['mem_revisit_s']:
            return 0
        slot = self.bins[b]
        if len(slot) < int(self.c['mem_per_bin']) and self.n < len(self.buf):
            j, kind = self.n, 1
            self.n += 1
            slot.append(j)
            self.n_added += 1
        elif slot:
            j, kind = slot.pop(0), 2
            slot.append(j)
            self.n_replaced += 1
        else:
            return 0                                             # buffer full and nothing to replace here
        self.buf[j] = emb_vec
        tracker.db_put(j, float(tracker.x[0]), float(tracker.x[1]), float(tracker.x[2]))
        self.last_t[b] = t_s
        return kind
