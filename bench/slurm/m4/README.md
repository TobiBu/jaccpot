# M4 on HoreKa: the LET exchange and work-balanced cuts at scale

The two opt-in switches of track M4 (`docs/production_2026-11.md` `## M4`), on dedicated nodes, as
N processes x 1 GPU through the production runner (`bench/m4_exchange_cuts.py`):

- `JACCPOT_CROSS_EXCHANGE=let`: each peer receives only the part of a device's summary tree its cross walk
  opens (exact: bitwise to all_gather, pinned locally on 2-4 A100s and 2-16 forced CPU devices);
- `JACCPOT_MESH_CUTS=work`: the rebalance before each segment cuts at equal estimated work (near
  particle-particle interactions + far pairs x `JACCPOT_MESH_WORK_FAR_WEIGHT`, default 16) instead of equal counts.

| job | nodes x tasks | arms | expected wall | `--time` |
| --- | --- | --- | --- | --- |
| `node1` (`m4_node1`) | 1 x 4 | disc + bulge, galaxy preset: 1e7 (2.5e6 per GPU) x {all_gather, let} x {count, work}; 1e8 (2.5e7 per GPU) default vs let + work | ~1.5-2 h | 3:00 |
| `node2` (`m4_node2`) | 2 x 4 | the same per GPU on 8 GPUs (1e7 at 1.25e6 per GPU; 2e8 at 2.5e7 per GPU) | ~1.5-2 h | 3:00 |
| `m5_scaling.sbatch` | 1-8 x 4 | M5 TEMPLATE (optional, not in the default set): weak (3.125e7 per GPU, 1e9 on 8 nodes) and strong (1e9) scaling, the default vs let + work | ~15-20 node-h weak, ~12 strong | per submission |

Per arm `result.json`: the edges' walls (the first edge holds the compiles; the summary's median skips it), per
segment the live rows per device, the work per device before / after each rebalance (work cuts), the LET rows
(the needs: max over the mesh) next to the all_gather's `(ndev - 1) x max_summary_nodes`, bits, retries, the
momentum split at the start and the end, each card's peak.

**What to read:** (1) `let` vs `all_gather` step wall at 4 and 8 GPUs (the request walks cost about one cross
walk per round against fewer bytes on NVLink / InfiniBand: the LET-default decision); (2) the LET rows per device
from 4 to 8 GPUs at fixed N per GPU (flat is the claim); (3) work vs count cuts: the imbalance after the rebalance
and the step wall (the slowest device's); (4) momentum alike in every arm.
