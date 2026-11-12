"""Figure 10 data -- communication vs compute fraction. NOT IMPLEMENTABLE YET.

This script refuses to run, on purpose, and explains why rather than emitting a
plausible-looking artifact.

The measurement needs per-stage timings on the distributed force path. Those do
not exist. The driver's per-device diagnostic vector (``DIAG_FIELDS`` in
``jaccpot/distributed/fmm.py``) carries interaction counts and overflow flags and
no timers whatsoever, and there is no other instrumentation in the ``shard_map``
pipeline. The single-device breakdown behind figure 06 is not transferable: it
reads ``_refresh_timing_*`` counters that exist only on the strict refresh path
with fusion disabled -- "fusing the stages is exactly what makes them
unmeasurable" -- and the distributed path has no analogue.

The scaffold this replaces declared eight stage names (``local_tree_build``,
``self_m2l_near``, ``all_gather_coarse``, ``coarse_m2m``, ``cross_walk``,
``halo_import``, ``remote_m2l``, ``p2p_combined``) and a ``COMM_STAGES`` subset of
them. Nothing in the code emits any of those names. Anything built on them would
have been reporting invented structure, which is worse than reporting nothing, so
they are gone rather than carried forward.

What would actually be needed, in rough order of intrusiveness:

1. A host callback per stage inside the sharded region, guarded so it compiles
   out when profiling is off. Direct, but callbacks inside ``shard_map`` serialise
   the thing being measured, so the timings would perturb the total.
2. A ``jax.profiler`` trace of one evaluation, with XLA ops attributed to stages
   by name. Non-invasive and gives real device time, but the attribution is
   brittle against fusion and needs re-checking whenever the kernels change.
3. Stage ablation: build evaluators with individual stages disabled and difference
   the wall clocks. Cheapest to write and the least trustworthy, because removing
   a stage changes what XLA fuses around it.

Option 2 is the most promising and is library/profiling work rather than
analysis, so it belongs in its own change with its own validation -- not folded
into a figure run.

Until then, section 5 states which stages dominate communication only if that
claim is backed by a measurement. It currently is not.
"""

from __future__ import annotations

import sys

MESSAGE = __doc__


def main() -> int:
    """Explain why the comm/compute split cannot be measured yet, and refuse.

    Returns
    -------
    int
        Always 2; there is nothing to produce.
    """

    print(MESSAGE, file=sys.stderr)
    print(
        "REFUSING: no per-stage timing exists on the distributed path, so any "
        "comm/compute split this script produced would be fabricated.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
