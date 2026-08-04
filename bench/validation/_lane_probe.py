"""Read the accept mask on any lane, including the ones that discard it.

The far-pair count is the only sharp instrument for "was the criterion applied?".
Accelerations are not: the large-N lane runs fp32 and the generic lane fp64, so a
relative difference of ~1e-6 between them is precision, not policy.

Getting the count off the prepared state does not work on the large-N lane.
``_apply_large_n_gpu_production_contract`` pins ``retain_traversal_result=False``
and ``retain_interactions=False`` regardless of what the caller asked for, and
``compact_far_pairs`` survives onto the state only under ``adaptive_order`` or an
explicit ``retain_far_pairs_for_grad``. The far field is consumed into the
downward locals during prepare, so by the time you hold a
``LargeNPreparedState`` the accept mask is gone. A bench that reads
``state.interactions`` there silently reports **zero** far pairs -- which then
trips every ``--min-far-pairs`` guard and reads as "this configuration measures
nothing" when in fact the measurement plumbing is what is missing.

So hook ``_build_dual_tree_artifacts`` in ``fmm_prepare``'s namespace, which is
upstream of every discard, and record per call:

* the accepted far-pair count,
* whether a ``pair_policy`` was actually installed -- the difference between
  running the criterion and running the geometric MAC underneath it, which is
  invisible in cost, and
* the per-node force scale's range, so the silent ``jnp.ones(...)`` fallback
  ``build_adaptive_policy_state`` substitutes for ``None`` cannot pass unnoticed.

A criterion prepare makes two calls: the geometric force-scale prepass (no
policy, no scale -- it must not recurse into the criterion it is measuring) and
then the real build. ``final`` is the last one.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import numpy as np


def far_pairs_from_artifacts(artifacts: Any) -> int:
    """Accepted far-pair count from whichever payload the traversal produced.

    ``traversal_result`` first: it carries an explicit ``far_pair_count``. The node
    interaction list and compact far pairs are fixed-capacity padded, so they are
    masked instead -- and for compact pairs the padding fill is unspecified, so
    ``far_pair_count`` is honoured when present rather than trusting ``-1``.
    """

    result = getattr(artifacts, "traversal_result", None)
    if result is not None:
        return int(np.asarray(jax.device_get(result.far_pair_count)).reshape(()))
    inter = getattr(artifacts, "interactions", None)
    if inter is not None:
        src = np.asarray(jax.device_get(inter.sources)).ravel()
        tgt = np.asarray(jax.device_get(inter.targets)).ravel()
        return int(np.sum((src >= 0) & (tgt >= 0)))
    compact = getattr(artifacts, "compact_far_pairs", None)
    if compact is not None:
        count = getattr(compact, "far_pair_count", None)
        if count is not None:
            return int(np.asarray(jax.device_get(count)).reshape(()))
        src = np.asarray(jax.device_get(compact.sources)).ravel()
        return int(np.sum(src >= 0))
    return -1


def near_leaf_pairs_from_artifacts(artifacts: Any) -> int:
    neighbors = getattr(artifacts, "neighbor_list", None)
    if neighbors is None:
        return -1
    return int(np.asarray(jax.device_get(neighbors.counts)).sum())


class DualBuildProbe:
    """Context manager recording every dual-tree build of a ``prepare_state``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._module: Any = None
        self._original: Any = None

    def __enter__(self) -> "DualBuildProbe":
        from jaccpot.runtime import fmm_prepare as module

        self._module = module
        self._original = module._build_dual_tree_artifacts
        original = self._original
        calls = self.calls

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            artifacts = out[0] if isinstance(out, tuple) else out
            policy_state = kwargs.get("policy_state")
            record: dict[str, Any] = {
                "far_pairs": far_pairs_from_artifacts(artifacts),
                "near_leaf_pairs": near_leaf_pairs_from_artifacts(artifacts),
                "pair_policy_installed": kwargs.get("pair_policy") is not None,
                "policy_state_installed": policy_state is not None,
            }
            # `AdaptivePolicyState` exposes the criterion's right-hand side only as
            # `target_accept_threshold` = max(eps * force_scale, 1e-24); the scale
            # itself is not kept. So the unit-scale fallback shows up here as a
            # *constant* threshold (eps everywhere) rather than as ones.
            threshold = getattr(policy_state, "target_accept_threshold", None)
            if threshold is not None:
                arr = np.asarray(jax.device_get(threshold))
                record["threshold"] = {
                    "dtype": str(arr.dtype),
                    "nodes": int(arr.size),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "constant": bool(arr.size > 1 and arr.min() == arr.max()),
                }
            else:
                record["threshold"] = None
            calls.append(record)
            return out

        module._build_dual_tree_artifacts = wrapper
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._module is not None:
            self._module._build_dual_tree_artifacts = self._original

    @property
    def final(self) -> dict[str, Any]:
        return self.calls[-1] if self.calls else {}

    def check_criterion_was_applied(self, *, context: str) -> None:
        """Raise unless the last build really carried the criterion.

        Both failures this guards are silent and make the run *faster*: a build
        with no ``pair_policy`` runs the plain geometric MAC, and a policy state
        built from ``force_scale_nodes=None`` compares against ``eps * 1`` instead
        of ``eps * min_b |a_b|``.
        """

        final = self.final
        if not final:
            raise RuntimeError(f"{context}: no dual-tree build was observed at all")
        if not final.get("pair_policy_installed"):
            raise RuntimeError(
                f"{context}: the final dual build ran with no pair_policy, so it "
                "answered the geometric MAC rather than the Dehnen criterion"
            )
        threshold: Optional[dict[str, Any]] = final.get("threshold")
        if threshold is None:
            raise RuntimeError(
                f"{context}: the final dual build carried no policy state, so the "
                "criterion had no acceptance threshold to compare against"
            )
        if threshold.get("constant"):
            raise RuntimeError(
                f"{context}: the acceptance threshold is constant across all "
                f"{threshold['nodes']} nodes at {threshold['min']:.3e} -- that is the "
                "signature of the unit force-scale fallback (eps*1), not a prepass"
            )
