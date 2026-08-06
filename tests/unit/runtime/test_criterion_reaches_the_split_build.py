"""The criterion must actually *reach* the split/streamed build, not merely be allowed in.

``_can_split_dual_tree_build`` no longer refuses a pair policy, but that alone was
not enough: ``need_traversal_result`` was forced true by ``use_paper_fixed_policy``,
and the split build refuses whenever the traversal result is needed. So the
criterion still always took the monolithic build and materialised
``num_nodes x max_interactions_per_node`` -- ~2.5 GiB at N=1e7 / leaf 256, which is
the binding constraint there.

Nothing consumed that traversal result. It feeds
``_prepare_state_extract_adaptive_far_pairs``, which runs only under
``adaptive_order``, and ``use_paper_fixed_policy`` requires ``not adaptive_order``.

Dropping the forcing changes the far-pair payload from a node interaction list to
compact COO pairs, which is a genuinely different downstream path, so the property
worth pinning is that the two paths agree: same accept mask, same accelerations to
round-off.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime._interaction_cache as interaction_cache
from jaccpot.config import FMMAdvancedConfig
from jaccpot.solver import FastMultipoleMethod

# Compile-bound: builds solvers and runs full prepares. Marked slow to match the
# sibling MAC tests (see test_dehnen_mac_gradients.py).
pytestmark = pytest.mark.slow

N = 4096
LEAF = 32
ORDER = 4
EPS = 1e-3


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(N, 3))
    mass = rng.uniform(0.5, 1.5, size=N)
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


def _solver(*, streamed: bool, retain: bool, force_scale_mode: str = "paper_cached"):
    cfg = FMMAdvancedConfig()
    cfg = replace(
        cfg,
        mac_type="dehnen_error",
        tree=replace(cfg.tree, tree_type="radix"),
        farfield=replace(cfg.farfield, streamed_far_pairs=streamed),
        runtime=replace(
            cfg.runtime,
            retain_traversal_result=retain,
            retain_interactions=retain,
            prepare_stage_memory_split_enabled=True,
        ),
    )
    return FastMultipoleMethod(
        theta=1.0,
        adaptive_eps=EPS,
        expansion_basis="solidfmm",
        softening=0.0,
        mac_force_scale_mode=force_scale_mode,
        advanced=cfg,
    )


@pytest.fixture
def _build_spy(monkeypatch):
    """Count split vs monolithic dual builds, and whether a policy rode along."""

    counts = {"split": 0, "split_with_policy": 0, "monolithic": 0}
    original_split = interaction_cache._build_dual_tree_artifacts_split
    original_raw = interaction_cache._dual_tree_build_raw

    def split_spy(**kwargs):
        counts["split"] += 1
        if kwargs.get("pair_policy") is not None:
            counts["split_with_policy"] += 1
        return original_split(**kwargs)

    def raw_spy(**kwargs):
        counts["monolithic"] += 1
        return original_raw(**kwargs)

    monkeypatch.setattr(
        interaction_cache, "_build_dual_tree_artifacts_split", split_spy
    )
    monkeypatch.setattr(interaction_cache, "_dual_tree_build_raw", raw_spy)
    return counts


def test_the_streamed_config_reaches_the_split_build_with_a_policy(_build_spy):
    positions, masses = _problem()
    fmm = _solver(streamed=True, retain=False)
    fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)

    assert _build_spy["split_with_policy"] >= 1, (
        "the criterion never reached the split build with a policy installed "
        f"(counts={_build_spy}); need_traversal_result is forcing the monolithic "
        "build again"
    )
    assert (
        _build_spy["monolithic"] == 0
    ), f"the criterion still took the monolithic build (counts={_build_spy})"


def test_the_retaining_config_still_takes_the_monolithic_build(_build_spy):
    """Control: asking for the traversal result must still get it.

    Without this the test above could pass because the split build had become
    unconditional, which would silently drop ``state.dual_tree_result`` for every
    caller that asked to retain it.
    """

    positions, masses = _problem()
    fmm = _solver(streamed=False, retain=True)
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)

    assert _build_spy["monolithic"] >= 1, (
        f"a caller that retained the traversal result got the split build "
        f"(counts={_build_spy})"
    )
    assert state.dual_tree_result is not None
    assert state.interactions is not None


def test_the_two_far_pair_payloads_agree():
    """Compact COO pairs and the node interaction list must give the same force."""

    positions, masses = _problem()

    accelerations = []
    far_counts = []
    for streamed, retain in ((False, True), (True, False)):
        fmm = _solver(streamed=streamed, retain=retain)
        state = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
        accelerations.append(
            np.asarray(
                jax.device_get(
                    fmm.evaluate_prepared_state(state, return_potential=False)
                ),
                dtype=np.float64,
            )
        )
        interactions = state.interactions
        if interactions is not None:
            src = np.asarray(jax.device_get(interactions.sources))
            tgt = np.asarray(jax.device_get(interactions.targets))
            far_counts.append(int(((src >= 0) & (tgt >= 0)).sum()))

    assert far_counts and far_counts[0] > 0, (
        "the retaining arm accepted no far pairs, so this comparison is vacuous "
        "(trap 5: eq (16a) is very conservative at small N)"
    )

    node_path, streamed_path = accelerations
    norms = np.linalg.norm(node_path, axis=1)
    rel = np.linalg.norm(streamed_path - node_path, axis=1) / np.where(
        norms > 0, norms, 1.0
    )
    # Round-off from a different summation order, not a different accept mask. A
    # changed mask shows up orders of magnitude above this.
    assert rel.max() < 1e-12, (
        f"the streamed and node-interaction far-pair payloads disagree by "
        f"{rel.max():.3e} -- that is a different accept mask, not round-off"
    )


def test_eq_16b_runs_on_the_streamed_path_and_keeps_its_far_term():
    """eq (16b)'s prepass must survive the streamed payload, far term included.

    This is the failure that killed the first N=1e6 sweep on the lane. The prepass
    sums exact scalar terms over near pairs and monopoles over far ones, and the
    streamed build produced compact far pairs and then *discarded* them before the
    estimator ran -- so `_compute_force_scale_fb_prepass_from_tree_artifacts` saw
    neither a node interaction list nor compact pairs and raised.

    The assertion that matters is not "it runs" but "the far term is still there":
    a near-only `f_b` captures only 53-66% of the true value once there are enough
    leaves for the far field to matter, which reads as an ordinary under-estimate
    rather than a missing term. So compare against the node-interaction path, which
    is the configuration eq (16b) was validated on.
    """

    positions, masses = _problem()

    scales = []
    for streamed, retain in ((False, True), (True, False)):
        fmm = _solver(streamed=streamed, retain=retain, force_scale_mode="paper_fb")
        engine = getattr(fmm, "_impl", fmm)
        fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
        scale = engine._last_force_scale_nodes
        assert scale is not None, "the eq (16b) prepass produced no force scale"
        scales.append(np.asarray(jax.device_get(scale), dtype=np.float64))

    node_path, streamed_path = scales
    assert node_path.min() != node_path.max(), (
        "the eq (16b) force scale is constant, so this comparison cannot see a "
        "missing far term"
    )
    rel = np.abs(streamed_path - node_path) / np.maximum(np.abs(node_path), 1e-300)
    assert rel.max() < 1e-6, (
        f"the streamed path's f_b differs from the node-interaction path's by "
        f"{rel.max():.3e}; a dropped far term shows up here as a systematic "
        "under-estimate of tens of percent"
    )
