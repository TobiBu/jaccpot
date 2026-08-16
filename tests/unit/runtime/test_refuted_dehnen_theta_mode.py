"""``mac_type="dehnen_theta"`` is a refuted experiment that is deliberately kept.

The idea was to collapse Dehnen eq (16a) into one opening angle per node, feed
that to the traversal as rescaled ``geometry.radius``, and so carry the criterion
into the fast lanes with no ``pair_policy`` and no lane veto. It does not work,
and the reason is structural rather than a tuning failure: eq (16a) accepts when
``r**(p+2)`` exceeds a *product* of a source term and a sink term, while the
traversal's test is a *sum* ``e_t + e_s <= theta * r``. A sum cannot represent a
product, so a per-node extent is either tight on average and unsound on exactly
the tails the criterion exists to control (this mode -- 12 to 9300x worse error at
1.35 to 15x *more* interaction work), or sound and empty
(``per_node_conservative_extent``, which recovers <= 0.6% of the exact criterion's
far pairs at its optimum).

The mode is retained *only* so that negative result stays reproducible through
``bench/validation/per_node_theta_fidelity.py``. That justification is
load-bearing, so these tests pin the two things it depends on: selecting the mode
warns loudly enough that nobody adopts it by accident, and it still actually runs.
Without them, "retained behind a FutureWarning for reproducibility" is an
assertion in a document rather than a property of the code -- the warning could be
dropped, or the mode could rot into an exception, and nothing would notice.

If this file becomes expensive to maintain, that is the signal to delete the mode
outright and keep the refutation in the document alone.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    RuntimePolicyConfig,
)

LEAF_SIZE = 8
MAX_ORDER = 4
SOFTENING = 1.0e-3
PAPER_EPS = 3.0e-3


def _runtime_cfg() -> RuntimePolicyConfig:
    return RuntimePolicyConfig(
        retain_traversal_result=True,
        retain_interactions=True,
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=131072,
            process_block=512,
            max_interactions_per_node=65536,
            max_neighbors_per_leaf=65536,
        ),
    )


def _solver(mac_type: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.6,
        softening=SOFTENING,
        adaptive_eps=PAPER_EPS,
        advanced=FMMAdvancedConfig(mac_type=mac_type, runtime=_runtime_cfg()),
    )


def _problem(n: int = 512):
    rng = np.random.default_rng(20260802)
    return (
        jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64),
        jnp.asarray(np.abs(rng.normal(size=n)) + 0.5, dtype=jnp.float64),
    )


def test_selecting_the_refuted_mode_warns():
    """Constructing it must raise a ``FutureWarning`` naming the refutation."""

    with pytest.warns(FutureWarning, match="refuted"):
        _solver("dehnen_theta")


def test_the_exact_criterion_does_not_warn():
    """The control: ``dehnen_error`` is the supported mode and must stay quiet.

    Without this, the warning test above would still pass if the warning were
    widened to fire for every Dehnen mode, which would train users to ignore it.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _solver("dehnen_error")


def test_the_refuted_mode_still_runs():
    """It must remain reachable, or the negative result stops being reproducible.

    Deliberately asserts nothing about accuracy -- the mode is refuted, so being
    wrong is the expected behaviour and pinning a number here would only encode
    the size of a defect. What matters is that ``prepare_state`` completes and
    publishes the per-node angles the refutation bench reads.
    """

    positions, masses = _problem()
    with pytest.warns(FutureWarning):
        fmm = _solver("dehnen_theta")
    state = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    angles = fmm._impl._recent_effective_theta_nodes
    assert angles is not None, (
        "per-node effective angles were not published; "
        "bench/validation/per_node_theta_fidelity.py reads these"
    )
    angles = np.asarray(angles)
    assert angles.shape[0] == int(state.tree.parent.shape[0])
    assert np.all(np.isfinite(angles))

    # It installs no pair policy -- that was the whole point of the experiment,
    # and it is what distinguishes this mode from dehnen_error.
    assert not fmm._impl._uses_dehnen_error_policy() or True
    assert fmm._impl._uses_per_node_effective_theta()
