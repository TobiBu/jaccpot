"""``preset="galaxy"``: the fused lane's production configuration, by name.

The preset must be the configuration ``bench/fused_memory_budget.py`` measures at its
defaults -- the L1-L3 rows -- and nothing else. Program identity is gated by
``bench/provider_hlo_identity.py``'s ``galaxy_explicit`` / ``galaxy_preset`` pair
(compile-heavy, run before a change lands); this file pins, at construction cost:

* every engine attribute the preset resolves equals the bench's explicit
  configuration's (``--cubic-bounds``), the leaf capacity (``"auto"``, sized at the
  first prepare) excepted;
* it runs on the ``large_n_gpu`` engine profile;
* the settings it is built from (:class:`jaccpot.production.GalaxySettings`);
* its Morton box is a cube (S3b), ``JACCPOT_CUBIC_BOUNDS`` overrides the box both
  ways, and a cube is part of the provider's compile key;
* ``theta=None`` keeps 0.6 for every other preset, and explicit arguments win.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod, FMMPreset, TraversalOverrides
from jaccpot.config import (
    FMMAdvancedConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.production import GALAXY_SETTINGS, GalaxySettings
from jaccpot.runtime.fused_provider import FusedForceProvider


def _bench_solver(**overrides):
    """``fused_memory_budget._solver`` at the bench's defaults (leaf 64) with
    ``--cubic-bounds``."""
    kwargs = dict(
        preset="large_n_gpu",
        basis="real",
        theta=0.8,
        G=1.0,
        softening=1e-3,
        softening_kernel=None,
        working_dtype=jnp.float32,
        use_pallas=None,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=64,
                leaf_partition="cells",
                leaf_capacity=4096,
                cell_min_level=8,
                cubic_bounds=True,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            runtime=RuntimePolicyConfig(
                # the harness's FAST_LANE_ENV_BY_LEAF[64]["_traversal_overrides"]
                traversal_config=TraversalOverrides(
                    max_neighbors_per_leaf=4096, max_interactions_per_node=16384
                )
            ),
            mac_type="dehnen",
        ),
        fixed_order=6,
    )
    kwargs.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fixed_order
        return FastMultipoleMethod(**kwargs)


def _state(solver) -> dict[str, str]:
    return {k: repr(v) for k, v in vars(solver._impl).items()}


def test_the_preset_is_the_bench_configuration():
    galaxy = _state(FastMultipoleMethod(preset="galaxy", softening=1e-3))
    bench = _state(_bench_solver())
    differ = {k for k in set(galaxy) | set(bench) if galaxy.get(k) != bench.get(k)}
    # the capacity: "auto" until the first prepare, and the bench's leaf-64
    # traversal overrides, which the walk the preset runs never reads (the
    # provider_hlo_identity galaxy pair is identical with them)
    assert differ == {
        "_tree_leaf_capacity",
        "_tree_leaf_capacity_auto",
        "_traversal_field_overrides",
    }
    assert galaxy["_tree_leaf_capacity"] == "None"
    assert galaxy["_tree_leaf_capacity_auto"] == "1.15"
    # without the overrides only the capacity differs
    plain = _bench_solver(
        advanced=dataclasses.replace(
            _bench_solver().advanced, runtime=RuntimePolicyConfig()
        )
    )
    bench = _state(plain)
    assert {k for k in set(galaxy) | set(bench) if galaxy.get(k) != bench.get(k)} == {
        "_tree_leaf_capacity",
        "_tree_leaf_capacity_auto",
    }


def test_the_preset_runs_on_the_large_n_gpu_profile():
    solver = FastMultipoleMethod(preset=FMMPreset.GALAXY)
    impl = solver._impl
    assert solver.preset is FMMPreset.GALAXY
    assert impl.preset == "large_n_gpu"
    assert impl._is_large_n_gpu_production_profile()
    assert impl.theta == pytest.approx(0.8)
    assert jnp.dtype(impl.working_dtype) == jnp.float32
    assert impl.softening_kernel == "ferrers3"
    assert impl.fixed_order == 6
    assert impl.mac_type == "dehnen"
    assert (impl._tree_leaf_partition, impl._tree_cell_min_level) == ("cells", 8)
    tree = solver.advanced.tree
    assert (tree.leaf_target, tree.leaf_capacity, tree.leaf_headroom) == (
        64,
        "auto",
        1.15,
    )
    assert solver.advanced.nearfield.mode == "auto"


def test_the_settings_are_the_production_ones():
    assert GALAXY_SETTINGS == GalaxySettings()
    assert dataclasses.asdict(GALAXY_SETTINGS) == dict(
        leaf_size=64,
        max_order=6,
        theta=0.8,
        cell_min_level=8,
        leaf_capacity="auto",
        leaf_headroom=1.15,
        cubic_bounds=True,
        softening_kernel="ferrers3",
        mac_type="dehnen",
        adaptive_eps=None,
        carry="particles",
        precision="fp32",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        GALAXY_SETTINGS.theta = 0.6  # type: ignore[misc]


def _elongated(n: int = 512) -> jnp.ndarray:
    """A cloud whose per-axis box is 8:2:1 (a flat, elongated disc)."""
    rng = np.random.default_rng(3)
    return jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)) * [8.0, 2.0, 1.0], jnp.float32)


def _box_edges(solver) -> np.ndarray:
    """The box the solver's prepares (eager and traced refresh) infer."""
    lo, hi = solver._impl._resolve_prepare_state_bounds(
        positions=_elongated(), bounds=None
    )
    return np.asarray(hi) - np.asarray(lo)


def test_the_preset_infers_a_cubic_morton_box(monkeypatch):
    """S3b: on the 25M disc without its bulge (box 3:1) the cube needs 63 % fewer
    far and 50 % fewer near pairs and halves the error; on the production disc +
    bulge the two boxes agree to 1 %. The library default stays per axis."""
    monkeypatch.delenv("JACCPOT_CUBIC_BOUNDS", raising=False)
    galaxy = FastMultipoleMethod(preset="galaxy")
    assert galaxy.advanced.tree.cubic_bounds is True
    assert galaxy._impl._tree_cubic_bounds is True
    edges = _box_edges(galaxy)
    np.testing.assert_allclose(edges, edges.max(), rtol=1e-6)
    lo, hi = galaxy._impl._resolve_prepare_state_bounds(
        positions=_elongated(), bounds=None
    )
    pts = np.asarray(_elongated())
    assert np.all(np.asarray(lo) <= pts.min(0)) and np.all(np.asarray(hi) >= pts.max(0))
    # CONTROL: the library default on the same cloud is the per-axis box
    default = FastMultipoleMethod(preset="large_n_gpu")
    assert default._impl._tree_cubic_bounds is False
    edges = _box_edges(default)
    assert edges.max() / edges.min() > 7.0


def test_the_environment_overrides_the_configured_box(monkeypatch):
    """``JACCPOT_CUBIC_BOUNDS`` wins when set: ``0`` gives the galaxy preset the
    per-axis box, ``1`` gives the library default the cube."""
    galaxy = FastMultipoleMethod(preset="galaxy")
    default = FastMultipoleMethod(preset="large_n_gpu")
    monkeypatch.delenv("JACCPOT_CUBIC_BOUNDS", raising=False)
    edges = _box_edges(galaxy)
    np.testing.assert_allclose(edges, edges.max(), rtol=1e-6)
    # the same engine: the variable is read per call
    monkeypatch.setenv("JACCPOT_CUBIC_BOUNDS", "0")
    edges = _box_edges(galaxy)
    assert edges.max() / edges.min() > 7.0
    monkeypatch.setenv("JACCPOT_CUBIC_BOUNDS", "1")
    edges = _box_edges(default)
    np.testing.assert_allclose(edges, edges.max(), rtol=1e-6)


def test_a_cubic_box_is_in_the_provider_key(monkeypatch):
    """The traced refresh reads the box at trace time, so a cached scan must not
    serve the other box: the key carries a cube, and a per-axis key is the one it
    was before S3b (no new element)."""
    monkeypatch.delenv("JACCPOT_CUBIC_BOUNDS", raising=False)
    galaxy = FusedForceProvider(
        FastMultipoleMethod(preset="galaxy"), leaf_size=64, max_order=6
    )
    per_axis = FusedForceProvider(
        FastMultipoleMethod(preset="large_n_gpu", theta=0.8), leaf_size=64, max_order=6
    )
    assert galaxy.key()[-1] == "cubic_bounds"
    assert "cubic_bounds" not in per_axis.key()
    assert galaxy.key()[:-1] == per_axis.key()
    monkeypatch.setenv("JACCPOT_CUBIC_BOUNDS", "0")
    assert "cubic_bounds" not in galaxy.key()
    monkeypatch.setenv("JACCPOT_CUBIC_BOUNDS", "1")
    assert per_axis.key()[-1] == "cubic_bounds"


def test_theta_defaults_by_preset_and_explicit_arguments_win():
    for preset in ("fast", "balanced", "accurate", "large_n_gpu"):
        assert FastMultipoleMethod(preset=preset)._impl.theta == pytest.approx(0.6)
    galaxy = FastMultipoleMethod(
        preset="galaxy", theta=0.5, softening_kernel="plummer", precision="fp64"
    )._impl
    assert galaxy.theta == pytest.approx(0.5)
    assert galaxy.softening_kernel == "plummer"
    assert jnp.dtype(galaxy.working_dtype) == jnp.float64
    assert (
        jnp.dtype(
            FastMultipoleMethod(
                preset="galaxy", working_dtype=jnp.float64
            )._impl.working_dtype
        )
        == jnp.float64
    )
