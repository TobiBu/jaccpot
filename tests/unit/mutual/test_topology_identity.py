"""Topology identity: what it must catch, and what it must not confuse.

The facility exists because four things came close to answering "did the
discrete structure change?" and none did. These tests pin the distinctions,
starting with the one that motivates the module: equal size counters do not
mean equal structure.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual import (
    TopologySwitchCounter,
    build_mutual_topology,
    diff_topologies,
    fingerprint_topology,
    topologies_identical,
)

THETA = 0.5
ORDER = 4
LEAF = 8


def _system(n, seed):
    """Return (positions, masses) for a uniform cube.

    JAX arrays, not NumPy: the builder runs a solver whose runtime type checks
    require them. The topology it returns is host-side NumPy regardless, which
    is what the identity facility consumes.
    """
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=n), dtype=jnp.float64)
    return positions, masses


def _topology(n=256, seed=0, theta=THETA, leaf_size=LEAF):
    """Build one topology for a uniform cube."""
    positions, masses = _system(n, seed)
    topology, _ = build_mutual_topology(
        positions, masses, theta=theta, order=ORDER, leaf_size=leaf_size
    )
    return topology


# Every test below except the two that need independent builds of one input (the
# determinism test, and one entry of the entry-point agreement matrix) reads its
# topologies from this module-scoped memo, so each distinct (n, seed, theta, leaf)
# is built once: four memoized builds plus three independent ones, where there
# were 22. The configurations share one seed and two leaf sizes for the same reason
# (2026-10-12: seeds 0/1/3, leaves 8/16/32). No test mutates a topology; the ones that need a variant build it with
# `dataclasses.replace` on copied arrays. A module fixture is built once per xdist
# worker that asks for it, so its users share a group (CI: `-n 2 --dist loadgroup`).
_MEMO_USERS = pytest.mark.xdist_group("mutual_topology_identity_memo")


@pytest.fixture(scope="module")
def cached_topology():
    """The memoized ``_topology``: same signature, each configuration built once.

    Returns
    -------
    Callable[..., MutualTopology]
        ``cached_topology(n=256, seed=0, theta=THETA, leaf_size=LEAF)``.
    """
    built = {}

    def get(n=256, seed=0, theta=THETA, leaf_size=LEAF):
        key = (n, seed, theta, leaf_size)
        if key not in built:
            built[key] = _topology(n=n, seed=seed, theta=theta, leaf_size=leaf_size)
        return built[key]

    return get


def test_the_same_positions_give_an_identical_fingerprint():
    """The builder must be deterministic, or a switch count is meaningless.

    Every other test here rests on this: if two builds from identical input
    could differ, a nonzero switch rate would say nothing about the
    optimisation path. Two independent builds, deliberately not the memo.
    """
    first = _topology(seed=0)
    second = _topology(seed=0)
    assert fingerprint_topology(first) == fingerprint_topology(second)
    assert fingerprint_topology(first).digest == fingerprint_topology(second).digest
    assert topologies_identical(first, second)
    assert diff_topologies(first, second).identical


@_MEMO_USERS
def test_equal_size_counters_do_not_mean_equal_structure(cached_topology):
    """The reason this module exists.

    ``MutualTopology.summary()`` returns size counters and documents them as
    safe to compare across runs. They are -- but they cannot detect a structure
    that changed without changing any count, and that is exactly the
    coincidence a switch diagnostic must not read as "no switch".

    Constructed rather than searched for, so the test is deterministic: swapping
    two entries of the near list changes the structure and leaves every count,
    shape and dtype untouched.
    """
    topology = cached_topology()
    assert topology.num_near_pairs >= 2, "need two near pairs to swap"

    near_a = topology.near_a.copy()
    near_a[[0, 1]] = near_a[[1, 0]]
    reordered = dataclasses.replace(topology, near_a=near_a)

    assert reordered.summary() == topology.summary()
    assert not topologies_identical(topology, reordered)

    diff = diff_topologies(topology, reordered)
    assert not diff.identical
    assert diff.changed == ("near_pairs",)
    assert diff.interaction_lists_changed


@_MEMO_USERS
def test_a_permutation_only_change_is_not_an_interaction_change(cached_topology):
    """Reordering particles is a switch, but not of the interaction lists.

    Both are real events with different consequences for a frozen-topology
    gradient, so the facility reports them separately rather than collapsing
    both to "switched".
    """
    topology = cached_topology()
    forward = topology.forward_permutation.copy()
    forward[[0, 1]] = forward[[1, 0]]
    permuted = dataclasses.replace(topology, forward_permutation=forward)

    diff = diff_topologies(topology, permuted)
    assert diff.changed == ("permutation",)
    assert not diff.interaction_lists_changed
    assert not diff.identical


@_MEMO_USERS
def test_different_positions_change_the_structure(cached_topology):
    """A genuinely different system must not fingerprint the same."""
    first = cached_topology(seed=0)
    second = cached_topology(seed=1)
    diff = diff_topologies(first, second)
    assert not diff.identical
    assert not topologies_identical(first, second)


@_MEMO_USERS
def test_theta_is_compared_exactly_not_by_rounding(cached_topology):
    """Two acceptance parameters that differ in the last bit are not the same.

    ``theta`` gates which pairs are far, so a fingerprint that rounded it could
    call two different criteria identical -- the same class of silent error
    ``pair_policy_cache_identity`` exists to prevent on the cache path.
    """
    topology = cached_topology()
    nudged = dataclasses.replace(topology, theta=np.nextafter(topology.theta, 1.0))
    diff = diff_topologies(topology, nudged)
    assert diff.changed == ("theta",)
    assert fingerprint_topology(topology).digest != fingerprint_topology(nudged).digest


@_MEMO_USERS
def test_max_leaf_size_is_part_of_identity(cached_topology):
    """Padding width sets kernel shapes, so a change in it is a change."""
    narrow = cached_topology(leaf_size=8)
    wide = cached_topology(leaf_size=16)
    diff = diff_topologies(narrow, wide)
    assert not diff.identical
    assert "max_leaf_size" in diff.changed


@_MEMO_USERS
def test_the_two_entry_points_never_disagree(cached_topology):
    """``topologies_identical`` and ``diff_topologies`` must agree.

    They take different routes -- one short-circuits on raw arrays, the other
    compares digests -- so a disagreement would mean one of them is wrong. The
    second ``seed=0`` entry is an independent build, so identical-but-distinct
    objects are compared too, not only an object with itself.
    """
    candidates = [
        cached_topology(seed=0),
        _topology(seed=0),
        cached_topology(seed=1),
        cached_topology(leaf_size=16),
        cached_topology(n=512),
    ]
    for left in candidates:
        for right in candidates:
            assert topologies_identical(left, right) == (
                diff_topologies(left, right).identical
            )


@_MEMO_USERS
def test_a_fingerprint_holds_no_arrays(cached_topology):
    """A long path retains one fingerprint per iteration, so it must be small."""
    fingerprint = fingerprint_topology(cached_topology())
    for value in dataclasses.asdict(fingerprint).values():
        assert isinstance(value, (int, float, str))


def test_non_numpy_arrays_are_refused():
    """Host-side only: a non-NumPy input is refused, by one of two enforcers.

    With ``JACCPOT_RUNTIME_TYPECHECK`` off, ``_feed_array``'s guard raises a
    ``TypeError`` saying the facility is host-side. With it on, the package-wide
    jaxtyping hook checks ``_digest``'s own annotation first and raises
    ``jaxtyping.TypeCheckError`` -- a ``TypeError`` subclass -- before the guard
    is reached. Same contract, two enforcers, so the test accepts either
    message. It exercises the hasher directly rather than smuggling a list into
    a ``MutualTopology`` via ``dataclasses.replace``: under the hook the
    dataclass ``__init__`` is instrumented too, and beartype's violation there
    is *not* a ``TypeError``, so that route asserts the wrong thing.
    """
    from jaccpot.mutual.identity import _digest

    either_enforcer = "host-side|Type-check error"
    with pytest.raises(TypeError, match=either_enforcer):
        _digest([1, 2, 3])
    with pytest.raises(TypeError, match=either_enforcer):
        _digest(jnp.asarray([1, 2, 3]))


@_MEMO_USERS
def test_the_switch_counter_counts_consecutive_changes(cached_topology):
    """Switch arithmetic on a known sequence A A B B A."""
    first = fingerprint_topology(cached_topology(seed=0))
    second = fingerprint_topology(cached_topology(seed=1))
    counter = TopologySwitchCounter()

    assert counter.observe(first) is None  # nothing to compare against
    for fingerprint in (first, second, second, first):
        counter.observe(fingerprint)

    assert counter.observations == 5
    assert counter.comparisons == 4
    assert counter.switches == 2  # A->B and B->A
    assert counter.switch_rate == pytest.approx(0.5)
    assert counter.unique_topologies == 2  # the path revisits A


@_MEMO_USERS
def test_the_switch_counter_starts_empty_without_dividing_by_zero(cached_topology):
    """A counter that has seen nothing reports a rate of zero, not a crash."""
    counter = TopologySwitchCounter()
    assert counter.observations == 0
    assert counter.comparisons == 0
    assert counter.switch_rate == 0.0
    assert counter.unique_topologies == 0

    counter.observe(cached_topology())
    assert counter.observations == 1
    assert counter.comparisons == 0
    assert counter.switch_rate == 0.0
    assert counter.switches == 0


@_MEMO_USERS
def test_the_switch_counter_summary_is_json_safe(cached_topology):
    """The summary goes straight into a results file, so no arrays."""
    import json

    counter = TopologySwitchCounter()
    counter.observe(cached_topology(seed=0))
    counter.observe(cached_topology(seed=1))
    summary = counter.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert summary["switches"] == 1
    assert summary["comparisons"] == 1
    assert set(summary["changed_components"]) <= {
        "num_particles",
        "num_nodes",
        "num_internal",
        "max_leaf_size",
        "order",
        "theta",
        "permutation",
        "tree_shape",
        "leaf_partition",
        "node_ranges",
        "far_pairs",
        "near_pairs",
    }


@_MEMO_USERS
def test_interaction_switches_never_exceed_switches(cached_topology):
    """An interaction change is a switch, so the narrower count cannot lead."""
    counter = TopologySwitchCounter()
    other = cached_topology(seed=1)
    topology = cached_topology()
    forward = topology.forward_permutation.copy()
    forward[[0, 1]] = forward[[1, 0]]
    for candidate in (
        topology,
        dataclasses.replace(topology, forward_permutation=forward),
        other,
        topology,
    ):
        counter.observe(candidate)
    assert counter.interaction_switches <= counter.switches
    assert counter.switches == 3
    assert counter.interaction_switches == 2  # the permutation swap is not one
