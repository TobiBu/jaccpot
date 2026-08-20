"""Pin the large-N near-field array shapes that nothing else observes.

``runtime/kernels/_evaluate.py::_evaluate_tree_compiled_impl`` annotates these
arrays, and its ``@jaxtyped`` decorator enforces the annotations on every trace.
But the arrays are filled only from ``_large_n_pipeline.py``, and that lane
refuses to dispatch off a GPU -- ``can_use_large_n_prepare_path`` gates on
``jax.default_backend()``. So on CPU they arrive as zero-sized sentinels, every
single time, and the annotations covering them were unverifiable: audit F27
records the lane at 0% CPU coverage.

This test is the missing observation. It stubs the backend probe the way the
sibling large-N tests do -- everything past that gate is backend-independent, so
the prepared state is real rather than a double -- and asserts the axes the
annotations name:

* ``nearfield_leaf_particle_indices`` / ``_mask`` are ``(leaves, w)``. Both axes
  are pinned, by varying ``leaf_size`` across cases: a single case would leave
  ``(16, 32)`` unable to distinguish ``w`` from a coincidence.
* the target-block pair share their second axis, and it is the configured block
  size rather than ``w`` -- which is why they cannot reuse ``w``.

What this test deliberately does NOT assert: a non-zero target-block count. The
first axis was 0 in every configuration probed, so ``blocks`` is bound but its
extent is unobserved, and ``precomputed_target_block_leaf_ids`` stays rank-only
for the same reason. Closing that needs the target-block lane actually engaging,
which is GPU work (audit F27 / 2.6).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._large_n_types import LargeNPreparedState
from jaccpot.runtime.fmm import FMMEngine

# Compile-bound: each case is a full large-N prepare.
pytestmark = pytest.mark.slow

ORDER = 2
# (n_particles, leaf_size). leaf_size VARIES on purpose -- it is what separates
# the `w` axis from a coincidence with the leaf count.
CASES = ((512, 32), (512, 16), (1024, 64))


def _prepare(n_particles: int, leaf_size: int) -> LargeNPreparedState:
    """Build a real large-N prepared state on CPU.

    Parameters
    ----------
    n_particles : int
        Particle count.
    leaf_size : int
        Leaf capacity, which becomes the ``w`` axis.

    Returns
    -------
    LargeNPreparedState
        A state from the real large-N path.
    """
    rng = np.random.default_rng(0)
    positions = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(n_particles, 3)), dtype=jnp.float32
    )
    masses = jnp.ones((n_particles,), dtype=jnp.float32)
    engine = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        fixed_order=ORDER,
    )
    return engine.prepare_state(positions, masses, leaf_size=leaf_size, max_order=ORDER)


@pytest.mark.parametrize(("n_particles", "leaf_size"), CASES)
def test_leaf_particle_tables_are_leaves_by_w(
    monkeypatch: pytest.MonkeyPatch, n_particles: int, leaf_size: int
) -> None:
    """``nearfield_leaf_particle_*`` must be ``(leaves, w)``, both axes pinned."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    state = _prepare(n_particles, leaf_size)
    assert isinstance(state, LargeNPreparedState), (
        "the large-N prepare path declined; this test would otherwise assert "
        "a different lane's shapes"
    )

    leaves = int(jnp.asarray(state.neighbor_list.leaf_indices).shape[0])
    for name in ("nearfield_leaf_particle_indices", "nearfield_leaf_particle_mask"):
        arr = getattr(state, name)
        assert arr.shape == (leaves, leaf_size), (
            f"{name} is {arr.shape}, expected (leaves={leaves}, w={leaf_size}); "
            "the `leaves w` annotation in kernels/_evaluate.py depends on this"
        )
    assert state.nearfield_leaf_particle_mask.dtype == jnp.bool_


@pytest.mark.parametrize("block_size", (8, 16, 64))
def test_target_block_second_axis_is_the_block_size(
    monkeypatch: pytest.MonkeyPatch, block_size: int
) -> None:
    """The target-block pair share a second axis, and it is the block size.

    Not ``w``: this is why they cannot reuse that name. Verified by varying the
    env var while holding ``leaf_size`` fixed, so the two cannot be confused.
    """
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", str(block_size))
    state = _prepare(512, 32)
    assert isinstance(state, LargeNPreparedState)

    src = state.nearfield_target_block_source_leaf_ids
    mask = state.nearfield_target_block_valid_mask
    if src is None or mask is None:
        pytest.skip("this configuration does not materialise the target-block pair")

    assert src.shape[1] == block_size, (
        f"target-block second axis is {src.shape[1]}, expected the configured "
        f"block size {block_size}"
    )
    assert src.shape == mask.shape, (
        "the target-block source ids and validity mask must share both axes; "
        f"got {src.shape} and {mask.shape}"
    )


def test_kernel_annotation_matches_the_measured_axes() -> None:
    """The kernel's annotation must name the axes this module measured.

    Why this exists, and it is not belt-and-braces. The arrays above reach
    ``_evaluate_tree_compiled_impl`` as ``(0, 0)`` sentinels on CPU, and a
    zero-sized array satisfies ``"leaves w"`` and ``"w leaves"`` equally -- both
    axes bind to 0. Verified: swapping the annotation to ``"w leaves"`` leaves the
    whole near-field suite green. So the ``@jaxtyped`` decorator, which does
    enforce this annotation once the lane is live, is enforcing something no CPU
    test can currently check.

    Comparing the annotation text against the axes measured above is what makes it
    checkable here. It is coarse -- it pins the spelling, not the semantics -- but
    the spelling is exactly what regressed in the ``leaf_nodes`` case.
    """
    import ast
    from pathlib import Path

    module = (
        Path(__file__).resolve().parents[3]
        / "jaccpot"
        / "runtime"
        / "kernels"
        / "_evaluate.py"
    )
    tree = ast.parse(module.read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_evaluate_tree_compiled_impl"
    )

    # `ast.unparse` normalises string quoting, so compare on a canonical form --
    # otherwise this fails on `"leaves w"` vs `'leaves w'`, which is not a drift.
    def canon(text: str) -> str:
        return text.replace('"', "'").replace(" ", "")

    annotations = {
        arg.arg: canon(ast.unparse(arg.annotation))
        for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        if arg.annotation is not None
    }

    expected = {
        name: canon(want)
        for name, want in {
            # Rank-only ON PURPOSE, even though the live shape measured above is
            # `(leaves, w)`. The sentinel is `(0, 0)`, and `leaves` is already
            # bound by `nearfield_leaf_nodes` in that signature, so naming the
            # first axis asserts the sentinel has `leaves` rows. Tried: 87 tests
            # fail, the characterization goldens among them. This entry exists so
            # nobody "fixes" it back using the measurement above as justification.
            "nearfield_leaf_particle_indices": "Int[Array, '_ _']",
            "nearfield_leaf_particle_mask": "Bool[Array, '_ _']",
            "precomputed_target_block_source_leaf_ids": "Int[Array, 'blocks blocksize']",
            "precomputed_target_block_valid_mask": "Bool[Array, 'blocks blocksize']",
        }.items()
    }
    wrong = {
        name: (annotations.get(name), want)
        for name, want in expected.items()
        if annotations.get(name) != want
    }
    assert not wrong, "\n".join(
        [
            "Kernel annotations disagree with the axes measured in this module.",
            "Either the measurement changed and this test needs updating, or the",
            "annotation drifted and the kernel is now asserting something untrue:",
            *(
                f"- {n}: is {got!r}, measured {want!r}"
                for n, (got, want) in wrong.items()
            ),
        ]
    )
