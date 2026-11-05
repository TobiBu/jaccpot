"""The forward-mode seams (``ad_mode="forward"``), one kernel at a time, on real operands.

Every production seam of the large-N gradient lane has a ``custom_jvp`` sibling in
:mod:`jaccpot.runtime._forward_seams` (P2M, M2M, L2L, L2P, the M2L CSR lanes and the
direct CSR near field). Their operands are recorded from the S9 G2 system (N = 512,
four blobs, total mass ~1, order 2, leaf 16, theta 0.6, float64; every Pallas kernel
in interpret mode on CPU) as ``FusedForceProvider.differentiable_step`` calls them in
forward mode, and per seam:

* the sibling's primal, alone and as ``jax.jvp``'s primal, is the ``custom_vjp``
  seam's forward BITWISE (the same production kernel);
* the jnp twin behind the reference tangent computes the production kernel's
  primal to round-off;
* the tangent is the reverse kernel's transpose: ``<w, J d> == <J^T w, d>`` against
  ``jax.vjp`` of the ``custom_vjp`` seam (its hand-written Pallas reverse) to
  round-off, each differentiable input separately -- for every tangent-kernel
  choice that exists (``"jnp"``: the reference or the kernel track's ``_jnp``;
  ``"auto"``: the Pallas tangent op once it lands);
* ``jax.vmap`` over three tangents (``jax.jacfwd``'s batching) equals three single
  jvps to round-off.

The far field is load-bearing: the recorded M2L list has live pairs.
"""

from __future__ import annotations

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 512
LEAF, ORDER, THETA, SOFTENING = 16, 2, 0.6, 1e-2

_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
_ENV_UNSET = (
    "JACCPOT_GRAD_AD_MODE",
    "JACCPOT_GRAD_FORWARD_TANGENT_KERNELS",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
)

#: seam -> (sibling name in _forward_seams, custom_vjp module, its name, the
#: differentiable positional arguments; the near seam's are keywords too)
_SEAMS = {
    "p2m": (
        "p2m_real_leaves_forward",
        "jaccpot.pallas.p2m_real_leaf",
        "p2m_real_leaves_pallas_cvjp",
        (0, 1, 2),
    ),
    "m2m": (
        "m2m_real_levels_forward",
        "jaccpot.pallas.cascade_real_level",
        "m2m_real_levels_pallas_cvjp",
        (0, 1),
    ),
    "l2l": (
        "l2l_real_levels_forward",
        "jaccpot.pallas.cascade_real_level",
        "l2l_real_levels_pallas_cvjp",
        (0, 1),
    ),
    "l2p": (
        "l2p_real_particles_forward",
        "jaccpot.pallas.l2p_real",
        "l2p_real_particles_pallas_cvjp",
        (0, 1, 2),
    ),
    "m2l": (
        "m2l_real_csr_lanes_forward",
        "jaccpot.pallas.m2l_real_csr_lanes",
        "m2l_real_csr_lanes_pallas_cvjp",
        (0, 1),
    ),
    "near": (
        "nearfield_leafpair_csr_sorted_direct_forward",
        "jaccpot.pallas.nearfield_leafpair_csr_direct_vjp",
        "nearfield_leafpair_csr_sorted_direct_pallas_cvjp",
        (0, 1, "softening_sq", "G"),
    ),
}

_RECORDED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and the large-N profile's backend gate open.

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    for key in _ENV_UNSET:
        patcher.delenv(key, raising=False)
    for key, value in _ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()
    _RECORDED.clear()
    jax.clear_caches()


def _system() -> tuple[Any, Any]:
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    rng = np.random.default_rng(1)
    rng.normal(size=(N, 3))  # the velocities of the G2 system
    masses = (np.abs(rng.normal(size=N)) + 0.5) / N
    return jnp.asarray(pos, jnp.float64), jnp.asarray(masses, jnp.float64)


def _provider() -> Any:
    from jaccpot import FastMultipoleMethod, FusedForceProvider
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )
    x, m = _system()
    prov, _ = FusedForceProvider.prepare(
        solver, x, m, leaf_size=LEAF, max_order=ORDER, theta=THETA, differentiable=True
    )
    return prov, x, m


def _recorded() -> dict:
    """Every seam's operands as the forward-mode step passes them (once).

    Returns
    -------
    dict
        seam -> ``(args, kwargs)`` with concrete arrays.
    """
    if _RECORDED:
        return _RECORDED
    from jaccpot._jax_compat import Tracer
    from jaccpot.runtime import _forward_seams as fs
    from jaccpot.runtime.grad_options import ad_mode_override

    store: dict = {}
    originals = {seam: getattr(fs, spec[0]) for seam, spec in _SEAMS.items()}

    def recorder(seam: str) -> Any:
        fn = originals[seam]

        def call(*args: Any, **kwargs: Any) -> Any:
            flat, tree = jax.tree_util.tree_flatten((args, kwargs))
            is_array = [isinstance(a, (jax.Array, Tracer)) for a in flat]

            def keep(*values: Any) -> None:
                it = iter(np.asarray(v) for v in values)
                leaves = [
                    jnp.asarray(next(it)) if a else leaf
                    for leaf, a in zip(flat, is_array)
                ]
                store[seam] = jax.tree_util.tree_unflatten(tree, leaves)

            jax.debug.callback(keep, *[a for a, i in zip(flat, is_array) if i])
            return fn(*args, **kwargs)

        return call

    prov, x, m = _provider()
    try:
        for seam, spec in _SEAMS.items():
            setattr(fs, spec[0], recorder(seam))
        with ad_mode_override("forward"):
            acc = jax.jit(lambda y: prov.differentiable_step(y, m)[0])(x)
            jax.block_until_ready(acc)
    finally:
        for seam, spec in _SEAMS.items():
            setattr(fs, spec[0], originals[seam])
    assert sorted(store) == sorted(_SEAMS)
    _RECORDED.update(store)
    return _RECORDED


def _functions(seam: str) -> tuple:
    """``(sibling, cvjp, primals)``: both as functions of the differentiable inputs."""
    import importlib

    from jaccpot.runtime import _forward_seams as fs

    sibling_name, module, cvjp_name, diff = _SEAMS[seam]
    sibling = getattr(fs, sibling_name)
    cvjp = getattr(importlib.import_module(module), cvjp_name)
    args, kwargs = _recorded()[seam]

    def bind(fn: Any) -> Any:
        def call(*values: Any) -> Any:
            a, kw = list(args), dict(kwargs)
            for slot, value in zip(diff, values):
                if isinstance(slot, str):
                    kw[slot] = value
                else:
                    a[slot] = value
            out = fn(*a, **kw)
            return out[0] if seam == "near" else out

        return call

    primals = tuple(
        jnp.asarray(kwargs[slot] if isinstance(slot, str) else args[slot])
        for slot in diff
    )
    return bind(sibling), bind(cvjp), primals


def _twin(seam: str) -> Any:
    """The jnp twin of the seam's primal, as a function of the differentiable inputs."""
    from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import _DirectStatic
    from jaccpot.runtime._forward_seams import JNP_TWINS

    args, kwargs = _recorded()[seam]
    if seam == "near":
        static = _DirectStatic(
            leaf_width=kwargs["leaf_width"],
            chunk=kwargs["chunk"],
            num_warps=None,
            num_stages=1,
            target_subtile=None,
            interpret=True,
            with_force_scale=False,
            row_limit=None,
            source_tile=None,
            source_flags=None,
            softening_kernel=kwargs["softening_kernel"],
            tile_skip=None,
        )
        return lambda x, m, e, g: JNP_TWINS["near"](
            x, m, *args[2:7], e, g, static=static
        )
    names = {
        "p2m": ("order", "num_internal", "total_nodes", "leaf_width"),
        "m2m": ("order", "num_internal", "num_levels", "level_batch_width"),
        "l2l": ("order", "num_levels", "level_batch_width"),
        "l2p": ("order",),
        "m2l": ("order",),
    }[seam]
    arrays = {"p2m": 4, "m2m": 7, "l2l": 7, "l2p": 6, "m2l": 5}[seam]
    static = dict(zip(names, args[arrays:]))
    if seam == "m2l":
        static.update(n_targets=args[10], presorted=args[11], has_active_count=True)
    diff = _SEAMS[seam][3]

    def twin(*values: Any) -> Any:
        a = list(args[:arrays])
        for slot, value in zip(diff, values):
            a[slot] = value
        return JNP_TWINS[seam](*a, **static)

    return twin


def _random_like(values: Any, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(size=np.shape(v)), jnp.asarray(v).dtype) for v in values
    )


def _vdot(a: Any, b: Any) -> float:
    return float(
        sum(
            jnp.vdot(x, y)
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )
    )


def test_the_far_field_has_live_pairs():
    args, _ = _recorded()["m2l"]
    assert int(args[4]) > 0, "the recorded M2L list has no live pair"


@pytest.mark.parametrize("seam", sorted(_SEAMS))
def test_the_seam_primal_is_the_production_kernel(seam):
    """The sibling is the custom_vjp seam's forward bitwise, alone and as the jvp's
    primal; the reference's jnp twin agrees with it to round-off."""
    from jaccpot.runtime.grad_options import ad_mode_override

    sibling, cvjp, primals = _functions(seam)
    want = np.asarray(jax.jit(cvjp)(*primals))
    tangents = _random_like(primals, 1)
    with ad_mode_override("forward", "jnp"):
        got = np.asarray(jax.jit(sibling)(*primals))
        primal_out, _ = jax.jit(lambda p, t: jax.jvp(sibling, p, t), static_argnums=())(
            primals, tangents
        )
        twin = np.asarray(jax.jit(_twin(seam))(*primals))
    assert np.array_equal(got, want)
    assert np.array_equal(np.asarray(primal_out), want)
    scale = float(np.max(np.abs(want)))
    assert scale > 0
    assert float(np.max(np.abs(twin - want))) <= 1e-13 * scale


def _tangent_choices(seam: str) -> list:
    """``"jnp"`` always; ``"auto"`` too when it resolves to another op (Pallas)."""
    from jaccpot.runtime._forward_seams import resolve_tangent

    choices = ["jnp"]
    if resolve_tangent(seam, "auto")[1] != resolve_tangent(seam, "jnp")[1]:
        choices.append("auto")
    return choices


@pytest.mark.parametrize("seam", sorted(_SEAMS))
def test_the_seam_tangent_is_the_reverse_kernels_transpose(seam):
    """``<w, J d> == <J^T w, d>`` per input, ``J^T`` the hand-written Pallas
    reverse; and three tangents through ``vmap`` equal three single jvps."""
    from jaccpot.runtime.grad_options import ad_mode_override

    sibling, cvjp, primals = _functions(seam)
    out, vjp = jax.vjp(cvjp, *primals)
    w = _random_like([out], 2)[0]
    cotangents = jax.jit(vjp)(w)
    directions = [_random_like(primals, 10 + k) for k in range(3)]
    for choice in _tangent_choices(seam):
        with ad_mode_override("forward", choice):
            jvp = jax.jit(lambda p, t: jax.jvp(sibling, p, t)[1])
            for i in range(len(primals)):
                d = tuple(
                    t if j == i else jnp.zeros_like(t)
                    for j, t in enumerate(directions[0])
                )
                lhs = _vdot(w, jvp(primals, d))
                rhs = _vdot(cotangents[i], d[i])
                assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs)), (
                    f"{seam} input {i} ({choice}): <w, J d> = {lhs!r}, "
                    f"<J^T w, d> = {rhs!r}"
                )
            stacked = tuple(
                jnp.stack([d[i] for d in directions]) for i in range(len(primals))
            )
            batched = np.asarray(
                jax.jit(jax.vmap(lambda *t: jax.jvp(sibling, primals, t)[1]))(*stacked)
            )
            for k, d in enumerate(directions):
                single = np.asarray(jvp(primals, d))
                scale = max(float(np.max(np.abs(single))), 1e-300)
                assert float(np.max(np.abs(batched[k] - single))) <= 1e-13 * scale


def test_the_tangent_resolution_order(monkeypatch):
    """``"jnp"`` takes the reference here; ``"auto"`` the Pallas op, else the
    kernel track's ``<op>_jnp``, else the reference; ``"pallas"`` refuses a seam
    without a Pallas op."""
    import importlib

    from jaccpot.runtime import _forward_seams as fs

    for seam, (module_name, op_name) in fs.SEAM_TANGENT_OPS.items():
        module = importlib.import_module(module_name)
        assert fs.resolve_tangent(seam, "jnp")[1] == "reference"
        if hasattr(module, op_name):
            assert fs.resolve_tangent(seam, "pallas") == (
                getattr(module, op_name),
                "pallas",
            )
            assert fs.resolve_tangent(seam, "auto")[1] == "pallas"
        else:
            with pytest.raises(NotImplementedError, match=op_name):
                fs.resolve_tangent(seam, "pallas")
            want = "jnp" if hasattr(module, op_name + "_jnp") else "reference"
            assert fs.resolve_tangent(seam, "auto")[1] == want


def test_the_solvers_step_fn_honours_grad_config_ad_mode():
    """``differentiable_step_fn(state, grad_config=GradConfig(ad_mode="forward"))``
    (the S8 seam on a ``LargeNPreparedState``): ``jax.jvp`` runs, its primal is the
    production evaluation bitwise (the reverse step function's, with traced ``G``
    and ``eps2``), and its tangent in ``G`` and the positions is the reverse step
    function's gradient (the adjoint identity, round-off)."""
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        ForceParams,
        GradConfig,
        TreeConfig,
    )

    x, m = _system()
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        G=1.0,
        softening=SOFTENING,
        use_pallas=True,
        working_dtype=jnp.float64,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity="auto",
            ),
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
        ),
    )
    state = fmm.prepare_state(x, m, leaf_size=LEAF, max_order=ORDER)
    forward = fmm.differentiable_step_fn(
        state, grad_config=GradConfig(ad_mode="forward")
    )
    reverse = fmm.differentiable_step_fn(state)
    w = _random_like([x], 21)[0]
    dx = _random_like([x], 22)[0]
    g0, e0 = jnp.asarray(1.0), jnp.asarray(SOFTENING**2)

    def acc(step: Any, xx: Any, g: Any) -> Any:
        return step(xx, m, ForceParams(G=g, eps2=e0))

    production = np.asarray(fmm.evaluate_prepared_state(state))
    plain, _ = jax.jvp(lambda xx: forward(xx, m), (x,), (dx,))
    assert np.array_equal(np.asarray(plain), production)
    primal, tangent = jax.jvp(
        lambda xx, g: acc(forward, xx, g), (x, g0), (dx, jnp.asarray(0.5))
    )
    assert np.array_equal(np.asarray(primal), np.asarray(acc(reverse, x, g0)))
    gx, gg = jax.grad(lambda xx, g: jnp.sum(w * acc(reverse, xx, g)), argnums=(0, 1))(
        x, g0
    )
    lhs = _vdot(w, tangent)
    rhs = _vdot(gx, dx) + 0.5 * float(gg)
    assert abs(lhs - rhs) <= 1e-12 * abs(rhs), (lhs, rhs)
    jax.clear_caches()
