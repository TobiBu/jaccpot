"""The escaper policy (OPT-IN): far particles leave the tree, ``ProductionRunner(escapers=)``.

WHY. A few particles far from the system set the Morton box: the 1e8 Sgr IC's
sampler tails (to 30 Mpc) made a 46 Mpc box, 22 pc level-21 cells and a radix
tree 68 levels deep (past yggdrax's 64, a ``TreeDepthError``), and needed a hand
clip. A long run's escapers grow the box the same way. With
``ProductionRunner(solver, escapers=EscaperPolicy(r_escape=...))`` such a system
runs WITHOUT a clip: particles farther than ``r_escape`` from the bound system's
centre of mass leave the tree and move on the monopole coupling below, with an
exact energy ledger (A2's prototype, wired into the runner in wave 3, ESC).

THE SPLIT (static shapes). The state keeps its N rows; the runner keeps them in
an internal order with the ``n_b`` tree ("bound") rows first and the escapers
last. ``n_b`` is a TRACED scalar (``StepParams.live``): every refresh builds the
tree with ``rebuild(num_valid=n_b)`` -- the multi-GPU lane's capacity-padded cut,
which gives the rows past ``n_b`` the sentinel Morton code and puts them in no
leaf -- in the box of the tree rows alone (:func:`live_bounds`), with the
escapers' masses zero. A changed escaper set is a new ``n_b`` and a row
permutation between segments: the provider and every program are kept, no
recompile. Callbacks, snapshots, restart files and the result see the caller's
row order (the runner gathers at an edge, only when something looks).

THE HAMILTONIAN. ``H = K + W_tree + W_c (+ W_ext)`` with ``W_tree`` the tree
rows' self-gravity (the FMM) and the coupling ``W_c = -G M_b sum_e m_e / |x_e -
X_b|`` between every escaper ``e`` and the bound system as a point mass ``M_b``
at its centre of mass ``X_b``. Its forces, every step (:func:`coupling`):

* an escaper: ``a_e = -G M_b (x_e - X_b) / |x_e - X_b|^3``;
* every tree row: the FMM's plus the SAME uniform ``a_u = G sum_e m_e (x_e -
  X_b) / |x_e - X_b|^3`` (``-dW_c/dx_i / m_i``, since ``dX_b/dx_i = m_i / M_b``).

The pair forces are equal and opposite (``M_b a_u = -sum_e m_e a_e``), so the
momentum is conserved up to the FMM's own error, and the leapfrog of ``H``
conserves ``E`` like any symplectic step. An external field acts on every row,
escapers included. NEGLECTED: the escapers' tidal field across the bound system
(relative ``(r_b / r_e)^2``) and the escapers' mutual gravity (``O(K^2)``; far
tails are tracers -- the user's decision, 2026-11-05).

THE LEDGER. A particle crossing ``r_escape`` outward, or ``r_return`` inward,
changes its representation at a segment edge: the energy of the SAME phase-space
state differs between the two by the multipole error of the far field. The
runner evaluates the energy on both sides of such an edge and records the jump;
``E - sum(jumps)`` is the conserved quantity (``energy_ledgered`` in the
diagnostics rows).

``dehnen_error``: eq (16b)'s ``f_b`` of an escaper row is its coupling
acceleration's magnitude (:func:`coupling`), so a returning escaper starts the
tree's thresholds from its monopole force rather than from nothing.

This module holds the policy and the TRACED pieces the force (``strict_run_v2(...,
live_rows=)``) and the runner's energy share; the loop is ``ProductionRunner``'s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "EscaperPolicy",
    "caller_rows",
    "coupling",
    "coupling_energy",
    "eager_split_inputs",
    "inverse_permutation",
    "live_bounds",
    "membership",
    "resplit",
    "split_permutation",
    "split_rebuild",
    "split_state",
]


@dataclass(frozen=True)
class EscaperPolicy:
    """Which particles leave the tree, and when they come back.

    Attributes
    ----------
    r_escape : float
        A tree row farther than this from the bound system's centre of mass
        becomes an escaper (checked at the run's start and at segment edges).
    r_return : Optional[float]
        An escaper nearer than this re-enters the tree; ``None``: ``r_escape /
        2`` (the hysteresis keeps a particle near ``r_escape`` from switching at
        every edge).
    every : int
        Segment edges between membership checks (``1``: every edge). A changed
        split costs two energy evaluations at that edge (the ledger's jump).
    """

    r_escape: float
    r_return: Optional[float] = None
    every: int = 1

    def __post_init__(self) -> None:
        if not float(self.r_escape) > 0:
            raise ValueError(f"r_escape must be positive, got {self.r_escape}")
        if self.r_return is not None and not 0 < float(self.r_return) <= float(
            self.r_escape
        ):
            raise ValueError("r_return must be in (0, r_escape]")
        if int(self.every) < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")

    @property
    def return_radius(self) -> float:
        """``r_return``, or ``r_escape / 2``.

        Returns
        -------
        float
            The re-entry radius.
        """
        return float(0.5 * self.r_escape if self.r_return is None else self.r_return)

    def record(self) -> dict:
        """The policy as JSON (restart files, fingerprints).

        Returns
        -------
        dict
            ``r_escape``, ``r_return`` (resolved) and ``every``.
        """
        return {
            "r_escape": float(self.r_escape),
            "r_return": self.return_radius,
            "every": int(self.every),
        }


def _live(n: int, n_b: Any) -> Any:
    """``[n]`` bool: row ``i`` is a tree row (``i < n_b``).

    Parameters
    ----------
    n : int
        Rows.
    n_b : Any
        Tree rows (traced or not).

    Returns
    -------
    Any
        The mask.
    """
    import jax.numpy as jnp

    return jnp.arange(int(n), dtype=jnp.int32) < jnp.asarray(n_b, jnp.int32)


def _wide(dtype: Any) -> Any:
    """The reduction dtype: fp64 when x64 is on (jaccpot needs it), else ``dtype``.

    Parameters
    ----------
    dtype : Any
        The state's dtype.

    Returns
    -------
    Any
        ``float64`` or ``dtype``.
    """
    import jax
    import jax.numpy as jnp

    return jnp.float64 if jax.config.jax_enable_x64 else dtype


def live_bounds(positions: Any, n_b: Any, *, cubic: bool) -> tuple:
    """The Morton box of the tree rows alone (the escapers' positions folded away).

    The box the solver would infer from the tree rows (``_infer_bounds``: the
    same padding, a cube when ``cubic``), with every escaper row folded onto row
    0 INSIDE the min/max reductions (as ``global_mesh_bounds`` does), so no
    second ``[N, 3]`` array is made. The eager prepare and every traced refresh
    of a split build in this box.

    Parameters
    ----------
    positions : Any
        ``[N, 3]``, internal order (tree rows first).
    n_b : Any
        Tree rows (``>= 1``).
    cubic : bool
        The solver's configured box shape (``TreeConfig.cubic_bounds``).

    Returns
    -------
    tuple
        ``(lower, upper)``.
    """
    import jax.numpy as jnp

    from jaccpot.runtime.kernels._evaluate import _infer_bounds

    pos = jnp.asarray(positions)
    live = _live(pos.shape[0], n_b)
    return _infer_bounds(
        jnp.where(live[:, None], pos, pos[0][None, :]), cubic=bool(cubic)
    )


def split_rebuild(
    prov: Any,
    like: Any,
    positions: Any,
    masses: Any,
    n_b: Any,
    *,
    force_scale: Any = None,
) -> Any:
    """``prov``'s rebuild over the first ``n_b`` rows: their box, escapers mass 0.

    Parameters
    ----------
    prov : Any
        The ``FusedForceProvider``.
    like : Any
        The prepared state the rebuild takes its shapes from (``prov.like()`` or
        the particle carry's materialised template).
    positions : Any
        ``[N, 3]``, internal order.
    masses : Any
        ``[N]`` true masses.
    n_b : Any
        Tree rows (traced).
    force_scale : Any
        ``mac_type='dehnen_error'``: the f_b carry (input order).

    Returns
    -------
    Any
        The ``FusedTopology``.
    """
    import jax.numpy as jnp

    live = _live(positions.shape[0], n_b)
    return prov._rebuild_like(
        like,
        positions,
        jnp.where(live, masses, jnp.zeros((), masses.dtype)),
        force_scale=force_scale,
        bounds=live_bounds(
            positions,
            n_b,
            cubic=bool(getattr(prov.engine, "_tree_cubic_bounds", False)),
        ),
        num_valid=jnp.asarray(n_b, jnp.int32),
    )


def _bound_frame(positions: Any, masses: Any, n_b: Any) -> tuple:
    """``(live, M_b, X_b)``: the tree rows' mass and centre (fp64 reductions).

    Parameters
    ----------
    positions : Any
        ``[N, 3]``.
    masses : Any
        ``[N]``.
    n_b : Any
        Tree rows.

    Returns
    -------
    tuple
        The mask, ``M_b`` and ``X_b`` (the wide dtype).
    """
    import jax.numpy as jnp

    wide = _wide(positions.dtype)
    live = _live(positions.shape[0], n_b)
    m_live = jnp.where(live, masses, jnp.zeros((), masses.dtype))
    m_b = jnp.sum(m_live, dtype=wide)
    x_b = jnp.sum(m_live[:, None] * positions, axis=0, dtype=wide) / jnp.maximum(
        m_b, jnp.asarray(1e-300 if wide == jnp.float64 else 1e-30, wide)
    )
    return live, m_b, x_b


def coupling(positions: Any, masses: Any, n_b: Any, G: Any) -> tuple:
    """The monopole coupling's accelerations and the escapers' force scale.

    Per row in the state's dtype; the sums (``M_b``, ``X_b``, ``a_u``) in fp64,
    so nothing of ``[N, 3]`` is made wider than the state.

    Parameters
    ----------
    positions : Any
        ``[N, 3]``, internal order.
    masses : Any
        ``[N]`` true masses.
    n_b : Any
        Tree rows (traced).
    G : Any
        Gravitational constant.

    Returns
    -------
    tuple
        ``(acc [N, 3], scale [N], live [N])``: the escapers' ``a_e`` and every
        tree row's uniform ``a_u``; ``|a_e|`` on the escaper rows (their f_b),
        zero on the tree rows; the mask.
    """
    import jax.numpy as jnp

    dtype = positions.dtype
    wide = _wide(dtype)
    live, m_b, x_b = _bound_frame(positions, masses, n_b)
    d = positions - x_b.astype(dtype)[None, :]
    r2 = jnp.sum(d * d, axis=1)
    # tree rows: no 1/r (a row at X_b would be inf, and inf * 0 is nan)
    inv_r = jnp.where(
        live, jnp.zeros((), dtype), 1.0 / jnp.sqrt(jnp.where(live, 1, r2))
    )
    inv_r3 = inv_r * inv_r * inv_r
    g = jnp.asarray(G, dtype)
    a_e = (-g * m_b.astype(dtype)) * d * inv_r3[:, None]
    m_e = jnp.where(live, jnp.zeros((), masses.dtype), masses)
    a_u = jnp.asarray(G, wide) * jnp.sum(
        (m_e * inv_r3)[:, None] * d, axis=0, dtype=wide
    )
    acc = jnp.where(live[:, None], a_u.astype(dtype)[None, :], a_e)
    scale = jnp.where(live, jnp.zeros((), dtype), jnp.sqrt(jnp.sum(a_e * a_e, axis=1)))
    return acc, scale, live


def coupling_energy(state: Any, masses: Any, n_b: Any, G: Any) -> dict:
    """The split's energies (traced; fp64 sums): ``W_c`` and the escapers' ``K``.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]``, internal order.
    masses : Any
        ``[N]``.
    n_b : Any
        Tree rows.
    G : Any
        Gravitational constant.

    Returns
    -------
    dict
        ``couple`` (``W_c``), ``kinetic_escapers``, ``mass_bound`` (``M_b``) and
        ``mass_escapers``.
    """
    import jax.numpy as jnp

    pos = state[:, 0]
    vel = state[:, 1]
    wide = _wide(pos.dtype)
    live, m_b, x_b = _bound_frame(pos, masses, n_b)
    d = pos.astype(wide) - x_b[None, :]
    r = jnp.sqrt(jnp.where(live, 1.0, jnp.sum(d * d, axis=1)))
    m = masses.astype(wide)
    m_e = jnp.where(live, 0.0, m)
    return {
        "couple": -jnp.asarray(G, wide) * m_b * jnp.sum(m_e / r),
        "kinetic_escapers": 0.5 * jnp.sum(m_e * jnp.sum(vel.astype(wide) ** 2, axis=1)),
        "mass_bound": m_b,
        "mass_escapers": jnp.sum(m_e),
    }


# ----------------------------------------------------------------------------
# Membership and the row order (the runner's edges).
# ----------------------------------------------------------------------------

_MEMBERSHIP_FN: dict = {}


def membership(state: Any, masses: Any, n_b: int, policy: EscaperPolicy) -> tuple:
    """Who is an escaper after this edge (device), and how many changed.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]``, internal order.
    masses : Any
        ``[N]``.
    n_b : int
        Tree rows now.
    policy : EscaperPolicy
        The radii.

    Returns
    -------
    tuple
        ``(escaper [N] bool on device, n_escapers int, n_changed int)``.
    """
    import jax
    import jax.numpy as jnp

    key = (policy.r_escape, policy.return_radius)
    fn = _MEMBERSHIP_FN.get(key)
    if fn is None:
        r_out, r_in = float(policy.r_escape), policy.return_radius

        def member(state_in: Any, masses_in: Any, n_b_in: Any) -> tuple:
            pos = state_in[:, 0]
            live, _, x_b = _bound_frame(pos, masses_in, n_b_in)
            d = pos - x_b.astype(pos.dtype)[None, :]
            r2 = jnp.sum(d * d, axis=1)
            escaper = jnp.where(live, r2 > r_out * r_out, ~(r2 < r_in * r_in))
            return (
                escaper,
                jnp.sum(escaper, dtype=jnp.int64),
                jnp.sum(escaper == live, dtype=jnp.int64),
            )

        fn = jax.jit(member)
        _MEMBERSHIP_FN[key] = fn
    escaper, count, changed = fn(state, masses, jnp.asarray(n_b, jnp.int32))
    count, changed = jax.device_get((count, changed))
    return escaper, int(count), int(changed)


def split_permutation(escaper: Any, n_escapers: int) -> Any:
    """``perm``: the tree rows (in their order) first, then the escapers (device).

    Parameters
    ----------
    escaper : Any
        ``[N]`` bool.
    n_escapers : int
        Its count.

    Returns
    -------
    Any
        ``int32[N]``; internal row ``i`` of the new order is old row ``perm[i]``.
    """
    import jax.numpy as jnp

    n = int(escaper.shape[0])
    n_b = n - int(n_escapers)
    keep = jnp.nonzero(~escaper, size=n_b)[0].astype(jnp.int32)
    out = jnp.nonzero(escaper, size=int(n_escapers))[0].astype(jnp.int32)
    return jnp.concatenate([keep, out])


def inverse_permutation(order: Any) -> np.ndarray:
    """The inverse of ``order`` (host): ``caller = internal[inverse]``.

    Parameters
    ----------
    order : Any
        ``int32[N]`` (host): internal row ``i`` is caller row ``order[i]``.

    Returns
    -------
    np.ndarray
        ``int32[N]``.
    """
    order = np.asarray(order)
    inverse = np.empty(order.shape, np.int32)
    inverse[order] = np.arange(order.shape[0], dtype=np.int32)
    return inverse


_EAGER_INPUTS_FN: dict = {}


def eager_split_inputs(positions: Any, masses: Any, n_b: int, *, cubic: bool) -> tuple:
    """The eager prepare's (and a re-plan's) inputs for a split, in one small program.

    Jitted, so the fold of the escaper rows stays inside the reductions (eagerly
    it would be a second ``[N, 3]`` array next to the prepare).

    Parameters
    ----------
    positions : Any
        ``[N, 3]``, internal order.
    masses : Any
        ``[N]`` true masses.
    n_b : int
        Tree rows.
    cubic : bool
        The solver's box shape.

    Returns
    -------
    tuple
        ``(tree masses [N] -- the escapers' zero --, (lower, upper))``.
    """
    import jax
    import jax.numpy as jnp

    fn = _EAGER_INPUTS_FN.get(bool(cubic))
    if fn is None:

        def inputs(pos: Any, m: Any, n_b_in: Any) -> tuple:
            live = _live(pos.shape[0], n_b_in)
            return (
                jnp.where(live, m, jnp.zeros((), m.dtype)),
                live_bounds(pos, n_b_in, cubic=bool(cubic)),
            )

        fn = jax.jit(inputs)
        _EAGER_INPUTS_FN[bool(cubic)] = fn
    return fn(positions, masses, jnp.asarray(int(n_b), jnp.int32))


def caller_rows(order: Any, *arrays: Any) -> tuple:
    """Internal-order arrays in the caller's row order (``None`` stays ``None``).

    Parameters
    ----------
    order : Any
        ``int32[N]`` on the host: internal row ``i`` is caller row ``order[i]``.
    *arrays : Any
        ``[N, ...]`` arrays in the internal order.

    Returns
    -------
    tuple
        The arrays gathered by :func:`inverse_permutation`.
    """
    import jax.numpy as jnp

    inverse = jnp.asarray(inverse_permutation(order))
    return tuple(None if a is None else a[inverse] for a in arrays)


def split_state(state: Any, masses: Any, split: dict, policy: EscaperPolicy) -> tuple:
    """The run's start: the escapers of the initial state, rows re-ordered.

    The membership from a state with every row in the tree uses the centre of
    mass of EVERY particle; it is taken again from the new tree rows' centre
    until nothing changes (at most four passes).

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]`` in the caller's order.
    masses : Any
        ``[N]``.
    split : dict
        The runner's split (``order``, ``n_bound``), updated in place.
    policy : EscaperPolicy
        The radii.

    Returns
    -------
    tuple
        ``(state, masses, moved)``: internal order; ``moved`` when the rows were
        permuted (a copy, the runner's own).

    Raises
    ------
    RuntimeError
        If no particle would stay in the tree.
    """
    n = int(state.shape[0])
    moved = False
    for _ in range(4):
        escaper, count, changed = membership(state, masses, split["n_bound"], policy)
        if changed == 0:
            break
        if count >= n:
            raise RuntimeError(
                f"the escaper policy (r_escape {policy.r_escape}) leaves no "
                "particle in the tree"
            )
        perm = split_permutation(escaper, count)
        del escaper
        state, masses = state[perm], masses[perm]
        split["order"] = _composed(split["order"], perm)
        split["n_bound"] = n - count
        split["transfers"] = int(split.get("transfers", 0)) + int(changed)
        moved = True
    return state, masses, moved


def _composed(order: Optional[np.ndarray], perm: Any) -> np.ndarray:
    """The run's order after ``perm`` (on the host: no ``[N]`` array on the device
    through the segments).

    Parameters
    ----------
    order : Optional[np.ndarray]
        ``int32[N]`` or ``None`` (the identity).
    perm : Any
        ``int32[N]`` on the device.

    Returns
    -------
    np.ndarray
        ``int32[N]``.
    """
    import jax

    perm_h = np.asarray(jax.device_get(perm), np.int32)
    return perm_h if order is None else np.asarray(order)[perm_h]


def _free(*arrays: Any) -> None:
    """Delete device buffers now (each once; ``None`` and host arrays skipped).

    Parameters
    ----------
    *arrays : Any
        Arrays the runner owns and no longer reads.
    """
    import jax

    for value in arrays:
        if isinstance(value, jax.Array) and not value.is_deleted():
            value.delete()


def _permuted_handle(handle: Any, perm: Any) -> Any:
    """The handle after a re-split: its f_b re-ordered, no start force.

    The self-gravity (and total) belong to the old split, so they are dropped:
    the next segment evaluates its start force from the new one
    (``strict_run_v2``: a handle without ``self_acceleration``).

    Parameters
    ----------
    handle : Any
        The ``StrictParticleCarry``.
    perm : Any
        ``int32[N]``.

    Returns
    -------
    Any
        The new handle.
    """
    import dataclasses

    return dataclasses.replace(
        handle,
        self_acceleration=None,
        acceleration=None,
        external_key=None,
        force_scale=None if handle.force_scale is None else handle.force_scale[perm],
    )


def _total_potential(potential: dict) -> Optional[float]:
    """``W_self + W_ext + W_c`` of a fitted evaluation, else ``None``.

    Parameters
    ----------
    potential : dict
        ``ProductionRunner.potential_energy``.

    Returns
    -------
    Optional[float]
        The potential energy.
    """
    if not potential["ok"]:
        return None
    return (
        float(potential["self"])
        + float(potential["external"])
        + float(potential.get("couple", 0.0))
    )


def resplit(
    runner: Any,
    state: Any,
    masses: Any,
    handle: Any,
    split: dict,
    policy: EscaperPolicy,
    *,
    energy: bool,
    external_params: Any = None,
) -> tuple:
    """A segment edge's membership check; on a change, the rows re-ordered and the jump.

    The jump is the potential energy of the SAME state in its new representation
    minus its old one (the kinetic energy is the same): both evaluated with
    ``runner.potential_energy`` on the handle's plan, the escapers' ``f_b``
    their coupling's. ``split["ledger"]`` becomes ``None`` once a jump is
    unknown (an energy that did not fit, or the energy off).

    Parameters
    ----------
    runner : Any
        The ``ProductionRunner``.
    state : Any
        ``[N, 2, 3]``, internal order.
    masses : Any
        ``[N]``.
    handle : Any
        The last segment's ``StrictParticleCarry``.
    split : dict
        The run's split, updated in place.
    policy : EscaperPolicy
        The radii.
    energy : bool
        Evaluate the jump (the run's energy diagnostic is on).
    external_params : Any
        The external potential's parameters.

    Returns
    -------
    tuple
        ``(state, masses, handle, edge)``; ``edge`` holds ``transfers`` and, after
        a change, ``jump`` and ``potential`` (the new split's evaluation, which
        the edge's diagnostics row reuses).

    Raises
    ------
    RuntimeError
        If no particle would stay in the tree.
    """
    n = int(state.shape[0])
    n_b = int(split["n_bound"])
    escaper, count, changed = membership(state, masses, n_b, policy)
    if changed == 0:
        return state, masses, handle, {"transfers": 0}
    if count >= n:
        raise RuntimeError(
            f"the escaper policy (r_escape {policy.r_escape}) leaves no particle "
            "in the tree"
        )

    def live_kw(rows: int) -> dict:
        return {} if rows == n else {"live_rows": rows}

    before = None
    if energy:
        before = _total_potential(
            runner.potential_energy(
                state,
                masses,
                handle,
                external_params=external_params,
                **live_kw(n_b),
            )
        )
    perm = split_permutation(escaper, count)
    del escaper
    old_state, old_handle = state, handle
    state, masses = state[perm], masses[perm]
    split["order"] = _composed(split["order"], perm)
    handle = _permuted_handle(handle, perm)
    del perm
    # the old split's arrays are the runner's own (a segment's outputs): freed
    # now, not when the caller's references go (they would sit next to the new
    # ones through the second energy evaluation, ~40 B per particle). The masses
    # may be the caller's array: kept.
    _free(
        old_state,
        old_handle.self_acceleration,
        old_handle.acceleration,
        old_handle.force_scale,
    )
    del old_state, old_handle
    split["n_bound"] = n - count
    split["transfers"] = int(split.get("transfers", 0)) + int(changed)
    edge: dict = {"transfers": int(changed), "jump": None}
    if energy:
        after = runner.potential_energy(
            state,
            masses,
            handle,
            external_params=external_params,
            **live_kw(n - count),
        )
        edge["potential"] = after
        after_w = _total_potential(after)
        if before is not None and after_w is not None:
            edge["jump"] = after_w - before
    if edge["jump"] is None or split["ledger"] is None:
        split["ledger"] = None
    else:
        split["ledger"] = float(split["ledger"]) + float(edge["jump"])
    return state, masses, handle, edge


def _write_split_arrays(current: dict, handle: Any, split: dict) -> dict:
    """A restart file's arrays under the escaper split: the CALLER's rows.

    Parameters
    ----------
    current : dict
        The runner's position at its last edge (``state``, ``masses``).
    handle : Any
        The ``StrictParticleCarry``.
    split : dict
        The split (``order``).

    Returns
    -------
    dict
        ``state``, ``masses``, ``self_acceleration`` (absent after a changed
        split: the continuation evaluates it), ``force_scale`` and
        ``acceleration`` where the run has them, all in the caller's row order,
        and ``escaper_order`` (``int32[N]``) when the order is not the identity.
    """
    import jax

    total = handle.acceleration
    names = ["state", "masses", "self_acceleration", "force_scale", "acceleration"]
    values = [
        current["state"],
        current["masses"],
        handle.self_acceleration,
        handle.force_scale,
        None if total is None or total is handle.self_acceleration else total,
    ]
    order = split["order"]
    if order is not None:
        values = list(caller_rows(order, *values))
    arrays = {
        name: np.asarray(jax.device_get(value))
        for name, value in zip(names, values)
        if value is not None
    }
    if order is not None:
        arrays["escaper_order"] = np.asarray(order, np.int32)
    return arrays
