"""The escaper policy (A2 prototype, OPT-IN): far particles leave the tree.

WHY. A few particles far from the system set the Morton box: the 1e8 Sgr IC's
sampler tails (to 30 Mpc) made a 44 Mpc box, 22 pc level-21 cells and a radix
tree 68 levels deep (past yggdrax's 64), and needed a hand clip. A long run's
escapers grow the box the same way. This module runs such a system WITHOUT a
clip: particles farther than ``r_escape`` from the bound system's centre of mass
leave the tree and move on a separate path, with an exact energy ledger.

THE SPLIT (static shapes). The state keeps its N rows; the runner keeps them in
an internal order with the ``n_b`` tree ("bound") rows first and the escapers
last. ``n_b`` is a TRACED scalar: the tree is built with
``FusedForceProvider.rebuild(num_valid=n_b)`` -- the multi-GPU lane's
capacity-padded cut, which puts the rows past ``n_b`` in no leaf -- on positions
whose escaper rows are mapped onto the first row (so the inferred box is the
bound rows'), with their masses zero. A changed escaper set is a new ``n_b`` and
a row permutation between segments: no recompile.

THE HAMILTONIAN. ``H = K + W_tree + W_c`` with ``W_tree`` the tree rows'
self-gravity (the FMM) and the coupling ``W_c = -G M_b sum_e m_e / |x_e - X_b|``
between every escaper ``e`` and the bound system as a point mass ``M_b`` at its
centre of mass ``X_b``. Its forces, evaluated every step from the state:

* an escaper: ``a_e = -G M_b (x_e - X_b) / |x_e - X_b|^3``;
* every tree row: the FMM's plus the SAME uniform ``a_u = G sum_e m_e (x_e -
  X_b) / |x_e - X_b|^3`` (``-dW_c/dx_i / m_i``, since ``dX_b/dx_i = m_i / M_b``).

The pair forces are equal and opposite (``M_b a_u = -sum_e m_e a_e``), so the
momentum is conserved exactly up to the FMM's own; the leapfrog of ``H``
conserves ``E = K + W_tree + W_c`` like any symplectic step. Neglected: the
escapers' tidal field across the bound system (relative ``(r_b / r_e)^2``) and
the escapers' own mutual gravity (``O(K^2)``; far tails are tracers).

THE LEDGER. A particle crossing ``r_escape`` outward, or ``r_return`` inward,
changes its representation at a segment edge: the energy of the SAME phase-space
state differs between the two by the multipole error of the far field. The
ledger records that jump (``E_after - E_before`` at the edge), so ``E -
sum(jumps)`` is the conserved quantity; the row also carries the kinetic and
coupling energy the escapers hold (what they carried away).

SCOPE OF THE PROTOTYPE. One card; the geometric MAC (``mac_type="dehnen"``; the
``dehnen_error`` f_b carry is not threaded); no external field; its own segment
loop (:class:`EscaperRunner`: one jitted scan per segment, the segment re-run on
a capacity overflow with ``FusedForceProvider.replanned``), not
``ProductionRunner``'s restarts, callbacks or mesh. Production wiring is the
follow-up (``docs/production_2026-11.md``, A2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "EscaperPolicy",
    "EscaperResult",
    "EscaperRunner",
    "escaper_energy",
]


@dataclass(frozen=True)
class EscaperPolicy:
    """Which particles leave the tree, and when they come back.

    Attributes
    ----------
    r_escape : float
        A tree row farther than this from the bound system's centre of mass
        becomes an escaper (checked at segment edges, and at the start).
    r_return : Optional[float]
        An escaper nearer than this re-enters the tree; ``None``: ``r_escape /
        2`` (the hysteresis keeps a particle near ``r_escape`` from switching at
        every edge).
    """

    r_escape: float
    r_return: Optional[float] = None

    def __post_init__(self) -> None:
        if not float(self.r_escape) > 0:
            raise ValueError(f"r_escape must be positive, got {self.r_escape}")
        if self.r_return is not None and not 0 < float(self.r_return) <= float(
            self.r_escape
        ):
            raise ValueError("r_return must be in (0, r_escape]")

    @property
    def return_radius(self) -> float:
        """``r_return``, or ``r_escape / 2``.

        Returns
        -------
        float
            The re-entry radius.
        """
        return float(0.5 * self.r_escape if self.r_return is None else self.r_return)


@dataclass
class EscaperResult:
    """What :meth:`EscaperRunner.run` returns.

    Attributes
    ----------
    state : Any
        ``[N, 2, 3]`` in the CALLER's row order.
    masses : Any
        ``[N]``, the caller's order.
    edges : list
        One dict per segment edge (step 0 first): ``step``, ``n_escapers``,
        ``retries``, the energies of :func:`escaper_energy` (``kinetic``,
        ``w_tree``, ``w_couple``, ``energy``, ``kinetic_escapers``), ``jump``
        (this edge's representation change), ``ledger`` (the running sum of
        jumps), ``energy_conserved`` (``energy - ledger``), ``momentum`` and
        ``energy_ok`` (the potential's rebuild fitted).
    order : Any
        ``[N]``: internal row ``i`` is caller row ``order[i]`` at the end.
    n_bound : int
        Tree rows at the end.
    """

    state: Any
    masses: Any
    edges: list
    order: Any
    n_bound: int


def _coupling(pos: Any, masses: Any, n_b: Any, G: float) -> tuple:
    """The monopole coupling of the escaper rows (past ``n_b``) and the tree rows.

    Parameters
    ----------
    pos : Any
        ``[N, 3]`` positions, internal order.
    masses : Any
        ``[N]`` TRUE masses.
    n_b : Any
        Tree rows (traced).
    G : float
        Gravitational constant.

    Returns
    -------
    tuple
        ``(acc [N, 3], W_c, live [N] bool)``: escapers' ``a_e``, tree rows'
        uniform ``a_u`` (fp64 inside, the positions' dtype out).
    """
    import jax.numpy as jnp

    n = int(pos.shape[0])
    live = jnp.arange(n) < n_b
    p = pos.astype(jnp.float64)
    m = masses.astype(jnp.float64)
    m_live = jnp.where(live, m, 0.0)
    m_b = jnp.sum(m_live)
    x_b = jnp.sum(m_live[:, None] * p, axis=0) / jnp.maximum(m_b, 1e-300)
    d = p - x_b
    r2 = jnp.sum(d * d, axis=1)
    inv_r = jnp.where(live, 0.0, 1.0 / jnp.sqrt(jnp.where(live, 1.0, r2)))
    inv_r3 = inv_r**3
    m_e = jnp.where(live, 0.0, m)
    a_e = -float(G) * m_b * d * inv_r3[:, None]
    a_u = float(G) * jnp.sum((m_e * inv_r3)[:, None] * d, axis=0)
    w_c = -float(G) * m_b * jnp.sum(m_e * inv_r)
    acc = jnp.where(live[:, None], a_u[None, :], a_e)
    return acc.astype(pos.dtype), w_c, live


def _tree_inputs(pos: Any, masses: Any, n_b: Any) -> tuple:
    """Positions and masses the tree is built on: escaper rows onto row 0, mass 0.

    Parameters
    ----------
    pos : Any
        ``[N, 3]``, internal order.
    masses : Any
        ``[N]``.
    n_b : Any
        Tree rows.

    Returns
    -------
    tuple
        ``(positions, masses)`` for ``FusedForceProvider.rebuild(num_valid=n_b)``.
    """
    import jax.numpy as jnp

    live = (jnp.arange(int(pos.shape[0])) < n_b)[:, None]
    return (
        jnp.where(live, pos, pos[0][None, :]),
        jnp.where(live[:, 0], masses, jnp.zeros((), masses.dtype)),
    )


def escaper_energy(state: Any, masses: Any, n_b: int, phi_tree: Any, G: float) -> dict:
    """The ledger's energies of an internal-order state (host floats, fp64).

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]``, internal order.
    masses : Any
        ``[N]`` true masses.
    n_b : int
        Tree rows.
    phi_tree : Any
        ``[N]`` the tree's potential per unit mass at the tree rows (escaper rows
        ignored).
    G : float
        Gravitational constant.

    Returns
    -------
    dict
        ``kinetic`` (all rows), ``kinetic_escapers``, ``w_tree`` (``1/2 sum m
        phi`` over tree rows), ``w_couple`` (``W_c``), ``energy`` (their sum) and
        ``momentum`` (a list).
    """
    s = np.asarray(state, np.float64)
    m = np.asarray(masses, np.float64)
    phi = np.asarray(phi_tree, np.float64)
    live = np.arange(s.shape[0]) < int(n_b)
    v2 = np.sum(s[:, 1] ** 2, axis=1)
    kin = 0.5 * float(np.sum(m * v2))
    kin_e = 0.5 * float(np.sum((m * v2)[~live]))
    w_tree = 0.5 * float(np.sum((m * phi)[live]))
    m_b = float(np.sum(m[live]))
    x_b = np.sum(m[live, None] * s[live, 0], axis=0) / m_b
    r = np.linalg.norm(s[~live, 0] - x_b, axis=1)
    w_c = -float(G) * m_b * float(np.sum(m[~live] / r)) if r.size else 0.0
    return {
        "kinetic": kin,
        "kinetic_escapers": kin_e,
        "w_tree": w_tree,
        "w_couple": w_c,
        "energy": kin + w_tree + w_c,
        "momentum": np.sum(m[:, None] * s[:, 1], axis=0).tolist(),
    }


class EscaperRunner:
    """A leapfrog run with the escaper policy (prototype; see the module).

    Parameters
    ----------
    solver : Any
        ``FastMultipoleMethod`` on the fused lane with cell leaves and the
        geometric MAC.
    policy : EscaperPolicy
        Who leaves the tree.
    leaf_size : int
        Leaf size.
    max_order : int
        Expansion order.
    steps : int
        Steps per segment (one jitted scan).

    Raises
    ------
    NotImplementedError
        Under ``mac_type='dehnen_error'``.
    """

    def __init__(
        self,
        solver: Any,
        policy: EscaperPolicy,
        *,
        leaf_size: int,
        max_order: int,
        steps: int = 10,
    ) -> None:
        from jaccpot.runtime.fused_provider import _engine_of

        self.solver = solver
        self.engine = _engine_of(solver)
        if str(getattr(self.engine, "mac_type", "")) == "dehnen_error":
            raise NotImplementedError(
                "the escaper prototype runs the geometric MAC (mac_type='dehnen'): "
                "the dehnen_error f_b carry is not threaded through it"
            )
        self.policy = policy
        self.leaf_size = int(leaf_size)
        self.max_order = int(max_order)
        self.steps = int(steps)
        self.G = float(self.engine.G)
        self._programs: dict = {}

    # -- eager -------------------------------------------------------------

    def _prepare(self, state: Any, masses: Any, n_b: int) -> Any:
        """The provider at ``state`` over its first ``n_b`` rows (eager).

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]``, internal order.
        masses : Any
            ``[N]``.
        n_b : int
            Tree rows.

        Returns
        -------
        Any
            A ``FusedForceProvider``.
        """
        import jax.numpy as jnp

        from jaccpot.runtime.capacity_plan import eager_live_rows
        from jaccpot.runtime.fused_provider import FusedForceProvider

        pos, m = _tree_inputs(jnp.asarray(state)[:, 0], jnp.asarray(masses), n_b)
        with eager_live_rows(int(n_b)):
            prov, _ = FusedForceProvider.prepare(
                self.solver,
                pos,
                m,
                leaf_size=self.leaf_size,
                max_order=self.max_order,
            )
        return prov

    def _membership(self, state: Any, masses: Any, n_b: int) -> np.ndarray:
        """Which internal rows are escapers after this edge (host).

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]``, internal order.
        masses : Any
            ``[N]``.
        n_b : int
            Tree rows now.

        Returns
        -------
        np.ndarray
            ``[N]`` bool, ``True`` for an escaper.

        Raises
        ------
        RuntimeError
            If no particle would stay in the tree.
        """
        pos = np.asarray(state, np.float64)[:, 0]
        m = np.asarray(masses, np.float64)
        live = np.arange(pos.shape[0]) < int(n_b)
        x_b = np.sum(m[live, None] * pos[live], axis=0) / np.sum(m[live])
        r = np.linalg.norm(pos - x_b, axis=1)
        escaper = ~live.copy()
        escaper[live & (r > float(self.policy.r_escape))] = True
        escaper[(~live) & (r < self.policy.return_radius)] = False
        if escaper.all():
            raise RuntimeError("the escaper policy left no particle in the tree")
        return escaper

    # -- traced ------------------------------------------------------------

    def _force(self, prov: Any) -> Any:
        """``(pos, masses, n_b) -> (acc, ok, needs, phi)`` on ``prov`` (traced body).

        Parameters
        ----------
        prov : Any
            The provider.

        Returns
        -------
        Any
            The body.
        """
        import jax.numpy as jnp

        def force(pos: Any, masses: Any, n_b: Any, with_phi: bool = False) -> tuple:
            tree_pos, tree_m = _tree_inputs(pos, masses, n_b)
            topology = prov.rebuild(tree_pos, tree_m, num_valid=n_b)
            acc_tree = prov.accelerations(topology)
            acc_c, _, live = _coupling(pos, masses, n_b, self.G)
            acc = jnp.where(live[:, None], acc_tree.astype(pos.dtype), 0) + acc_c
            phi = prov.potential(topology) if with_phi else None
            return acc, prov.capacity_ok(topology), topology.needs(), phi

        return force

    def _program(self, prov: Any, kind: str) -> Any:
        """The jitted segment scan (``"scan"``) or edge evaluation (``"edge"``).

        Parameters
        ----------
        prov : Any
            The provider.
        kind : str
            ``"scan"`` or ``"edge"``.

        Returns
        -------
        Any
            The compiled function.
        """
        import jax
        import jax.numpy as jnp

        key = (kind, prov.key(), self.steps)
        program = self._programs.get(key)
        if program is not None:
            return program
        force = self._force(prov)
        steps = self.steps

        if kind == "edge":

            def edge(state: Any, masses: Any, n_b: Any) -> tuple:
                acc, ok, needs, phi = force(state[:, 0], masses, n_b, with_phi=True)
                return acc, ok, needs, phi

            program = jax.jit(edge)
        else:

            def scan(state: Any, acc: Any, masses: Any, n_b: Any, dt: Any) -> tuple:
                def step(carry: tuple, _: Any) -> tuple:
                    s, a, ok, needs = carry
                    v_half = s[:, 1] + 0.5 * dt * a
                    x_new = s[:, 0] + dt * v_half
                    a_new, ok_i, needs_i, _ = force(x_new, masses, n_b)
                    v_new = v_half + 0.5 * dt * a_new
                    s_new = jnp.stack([x_new, v_new], axis=1).astype(s.dtype)
                    needs_new = jnp.where(ok, jnp.maximum(needs, needs_i), needs)
                    return (s_new, a_new, ok & ok_i, needs_new), None

                needs0 = jnp.zeros_like(force(state[:, 0], masses, n_b)[2])
                out, _ = jax.lax.scan(
                    step, (state, acc, jnp.asarray(True), needs0), None, length=steps
                )
                return out

            program = jax.jit(scan)
        self._programs[key] = program
        return program

    # -- the run -----------------------------------------------------------

    def run(
        self, state: Any, masses: Any, *, dt: float, num_steps: int
    ) -> EscaperResult:
        """Run ``num_steps`` steps in segments of ``steps``.

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]``.
        masses : Any
            ``[N]`` (positive).
        dt : float
            Timestep.
        num_steps : int
            Steps (a multiple of ``steps``).

        Returns
        -------
        EscaperResult
            The final state in the caller's order and the ledger rows.

        Raises
        ------
        ValueError
            If ``num_steps`` is not a multiple of ``steps``.
        RuntimeError
            If a segment overflows after two re-plans.
        """
        import jax
        import jax.numpy as jnp

        if int(num_steps) % self.steps:
            raise ValueError("num_steps must be a multiple of steps")
        state = jnp.asarray(state)
        masses = jnp.asarray(masses)
        n = int(state.shape[0])
        order = np.arange(n)
        n_b = n
        prov = None
        acc = None
        ledger = 0.0
        edges: list = []
        #: the rows so far (kept on the runner for a run that raises)
        self.edges = edges
        for index in range(int(num_steps) // self.steps + 1):
            jump = 0.0
            if prov is None:
                # the start: the initial split, no jump
                escaper = self._membership(state, masses, n_b)
                changed = True
            else:
                acc_now, row, ok = self._edge(prov, state, masses, n_b)
                escaper = self._membership(state, masses, n_b)
                changed = bool(np.any(escaper != ~(np.arange(n) < n_b)))
                energy_before = row["energy"]
            if changed:
                perm = np.concatenate(
                    [np.flatnonzero(~escaper), np.flatnonzero(escaper)]
                )
                state, masses, order = state[perm], masses[perm], order[perm]
                n_b = int(np.sum(~escaper))
                if prov is None:
                    prov = self._prepare(state, masses, n_b)
                # (later changes keep the provider: its template is shapes only
                # and n_b is traced, so a new split compiles nothing; a plan the
                # new split outgrows is re-planned by the segment's retry)
                acc_now, row, ok = self._edge(prov, state, masses, n_b)
                if index > 0:
                    # the same phase-space state in its new representation
                    jump = row["energy"] - energy_before
                acc = acc_now
            ledger += jump
            row.update(
                step=index * self.steps,
                n_escapers=n - n_b,
                jump=jump,
                ledger=ledger,
                energy_conserved=row["energy"] - ledger,
                energy_ok=bool(ok),
            )
            edges.append(row)
            if index == int(num_steps) // self.steps:
                break
            retries = 0
            while True:
                scan = self._program(prov, "scan")
                s_out, a_out, ok_out, needs = scan(
                    state, acc, masses, jnp.asarray(n_b), jnp.asarray(dt, state.dtype)
                )
                if bool(jax.device_get(ok_out)):
                    break
                mode = self.engine._segment_retry_mode(
                    np.asarray(jax.device_get(needs)).astype(np.int64), retries, False
                )
                if mode is None or retries >= 2:
                    raise RuntimeError(
                        f"EscaperRunner: segment {index + 1} overflowed its plan "
                        f"after {retries} re-plan(s); needs "
                        f"{np.asarray(jax.device_get(needs)).tolist()}"
                    )
                from jaccpot.runtime.capacity_plan import eager_live_rows

                pos0, m0 = _tree_inputs(state[:, 0], masses, n_b)
                with eager_live_rows(int(n_b)):
                    replanned = prov.replanned(
                        np.asarray(jax.device_get(needs)).astype(np.int64),
                        pos0,
                        m0,
                        walk_only=mode == "walk",
                    )
                if replanned is None:
                    raise RuntimeError(
                        f"EscaperRunner: segment {index + 1} overflowed a named cap "
                        "or nothing could be re-planned; needs "
                        f"{np.asarray(jax.device_get(needs)).tolist()}"
                    )
                prov = replanned[0]
                retries += 1
            edges[-1]["retries_next"] = retries
            state, acc = s_out, a_out
        inverse = np.empty_like(order)
        inverse[order] = np.arange(n)
        return EscaperResult(
            state=state[inverse],
            masses=masses[inverse],
            edges=edges,
            order=order,
            n_bound=n_b,
        )

    def _edge(self, prov: Any, state: Any, masses: Any, n_b: int) -> tuple:
        """The force and the ledger's energies at an edge.

        Parameters
        ----------
        prov : Any
            The provider.
        state : Any
            ``[N, 2, 3]``, internal order.
        masses : Any
            ``[N]``.
        n_b : int
            Tree rows.

        Returns
        -------
        tuple
            ``(acc, row, ok)``.
        """
        import jax
        import jax.numpy as jnp

        acc, ok, _, phi = self._program(prov, "edge")(state, masses, jnp.asarray(n_b))
        row = escaper_energy(
            jax.device_get(state),
            jax.device_get(masses),
            n_b,
            jax.device_get(phi),
            self.G,
        )
        return acc, row, bool(jax.device_get(ok))
