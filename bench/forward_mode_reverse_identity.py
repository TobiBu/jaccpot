"""The reverse programs are untouched by forward mode: lowered program text, A vs B.

Forward mode (``ad_mode="forward"``) is a second dispatch beside the reverse one,
and its rule is that the reverse (and default) programs stay byte-identical. This
script lowers -- nothing is compiled or run -- the gradient lane's programs on the
S9 G2 system (N = 512, four blobs, total mass ~1, order 2, leaf 16, theta 0.6,
float64, every Pallas kernel and the GPU's walk in interpret mode on CPU) and
saves ``lowered.as_text(debug_info=False)`` with its sha256:

* ``step_forward`` / ``step_grad`` -- ``jax.jit`` of
  ``FusedForceProvider.differentiable_step`` and ``jax.grad`` of a weighted sum of
  its accelerations in the positions, the masses, ``G`` and ``eps2``;
* ``rollout_forward`` / ``rollout_grad`` -- the 3-step
  ``differentiable_rollout(checkpoint="step")`` with a traced field constant, and
  ``jax.value_and_grad`` of its loss in every input;
* ``step_grad_dehnen_error`` -- the step's gradient under
  ``mac_type='dehnen_error'`` (seeded f_b).

Usage (each ``record`` in its own process, with the jaccpot to measure first on
the path)::

    python bench/forward_mode_reverse_identity.py record --tag base --root DIR
    python bench/forward_mode_reverse_identity.py record --tag mine --root DIR
    python bench/forward_mode_reverse_identity.py compare base mine --root DIR

``compare`` exits 1 on any difference (a program more or fewer, one character).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

N = 512
LEAF, ORDER, THETA, SOFTENING = 16, 2, 0.6, 1e-2
DT, STEPS, K_EXT = 1e-3, 3, 0.37

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


def _programs() -> dict[str, Callable[[], Any]]:
    """Name -> a thunk that lowers that program (the environment set first)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import FastMultipoleMethod, ForceParams, FusedForceProvider
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def solver(mac_type: str) -> Any:
        criterion: dict[str, Any] = (
            dict(
                adaptive_eps=1e-5,
                adaptive_error_model="dehnen_paper",
                mac_force_scale_mode="paper_fb",
            )
            if mac_type == "dehnen_error"
            else {}
        )
        return FastMultipoleMethod(
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
                mac_type=mac_type,
            ),
            fixed_order=ORDER,
            **criterion,
        )

    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    rng = np.random.default_rng(1)
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = (np.abs(rng.normal(size=N)) + 0.5) / N
    weight = rng.normal(size=(N, 3))
    x, v, m, w = (jnp.asarray(a, jnp.float64) for a in (pos, vel, masses, weight))
    lanes: dict = {}

    def lane(mac_type: str) -> tuple:
        if mac_type not in lanes:
            s = solver(mac_type)
            prov, top = FusedForceProvider.prepare(
                s,
                x,
                m,
                leaf_size=LEAF,
                max_order=ORDER,
                theta=THETA,
                differentiable=True,
            )
            aux0 = None
            if prov.scale_carry:
                fb = jnp.asarray(s._impl._last_force_scale_particles, jnp.float64)
                perm = jnp.asarray(top.prepared.tree.particle_indices)
                aux0 = (jnp.zeros_like(fb).at[perm].set(fb), None, None)
            lanes[mac_type] = (prov, aux0)
        return lanes[mac_type]

    g0, e0, k0 = jnp.asarray(1.0), jnp.asarray(SOFTENING**2), jnp.asarray(K_EXT)

    def step_forward() -> Any:
        prov, _ = lane("dehnen")
        return jax.jit(prov.differentiable_step).lower(x, m)

    def step_loss(prov: Any, aux0: Any, params: bool) -> Callable[..., Any]:
        def loss(xx, mm, g, e):
            p = ForceParams(G=g, eps2=e) if params else None
            return jnp.sum(w * prov.differentiable_step(xx, mm, p, aux0)[0])

        return loss

    def step_grad() -> Any:
        prov, _ = lane("dehnen")
        f = jax.grad(step_loss(prov, None, True), argnums=(0, 1, 2, 3))
        return jax.jit(f).lower(x, m, g0, e0)

    def step_grad_dehnen_error() -> Any:
        prov, aux0 = lane("dehnen_error")
        f = jax.grad(step_loss(prov, aux0, False), argnums=(0, 1))
        return jax.jit(f).lower(x, m, g0, e0)

    def rollout_loss(xx, vv, mm, g, e, k):
        prov, _ = lane("dehnen")
        out = differentiable_rollout(
            prov.differentiable_step,
            xx,
            vv,
            mm,
            DT,
            STEPS,
            params=ForceParams(G=g, eps2=e),
            external_fn=lambda rows, a: -a["k"] * rows[:, 0],
            external_args={"k": k},
            checkpoint="step",
        )
        return jnp.sum(w * out.positions) + jnp.sum(w * out.velocities)

    def rollout_forward() -> Any:
        return jax.jit(rollout_loss).lower(x, v, m, g0, e0, k0)

    def rollout_grad() -> Any:
        f = jax.value_and_grad(rollout_loss, argnums=tuple(range(6)))
        return jax.jit(f).lower(x, v, m, g0, e0, k0)

    return {
        "step_forward": step_forward,
        "step_grad": step_grad,
        "step_grad_dehnen_error": step_grad_dehnen_error,
        "rollout_forward": rollout_forward,
        "rollout_grad": rollout_grad,
    }


def record(tag: str, root: Path) -> int:
    """Lower every program and save its text under ``root/tag``.

    Parameters
    ----------
    tag : str
        The record's name.
    root : Path
        Where records live.

    Returns
    -------
    int
        ``0``.
    """
    import jax

    for key, value in _ENV.items():
        os.environ[key] = value
    os.environ.pop("JACCPOT_GRAD_AD_MODE", None)
    jax.default_backend = lambda: "gpu"  # the large-N profile's backend gate
    out = root / tag
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, lower in _programs().items():
        t0 = time.time()
        text = lower().as_text(debug_info=False)
        (out / f"{name}.txt").write_text(text)
        digests[name] = hashlib.sha256(text.encode()).hexdigest()
        print(
            f"[{tag}] {name}: {len(text)} chars, {time.time() - t0:.1f} s", flush=True
        )
    (out / "digests.json").write_text(json.dumps(digests, indent=1, sort_keys=True))
    return 0


def compare(a: str, b: str, root: Path) -> int:
    """Compare two records program by program.

    Parameters
    ----------
    a : str
        The first record.
    b : str
        The second record.
    root : Path
        Where records live.

    Returns
    -------
    int
        ``0`` when every program's text is identical, else ``1``.
    """
    da = json.loads((root / a / "digests.json").read_text())
    db = json.loads((root / b / "digests.json").read_text())
    bad = 0
    for name in sorted(set(da) | set(db)):
        same = da.get(name) == db.get(name)
        bad += not same
        print(f"{'same' if same else 'DIFFERS'}  {name}")
    print(f"{len(set(da) | set(db)) - bad}/{len(set(da) | set(db))} identical")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    """Parse the command line and run it.

    Parameters
    ----------
    argv : list[str]
        The arguments.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--tag", required=True)
    rec.add_argument("--root", type=Path, required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "record":
        return record(args.tag, args.root)
    return compare(args.a, args.b, args.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
