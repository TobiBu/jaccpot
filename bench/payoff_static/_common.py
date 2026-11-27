"""Command-line pieces the section-7 (static payoff) benches share.

Two things live here because every ``bench/payoff_static`` script needs them the
same way, and a copy per script is how one of them drifts:

* **The lane, chosen explicitly.** ``make_forward_operator`` still defaults to
  ``lane="radix"`` (the general path, section 3's validation configuration).
  Section 7 runs on ``lane="fused"`` -- the production fused lane, fp64 by
  default, geometric ``"dehnen"`` MAC because ``dehnen_error`` is mass-dependent
  and refused with zero-mass tracers. So every bench takes ``--lane``
  (default ``fused``) and passes it through, and the results JSON records it. A
  bench that relied on the library default would silently measure the radix
  lane.
* **The CPU route of the mesh lane.** On a CPU backend the mesh gradient runs
  only under the S9 interpret set of ``tests/distributed/test_mesh_grad_gates.py``
  (the lane's Pallas kernels and the GPU's walk interpreted,
  ``jax.default_backend`` reporting ``"gpu"``, the ragged exchange on ``buf``).
  ``--cpu-interpret`` applies it, with forced host devices; it is for smoke runs
  only, and the config records that it was on. Nothing measured under it is a
  timing.

Nothing here imports JAX at module level: the environment edits must happen
before the first ``import jax``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

#: The evaluation lanes of ``density_reconstruction.forward.LANES``, duplicated
#: so the argument parser does not import jaccpot (and with it JAX) before
#: ``runmeta.select_gpu``. ``make_forward_operator`` validates the value again.
LANES = ("fused", "radix")

#: The S9 interpret set, as ``tests/applications/test_density_reconstruction_distributed.py``
#: and ``bench/multigpu_grad_gate.py --cpu-smoke`` apply it.
CPU_INTERPRET_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
CPU_INTERPRET_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_CROSS_WALK",
)


def add_lane_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--lane``, ``--mac`` and ``--preset``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The bench's parser.
    """
    parser.add_argument(
        "--lane",
        choices=LANES,
        default="fused",
        help=(
            "Evaluation lane passed to make_forward_operator. 'fused' (default) "
            "is the production fused lane section 7 runs on; 'radix' is the "
            "general path (the library's default, section 3's configuration). "
            "On 'fused' the working precision is --dtype"
        ),
    )
    parser.add_argument(
        "--mac",
        default=None,
        help=(
            "Multipole acceptance criterion; default: the lane's (fused: the "
            "geometric 'dehnen'; radix: the preset's). Mass-dependent criteria "
            "('dehnen_error', 'dehnen_theta') are refused with zero-mass tracers"
        ),
    )
    parser.add_argument(
        "--preset",
        default="accurate",
        help="jaccpot preset of the radix lane (ignored on the fused lane)",
    )


def operator_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the lane keywords for ``make_forward_operator``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line with ``lane``, ``mac``, ``preset`` and ``dtype``.

    Returns
    -------
    Dict[str, Any]
        ``lane`` and ``mac_type`` always; ``working_dtype`` on the fused lane,
        ``preset`` on the radix lane.
    """
    kwargs: Dict[str, Any] = {"lane": str(args.lane), "mac_type": args.mac}
    if str(args.lane) == "fused":
        kwargs["working_dtype"] = str(args.dtype)
    else:
        kwargs["preset"] = str(args.preset)
    return kwargs


def lane_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the lane fields for a results JSON's ``config``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    Dict[str, Any]
        ``lane``, the requested ``mac_type`` (``None``: the lane's default; each
        record's ``operator`` block names the one in use), ``working_dtype``, and
        ``preset`` on the radix lane.
    """
    record: Dict[str, Any] = {
        "lane": str(args.lane),
        "mac_type": args.mac,
        "working_dtype": str(args.dtype),
    }
    if str(args.lane) == "radix":
        record["preset"] = str(args.preset)
    return record


def add_cpu_interpret_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--cpu-interpret``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The bench's parser.
    """
    parser.add_argument(
        "--cpu-interpret",
        action="store_true",
        help=(
            "CPU smoke of the mesh lane: forced host devices and the S9 "
            "interpret set (the mesh gradient's only CPU route). Compiles for "
            "minutes; nothing measured under it is a timing"
        ),
    )


def apply_cpu_interpret(num_devices: int) -> None:
    """Set the CPU mesh route's environment. Call before the first ``import jax``.

    Parameters
    ----------
    num_devices : int
        Host devices to force (at least 1).

    Raises
    ------
    RuntimeError
        If JAX is already imported: the platform and device count would be
        silently ignored.
    """
    import sys

    if "jax" in sys.modules:
        raise RuntimeError("apply_cpu_interpret() must run before jax is imported")
    for key in CPU_INTERPRET_UNSET:
        os.environ.pop(key, None)
    os.environ.update(CPU_INTERPRET_ENV)
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        flags = (
            f"{flags} --xla_force_host_platform_device_count={max(int(num_devices), 1)}"
        )
    os.environ["XLA_FLAGS"] = flags.strip()


def patch_cpu_lane() -> None:
    """The CPU route's runtime patches. Call right after ``import jax``.

    ``jax.default_backend`` reports ``"gpu"`` (the large-N GPU profile) and the
    ragged exchange resolves to ``buf``, as in the WP-G CPU gates.
    """
    import jax
    import yggdrax.distributed.comm as ygg_comm

    jax.default_backend = lambda: "gpu"
    resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = lambda method="auto": (
        "buf" if method == "auto" else resolve(method)
    )
