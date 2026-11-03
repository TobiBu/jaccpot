"""H0's CPU smoke shim: on ``PYTHONPATH`` ONLY in ``SMOKE=1`` runs (``submit_all.sh``).

With ``H0_SMOKE_CPU=1`` and ``JAX_PLATFORMS=cpu`` it makes the fused production lane run
on the CPU the way jaccpot's lane goldens and ``bench/provider_hlo_identity.py`` do:
``jax.default_backend`` reports ``"gpu"`` (the large-N prepare declines off a GPU
otherwise) and the near field runs its Pallas kernels in interpret mode on the CSR lane
(``tests/unit/runtime/test_force_scale_evaluation.py``'s switches). Every process the
benches spawn inherits ``PYTHONPATH``, so the shim reaches them too. A real job never
has this directory on its path.

It then runs the next ``sitecustomize`` on ``sys.path``, if any (a site's own).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("H0_SMOKE_CPU") == "1" and os.environ.get("JAX_PLATFORMS") == "cpu":
    for _key, _val in (
        ("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1"),
        ("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1"),
        ("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", "0"),
    ):
        os.environ.setdefault(_key, _val)
    import jax

    jax.default_backend = lambda: "gpu"  # type: ignore[assignment]

_rest = [p for p in sys.path if p and os.path.abspath(p) != _HERE]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
