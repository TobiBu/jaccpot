"""H0's CPU smoke shim: on ``PYTHONPATH`` ONLY in ``SMOKE=1`` runs (``submit_all.sh``).

With ``H0_SMOKE_CPU=1`` and ``JAX_PLATFORMS=cpu`` it makes the fused production lane run
on the CPU the way jaccpot's lane goldens and ``bench/provider_hlo_identity.py`` do:
``jax.default_backend`` reports ``"gpu"`` (the large-N prepare declines off a GPU
otherwise), the Pallas near field reports itself supported (``use_pallas=None`` resolves
to True, as the CPU tests pass it) and runs its kernels in interpret mode on the CSR lane, and
yggdrax's ragged exchange keeps its CPU route
(``tests/unit/runtime/test_force_scale_evaluation.py``'s switches). Every process the
benches spawn inherits ``PYTHONPATH``, so the shim reaches them too. A real job never
has this directory on its path.

It then runs the next ``sitecustomize`` on ``sys.path``, if any (a site's own).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("H0_SMOKE_CPU") == "1" and os.environ.get("JAX_PLATFORMS") == "cpu":
    # every Pallas kernel of the lane in interpret mode, the prepacked near layout, the
    # GPU's walk: tests/integration/strict/test_grad_rollout_differentiable.py's set
    for _key, _val in (
        ("JACCPOT_CASCADE_PALLAS", "1"),
        ("JACCPOT_CASCADE_PALLAS_INTERPRET", "1"),
        ("JACCPOT_M2L_CSR_INTERPRET", "1"),
        ("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1"),
        ("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1"),
        ("JACCPOT_L2P_KERNEL", "interpret"),
        ("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", "0"),
        ("JACCPOT_WALK_PALLAS_INTERPRET", "1"),
    ):
        os.environ.setdefault(_key, _val)
    # the test conftest's XLA:CPU flag: interpret-mode kernels compile minutes faster
    if "xla_backend_extra_options" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_backend_extra_options=-vectorize-loops=false"
        ).strip()
    import jax

    jax.default_backend = lambda: "gpu"  # type: ignore[assignment]

    def _pallas_supported(module):
        # use_pallas=None resolves to True, as the CPU tests pass use_pallas=True
        module.pallas_nearfield_fused_supported = lambda: True

    def _ragged_on_cpu(module):
        # "auto" would read the patched backend and pick the native ragged_all_to_all,
        # which XLA:CPU cannot lower: keep the exchange on its CPU route ("buf")
        resolve = module.resolve_ragged_method
        module.resolve_ragged_method = lambda method="auto": (
            "buf" if method == "auto" else resolve(method)
        )

    _PATCHES = {
        "jaccpot.pallas.nearfield_fused_leaf": _pallas_supported,
        "yggdrax.distributed.comm": _ragged_on_cpu,
    }

    class _PatchAfterImport(importlib.abc.MetaPathFinder):
        """Run ``_PATCHES[name]`` on a module right after it is first executed."""

        def find_spec(self, name, path=None, target=None):
            if name not in _PATCHES:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def _exec(module):
                exec_module(module)
                _PATCHES[name](module)

            spec.loader.exec_module = _exec  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _PatchAfterImport())

_rest = [p for p in sys.path if p and os.path.abspath(p) != _HERE]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
