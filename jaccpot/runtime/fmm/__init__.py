"""Runtime FMM orchestrator package for Jaccpot.

This package will host the exploded ``_fmm_impl`` orchestrator (engine, config
resolution, prepare/strict-run/diagnostics) as the ``_fmm_impl`` monolith is
split up. During the mechanical-extraction phase it re-exports the current
runtime symbols so ``jaccpot.runtime.fmm`` keeps resolving identically and the
test suite stays green.

Note: this ``__init__`` must NOT import the engine class eagerly once it lands
in ``fmm/engine.py`` -- doing so would re-form the
``_interaction_cache -> fmm -> engine -> prepare -> _interaction_cache`` import
cycle. Consumers import the class explicitly from ``jaccpot.runtime.fmm.engine``.
"""

from .._fmm_impl import *  # noqa: F401,F403
from .._fmm_impl import build_interactions_and_neighbors  # noqa: F401
from ..kernels.core import _build_grouped_class_segments  # noqa: F401
from ..kernels.core import _infer_bounds  # noqa: F401
