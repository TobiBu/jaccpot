"""Reusable FMM numerical kernel library (leaf package).

This package is the destination for the free-function numerical core currently
living at the bottom of ``jaccpot/runtime/_fmm_impl.py`` -- the M2L / L2L batch
kernels, the solidfmm/real accumulate + propagate routines, the Pallas
fast-lane gates, the downward-sweep driver, and the tree/prepared-state
evaluation helpers.

Design contract (the hinge of the refactor): this package MUST remain a leaf --
it may depend on ``operators/``, ``downward/``, ``nearfield/``, ``upward/``,
``pallas/``, and the shared ``runtime/fmm_constants`` + ``runtime/fmm_caches``
modules, but it MUST NOT import the orchestrator (``runtime.fmm.engine`` or the
prepare pipeline). Keeping it a leaf is what lets ``distributed/``,
``experimental/``, ``_interaction_cache`` and ``_large_n_pipeline`` import the
kernels without dragging in the engine, dissolving the lazy-import cycles that
exist today.

Populated during Phase 2 (mechanical extraction); intentionally empty for now.
"""
