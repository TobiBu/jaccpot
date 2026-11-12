"""Shared scaffolding for the Jaccpot I paper pipeline.

Three layers, kept strictly separate (see the repo README's "Paper Branch"
section):

* ``bench/**/*.py`` computes and writes JSON. It imports :mod:`jsonio` and
  :mod:`runmeta` from here, and must never import :mod:`style` -- a benchmark
  that can import matplotlib is a benchmark that can grow a plot.
* ``examples/jaccpot_paper/fig_*.ipynb`` reads JSON and draws. It imports
  :mod:`jsonio` and :mod:`style`, and never recomputes.

:mod:`jsonio` and :mod:`runmeta` are deliberately import-safe before JAX is
configured, because every GPU bench script has to call ``autocvd`` *before* the
first ``import jax``. Neither imports jax at module scope.
"""

from __future__ import annotations

__all__ = ["jsonio", "runmeta", "style"]
