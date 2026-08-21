"""Single point of contact for the JAX internals jaccpot depends on.

Why this module exists: ``jax.core`` is not public. It is absent from
``jax.__all__``, its contents live in ``jax._src.core``, and there is no
sanctioned replacement on the supported range -- ``jax.extend.core.Tracer`` does
not exist in jax 0.10.2, which is the floor and effectively the ceiling
``pyproject.toml`` pins, for reasons written out in full there.

So the ``jax.core.Tracer`` references this module replaces were each two things
at once: a type-checker error (20 of them, audit E.4 bucket B) and an unpinned
dependency on a private module, spread across 13 files. Centralising them does
not make the dependency public, but it makes it countable and gives the eventual
move a single site to change.

Do not add to this module casually. It holds private-API dependencies that cannot
presently be avoided, and every entry should say why it cannot.
"""

from __future__ import annotations

import jax

# ``isinstance(x, Tracer)`` is how the runtime asks "am I inside a trace?" --
# ``runtime/fmm_caches.py::_contains_tracer`` is the main consumer. This resolves
# to ``jax._src.core.Tracer``; the suppression is for the missing public
# re-export, not for a missing attribute at runtime.
Tracer = jax.core.Tracer  # pyright: ignore[reportAttributeAccessIssue]

__all__ = ["Tracer"]
