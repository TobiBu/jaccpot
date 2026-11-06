"""The fused lane's platform: its CPU route (``RuntimePolicyConfig.cpu_fused_lane``).

The fused lane (the ``large_n_gpu`` profile: ``FusedForceProvider``,
``ProductionRunner``, ``strict_run_v2``) takes its lane policy from the backend:
the prepare declines a backend that is not a GPU (falling back to the general
path), and the near-field mode, chunk sizes, the split dual-tree build and the
streamed traversal caps follow the GPU's choices. Its kernels have pure-JAX forms
that run on a CPU unchanged, which is how jaccpot's own CPU tests run the lane:
they make :func:`jax.default_backend` report ``"gpu"`` for the whole process.

The CPU route is the supported form of that, per engine and opt-in:

* :func:`lane_policy_backend` is what the engine's lane gates read instead of
  :func:`jax.default_backend`: ``"gpu"`` on a CPU backend for an engine built with
  ``RuntimePolicyConfig(cpu_fused_lane=True)``, the real backend otherwise. On a
  GPU (and for every engine without the flag) it IS ``jax.default_backend()``, so
  the programs there are unchanged. Kernel availability is not lane policy: the
  Pallas gates (``pallas_*_supported``) keep reading the real backend, so on the
  route the kernels are the pure-JAX ones.
* :func:`lane_platform_scope`: with ``use_pallas=True`` on the route, the near
  field's Pallas kernel runs in interpret mode on the CSR lane (the lane
  ``mac_type="dehnen_error"``'s force scale needs), with the rectangle payload
  declined. Those switches are environment knobs read at trace time across the
  near-field modules; the scope gives them these values as context-local
  defaults (:func:`jaccpot._env.scoped_env_defaults`) -- an explicit shell value
  still wins, ``os.environ`` is never written, other threads do not see it. The
  runner's and the provider's entry points enter it (:func:`on_lane_platform`).
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator, TypeVar

import jax

from jaccpot._env import scoped_env_defaults

__all__ = [
    "CPU_ROUTE_PALLAS_ENV",
    "cpu_route_active",
    "lane_platform_scope",
    "lane_policy_backend",
    "on_lane_platform",
    "on_lane_platform_of_solver",
]

#: The near-field switches the CPU route scopes under ``use_pallas=True``: the
#: Pallas kernels in interpret mode, the CSR row-chunk lane (whose direct kernel
#: accumulates ``dehnen_error``'s force scale), the rectangle payload declined
#: (its 1024 MB default would otherwise be built and read on small N).
CPU_ROUTE_PALLAS_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}

_F = TypeVar("_F", bound=Callable[..., Any])


def _engine(obj: Any) -> Any:
    """The engine behind a facade, provider, runner or engine.

    Parameters
    ----------
    obj : Any
        ``FastMultipoleMethod``, ``FMMEngine``, or an object with ``.engine``.

    Returns
    -------
    Any
        The engine (``obj`` itself when it has neither ``engine`` nor ``_impl``).
    """
    engine = getattr(obj, "engine", None)
    if engine is not None:
        return engine
    return getattr(obj, "_impl", obj)


def cpu_route_active(obj: Any) -> bool:
    """Whether ``obj``'s engine runs the fused lane's CPU route now.

    Parameters
    ----------
    obj : Any
        An engine, a facade, or an object with ``.engine``.

    Returns
    -------
    bool
        ``cpu_fused_lane`` is set and the backend is a CPU.
    """
    return (
        bool(getattr(_engine(obj), "cpu_fused_lane", False))
        and jax.default_backend() == "cpu"
    )


def lane_policy_backend(obj: Any) -> str:
    """The backend the fused lane's policy follows (module docstring).

    Parameters
    ----------
    obj : Any
        An engine, a facade, or an object with ``.engine``.

    Returns
    -------
    str
        ``"gpu"`` on the CPU route, else :func:`jax.default_backend`.
    """
    backend = jax.default_backend()
    if backend == "cpu" and bool(getattr(_engine(obj), "cpu_fused_lane", False)):
        return "gpu"
    return backend


@contextlib.contextmanager
def lane_platform_scope(obj: Any) -> Iterator[None]:
    """The near-field switches of the CPU route under ``use_pallas`` (module docstring).

    A no-op off the route and with ``use_pallas=False``.

    Parameters
    ----------
    obj : Any
        An engine, a facade, or an object with ``.engine``.

    Yields
    ------
    None
    """
    engine = _engine(obj)
    if cpu_route_active(engine) and bool(getattr(engine, "use_pallas", False)):
        with scoped_env_defaults(CPU_ROUTE_PALLAS_ENV):
            yield
    else:
        yield


def on_lane_platform(method: _F) -> _F:
    """Run a method of an object with ``.engine`` inside :func:`lane_platform_scope`.

    Parameters
    ----------
    method : _F
        The method; its ``self`` has ``.engine`` (or is an engine).

    Returns
    -------
    _F
        The wrapped method.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with lane_platform_scope(self):
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def on_lane_platform_of_solver(method: _F) -> _F:
    """:func:`on_lane_platform` for a classmethod whose first argument is the solver.

    Parameters
    ----------
    method : _F
        ``method(cls, solver, ...)``.

    Returns
    -------
    _F
        The wrapped method (apply ``@classmethod`` on top).
    """

    @functools.wraps(method)
    def wrapper(cls: Any, solver: Any, *args: Any, **kwargs: Any) -> Any:
        with lane_platform_scope(solver):
            return method(cls, solver, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
