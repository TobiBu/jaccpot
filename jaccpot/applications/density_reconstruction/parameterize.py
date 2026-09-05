"""Parameter pytrees and the mapping to source positions.

Two parameterisations, and **the contrast between them is the section**:

``positions``
    Hard decision 2: the ``3N`` source coordinates, free. The headline. The map
    to positions is the identity.
``parametric``
    7 free parameters with no perturber, 11 with one -- scale lengths, a halo
    flattening, an orientation, and the perturber's place and size. The
    baseline finite differences could also have done.

The parametric map is smooth and exactly differentiable because the *draws* are
held fixed and only the *scales* are free: every radial distribution in the
composite is a scale parameter times a unit-space deviate, so
:func:`~jaccpot.applications.density_reconstruction.truth.sample_unit_template`
freezes the deviates as constants and this module multiplies. That is the
standard reparameterisation, and it means both parameterisations feed the
identical observation operator and the identical loss -- the only difference is
the width of the pytree that reaches ``jax.grad``.

Masses appear in no pytree here, under any name, and
:func:`~jaccpot.applications.density_reconstruction.forward.assert_masses_frozen_and_equal`
is what enforces it (hard decision 1). Note what the parametric case makes
explicit: with masses frozen *and equal*, the total mass is fixed by ``N`` alone,
so none of these parameters is a mass and the fit cannot trade profile shape
against normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import jax.numpy as jnp
import numpy as np

from jaccpot.applications.density_reconstruction.truth import (
    TruthConfig,
    UnitTemplate,
    sample_unit_template,
)
from jaccpot.operators._precision import highest_matmul_precision

__all__ = [
    "PARAMETRIC_NAMES",
    "PARAMETRIC_NAMES_WITH_PERTURBER",
    "InitialGuessMode",
    "ParametricParameterization",
    "ParameterizationKind",
    "PositionsParameterization",
    "initial_positions",
    "make_parameterization",
    "positions_params",
    "to_positions",
]

InitialGuessMode = Literal[
    "perturbed_truth", "smooth_wrong", "isotropized_truth", "uniform_sphere"
]
ParameterizationKind = Literal["positions", "parametric"]

#: The parametric model's free parameters when there is no perturber. Scales are
#: carried as logarithms so they cannot go negative under an unconstrained
#: optimiser, and so a step is multiplicative -- which is what a scale length
#: wants.
PARAMETRIC_NAMES: Tuple[str, ...] = (
    "log_bulge_scale",
    "log_disc_scale_length",
    "log_disc_scale_height",
    "log_halo_scale",
    "halo_flattening",
    "tilt_x",
    "tilt_y",
)

#: With a perturber, its place and size join them: 11 parameters.
PARAMETRIC_NAMES_WITH_PERTURBER: Tuple[str, ...] = PARAMETRIC_NAMES + (
    "perturber_offset_x",
    "perturber_offset_y",
    "perturber_offset_z",
    "log_perturber_scale",
)


def positions_params(source_positions: Any) -> Dict[str, jnp.ndarray]:
    """Wrap positions as the free-parameter pytree.

    Parameters
    ----------
    source_positions : Any
        ``(N, 3)`` positions.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"positions": (N, 3) float64}``. The single-key dict is the seam the
        parametric case shares; nothing else belongs in here.
    """
    return {"positions": jnp.asarray(source_positions, dtype=jnp.float64)}


def to_positions(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Map a positions pytree to source positions.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Output of :func:`positions_params`.

    Returns
    -------
    jnp.ndarray
        ``(N, 3)`` positions. The identity for this parameterisation.
    """
    return params["positions"]


@dataclass(frozen=True)
class PositionsParameterization:
    """The ``3N`` free-coordinate parameterisation -- the section's headline.

    Attributes
    ----------
    num_sources : int
        ``N``.
    kind : ParameterizationKind
        ``"positions"``. Recorded in every results JSON.
    """

    num_sources: int
    kind: ParameterizationKind = "positions"

    @property
    def num_free(self: "PositionsParameterization") -> int:
        """Return the free-parameter count.

        Returns
        -------
        int
            ``3N``.
        """
        return 3 * int(self.num_sources)

    def pack(
        self: "PositionsParameterization", positions: Any
    ) -> Dict[str, jnp.ndarray]:
        """Wrap positions as the parameter pytree.

        Parameters
        ----------
        positions : Any
            ``(N, 3)`` starting positions.

        Returns
        -------
        Dict[str, jnp.ndarray]
            The pytree ``jax.grad`` will differentiate.
        """
        return positions_params(positions)

    def to_positions(
        self: "PositionsParameterization", params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Map the pytree to source positions.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            The parameter pytree.

        Returns
        -------
        jnp.ndarray
            ``(N, 3)`` positions.
        """
        return to_positions(params)

    def record(self: "PositionsParameterization") -> Dict[str, Any]:
        """Return the JSON-safe description of this parameterisation.

        Returns
        -------
        Dict[str, Any]
            Kind, free-parameter count and ``N``.
        """
        return {
            "parameterization": self.kind,
            "num_free_parameters": self.num_free,
            "N": int(self.num_sources),
        }


@highest_matmul_precision
def _rotation(tilt_x: jnp.ndarray, tilt_y: jnp.ndarray) -> jnp.ndarray:
    """Build ``Rx(tilt_x) @ Ry(tilt_y)`` as a differentiable 3x3 matrix.

    Parameters
    ----------
    tilt_x : jnp.ndarray
        Rotation about ``x`` in radians.
    tilt_y : jnp.ndarray
        Rotation about ``y`` in radians.

    Returns
    -------
    jnp.ndarray
        ``(3, 3)`` rotation. Two angles rather than three: the composite is
        axisymmetric about its own ``z`` before rotation, so a third angle
        would be a pure degeneracy, and the section has enough of those
        already.
    """
    cx, sx = jnp.cos(tilt_x), jnp.sin(tilt_x)
    cy, sy = jnp.cos(tilt_y), jnp.sin(tilt_y)
    one, zero = jnp.ones_like(cx), jnp.zeros_like(cx)
    rx = jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([zero, cx, -sx]),
            jnp.stack([zero, sx, cx]),
        ]
    )
    ry = jnp.stack(
        [
            jnp.stack([cy, zero, sy]),
            jnp.stack([zero, one, zero]),
            jnp.stack([-sy, zero, cy]),
        ]
    )
    return rx @ ry


@dataclass(frozen=True)
class ParametricParameterization:
    """The 7- or 11-parameter smooth model -- the finite-difference baseline.

    The unit draws are constants; the parameters are the scales, the halo
    flattening, the orientation and the perturber's place and size. Both this
    and :class:`PositionsParameterization` produce ``(N, 3)`` positions for the
    same operator, which is what makes the cost comparison in fig16 a
    comparison of parameter count and nothing else.

    Attributes
    ----------
    template : UnitTemplate
        The frozen unit-space draws.
    has_perturber : bool
        Whether the perturber's four parameters are free.
    kind : ParameterizationKind
        ``"parametric"``. Recorded in every results JSON.
    """

    template: UnitTemplate
    has_perturber: bool
    kind: ParameterizationKind = "parametric"

    @property
    def names(self: "ParametricParameterization") -> Tuple[str, ...]:
        """Return the free-parameter names, in a fixed order.

        Returns
        -------
        Tuple[str, ...]
            :data:`PARAMETRIC_NAMES`, extended by the perturber's four when it
            is present.
        """
        return (
            PARAMETRIC_NAMES_WITH_PERTURBER if self.has_perturber else PARAMETRIC_NAMES
        )

    @property
    def num_free(self: "ParametricParameterization") -> int:
        """Return the free-parameter count.

        Returns
        -------
        int
            7 without a perturber, 11 with one.
        """
        return len(self.names)

    @property
    def num_sources(self: "ParametricParameterization") -> int:
        """Return ``N``, the particle count the model realises.

        Returns
        -------
        int
            Particle count, which is *not* a free parameter -- it fixes the
            per-particle mass and so the total mass.
        """
        return self.template.num_particles

    def true_params(
        self: "ParametricParameterization", config: TruthConfig
    ) -> Dict[str, jnp.ndarray]:
        """Return the parameter values that reproduce ``config``'s truth.

        Parameters
        ----------
        config : TruthConfig
            The configuration the template was drawn from.

        Returns
        -------
        Dict[str, jnp.ndarray]
            One 0-d array per name. The tilts are zero and the halo flattening
            is one, because the truth's halo is drawn isotropically and its
            disc lies in the ``z = 0`` plane -- so the true parameter vector is
            a genuine interior point, not a value clipped at a bound.
        """
        values: Dict[str, jnp.ndarray] = {
            "log_bulge_scale": jnp.asarray(np.log(config.bulge_scale)),
            "log_disc_scale_length": jnp.asarray(np.log(config.disc_scale_length)),
            "log_disc_scale_height": jnp.asarray(np.log(config.disc_scale_height)),
            "log_halo_scale": jnp.asarray(np.log(config.halo_scale)),
            "halo_flattening": jnp.asarray(1.0),
            "tilt_x": jnp.asarray(0.0),
            "tilt_y": jnp.asarray(0.0),
        }
        if self.has_perturber:
            offset = np.asarray(config.perturber_offset, dtype=np.float64)
            values["perturber_offset_x"] = jnp.asarray(offset[0])
            values["perturber_offset_y"] = jnp.asarray(offset[1])
            values["perturber_offset_z"] = jnp.asarray(offset[2])
            values["log_perturber_scale"] = jnp.asarray(np.log(config.perturber_scale))
        return values

    def pack(
        self: "ParametricParameterization", values: Dict[str, Any]
    ) -> Dict[str, jnp.ndarray]:
        """Coerce a name-to-value mapping into the parameter pytree.

        Parameters
        ----------
        values : Dict[str, Any]
            One scalar per name in :attr:`names`.

        Returns
        -------
        Dict[str, jnp.ndarray]
            The pytree ``jax.grad`` will differentiate.

        Raises
        ------
        ValueError
            If a name is missing or an extra name is supplied. Silently
            defaulting a scale length would make a fit unreproducible from its
            own results file.
        """
        expected = set(self.names)
        supplied = set(values)
        if expected != supplied:
            raise ValueError(
                "parametric pytree must carry exactly "
                f"{sorted(expected)}; missing {sorted(expected - supplied)}, "
                f"unexpected {sorted(supplied - expected)}"
            )
        return {
            name: jnp.asarray(values[name], dtype=jnp.float64) for name in self.names
        }

    @highest_matmul_precision
    def to_positions(
        self: "ParametricParameterization", params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Realise the model's particle positions from its parameters.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            The parameter pytree.

        Returns
        -------
        jnp.ndarray
            ``(N, 3)`` positions, in the same component block order the truth
            uses -- bulge, disc, halo, perturber -- so the two are comparable
            block by block.
        """
        t = self.template
        rotation = _rotation(params["tilt_x"], params["tilt_y"])

        bulge = (
            jnp.exp(params["log_bulge_scale"])
            * jnp.asarray(t.bulge_unit_radii)[:, None]
            * jnp.asarray(t.bulge_directions)
        )

        disc_r = jnp.exp(params["log_disc_scale_length"]) * jnp.asarray(
            t.disc_unit_radii
        )
        disc_z = jnp.exp(params["log_disc_scale_height"]) * jnp.asarray(
            t.disc_unit_heights
        )
        phi = jnp.asarray(t.disc_phi)
        disc = jnp.stack([disc_r * jnp.cos(phi), disc_r * jnp.sin(phi), disc_z], axis=1)

        halo = (
            jnp.exp(params["log_halo_scale"])
            * jnp.asarray(t.halo_unit_radii)[:, None]
            * jnp.asarray(t.halo_directions)
        )
        # Flattening acts on the symmetry axis before the orientation is
        # applied, so the two parameters are not entangled.
        halo = halo * jnp.stack([jnp.ones(()), jnp.ones(()), params["halo_flattening"]])

        # The bulge is spherical, so rotating it is a no-op; the disc and the
        # flattened halo share the composite's symmetry axis and turn together.
        parts = [bulge, disc @ rotation.T, halo @ rotation.T]

        if self.has_perturber:
            offset = jnp.stack(
                [
                    params["perturber_offset_x"],
                    params["perturber_offset_y"],
                    params["perturber_offset_z"],
                ]
            )
            perturber = (
                jnp.exp(params["log_perturber_scale"])
                * jnp.asarray(t.perturber_unit_radii)[:, None]
                * jnp.asarray(t.perturber_directions)
            ) + offset
            parts.append(perturber)

        return jnp.concatenate(parts, axis=0)

    def record(self: "ParametricParameterization") -> Dict[str, Any]:
        """Return the JSON-safe description of this parameterisation.

        Returns
        -------
        Dict[str, Any]
            Kind, free-parameter count, the names, and ``N``.
        """
        return {
            "parameterization": self.kind,
            "num_free_parameters": self.num_free,
            "parameter_names": list(self.names),
            "N": int(self.num_sources),
        }


def make_parameterization(
    kind: ParameterizationKind,
    *,
    config: TruthConfig,
) -> Any:
    """Build the requested parameterisation for one truth configuration.

    Parameters
    ----------
    kind : ParameterizationKind
        ``"positions"`` or ``"parametric"``.
    config : TruthConfig
        The truth configuration. The parametric case draws its unit template
        from it, which is what ties the model's realisation to the truth's.

    Returns
    -------
    Any
        A :class:`PositionsParameterization` or
        :class:`ParametricParameterization`.

    Raises
    ------
    ValueError
        If ``kind`` is not recognised.
    """
    if kind == "positions":
        return PositionsParameterization(num_sources=int(config.num_particles))
    if kind == "parametric":
        return ParametricParameterization(
            template=sample_unit_template(config),
            has_perturber=config.perturber != "none",
        )
    raise ValueError(
        f"unknown parameterization {kind!r}; expected 'positions' or 'parametric'"
    )


def initial_positions(
    truth_positions: np.ndarray,
    *,
    mode: InitialGuessMode,
    seed: int,
    perturbation: float = 0.1,
) -> np.ndarray:
    """Construct an initial guess (section 2: the starting point is a result).

    Parameters
    ----------
    truth_positions : np.ndarray
        ``(N, 3)`` true positions. Never returned unmodified.
    mode : InitialGuessMode
        ``"perturbed_truth"`` adds isotropic Gaussian displacement of scale
        ``perturbation``; ``"smooth_wrong"`` replaces the truth with a smooth
        single-component Hernquist sphere matched only in half-mass radius, so
        it is structurally wrong -- no disc, no perturber, wrong profile;
        ``"isotropized_truth"`` keeps each radius and redraws its direction;
        ``"uniform_sphere"`` ignores the truth beyond its outer radius.
    seed : int
        Seed for the draw.
    perturbation : float
        Displacement scale for ``"perturbed_truth"``.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` float64 initial positions.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(truth_positions, dtype=np.float64)
    n = x.shape[0]
    if mode == "perturbed_truth":
        return x + perturbation * rng.standard_normal(x.shape)
    radii = np.linalg.norm(x, axis=1)
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    directions = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    if mode == "isotropized_truth":
        return radii[:, None] * directions
    if mode == "uniform_sphere":
        r_max = float(radii.max())
        r = r_max * np.cbrt(rng.uniform(0.0, 1.0, size=n))
        return r[:, None] * directions
    if mode == "smooth_wrong":
        # A Hernquist sphere whose scale radius reproduces only the truth's
        # half-mass radius: r_half = a (1 + sqrt(2)). Right size, wrong
        # everything else -- the guess that tests whether the fit can find
        # structure it was not handed.
        r_half = float(np.median(radii))
        scale = r_half / (1.0 + np.sqrt(2.0))
        u = rng.uniform(0.0, 0.98, size=n)
        root = np.sqrt(u)
        return (scale * root / (1.0 - root))[:, None] * directions
    raise ValueError(f"unknown initial-guess mode {mode!r}")
