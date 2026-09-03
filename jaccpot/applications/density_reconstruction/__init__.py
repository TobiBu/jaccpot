"""Static density reconstruction: recover a discrete mass distribution from field samples.

The Paper I section-7 case study. Given accelerations observed at ``M`` tracer
positions, recover the ``N`` equal-mass source positions that produced them, by
gradient descent through the FMM. Positions are the **only** free parameters;
masses are frozen and equal; there are no velocities and no dynamical statement
(see :mod:`~jaccpot.applications.density_reconstruction.forward` for the
invariants and the contract they enforce).

What this package demonstrates is that the optimisation *machinery* scales to
O(10^7) free coordinates. It does not claim the inverse problem is well posed --
recovering discrete source positions from external field samples has continuous
degeneracies -- so the primary metric throughout is the field-space residual,
with recovered-density agreement secondary and per-particle position error an
explicitly degenerate tertiary diagnostic.
"""

from jaccpot.applications.density_reconstruction.diagnostics import (
    density_agreement,
    field_residual,
    moment_drift,
    position_error,
)
from jaccpot.applications.density_reconstruction.forward import (
    ForwardOperator,
    assert_masses_frozen_and_equal,
    make_forward_operator,
)
from jaccpot.applications.density_reconstruction.loss import (
    Regularization,
    data_misfit,
    leaf_blocks_from_state,
    regularization_terms,
    total_loss,
)
from jaccpot.applications.density_reconstruction.parameterize import (
    ParametricParameterization,
    PositionsParameterization,
    initial_positions,
    make_parameterization,
)
from jaccpot.applications.density_reconstruction.topology import (
    ChurnRates,
    RadixStructure,
    SwitchLog,
    churn_between,
    fingerprint_prepared_state,
    radix_structure,
)
from jaccpot.applications.density_reconstruction.truth import (
    GroundTruth,
    make_ground_truth,
)

__all__ = [
    "ChurnRates",
    "ForwardOperator",
    "GroundTruth",
    "ParametricParameterization",
    "PositionsParameterization",
    "RadixStructure",
    "Regularization",
    "SwitchLog",
    "assert_masses_frozen_and_equal",
    "churn_between",
    "data_misfit",
    "density_agreement",
    "field_residual",
    "fingerprint_prepared_state",
    "initial_positions",
    "leaf_blocks_from_state",
    "make_forward_operator",
    "make_ground_truth",
    "make_parameterization",
    "moment_drift",
    "position_error",
    "radix_structure",
    "regularization_terms",
    "total_loss",
]
