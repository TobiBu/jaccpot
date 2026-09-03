"""Application layers built on the jaccpot solver.

Each subpackage here is a *consumer* of :class:`jaccpot.FastMultipoleMethod`:
it composes the public solver API into a specific scientific task and owns the
assertions that task needs, without reaching into ``jaccpot.runtime``. Nothing
in the solver imports from here.
"""
