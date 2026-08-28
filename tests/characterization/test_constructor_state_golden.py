"""Golden snapshot of every attribute ``FMMEngine.__init__`` resolves.

**Why this exists, and why it is in `characterization/`.** The engine constructor is
722 lines resolving 60 parameters into 272 instance attributes, and the *order* of
that resolution is load-bearing: commits ``b462e45`` and ``dee46d6`` are both bugs in
exactly this, where a later default silently overwrote an earlier decision. Nothing
guarded it. The forward and gradient goldens run one configuration each, so they
cannot see a preset resolving differently, and audit item 2.1 names that gap as the
reason the constructor could not safely be refactored.

This is the same role the numerics goldens play -- a tripwire for a silent change --
applied to configuration instead of to floats, so it lives beside them.

**What makes it total.** Constructor state is 272 attributes of plain Python types
(int, bool, float, None, str, tuple, dict, set) plus one dataclass. Measured: **no
JAX arrays at all**, and two identical constructions produce byte-identical ``repr``
output. So a ``repr``-based snapshot is exact rather than approximate -- there is no
tolerance here, and there should not be one.

Regenerate intentionally with ``JACCPOT_REGEN_GOLDEN=1 pytest ...``, the same switch
the other goldens use. Regenerating is how an *intended* config change is recorded;
the diff is plain text, so a reviewer can read what moved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import jaccpot
from jaccpot import ComplexSHBasis, RealSHBasis
from jaccpot.config import NearFieldConfig, RuntimePolicyConfig
from jaccpot.runtime._fmm_impl import FMMEngine

GOLDEN_PATH = Path(__file__).parent / "golden_modes" / "constructor_state.json"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

#: ``mac_type="dehnen_error"`` is refused without an explicit relative force-accuracy
#: target -- the theta-derived default is a far looser tail_proxy heuristic. That
#: refusal is deliberate (STYLE_GUIDE §9), so the case supplies one.
PAPER_EPS = 1e-3

#: Marker key in a matrix entry: construct through the public `jaccpot`
#: facade and snapshot its engine, rather than constructing the engine directly.
#: Not a constructor argument -- stripped before the call.
_VIA_FACADE = "__via_facade__"

#: The matrix. Each entry names one axis of the resolution logic that the existing
#: goldens do not vary. Chosen from the branches in `__init__` rather than from the
#: parameter list: a parameter that only lands in an attribute untouched by any
#: branch adds a row of snapshot and no coverage.
CONFIGS: dict[str, dict[str, Any]] = {
    # --- basis, which selects entire kernel families ---
    "cartesian_default": {},
    "solidfmm_complex": {"expansion_basis": "solidfmm"},
    "solidfmm_real": {"expansion_basis": "solidfmm", "basis_impl": RealSHBasis()},
    "solidfmm_explicit_complex": {
        "expansion_basis": "solidfmm",
        "basis_impl": ComplexSHBasis(),
    },
    # --- presets: the gap audit 2.1 names explicitly. Routed through the PUBLIC
    # facade (`_VIA_FACADE`), because that is the only path that resolves all four.
    # The engine's own `preset=` argument accepts any `FMMPreset` but `get_preset_config`
    # only implements FAST and LARGE_N_GPU, so `Engine(preset="balanced")` raises
    # `AssertionError: Missing config for preset`. The facade never passes those two
    # through -- it resolves them into explicit engine kwargs -- so production is
    # unaffected, and snapshotting via the facade is what covers them.
    "preset_fast": {"preset": "fast", _VIA_FACADE: True},
    "preset_balanced": {"preset": "balanced", _VIA_FACADE: True},
    "preset_accurate": {"preset": "accurate", _VIA_FACADE: True},
    "preset_large_n_gpu": {"preset": "large_n_gpu", _VIA_FACADE: True},
    # `basis="complex"`, not `"real"`: the FACADE defaults to real, so a "real" row
    # would duplicate `preset_accurate` exactly. Caught by the distinctness guard,
    # which is the point of having one.
    "preset_accurate_complex": {
        "preset": "accurate",
        "basis": "complex",
        _VIA_FACADE: True,
    },
    # The engine's own two supported values, direct, so the engine-level `preset`
    # attribute is snapshotted too (it stays None on every facade route).
    "engine_preset_fast": {"preset": "fast"},
    "engine_preset_large_n_gpu": {"preset": "large_n_gpu"},
    # --- MAC family, including the jaccpot-level policy value ---
    # `bh` is the engine default, so `cartesian_default` already covers it -- a
    # `mac_bh` row would be a duplicate, which the distinctness guard rejects.
    "mac_engblom": {"mac_type": "engblom"},
    "mac_dehnen": {"mac_type": "dehnen"},
    "mac_dehnen_error": {"mac_type": "dehnen_error", "adaptive_eps": PAPER_EPS},
    "mac_dehnen_scaled": {"mac_type": "dehnen", "dehnen_radius_scale": 1.3},
    # --- adaptive order and its error model. Domains read from the validation in
    # `__init__` rather than guessed: `adaptive_error_model` is
    # tail_proxy|dehnen_degree|dehnen_paper and `mac_force_scale_mode` is
    # prev|prepass|paper|paper_cached.
    "adaptive_order_on": {"adaptive_order": True},
    "adaptive_model_dehnen_degree": {"adaptive_error_model": "dehnen_degree"},
    # `dehnen_paper` carries the same eq (16a) requirement as `mac_type="dehnen_error"`:
    # both refuse a theta-derived eps. Consistent, and the case has to supply one.
    "adaptive_model_dehnen_paper": {
        "adaptive_error_model": "dehnen_paper",
        "adaptive_eps": PAPER_EPS,
    },
    "force_scale_prepass": {"mac_force_scale_mode": "prepass"},
    "force_scale_paper": {"mac_force_scale_mode": "paper"},
    "force_scale_paper_cached": {"mac_force_scale_mode": "paper_cached"},
    # --- memory policy, which drives several derived budgets ---
    "memory_throughput": {
        "runtime_policy": RuntimePolicyConfig(memory_objective="throughput")
    },
    "memory_minimum": {
        "runtime_policy": RuntimePolicyConfig(memory_objective="minimum_memory")
    },
    "memory_budgeted": {
        "runtime_policy": RuntimePolicyConfig(memory_budget_bytes=2 * 1024**3)
    },
    # --- far/near field and runtime lane selection ---
    "streamed_far_pairs_on": {"streamed_far_pairs": True},
    "streamed_far_pairs_off": {"streamed_far_pairs": False},
    "farfield_dense": {"use_dense_interactions": True},
    "farfield_pair_grouped": {"farfield_mode": "pair_grouped"},
    "farfield_class_major": {"farfield_mode": "class_major"},
    "nearfield_baseline": {"nearfield": NearFieldConfig(mode="baseline")},
    "nearfield_bucketed": {"nearfield": NearFieldConfig(mode="bucketed")},
    "runtime_path_large_n": {"runtime_path": "large_n"},
    "backend_radix": {"runtime_policy": RuntimePolicyConfig(execution_backend="radix")},
    "backend_octree": {
        "runtime_policy": RuntimePolicyConfig(execution_backend="octree")
    },
    "grouped_interactions": {
        "grouped_interactions": True,
        "expansion_basis": "solidfmm",
    },
    "mixed_order": {"mixed_order_farfield": True, "mixed_order_min_order": 2},
    # --- tree ---
    "tree_fixed_depth": {"tree_build_mode": "fixed_depth"},
    "tree_refine_local": {"refine_local": True, "max_refine_levels": 3},
    "host_refine_on": {"runtime_policy": RuntimePolicyConfig(host_refine_mode="on")},
    "host_refine_off": {"runtime_policy": RuntimePolicyConfig(host_refine_mode="off")},
    # --- EXPLICIT overrides of hardware-dependent defaults. These exist because a
    # snapshot taken on CPU cannot see a clobber of an attribute whose CPU-resolved
    # value already equals the clobbered one. Mutation-tested: injecting a later
    # `self.use_pallas = False` (the b462e45 shape) is INVISIBLE without these rows,
    # since `pallas_nearfield_fused_supported()` is already False on CPU. With
    # `use_pallas=True` pinned, the clobber is caught.
    # Only the forced-ON row is here. `use_pallas=False` IS the CPU-resolved default,
    # so a forced-off row is byte-identical to `cartesian_default` and the distinctness
    # guard rejects it -- it would only be distinct on Pallas-capable hardware, which
    # a CPU-generated golden cannot represent. Recorded rather than worked around.
    "use_pallas_forced_on": {"use_pallas": True},
    # --- misc switches that gate whole code paths ---
    "fail_fast": {"runtime_policy": RuntimePolicyConfig(fail_fast=True)},
    "retain_far_pairs_for_grad": {"retain_far_pairs_for_grad": True},
    "autotune_m2l": {"runtime_policy": RuntimePolicyConfig(autotune_m2l_chunk=True)},
    "no_interaction_cache": {
        "runtime_policy": RuntimePolicyConfig(enable_interaction_cache=False)
    },
    "fixed_order_and_leaf": {"fixed_order": 4, "fixed_max_leaf_size": 32},
}


def _state(**kwargs: Any) -> dict[str, str]:
    """Every resolved constructor attribute, as text.

    ``repr`` rather than the values themselves because the snapshot has to survive a
    JSON round-trip, and because it makes the golden diffable -- a reviewer can see
    that ``memory_budget_bytes`` moved without owning a reader for the format.

    Parameters
    ----------
    **kwargs : Any
        Constructor arguments for this case.

    Returns
    -------
    dict[str, str]
        Attribute name to ``repr`` of its resolved value, every attribute in
        ``vars()``. Sorted by the caller via ``json.dump(sort_keys=True)``.
    """
    kwargs = dict(kwargs)
    via_facade = kwargs.pop(_VIA_FACADE, False)
    if via_facade:
        # `jaccpot.FastMultipoleMethod` is the FACADE; `FMMEngine` below is the engine.
        # This file names both, so a blanket rename over it is wrong -- audit 2.4 did
        # exactly that here once and this line is why the golden went red.
        # Snapshot the ENGINE either way, since that is what the refactor touches.
        engine = jaccpot.FastMultipoleMethod(**kwargs)._impl
    else:
        engine = FMMEngine(**kwargs)
    return {name: repr(value) for name, value in vars(engine).items()}


def _all_states() -> dict[str, dict[str, str]]:
    """Build the whole matrix.

    Returns
    -------
    dict[str, dict[str, str]]
        Case id to its attribute snapshot.
    """
    return {case: _state(**kwargs) for case, kwargs in CONFIGS.items()}


#: The case every other case is stored as a diff against. Full state for this one,
#: changed attributes only for the rest.
BASE_CASE = "cartesian_default"


def _to_golden(states: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Compress the matrix to a base state plus per-case diffs.

    Stored this way for two reasons, and the second matters more. Size: 45 near
    identical 272-attribute blocks was 549 KB, against ~190 KB for every other golden
    in this directory combined. Readability: a reviewer now sees exactly which
    attributes each configuration changes, so an unexpected line in a diff is
    self-explaining instead of one entry among 12,240.

    Parameters
    ----------
    states : dict[str, dict[str, str]]
        Full snapshots per case.

    Returns
    -------
    dict[str, Any]
        ``{"base_case", "base_state", "diffs"}``. Every case has the same attribute
        set (asserted separately), so a diff carries changed values only.
    """
    base = states[BASE_CASE]
    return {
        "base_case": BASE_CASE,
        "base_state": base,
        "diffs": {
            case: {a: v for a, v in state.items() if base.get(a) != v}
            for case, state in states.items()
            if case != BASE_CASE
        },
    }


def _from_golden(golden: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Rebuild full per-case snapshots from the stored base plus diffs.

    Parameters
    ----------
    golden : dict[str, Any]
        As written by :func:`_to_golden`.

    Returns
    -------
    dict[str, dict[str, str]]
        Full snapshot per case, so the comparison downstream is still total.
    """
    base = golden["base_state"]
    out = {golden["base_case"]: dict(base)}
    for case, diff in golden["diffs"].items():
        merged = dict(base)
        merged.update(diff)
        out[case] = merged
    return out


def test_every_case_sets_the_same_attribute_names() -> None:
    """All cases must share one attribute set -- which is what licenses diff storage.

    If a configuration could add or drop an attribute, a diff of changed *values*
    would silently lose that fact, so this is a precondition of the golden format
    rather than an incidental tidiness check.
    """
    states = _all_states()
    reference = set(states[BASE_CASE])
    for case, state in states.items():
        assert set(state) == reference, (
            f"{case} does not set the same attributes as {BASE_CASE}: "
            f"extra {sorted(set(state) - reference)}, "
            f"missing {sorted(reference - set(state))}"
        )


def test_constructor_state_matches_the_committed_golden() -> None:
    """No constructor attribute may move without the golden moving with it.

    This is the gate audit 2.1 was missing. It fails on a changed *default*, a changed
    resolution *order* that alters an outcome, an attribute that stops being set, or a
    new attribute appearing -- across every configuration in the matrix, not just the
    one the numerics goldens happen to run.
    """
    states = _all_states()

    if REGEN or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(_to_golden(states), indent=1, sort_keys=True) + "\n"
        )
        if not REGEN:
            pytest.skip(f"generated missing golden {GOLDEN_PATH.name} (commit it)")
        return

    golden = _from_golden(json.loads(GOLDEN_PATH.read_text()))

    assert set(states) == set(golden), (
        "the config matrix changed shape: "
        f"added {sorted(set(states) - set(golden))}, "
        f"removed {sorted(set(golden) - set(states))}"
    )

    problems: list[str] = []
    for case in sorted(states):
        got, want = states[case], golden[case]
        for attr in sorted(set(got) | set(want)):
            if attr not in want:
                problems.append(f"{case}: NEW attribute {attr} = {got[attr]}")
            elif attr not in got:
                problems.append(
                    f"{case}: attribute {attr} NO LONGER SET (was {want[attr]})"
                )
            elif got[attr] != want[attr]:
                problems.append(
                    f"{case}: {attr}\n     was {want[attr]}\n     now {got[attr]}"
                )
    assert not problems, (
        "constructor state drifted from the golden:\n  "
        + "\n  ".join(problems[:40])
        + (f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else "")
    )


def test_every_matrix_case_resolves_to_a_distinct_state() -> None:
    """No two cases may resolve identically -- a duplicate row is not coverage.

    The D.12 lesson applied to configuration: a snapshot that runs 32 cases looks like
    32 cases of coverage even if several of them differ in nothing the constructor
    reads. If this fails, either the axis in question does not reach any attribute (so
    the case should go) or a parameter is being dropped on the floor (so the
    constructor should be fixed).
    """
    states = _all_states()
    by_fingerprint: dict[str, list[str]] = {}
    for case, state in states.items():
        key = json.dumps(state, sort_keys=True)
        by_fingerprint.setdefault(key, []).append(case)

    collisions = {tuple(v) for v in by_fingerprint.values() if len(v) > 1}
    assert not collisions, (
        "these matrix cases resolve to byte-identical constructor state, so all but "
        f"one of each group adds no coverage: {sorted(collisions)}"
    )


def test_construction_is_deterministic() -> None:
    """Constructing twice with the same arguments must give the same state.

    Cheap, and it is what makes the golden above meaningful: a time-derived value, a
    set iteration order leaking into a tuple, or a global counter would all make the
    snapshot flaky rather than wrong, and this says which of the two is happening.
    """
    for case, kwargs in CONFIGS.items():
        first = _state(**kwargs)
        second = _state(**kwargs)
        drifted = [a for a in first if first[a] != second[a]]
        assert not drifted, (
            f"{case}: constructing twice gave different state for {drifted} -- the "
            f"constructor is not deterministic, so the golden cannot be trusted"
        )


def test_the_snapshot_covers_every_attribute_the_constructor_sets() -> None:
    """The snapshot is total: no attribute is filtered out of the comparison.

    Guards the one way `_state` could quietly stop being a full-state check -- if it
    grew a skip list, or if an attribute type stopped surviving `repr`. Also pins the
    "no JAX arrays in constructor state" property the module docstring relies on,
    since an array's `repr` is truncated and would make the comparison partial.
    """
    fmm = FMMEngine(expansion_basis="solidfmm")
    live = vars(fmm)
    snapshot = _state(expansion_basis="solidfmm")
    assert set(snapshot) == set(live), (
        "snapshot does not cover every attribute: missing "
        f"{sorted(set(live) - set(snapshot))}"
    )
    truncated = [name for name, text in snapshot.items() if "..." in text]
    assert not truncated, (
        f"these attributes have a truncated repr, so comparing them is only partial: "
        f"{truncated}. A JAX array or a large container reached constructor state; "
        f"snapshot it structurally instead of by repr."
    )
