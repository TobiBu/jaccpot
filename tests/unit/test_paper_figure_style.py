"""The paper's figure palette stays colour-vision- and print-safe.

``style.CATEGORICAL`` is Okabe-Ito in an order chosen by searching every
permutation for the best worst-case adjacent separation. That order is load
bearing and invisible: someone reordering the tuple for aesthetic reasons, or
swapping in a nicer-looking hue, would not see anything break. This test is what
breaks.

Thresholds come from :mod:`palette_check`, which ports the six-check validator;
the two documented WARNs (all-pairs green/purple in the 6-8 floor band, and two
hues below 3:1 against white) are asserted as *bounds* rather than silently
tolerated, so a change that makes either worse fails here.

No matplotlib import: these are pure colour computations, so the test runs
wherever the suite does.
"""

from __future__ import annotations

from examples.jaccpot_paper.common import palette_check as P
from examples.jaccpot_paper.common import style


def test_categorical_palette_passes_the_adjacent_checks() -> None:
    result = P.validate_categorical(style.CATEGORICAL)
    assert result["ok"], P.format_report("categorical (adjacent)", result)
    # The searched order achieves 18.0; anything materially below it means the
    # order was changed without rerunning the search.
    assert result["worst_cvd"] >= P.CVD_TARGET
    assert result["worst_normal"] >= P.NORMAL_FLOOR


def test_categorical_palette_all_pairs_stays_in_the_documented_band() -> None:
    """Scatter-style use compares any two series, not just neighbours.

    The green/purple pair sits at 7.6, inside the 6-8 floor band that is legal
    only with secondary encoding -- which these figures always carry (distinct
    markers, and direct labels where there is room). Below 6 it would be a hard
    fail and the palette would need re-stepping.
    """

    result = P.validate_categorical(style.CATEGORICAL, pairs="all")
    assert result["ok"], P.format_report("categorical (all pairs)", result)
    assert result["worst_cvd"] >= P.CVD_FLOOR
    assert result["worst_normal"] >= P.NORMAL_FLOOR


def test_sequential_ramp_is_a_valid_ordinal_scale() -> None:
    result = P.validate_ordinal(style.SEQUENTIAL)
    assert result["ok"], P.format_report("sequential ramp", result)


def test_entity_colours_come_from_the_validated_palette() -> None:
    """Colour follows the entity, so every entity must map into the palette.

    A hand-written hex here would be outside everything the validator checked.
    """

    unknown = {
        name: value
        for name, value in style.ENTITY.items()
        if value not in style.CATEGORICAL
    }
    assert not unknown, f"entity colours outside the validated palette: {unknown}"


def test_distinct_entities_that_share_a_panel_are_distinguishable() -> None:
    """Series drawn together must clear the CVD floor against each other.

    Grouped by the panel they co-occur in; `ENTITY` deliberately reuses hues
    across *different* figures (blue is both `jaccpot` and `gpu`), which is fine
    because those never share an axis.
    """

    panels = {
        "figure 04 runners": ("jaccpot", "direct", "jaxfmm"),
        "figure 03 MAC arms": ("geometric", "mass", "mass_16b"),
        "figure 01 bases": ("real", "solidfmm", "complex"),
        "figure 12/13 targets": ("positions", "masses"),
    }
    for panel, names in panels.items():
        colours = [style.ENTITY[n] for n in names]
        assert len(set(colours)) == len(colours), f"{panel}: repeated colour"
        result = P.validate_categorical(colours, pairs="all")
        assert result["worst_cvd"] >= P.CVD_FLOOR, (
            f"{panel}: worst CVD separation {result['worst_cvd']:.1f} is below the "
            f"floor {P.CVD_FLOOR}\n" + P.format_report(panel, result)
        )


def test_required_config_keys_are_what_the_figures_annotate() -> None:
    """The traceability contract is a single list; pin it deliberately.

    Dropping a key here would let an artifact be written without recording the
    axis it was measured at, which is how a manuscript number stops being
    reproducible.
    """

    from examples.jaccpot_paper.common.jsonio import REQUIRED_CONFIG_KEYS

    assert set(REQUIRED_CONFIG_KEYS) == {
        "n",
        "theta",
        "order",
        "basis",
        "seed",
        "device",
        "precision",
    }
