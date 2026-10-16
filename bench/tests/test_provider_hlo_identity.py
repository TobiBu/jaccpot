"""`bench/provider_hlo_identity.py`'s capture and comparison, on toy programs.

The S2 refactor is gated on this tool finding NO difference, so the ways it could
find none while missing something -- or perturb what it records -- are pinned here:

* a program jitted while the wrapper is active is captured when it runs at the top
  level, and traced ONCE: the lowering the tool saves and the call share jax's trace
  cache. If they did not, the library's trace-time side effects (the engine's traced
  caps, read by the next trace) would run twice and the second trace -- the one that
  runs -- could differ from the text saved;
* a jitted function called inside another trace is part of that program, not a
  program of its own;
* a donated argument is digested before the call consumes it;
* ``compare`` fails on one bit of one array (and counts the elements), on one
  character of a program's text, and on a renamed program, and says when the module
  name is the only difference;
* ``--control`` forgives only what the A-vs-A control also shows;
* the compiled text carries no source lines (any moved line on the stack, the
  tool's own included, changed it between two otherwise identical GPU records).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bench import provider_hlo_identity as phi


def _run(fn, *args, keep: bool = True):
    cap = phi.Capture(keep_arrays=keep)
    with phi.wrapped_jit():
        jitted = jax.jit(fn)
        with phi.capturing(cap):
            out = jitted(*args)
    return cap, out


def test_a_top_level_program_is_captured_and_traced_once():
    traces = []

    def body(x):
        traces.append(1)
        return x * 2.0

    real = jax.jit
    cap, out = _run(body, jnp.arange(3.0))
    assert jax.jit is real  # restored
    assert len(traces) == 1
    assert [p.label for p in cap.programs] == ["00_jit_body"]
    assert "stablehlo.multiply" in cap.programs[0].lowered
    np.testing.assert_array_equal(np.asarray(out), [0.0, 2.0, 4.0])
    (out_key,) = cap.programs[0].outputs
    np.testing.assert_array_equal(cap.arrays[out_key], [0.0, 2.0, 4.0])


def test_a_jit_called_inside_a_trace_is_part_of_that_program():
    cap = phi.Capture()
    with phi.wrapped_jit():
        inner = jax.jit(lambda x: x + 1.0)

        @partial(jax.jit, donate_argnums=(0,))
        def outer(x):
            return inner(x) * 3.0

        with phi.capturing(cap):
            x = jnp.ones(4)
            outer(x)
    assert [p.module for p in cap.programs] == ["jit_outer"]
    (in_key,) = cap.programs[0].inputs
    # digested before the donating call
    np.testing.assert_array_equal(cap.arrays[in_key], np.ones(4))


def test_outside_the_capture_window_nothing_is_recorded():
    cap = phi.Capture()
    with phi.wrapped_jit():
        f = jax.jit(lambda x: x - 1.0)
    f(jnp.ones(2))  # a cached runner called after the window
    with phi.wrapped_jit(), phi.capturing(cap):
        f(jnp.ones(2))  # created in an earlier window, run in this one
    assert len(cap.programs) == 1


def _record(tmp_path, name, fn, x):
    cap, _ = _run(fn, x)
    rec = {"programs": phi.program_rows(cap), "returns": {}}
    phi.write_variant(tmp_path / name, rec, cap)
    return tmp_path / name


def test_compare_passes_identical_records_and_counts_flipped_bits(tmp_path):
    def body(x):
        return x * 2.0

    x = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    a = _record(tmp_path, "a", body, x)
    b = _record(tmp_path, "b", body, x)
    assert phi.compare_variant(a, b)["pass"]

    flipped = np.asarray(x).copy()
    flipped[3] = np.nextafter(flipped[3], np.float32(2.0))
    c = _record(tmp_path, "c", body, jnp.asarray(flipped))
    entry = phi.compare_variant(a, c)
    assert not entry["pass"]
    assert "texts" not in entry  # the same program
    counts = {key: row.get("n_diff") for key, row in entry["arrays"].items()}
    assert sorted(counts.values()) == [1, 1]  # the input and the output


def test_compare_fails_on_one_character_of_text_and_names_a_rename(tmp_path):
    x = jnp.ones(4)

    def body(x):
        return x * 2.0

    def renamed(x):
        return x * 2.0

    def other(x):
        return x * 3.0

    a = _record(tmp_path, "a", body, x)
    entry = phi.compare_variant(a, _record(tmp_path, "b", renamed, x))
    assert not entry["pass"] and "programs" in entry
    (diff,) = entry["texts"].values()
    assert diff["only_module_name"]

    entry = phi.compare_variant(a, _record(tmp_path, "c", other, x))
    assert not entry["pass"]
    (diff,) = entry["texts"].values()
    assert not diff["only_module_name"]
    assert any("3.0" in line for line in diff["diff_head"])


def test_a_control_turns_its_own_differences_into_noise(tmp_path):
    def body(x):
        return x * 2.0

    x = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    flipped = np.asarray(x).copy()
    flipped[3] = np.nextafter(flipped[3], np.float32(2.0))
    a = _record(tmp_path, "a", body, x)
    same = _record(tmp_path, "same", body, x)
    moved = _record(tmp_path, "moved", body, jnp.asarray(flipped))
    # the difference is reproducible (absent from an identical control): it fails
    assert not phi.compare_variant(a, moved, control_dir=same)["pass"]
    # the control differs the same way: run-to-run noise, reported, not failed
    entry = phi.compare_variant(a, moved, control_dir=moved)
    assert entry["pass"] and len(entry["noise"]) == 2


def test_the_compiled_text_carries_no_source_lines():
    def body(x):
        return jnp.sin(x) * 2.0

    compiled = jax.jit(body).lower(jnp.ones(4)).compile()
    assert "source_line" in compiled.as_text() or "FileNames" in compiled.as_text()
    text = phi._compiled_text(compiled)
    assert "source_line" not in text and "FileNames" not in text
    assert "sine" in text


def test_pair_compares_two_variants_of_one_record(tmp_path, capsys):
    """``compare TAG TAG --pair A:B`` (S3's galaxy pair): A against B, nothing else."""
    import json

    def body(x):
        return x * 2.0

    def other(x):
        return x * 3.0

    x = jnp.ones(4)
    tag = tmp_path / "tag"
    _record(tag, "explicit", body, x)
    _record(tag, "preset", body, x)
    _record(tag, "drifted", other, x)
    (tag / "manifest.json").write_text(
        json.dumps({"variants": {"explicit": {}, "preset": {}, "drifted": {}}})
    )
    argv = ["compare", "tag", "tag", "--root", str(tmp_path), "--pair"]
    assert phi.main(argv + ["explicit:preset"]) == 0
    assert "PASS: 1/1" in capsys.readouterr().out
    assert phi.main(argv + ["explicit:drifted"]) == 1
    assert "differ: explicit:drifted" in capsys.readouterr().out
