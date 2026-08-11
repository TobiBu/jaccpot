"""The large-N lane's environment configuration, read once per solver.

WHY THIS IS ITS OWN MODULE. ``_large_n_pipeline.py`` is the least-verified
production lane in the tree -- ``prepare_large_n_state`` is 1253 lines behind a
``jax.default_backend() == "gpu"`` gate, and its reverse path
(``_large_n_grad.py``) sits at 0% coverage on CPU CI. A.9 argues, and this change
respects, that the *rest* of that module stays whole until it has a
characterization net. This reader is the one piece that is pure host-side
configuration: it touches no array, traces nothing, and is fully exercisable on a
CPU, so it can be moved and read on its own.

WHY THE PARSERS ARE LOCAL AND HAND-ROLLED. The three closures below
(``_env_bool``, ``_env_pos_int``, ``_canonical_static_int``) deliberately do not
route through :mod:`jaccpot._env`. Their malformed-value semantics differ: these
fall back to the *stated default*, while ``_env.env_flag`` returns ``False``. For
a default-on knob that is the difference between "a typo left the feature on" and
"a typo silently turned it off", and on this lane the knobs select which kernel
and which capacity a multi-million-particle run uses. A.4/F13 records this as a
decision that needs sign-off (Tier 2.2), not a cleanup -- so the semantics are
unchanged here.

``_canonical_static_int`` additionally snaps a requested value up to the nearest
member of an allowed ladder, because each distinct capacity is a distinct
compiled shape: a free-running integer would recompile per N.

Extracted verbatim from ``_large_n_pipeline.py`` (Tier 1.8).
"""

from __future__ import annotations

import os
from typing import Any

__all__: list[str] = []


def _read_large_n_env_config() -> dict[str, Any]:
    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _env_pos_int(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except Exception:
            value = int(default)
        return max(1, int(value))

    def _canonical_static_int(
        value_env: str,
        default_value: int,
        options_env: str,
        default_options: str,
    ) -> int:
        try:
            raw_value = int(os.environ.get(value_env, str(default_value)))
        except Exception:
            raw_value = int(default_value)
        options_raw = str(os.environ.get(options_env, default_options)).strip()
        options: list[int] = []
        for token in options_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                val = int(token)
            except Exception:
                continue
            if val > 0 and val not in options:
                options.append(val)
        if not options:
            options = [int(default_value)]
        if raw_value in options:
            return int(raw_value)
        return int(min(options, key=lambda v: (abs(v - raw_value), v)))

    overflow_profile_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM",
        "2.0",
    )
    try:
        overflow_profile_headroom = max(1.0, float(overflow_profile_headroom_raw))
    except Exception:
        overflow_profile_headroom = 2.0
    overflow_profile_caps_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_CAP_OPTIONS",
        "64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    overflow_profile_caps: list[int] = []
    for token in str(overflow_profile_caps_raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in overflow_profile_caps:
            overflow_profile_caps.append(value)
    overflow_profile_caps = sorted(overflow_profile_caps)
    if not overflow_profile_caps:
        overflow_profile_caps = [64, 128, 256, 512, 1024]

    neighbor_profile_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM",
        "1.0",
    )
    try:
        neighbor_profile_headroom = max(1.0, float(neighbor_profile_headroom_raw))
    except Exception:
        neighbor_profile_headroom = 1.0
    neighbor_profile_caps_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_CAP_OPTIONS",
        "4096,8192,12288,16384,20480,24576,28672,32768,49152,65536,98304,131072",
    )
    neighbor_profile_caps: list[int] = []
    for token in str(neighbor_profile_caps_raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in neighbor_profile_caps:
            neighbor_profile_caps.append(value)
    neighbor_profile_caps = sorted(neighbor_profile_caps)
    if not neighbor_profile_caps:
        neighbor_profile_caps = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
    neighbor_profile_bootstrap_cap_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP",
        "0",
    )
    try:
        neighbor_profile_bootstrap_cap = max(
            0,
            int(neighbor_profile_bootstrap_cap_raw),
        )
    except Exception:
        neighbor_profile_bootstrap_cap = 0
    overflow_profile_bootstrap_cap_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP",
        "0",
    )
    try:
        overflow_profile_bootstrap_cap = max(
            0,
            int(overflow_profile_bootstrap_cap_raw),
        )
    except Exception:
        overflow_profile_bootstrap_cap = 0

    static_runtime_fixed_sizing = _env_bool(
        "JACCPOT_STATIC_RUNTIME_FIXED_SIZING",
        True,
    )
    try:
        overflow_profile_fixed_cap = max(
            0,
            int(
                os.environ.get(
                    "JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP",
                    str(int(overflow_profile_bootstrap_cap)),
                )
            ),
        )
    except Exception:
        overflow_profile_fixed_cap = int(max(0, int(overflow_profile_bootstrap_cap)))
    try:
        neighbor_profile_fixed_cap = max(
            0,
            int(
                os.environ.get(
                    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
                    str(int(neighbor_profile_bootstrap_cap)),
                )
            ),
        )
    except Exception:
        neighbor_profile_fixed_cap = int(max(0, int(neighbor_profile_bootstrap_cap)))

    # Static target-block cap. Supports "auto" (data-driven sizing; sentinel 0)
    # in addition to explicit ints, and — for any value — auto-grows to fit the
    # densest leaf at build time (see _large_n_pipeline static-block region),
    # mirroring the neighbor/overflow cap profiling (headroom + caps ladder).
    static_target_blocks_cap_raw = (
        str(os.environ.get("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32"))
        .strip()
        .lower()
    )
    static_target_blocks_auto = static_target_blocks_cap_raw in {"auto", "-1"}
    if static_target_blocks_auto:
        static_target_blocks_max_per_leaf = 0
    else:
        try:
            static_target_blocks_max_per_leaf = max(
                1, int(static_target_blocks_cap_raw)
            )
        except Exception:
            static_target_blocks_max_per_leaf = 0
            static_target_blocks_auto = True
    static_target_blocks_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_HEADROOM", "1.25"
    )
    try:
        static_target_blocks_headroom = max(
            1.0, float(static_target_blocks_headroom_raw)
        )
    except Exception:
        static_target_blocks_headroom = 1.25
    static_target_blocks_cap_options: list[int] = []
    for token in str(
        os.environ.get(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF_OPTIONS",
            "8,16,32,64,128,256,512,1024,2048,4096",
        )
    ).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in static_target_blocks_cap_options:
            static_target_blocks_cap_options.append(value)
    static_target_blocks_cap_options = sorted(static_target_blocks_cap_options)
    if not static_target_blocks_cap_options:
        static_target_blocks_cap_options = [8, 16, 32, 64, 128, 256, 512, 1024]

    return {
        "nearfield_delayed_scatter_chunks_per_superchunk": _env_pos_int(
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS", 1
        ),
        "nearfield_chunk_scan_batch_size": _env_pos_int(
            "JACCPOT_LARGE_N_CHUNK_SCAN_BATCH_SIZE", 1
        ),
        "nearfield_chunk_scan_unroll": _env_pos_int(
            "JACCPOT_LARGE_N_CHUNK_SCAN_UNROLL", 1
        ),
        "nearfield_superchunk_scan_unroll": _env_pos_int(
            "JACCPOT_LARGE_N_SUPERCHUNK_SCAN_UNROLL", 1
        ),
        "nearfield_sorted_scatter_hint": _env_bool(
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT", False
        ),
        "nearfield_grouped_sorted_scatter": _env_bool(
            "JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER", False
        ),
        "nearfield_superchunk_target_reduce": _env_bool(
            "JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE", False
        ),
        "nearfield_disable_chunk_cond": _env_bool(
            "JACCPOT_LARGE_N_DISABLE_CHUNK_COND", True
        ),
        "nearfield_target_leaf_batch_size": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE",
            16,
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_OPTIONS",
            "16,32,64",
        ),
        "nearfield_target_block_tile_size": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE",
            4,
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_OPTIONS",
            "4,8,16",
        ),
        "nearfield_target_block_tile_scan_unroll": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL",
            1,
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL_OPTIONS",
            "1,2,4",
        ),
        "nearfield_target_block_batch_scan_unroll": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL",
            1,
            "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL_OPTIONS",
            "1,2,4",
        ),
        "nearfield_target_block_overflow_fast_max_blocks": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS",
            65536,
            "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS_OPTIONS",
            "16384,32768,65536,131072",
        ),
        "static_target_blocks_enabled": _env_bool(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", True
        ),
        "static_target_blocks_max_per_leaf": int(static_target_blocks_max_per_leaf),
        "static_target_blocks_auto": bool(static_target_blocks_auto),
        "static_target_blocks_headroom": float(static_target_blocks_headroom),
        "static_target_blocks_cap_options": tuple(
            int(v) for v in static_target_blocks_cap_options
        ),
        "overflow_profile_headroom": float(overflow_profile_headroom),
        "overflow_profile_caps": tuple(int(v) for v in overflow_profile_caps),
        "neighbor_profile_headroom": float(neighbor_profile_headroom),
        "neighbor_profile_caps": tuple(int(v) for v in neighbor_profile_caps),
        "neighbor_profile_bootstrap_cap": int(neighbor_profile_bootstrap_cap),
        "overflow_profile_bootstrap_cap": int(overflow_profile_bootstrap_cap),
        "static_runtime_fixed_sizing": bool(static_runtime_fixed_sizing),
        "overflow_profile_fixed_cap": int(overflow_profile_fixed_cap),
        "neighbor_profile_fixed_cap": int(neighbor_profile_fixed_cap),
        "disable_specialized_large_n_nearfield": _env_bool(
            "JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD", False
        ),
    }


def _large_n_env_config_for_fmm(fmm: object) -> dict[str, Any]:
    cfg = getattr(fmm, "_large_n_env_config_cached", None)
    if cfg is None:
        cfg = _read_large_n_env_config()
        setattr(fmm, "_large_n_env_config_cached", cfg)
    return cfg
