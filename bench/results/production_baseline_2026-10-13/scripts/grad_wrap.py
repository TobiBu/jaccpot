"""Run bench/grad_step_profile.py in-process, then print the device peak (the bench records none)."""

import json
import runpy
import sys

out_peak = sys.argv[1]
sys.argv = ["grad_step_profile.py"] + sys.argv[2:]
try:
    runpy.run_path("bench/grad_step_profile.py", run_name="__main__")
except SystemExit as e:
    print(f"[grad_wrap] exit {e.code}", flush=True)
finally:
    import jax

    st = jax.devices()[0].memory_stats() or {}
    pk = st.get("peak_bytes_in_use", 0)
    print(f"[grad_wrap] process device peak {pk/2**30:.2f} GiB", flush=True)
    json.dump({"peak_bytes": pk}, open(out_peak, "w"))
