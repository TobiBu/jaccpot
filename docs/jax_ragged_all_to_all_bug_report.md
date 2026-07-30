# Draft upstream bug report: jit + ragged_all_to_all + grad

Outbound draft for <https://github.com/jax-ml/jax/issues>, not project
documentation. Paste as-is (trim to taste); the reproducer is
`bench/repro_jax_ragged_all_to_all_grad.py`. Background and how it was found:
`docs/differentiable_fmm_distributed_audit.md`.

No matching issue was found on 2026-07-30 — search before filing in case one has
appeared since.

---

**Title:** `jax.lax.ragged_all_to_all` under `jit(shard_map(...))` returns the
un-written output buffer on every call after a `grad` has executed

### Description

Take a `jax.jit`-wrapped `shard_map` containing a single
`jax.lax.ragged_all_to_all`. Evaluate it, execute a gradient of it, then evaluate
it again: the second evaluation returns the *fill value* the output operand was
initialised to, i.e. the exchange no longer writes its output. The damage is
persistent for the process and is not confined to the callable that was
differentiated.

The un-jitted `shard_map` is unaffected, and merely *tracing* the gradient
(`jax.make_jaxpr(jax.grad(f))`) does no damage — only executing it.

### Reproducer

```python
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map

NDEV, CAP, FILL = 2, 4, -1.0
mesh = Mesh(np.array(jax.devices()[:NDEV]), ("gpus",))
so = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))
io = jnp.asarray(np.array([[0, 2], [0, 2]], np.int32))
oo = jnp.asarray(np.array([[0, 0], [2, 2]], np.int32))
ro = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))

def body(x, s, i, o, r):
    out = jnp.full((CAP,), FILL, x.dtype)
    return jax.lax.ragged_all_to_all(x, out, i[0], s[0], o[0], r[0], axis_name="gpus")

run = jax.jit(lambda x: shard_map(          # remove jax.jit -> problem disappears
    body, mesh=mesh, in_specs=(P("gpus"),) * 5, out_specs=P("gpus"),
    check_vma=False)(x, so, io, oo, ro))

x = jnp.asarray(np.arange(1.0, NDEV * CAP + 1.0))
print("before:", run(x))
jax.block_until_ready(jax.grad(lambda v: jnp.sum(run(v) ** 2))(x))
print("after :", run(x))
```

### Observed

```
before: [1. 2. 5. 6. 3. 4. 7. 8.]
after : [1. 2. 5. 6. -1. -1. -1. -1.]     # -1.0 is the fill value
```

Expected: `after == before`.

### Notes from bisecting it

* **`jax.jit` is the trigger.** The bare `shard_map` is clean; wrapping it in
  `jax.jit` is what breaks it.
* Deterministic per process: 6/6 runs corrupt. But **run one configuration per
  process** when checking — several jit+grad sequences in one process perturb each
  other, and a case that corrupts in isolation can look clean if another ran first.
* *Which* rows are lost varies run to run (either device's half, sometimes both);
  that a jitted run corrupts does not vary.
* Not sensitive to whether the output operand is a compile-time constant: tying it
  to a traced input (`jnp.full(...) + jnp.zeros_like(x[:CAP]) * x[:CAP]`) corrupts
  identically, and there every row comes back as fill.
* The gradient *values* are correct — `jax.grad` matches finite differences to
  ~1e-10. Only later forwards are affected.
* Substituting an `all_gather`-based or `all_to_all`-based exchange for the same
  logical operation is correct and stable under the same jit+grad usage, so this
  looks specific to `ragged_all_to_all`.
* Not reproduced with `psum` / `all_gather` / `all_to_all` / `ppermute`, all of
  which transpose correctly here.

### Environment

* jax 0.9.0.1, jaxlib 0.9.0.1, CUDA backend
* 2x NVIDIA A100-PCIE-40GB
* Linux 6.8.0, Python 3.12, `JAX_ENABLE_X64=1`

### Impact

This silently produces wrong *forward* results after any gradient step, with no
error — in our case a multi-GPU N-body solver lost its entire cross-domain
near-field interaction and reproduced a local-only force to 2e-16, i.e. a 42%
error against the true force, on evaluators built before the gradient as well.
Anything that alternates gradients and forward evaluations (an optimisation loop,
training) is exposed.
