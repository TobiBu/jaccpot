"""Keep one GPU reserved between benchmark runs.

usage: CUDA_VISIBLE_DEVICES=6 python guard.py CTL ACK [HOLD_GIB]

Keeps a CUDA context on the card for the whole series, so the card never looks
free to schedulers (autocvd treats any compute process as busy). It reads the
word in CTL every half second:

* ``hold``  -- cudaMalloc up to HOLD_GIB (1 GiB chunks) so least-used schedulers
  look elsewhere between runs;
* ``yield`` -- cudaFree every chunk and keep only the context, while a measured
  run owns the card;
* ``exit``  -- release everything and stop.

After each transition it writes the state it reached to ACK. Raw cudaMalloc /
cudaFree through ctypes: memory goes back to the driver at once (jax's own
allocator kept the freed blocks).
"""

import ctypes
import os
import sys
import time

LIB = "/export/home/tbuck/jaccpot/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
cuda = ctypes.CDLL(LIB)
ctl, ack = sys.argv[1], sys.argv[2]
hold_gib = int(sys.argv[3]) if len(sys.argv) > 3 else 32
assert cuda.cudaFree(ctypes.c_void_p(0)) == 0, "no CUDA context"
held: list = []
state = None


def _ack(word: str) -> None:
    tmp = ack + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(word)
    os.replace(tmp, ack)


def _release() -> None:
    while held:
        cuda.cudaFree(held.pop())
    cuda.cudaDeviceSynchronize()


while True:
    try:
        want = open(ctl).read().strip()
    except OSError:
        want = "hold"
    if want not in ("hold", "yield", "exit"):
        want = "hold"
    if want != state:
        _release()
        if want == "exit":
            _ack("exit")
            break
        if want == "hold":
            for _ in range(hold_gib):
                ptr = ctypes.c_void_p()
                if cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(1 << 30)) != 0:
                    break  # hold what fits
                held.append(ptr)
        state = want
        _ack(f"{state} {len(held)}")
    time.sleep(0.5)
