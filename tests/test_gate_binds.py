#!/usr/bin/env python3
"""The LiveFlow gate must bind across forked workers (T3-0 4s.25, 4s.26).

Measured on dev2 before this fix: mean instantaneous read concurrency 56.0, peak 64, on a node
whose four per-rank gates were set to limit=1 -- a ceiling of 4. The gate was created on the first
make_loader call, so under the pool-once policy a worker forked by that pool could hold None and
read through __getitem__'s ungated branch, or hold a private copy that admits itself every time.

    LIVEFLOW_SHM_PYTHONPATH=<liveflow_shm/python> LIVEFLOW_MAX_READ_THREADS=24 \
        python3 tests/test_gate_binds.py
"""
import importlib.util
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "dlio_benchmark", "data_loader", "liveflow_data_loader.py")
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))


def _load():
    """Import the loader module. The repo root goes on the path so its own package imports resolve
    without a full DLIO install, which is what makes this runnable off-cluster."""
    p = os.environ.get("LIVEFLOW_SHM_PYTHONPATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)
    root = os.path.abspath(os.path.join(HERE, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("lfdl", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _peak_concurrency(m, limit, nworkers=24, iters=40):
    g = m.get_gate()
    g.set_limit(limit)
    peak, cur, lock = mp.Value("i", 0), mp.Value("i", 0), mp.Lock()

    def worker(_):
        gg = m.get_gate()
        if gg is None:
            os._exit(3)                      # a forked worker with no gate is the defect
        for _ in range(iters):
            tok = gg.acquire()
            with lock:
                cur.value += 1
                if cur.value > peak.value:
                    peak.value = cur.value
            time.sleep(0.002)
            with lock:
                cur.value -= 1
            gg.release(tok)

    ps = [mp.Process(target=worker, args=(i,)) for i in range(nworkers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return peak.value, [p.exitcode for p in ps]


def main():
    os.environ.setdefault("LIVEFLOW_MAX_READ_THREADS", "24")
    m = _load()
    g = m.get_gate()
    check("the gate exists at IMPORT time, before any fork", g is not None)
    check("and it is built wide open, so importing throttles nothing",
          g.limit == g._max, "limit=%s max=%s" % (g.limit, g._max))

    for limit in (4, 1):
        peak, codes = _peak_concurrency(m, limit)
        check("at limit=%d the peak concurrency across 24 forked workers is %d" % (limit, limit),
              peak == limit, "peak=%d" % peak)
        check("and every worker saw the shared gate", all(c == 0 for c in codes), str(codes))

    # A limit change made AFTER the workers exist must reach them: that is the pool-once contract.
    peak, _ = _peak_concurrency(m, 2)
    check("a limit change after the fork reaches every worker", peak == 2, "peak=%d" % peak)

    print("\n%d/%d checks passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
