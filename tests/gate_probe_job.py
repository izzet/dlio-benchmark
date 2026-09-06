#!/usr/bin/env python3
"""On-cluster proof that the fixed gate binds under fork (T3-0 4s.26).

Loads the REAL loader module from the gate-fix worktree, narrows the gate the way an arm does, forks
LIVEFLOW_MAX_READ_THREADS workers per rank, and reports the worker probe line and the peak
instantaneous concurrency the gate actually allowed. No corpus is read, so this is genuinely
gate-class: it costs no Lustre bandwidth and cannot confound a measurement beside it.
"""
import importlib.util
import multiprocessing as mp
import os
import socket
import sys
import time

OVERRIDE = os.environ["T30_DLIO_OVERRIDE"]
sys.path.insert(0, OVERRIDE)
p = os.environ.get("LIVEFLOW_SHM_PYTHONPATH")
if p:
    sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location(
    "lfdl", os.path.join(OVERRIDE, "dlio_benchmark", "data_loader", "liveflow_data_loader.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

HOST = socket.gethostname()
NW = int(os.environ.get("LIVEFLOW_MAX_READ_THREADS", "24"))
g = m.get_gate()
print("[%s] import-time gate: %s id=%s limit=%s max=%s"
      % (HOST, "present" if g else "NONE", hex(id(g)) if g else "n/a",
         g.limit if g else "n/a", getattr(g, "_max", "n/a") if g else "n/a"), flush=True)
if g is None:
    print("[%s] FAIL: no gate at import" % HOST, flush=True)
    sys.exit(2)

peak, cur, lock = mp.Value("i", 0), mp.Value("i", 0), mp.Lock()
ids = mp.Array("l", NW)


def worker(i):
    gg = m.get_gate()
    ids[i] = 0 if gg is None else id(gg)
    if gg is None:
        os._exit(3)
    for _ in range(60):
        tok = gg.acquire()
        with lock:
            cur.value += 1
            if cur.value > peak.value:
                peak.value = cur.value
        time.sleep(0.003)
        with lock:
            cur.value -= 1
        gg.release(tok)


for limit in (int(os.environ.get("PROBE_LIMIT_A", "4")), int(os.environ.get("PROBE_LIMIT_B", "1"))):
    g.set_limit(limit)
    peak.value = 0
    ps = [mp.Process(target=worker, args=(i,)) for i in range(NW)]
    for q in ps:
        q.start()
    for q in ps:
        q.join()
    shared = len({v for v in ids if v}) == 1
    ok = (peak.value == limit) and shared and all(q.exitcode == 0 for q in ps)
    print("[%s] limit=%d  workers=%d  PEAK=%d  shared_gate_id=%s  -> %s"
          % (HOST, limit, NW, peak.value, shared, "PASS" if ok else "FAIL"), flush=True)
