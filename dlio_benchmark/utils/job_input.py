"""Job-level, critical-path input exposure for one analysis window (LiveFlow job-level layer).

A bulk-synchronous step ends in an all-rank barrier, so one rank's rare input stall is paid by every
other rank as barrier wait.  A per-node analyzer only sees its own ranks' fetch waits, so it books that
wait as `comm`.  At each window boundary every rank therefore reduces its per-step fetch-wait vector
over the job (one Allreduce(MAX) over the K steps of the window, plus one scalar Allreduce(SUM)) and
emits two job-level window facts on its own SHM lane:

    input_step_max_sum_job  (cat input_exposure_job)  time = critical_input = sum_steps max_rank fetch_wait
    input_own_mean_job      (cat input_exposure_job)  time = own_input      = mean_rank sum_steps fetch_wait

Their difference, critical_input - own_input, is the peer-skew input the typical rank paid at the barrier.
Every rank holds the identical reduced values, so any node's analyzer sees the job-level number.  The
reduction reuses the application's own communicator at the window boundary (the same point at which
the window agreement already runs); no new process group or service.

Opt-in: LIVEFLOW_JOB_INPUT_SYNC=1.  Default off keeps every run free of the extra collectives.
"""
import os

_INT_TAG = lambda value: (0, int(value))
FAMILY = "input_exposure_job"
CRITICAL_NAME = "input_step_max_sum_job"
OWN_NAME = "input_own_mean_job"


def enabled():
    return os.environ.get("LIVEFLOW_JOB_INPUT_SYNC", "0") == "1"


def reduce_window(comm, steps, window_steps, op_max, op_sum):
    """Return (critical_s, own_mean_s, n_valid) from this rank's per-step waits in seconds.

    `window_steps` is the window's step count, identical on every rank by construction (counter-based
    windows).  A rank holding fewer waits than that pads with zeros and reports it through n_valid.
    """
    import numpy as np
    k = int(window_steps)
    local = np.zeros(k + 1, dtype=np.float64)
    n_local = min(len(steps), k)
    if n_local:
        local[:n_local] = steps[:n_local]
    local[k] = float(n_local)
    out = np.empty_like(local)
    comm.Allreduce(local, out, op=op_max)
    own = np.array([float(local[:k].sum())], dtype=np.float64)
    own_sum = np.empty_like(own)
    comm.Allreduce(own, own_sum, op=op_sum)
    size = max(int(comm.Get_size()), 1)
    return float(out[:k].sum()), float(own_sum[0]) / size, int(out[k])


def emit(tracer, epoch, step, critical_s, own_mean_s, n_valid):
    """One event per quantity on this rank's lane, ordered before window.stop by the caller."""
    now = tracer.get_time()
    for name, seconds, extra in ((CRITICAL_NAME, critical_s, {"step_count": _INT_TAG(n_valid)}),
                                 (OWN_NAME, own_mean_s, {})):
        dur_us = max(0, int(round(seconds * 1_000_000.0)))
        tracer.log_event(name=name, cat=FAMILY, start_time=now - dur_us, duration=dur_us,
                         int_args={"epoch": _INT_TAG(epoch), "step": _INT_TAG(step), **extra},
                         float_args={}, string_args={})


def sync_window(comm, loader, window, epoch, step, tracer):
    """Drain the loader's per-step waits, reduce them over the job and emit the window facts.

    Returns (critical_s, own_mean_s) or None when the loader keeps no per-step account.
    """
    drain = getattr(loader, "drain_step_stalls", None)
    if not callable(drain) or window is None:
        return None
    steps = drain()
    k = getattr(window, "pending_steps", 0) or len(steps)
    if comm is None or int(comm.Get_size()) <= 1:
        critical, own, n_valid = float(sum(steps)), float(sum(steps)), len(steps)
    else:
        from mpi4py import MPI
        critical, own, n_valid = reduce_window(comm, steps, k, MPI.MAX, MPI.SUM)
    emit(tracer, epoch, step, critical, own, n_valid)
    return critical, own


def node_max(comm, value):
    """Worst value over the node's ranks (they reach every window boundary together, make_loader's bcast already
    makes the boundary a node-local collective).  The analyzer requires the ranks of a node to publish an identical
    applied vector, apply cost included; a rank-local latency would make the whole vector discordant and the planner
    would receive nothing."""
    if comm is None or int(comm.Get_size()) <= 1:
        return value
    return comm.allreduce(value, op=max)
