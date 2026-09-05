"""LiveFlow data loader: build the workers ONCE, move a concurrency limit instead.

THE PROBLEM THIS EXISTS TO REMOVE
`read_threads` maps to torch's `num_workers`, a CONSTRUCTOR argument. Every change therefore
builds a new DataLoader and forks a new set of workers -- the storage-knob fork storm of
sec 71/72/84, where a single restart could leave 15 of 16 ranks done and take the job down.
Option C removed the LETHALITY (a re-forked worker inherits an mmap, so there is nothing to
re-initialise and nothing to hang) but not the cost: a rebuild still tears down and respawns
every worker mid-epoch, and `LiveDataLoader` has to carry a position-tracking sampler purely to
resume from where the old one stopped.

THE PATTERN IS ALREADY PROVEN HERE
sec 82's decode knob is the one knob that never needed a restart, because it does not resize
anything: a fixed-width pool is built once and `decode_threads` moves a concurrency LIMIT held in
an mmap inherited across fork (`_DECODE_LIMIT`, train.py:130). Three epochs, 1 -> 16, zero
restarts. This is that pattern applied to storage.

WHY WORKERS ARE THROTTLED AND NOT IDLED
torch's DataLoader hands index batches to workers round-robin and consumes results IN ORDER, so a
worker that declines to produce stalls the main loop when its turn comes. You cannot gate which
workers run, only what they do. So the DataLoader is built once at MAX width, every worker stays
in the rotation, and `ShmGate` bounds how many are inside the read at any instant. That is also
the more faithful reading of `read_threads`: concurrent pressure on the filesystem, not a process
count.

WHAT IT TRADES, PLAINLY
Lowering the limit does not remove work, it delays it -- a worker above the limit waits for a slot
rather than not existing. Against a rebuild: no fork, no teardown, no sampler resume, but blocked
workers instead of absent ones. Whether that is the better operating point is empirical, which is
what the ablation in liveflow_shm/ measures. Set LIVEFLOW_GATE=0 to fall back to rebuild
semantics and A/B the two.
"""

import math
import os
import sys
import time


def _mark(msg):
    """Unconditional stderr marker.

    DLIO's logger is reconfigured per rank and per worker and its info records did not reach the
    run log, which made "did this rebuild?" unanswerable from the artefacts. The whole claim of
    this loader is about what does NOT happen, so the evidence has to be unmissable.
    """
    sys.stderr.write("[LIVEFLOW_DL] %s\n" % msg)
    sys.stderr.flush()

import torch
from torch.utils.data import DataLoader

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.consumer_iterator import traced_consumer_batches
from dlio_benchmark.data_loader.torch_data_loader import TorchDataset, dlio_sampler
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai

dlp = Profile(MODULE_DATA_LOADER)

_GATE = None            # module-level so forked workers inherit the SAME object
_PREFETCH_DEPTH = None  # mp.Value, must exist before any fork for the same reason


def get_prefetch_depth():
    return _PREFETCH_DEPTH


def _make_prefetch_depth(initial, max_depth):
    """Shared prefetch depth. Created BEFORE any worker is forked, like the gate."""
    global _PREFETCH_DEPTH
    if _PREFETCH_DEPTH is None:
        import multiprocessing as _mp
        _PREFETCH_DEPTH = _mp.Value("i", max(1, int(initial)), lock=False)
        _PREFETCH_DEPTH_MAX[0] = int(max_depth)
    return _PREFETCH_DEPTH


_PREFETCH_DEPTH_MAX = [32]
# The pool width the NEXT rebuild should use, decided by reconfigure() and applied by read().
# Module level rather than an instance attribute because DLIO's make_loader asks the framework for
# a NEW loader object at every epoch boundary, so a decision stored on the object that made it
# would be dropped exactly when it is due to be honoured. One rank is one process, so this is one
# value per pool.
# _UNSET distinguishes "nothing pending" from a pending "build at MAX", which is spelled None.
_UNSET = object()
_pending_pool_width = [_UNSET]

# THE POOL IS BUILT ONCE PER PROCESS AND IS NEVER TORN DOWN DURING A RUN.
#
# Orchestrator decision 2026-09-04, after the node sweep (T3-0 section 4s.6) showed the retirement
# fatal is in MVAPICH's async event thread and not in any node: 15 of 17 screened pairs died within
# 800 retirement events, and a node that survived one screen failed the next. There is nothing to
# exclude, so the retirement storm is removed by construction instead. A knob change that cannot be
# applied inside the existing pool is REFUSED with a recorded reason rather than rebuilt.
#
# It lives at module scope because DLIO asks the framework for a NEW loader object at every epoch
# boundary. A pool held on the object would be rebuilt every epoch no matter what this file did,
# which is exactly the storm: nine boundaries times 32 ranks is 288 retirement events per 10-epoch
# run, at about 1 fatal in 480.
#
# Sample ORDER is unaffected: dlio_sampler computes a fixed index list at construction with no
# per-epoch shuffle, so a fresh loader per epoch was already iterating the same order.
_POOL = {"dataset": None, "workers": None, "prefetch": None, "prefetch_obj": None}
# Knob moves this policy refused, for the record and for the readout.
_REFUSALS = []


def get_gate():
    return _GATE


def _make_gate(limit, max_limit):
    """Create the process-wide gate. MUST run before any worker is forked.

    Building it lazily inside a worker would give each worker a private copy, and the gate would
    silently throttle nothing -- the failure mode that motivated `_DECODE_LIMIT` being created at
    import time rather than on first use.
    """
    global _GATE
    if _GATE is None:
        import sys
        p = os.environ.get("LIVEFLOW_SHM_PYTHONPATH")
        if p and p not in sys.path:
            sys.path.insert(0, p)
        from shm_gate import ShmGate
        _GATE = ShmGate(limit=limit, max_limit=max_limit)
    return _GATE


class GatedTorchDataset(TorchDataset):
    """TorchDataset whose read is bounded by the shared gate."""

    def __getitem__(self, image_idx):
        gate = get_gate()
        if gate is None:
            return super().__getitem__(image_idx)
        # Explicit token rather than `with gate:`. The context-manager form has to stash whether
        # it actually acquired (the gate has a bypass fast path when the limit cannot bind), and
        # stashing that on the gate instance is only safe because each DataLoader worker is its
        # own PROCESS calling this sequentially. Passing the token makes the contract local and
        # keeps it correct if a caller ever becomes threaded.
        token = gate.acquire()
        try:
            return super().__getitem__(image_idx)
        finally:
            gate.release(token)


class LiveFlowDataLoader(BaseDataLoader):
    """Builds its workers once at MAX width; knob changes move the gate, never the DataLoader."""

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, data_loader_type=DataLoaderType.PYTORCH):
        super().__init__(format_type, dataset_type, epoch, data_loader_type)
        self._dataset = None
        self._built_workers = None
        self._built_prefetch = None
        self._prefetch = None
        # None = build the worker pool at MAX. Set to an int only when reconfigure() decides a
        # deep cut is cheaper to fork than to park.
        self._pool_width = None
        # Critical-path accounting: time the training loop spent BLOCKED waiting for a batch.
        self._stall_s = 0.0
        self._steps = 0
        self._stall_mark = (0.0, 0)   # last reported (stall_s, steps), for per-window deltas
        self._consumer_boundary = None

    @dlp.log
    def read(self, read_threads=None, prefetch_size=None):
        if read_threads is None:
            read_threads = self._args.read_threads
        if prefetch_size is None:
            prefetch_size = self._args.prefetch_size

        max_workers = int(os.environ.get("LIVEFLOW_MAX_READ_THREADS", "8") or 8)
        use_gate = os.environ.get("LIVEFLOW_GATE", "1") == "1"

        if use_gate:
            # The gate carries the knob; the DataLoader is built at MAX and never rebuilt.
            gate = _make_gate(limit=max(1, int(read_threads)), max_limit=max_workers)
            before = gate.limit
            after = gate.set_limit(read_threads)
            if before != after:
                _mark("GATE %d -> %d  (knob applied without touching the DataLoader)"
                      % (before, after))
            # `_pool_width` is normally None, meaning "build at MAX and let the gate carry every
            # change". reconfigure() sets it only to narrow the pool for a deep cut, where parking
            # would cost more than forking (see the note there).
            # ADOPT THE PROCESS POOL IF ONE EXISTS. A new loader object at an epoch boundary
            # takes over the pool that is already running instead of forking a replacement, which
            # is what removes the retirement storm.
            if _POOL["dataset"] is not None:
                self._dataset = _POOL["dataset"]
                self._built_workers = _POOL["workers"]
                self._built_prefetch = _POOL["prefetch"]
                self._prefetch = _POOL["prefetch_obj"]
                _mark("ADOPT pool workers=%s prefetch=%s gate_limit=%s  (built once, never torn "
                      "down during the run)"
                      % (self._built_workers, self._built_prefetch,
                         get_gate().limit if use_gate else "n/a"))
                return
            workers = self._pool_width or max_workers
        else:
            workers = max(0, int(read_threads))

        if self._dataset is not None and self._built_workers == workers:
            # Already built at this width. Under the gate this is EVERY call after the first,
            # which is the entire point: the knob moved and no worker was harmed.
            _mark("REUSE workers=%d gate_limit=%s  (no rebuild, no fork)"
                  % (workers, get_gate().limit if use_gate else "n/a"))
            return

        dataset = GatedTorchDataset(self.format_type, self.dataset_type, self.epoch_number,
                                    self.num_samples, workers, self.batch_size) \
            if use_gate else \
            TorchDataset(self.format_type, self.dataset_type, self.epoch_number,
                         self.num_samples, workers, self.batch_size)
        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size,
                               self.num_samples, self._args.epochs)

        if workers >= 1:
            prefetch_factor = max(1, math.ceil(prefetch_size / workers))
        else:
            prefetch_factor = max(1, prefetch_size)

        kwargs = {}
        if workers > 0:
            kwargs = {"multiprocessing_context": self._args.multiprocessing_context,
                      "prefetch_factor": prefetch_factor,
                      "persistent_workers": True,
                      "worker_init_fn": dataset.worker_init}

        self._dataset = DataLoader(dataset,
                                   batch_size=self.batch_size,
                                   sampler=sampler,
                                   num_workers=workers,
                                   pin_memory=self._args.pin_memory,
                                   drop_last=True,
                                   **kwargs)
        self._built_workers = workers
        self._built_prefetch = prefetch_size
        _POOL["dataset"] = self._dataset
        _POOL["workers"] = workers
        _POOL["prefetch"] = prefetch_size

        if os.environ.get("LIVEFLOW_DYNAMIC_PREFETCH", "0") == "1":
            import sys as _sys
            _p = os.environ.get("LIVEFLOW_SHM_PYTHONPATH")
            if _p and _p not in _sys.path:
                _sys.path.insert(0, _p)
            from dynamic_prefetch import DynamicPrefetch
            depth = _make_prefetch_depth(prefetch_size or 1,
                                         int(os.environ.get("LIVEFLOW_MAX_PREFETCH", "32")))
            self._prefetch = DynamicPrefetch(self._dataset, depth=depth,
                                             max_depth=_PREFETCH_DEPTH_MAX[0])
            _POOL["prefetch_obj"] = self._prefetch
            _mark("DYNAMIC PREFETCH on, depth=%d (prefetch_size is now a live knob)"
                  % self._prefetch.depth)

        _mark("BUILT workers=%d gate=%s limit=%s"
              % (workers, "on" if use_gate else "off",
                 get_gate().limit if use_gate else "n/a"))

    @dlp.log
    def reconfigure(self, read_threads=None, prefetch_size=None):
        """Apply a knob change IN PLACE. This is what makes the change genuinely mid-epoch.

        DLIO's make_loader looks for this method: if the cached loader has one it is called and
        the SAME loader object is returned, so the `batch_iter` generator already running inside
        the training loop stays alive. Without it, a knob change constructs a new loader, the old
        iterator is discarded, and `LiveDataLoader` needs a position-tracking sampler purely to
        resume from where the old one stopped.

        Here the whole reconfigure is an integer store into a gate that every worker already
        shares. No fork, no teardown, no iterator restart, no sampler resume.
        """
        if read_threads is None:
            return
        gate = get_gate()
        if gate is None:
            # GATE OFF: this is the rebuild control arm, and it must actually REBUILD. Silently
            # returning here made DLIO take the reconfigure path and drop the knob entirely --
            # the control arm applied 1 of 3 knob changes and looked slower for the wrong reason.
            # An in-place reconfigure that does not reconfigure is worse than not having one.
            # The gate-off control arm is the one place a rebuild is the point, so it is the ABLATION
            # this policy is judged against. It carries the retirement risk by design.
            if os.environ.get("LIVEFLOW_ALLOW_POOL_TEARDOWN", "0") != "1":
                _REFUSALS.append(("rebuild_control_arm", None, read_threads,
                                  "teardown not permitted; set LIVEFLOW_ALLOW_POOL_TEARDOWN=1"))
                _mark("REFUSED gate=off rebuild: the control arm needs a teardown, which is not "
                      "permitted during a run. Set LIVEFLOW_ALLOW_POOL_TEARDOWN=1 to run it.")
                return
            _mark("RECONFIGURE gate=off -> falling back to a real rebuild (control arm)")
            self._dataset = None
            self._built_workers = None
            self.read(read_threads=read_threads, prefetch_size=prefetch_size)
            return
        # A DEEP CUT IS CHEAPER TO FORK THAN TO PARK, AND THAT IS MEASURED, NOT ASSUMED.
        #
        # Parking a worker does not remove its turn: torch's DataLoader delivers batches in WORKER
        # ORDER, so a parked worker head-of-line blocks the consumer while the rest of the pool has
        # capacity. Phase 20 priced this -- applying the same four plans cost +19.2% through the
        # gate against +3.2% through a rebuild, purely because the sequence cut concurrency to 2
        # against 8 forked workers.
        #
        # Making the gate FIFO did NOT fix it (phase 21). Fair admission orders waiters by ARRIVAL,
        # but the consumer is waiting on a specific WORKER, so a fairly-admitted worker is still
        # the wrong one. That is why the fix has to be here rather than in the gate.
        #
        # Shallow cuts are fine and must stay free: the pool is built at MAX and the steady state
        # already sits below it (read_threads=4 of 8 parks half the pool and costs nothing
        # measurable). Only a deep cut is worth a fork, so the pool tracks the limit only when the
        # limit drops below `LIVEFLOW_GATE_PARK_RATIO` of it. The trade is explicit: one fork now,
        # against every subsequent batch paying head-of-line blocking.
        built = self._built_workers or 0
        want = max(1, int(read_threads))
        before = gate.limit
        ratio = float(os.environ.get("LIVEFLOW_GATE_PARK_RATIO", "0.5"))
        shrink = os.environ.get("LIVEFLOW_GATE_SHRINK", "1") == "1"

        # RECONFIGURE MUST NOT TEAR DOWN THE POOL THE CONSUMER IS DRAWING FROM. THIS WEDGED TWO
        # RUNS AND COST TWO ALLOCATIONS.
        #
        # This method exists precisely because make_loader returns the SAME loader object so the
        # `batch_iter` generator already running inside the training loop stays alive. That
        # generator holds `src`, the DataLoader object captured when it started. Calling
        # shutdown_workers() here kills that object's worker processes and then sets
        # `self._dataset` to a NEW DataLoader, so the live generator is left waiting on a queue
        # that no worker will ever fill again. The consumer blocks forever, which is exactly what
        # was observed: un_forkgate_n2 (f5z2E1iB7wMq) and un_forkgate2_n2 (f5z3GQv8S7eF) both went
        # silent immediately after a reconfigure, completed about 7 of 432 steps, and had to be
        # killed by a wall limit. The epoch-boundary control un_forkgate3_n2 (f5z3nMDcP8ns), which
        # applies knobs only where the loader is rebuilt anyway, completed the same epoch in 544 s.
        #
        # AND THE SHRINK TEST FIRED WHEN NOTHING HAD CHANGED, which is what made it reachable at
        # all. It compared the requested limit against the FORKED WIDTH, but the pool is built at
        # MAX on purpose and the steady state legitimately sits below it, so at MAXW=24 with the
        # shipped read_threads=4 the test `4 < 24 * 0.5` is true on every single call. The plan
        # that wedged un_forkgate2_n2 moved PREFETCH and left read_threads at 4, and a pool
        # teardown ran anyway. The comparison belongs against the limit in force, which is what a
        # cut means.
        #
        # So: pool changes are DEFERRED to the next epoch boundary, where make_loader retires the
        # old pool and builds a new one with no iterator in flight. Mid-epoch the knob still moves,
        # through the gate, which is the whole design of this loader.
        if built and want > built:
            # REFUSED, and the gate takes what the pool can actually deliver. Growing the pool needs
            # a fork, a fork needs the old pool retired, and a retirement is what kills ranks.
            _REFUSALS.append(("read_threads", before, want, "would need a wider pool"))
            _mark("REFUSED read_threads %d -> %d: it exceeds the %d forked workers and growing the "
                  "pool would need a teardown. The gate takes %d, which is what the pool can "
                  "deliver, and the refusal is recorded rather than acked as applied."
                  % (before, want, built, built))
            gate.set_limit(built)
            return

        if shrink and built > 1 and want < max(1, int(before * ratio)):
            # The cut is APPLIED, by parking, and the pool is NOT shrunk. Phase 20 priced parking a
            # deep cut at +19.2% against +3.2% for a rebuild, and that trade is now refused: a
            # rebuild costs a retirement, a retirement kills a rank about once in 480, and a run
            # that dies is worth less than one that is 19% slow. The knob still moves.
            _REFUSALS.append(("pool_shrink", built, want, "teardown not permitted during a run"))
            _mark("PARKING a deep cut %d -> %d against %d forked workers, and NOT shrinking the "
                  "pool: the cut is applied through the gate and costs head-of-line blocking, "
                  "which is cheaper than a teardown that can kill the rank."
                  % (before, want, built))

        after = gate.set_limit(read_threads)
        _mark("RECONFIGURE gate %d -> %d mid-epoch (same loader, same iterator, no fork)"
              % (before, after))

        # PREFETCH IS NOT GATEABLE, AND SAYING SO MATTERS.
        #
        # torch's `prefetch_factor` is a constructor argument with no runtime setter, so unlike
        # read_threads it cannot be moved by a limit. Under the gate the DataLoader is never
        # rebuilt, which means a prefetch_size change would be SILENTLY ignored -- the knob would
        # appear to apply, the optimizer would ack it, the gates would record it as a successful
        # action, and nothing would have changed. That is the worst available outcome and exactly
        # the failure mode this work has been finding elsewhere.
        #
        # So it is reported, and LIVEFLOW_GATE_REBUILD_PREFETCH=1 opts into a real rebuild for
        # callers that would rather pay the fork than lose the knob.
        if prefetch_size is not None and self._built_prefetch is not None \
                and int(prefetch_size) != int(self._built_prefetch):
            if self._prefetch is not None:
                # Dynamic prefetch owns the depth, so this IS applicable after all.
                before_d = self._prefetch.depth
                after_d = self._prefetch.set_depth(prefetch_size)
                self._built_prefetch = prefetch_size
                _mark("PREFETCH %s -> %s applied via the dynamic buffer (no rebuild)"
                      % (before_d, after_d))
            elif os.environ.get("LIVEFLOW_GATE_REBUILD_PREFETCH", "0") == "1":
                # REFUSED. There is no longer a boundary at which a rebuild is safe, because the
                # pool is never torn down. Opting in asks for a teardown and gets a refusal.
                _REFUSALS.append(("prefetch_size", self._built_prefetch, prefetch_size,
                                  "prefetch_factor is a constructor argument"))
                _mark("REFUSED prefetch %s -> %s: prefetch_factor is a torch constructor argument "
                      "and applying it needs a rebuild, which is not permitted during a run. Set "
                      "LIVEFLOW_DYNAMIC_PREFETCH=1 to make prefetch depth a live knob instead."
                      % (self._built_prefetch, prefetch_size))
            else:
                _mark("PREFETCH %s -> %s NOT APPLIED: prefetch_factor is a torch constructor "
                      "argument and cannot be gated. Set LIVEFLOW_GATE_REBUILD_PREFETCH=1 to "
                      "rebuild instead of ignoring it."
                      % (self._built_prefetch, prefetch_size))

    @dlp.log
    def next(self):
        super().next()
        # THE CRITICAL-PATH MEASUREMENT (sec 83.1).
        #
        # `fetch_frac` in dlio.yaml is fetch's share of SUMMED time --
        # fetch / (fetch + compute) -- which is not the same question as whether fetch is
        # STALLING the step. A perfectly overlapped pipeline can spend most of its summed time
        # fetching and cost the training loop nothing, because the next batch is always ready
        # before it is asked for. The rule cannot tell those apart, which is how it reached
        # severity=1.000 "unclassified" and moved a knob that phase 19 proved does nothing.
        #
        # The distinction is directly observable right here: the time the consumer spends BLOCKED
        # in __next__ is, by construction, the part of fetch that is on the critical path. If it
        # is near zero, fetch is fully hidden and no amount of fetch pressure justifies acting.
        #
        # Counted per step and reported per epoch, so a window can carry a stall fraction rather
        # than a share of summed time.
        # OBSERVATIONAL EQUIVALENCE WITH THE STOCK LOADER IS NOT OPTIONAL.
        #
        # `dft_ai.dataloader.fetch.iter` is what produces the `fetch.iter` events the analyzer
        # aggregates into `fetch_iter_time_proc_max`, which is the metric BOTH terms of
        # dlio.fetch_pressure.v1 are built from. This loader originally iterated the source
        # directly and emitted none of them, so under the gated loader the rule had no input and
        # could not fire at all.
        #
        # That is the real reason the gate arm logged 0 applies in the first phase 20 attempt. I
        # recorded the cause as the gated loader presenting 8 worker processes where the stock
        # loader presents 4; that difference is real, but this one is sufficient on its own and
        # was the actual mechanism. A loader that changes what the observer can SEE cannot be
        # compared against one that does not, and this file is in the observation path.
        src = self._prefetch if self._prefetch is not None else self._dataset
        step = 1
        for batch in traced_consumer_batches(src, owner=self):
            dlp.update(step=step)
            dft_ai.update(step=step)
            step += 1
            yield batch
        if self._consumer_boundary is not None:
            self._stall_s, self._steps = self._consumer_boundary.totals
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    def emit_window_stall(self, epoch, step):
        if self._consumer_boundary is None:
            return 0.0
        emitted = self._consumer_boundary.emit_window_stall(epoch, step)
        self._stall_s, self._steps = self._consumer_boundary.totals
        return emitted

    def shutdown_workers(self, reason=""):
        """Stop this pool before another DataLoader forks a replacement.

        Persistent workers intentionally survive ordinary iterator exhaustion.  That is useful
        while a loader is cached inside one epoch, but it is unsafe to fork a fresh pool while
        the previous one is still alive: each rank temporarily doubles its worker count and every
        child retains the parent's fork-inherited tracing and gate state.  PyTorch's private
        shutdown hook is the supported lifecycle used by its DataLoader destructor; call it
        explicitly here so the old pool has exited before a replacement is constructed.
        """
        # THE RETIREMENT ITSELF, REFUSED BY DEFAULT.
        #
        # Every fabric-stack fatal this project has seen came from here: torch signals the worker
        # processes, MVAPICH's async event thread takes EINTR in ibv_get_cq_event and treats it as
        # a fatal CQ error, and the rank dies. The node sweep (4s.6) closed the last alternative
        # explanation, so the pool is not retired during a run and a knob that would need a
        # teardown is refused in reconfigure() instead.
        #
        # LIVEFLOW_ALLOW_POOL_TEARDOWN=1 restores the old behaviour exactly, the ablation.
        if os.environ.get("LIVEFLOW_ALLOW_POOL_TEARDOWN", "0") != "1":
            _REFUSALS.append(("teardown", None, None, reason or "unspecified"))
            _mark("REFUSED teardown reason=%s: the pool is built once and kept for the run. "
                  "Retiring 24 workers has killed a rank about once in 480 retirements "
                  "(T3-0 4s.5, 4s.6), and nothing about this pool needs replacing."
                  % (reason or "unspecified"))
            return

        dataset = self._dataset
        iterator = getattr(dataset, "_iterator", None) if dataset is not None else None
        if iterator is not None:
            _mark("RETIRE workers=%s reason=%s (shutdown before rebuild)"
                  % (getattr(dataset, "num_workers", "?"), reason or "unspecified"))
            # STAGGER THE EXITS. Retiring 24 workers at once delivers 24 SIGCHLDs to this process
            # in a burst, and MVAPICH's async event thread, which exists because DLIO initialises
            # MPI with THREAD_MULTIPLE, is blocked in ibv_get_cq_event. An interrupted wait there
            # is reported as `Got FATAL event CQ error on CQ (nil): Interrupted system call (4)`
            # and kills the rank. Measured on un_loop_n8_10e_r1 (f5z5EzMFcvTH): rank 14 died on the
            # sixth of six epoch-boundary retirements, about one fatal in 192 retirement events
            # across the run, so this is a race and not a deterministic fault.
            #
            # `_workers_done_event` is what tells a worker to leave its loop, and torch's own
            # shutdown sets it too. Setting it first and then joining each worker with a small
            # timeout spreads the exits over time instead of bunching them. Whatever has not left
            # is terminated by torch's own path below, unchanged.
            #
            # TESTED AND IT DOES NOT WORK, so it is OFF by default. `retire_storm_n2`
            # (f5z7nk9dgfvF), 2 nodes, 200 retire-and-rebuild cycles per rank: the unmitigated arm
            # died with the production fatal after about 60 cycles, and the staggered arm at 0.05 s
            # died the same way. Lowering the MPI thread level to SERIALIZED failed too
            # (f5z7pWMcexzb). The path is kept because it is one env variable away from being
            # re-tested, and it is not paid for by default: 24 workers at 0.05 s is 1.2 s per
            # retirement for no measured benefit. See T3-0 section 4s.5.
            gap = float(os.environ.get("LIVEFLOW_RETIRE_STAGGER_S", "0") or 0.0)
            workers = list(getattr(iterator, "_workers", None) or [])
            if gap > 0.0 and workers:
                try:
                    ev = getattr(iterator, "_workers_done_event", None)
                    if ev is not None:
                        ev.set()
                    left = 0
                    for w in workers:
                        w.join(timeout=gap)
                        if not w.is_alive():
                            left += 1
                    _mark("RETIRE staggered gap=%.3fs workers=%d exited_before_terminate=%d"
                          % (gap, len(workers), left))
                except Exception as exc:   # a stagger that fails must not stop the retirement
                    _mark("RETIRE stagger skipped: %s" % exc)
            try:
                iterator._shutdown_workers()
            except Exception as exc:  # a failed worker must not prevent rank teardown
                _mark("RETIRE workers shutdown failed: %s" % exc)
        self._dataset = None
        self._built_workers = None
        self._built_prefetch = None
        self._prefetch = None
        _POOL["dataset"] = None
        _POOL["workers"] = None
        _POOL["prefetch"] = None
        _POOL["prefetch_obj"] = None

    @dlp.log
    def finalize(self):
        # Deliberately does NOT tear down the loader: persistent workers are the point.
        #
        # Report what the gate actually DID. G1/G2 prove the mechanism throttles in a
        # microbenchmark; that is not the same as it doing real work inside DLIO, where the
        # limit may simply never bind. A gate that never blocks is not throttling anything, and
        # saying so is more useful than assuming it worked.
        # The stall fraction is the number sec 83.1 says the attribution rule is missing. Reported
        # unconditionally, including when it is ~0, because "fetch is fully hidden" is exactly the
        # case the rule currently cannot see and acts on anyway.
        if self._consumer_boundary is not None:
            self._stall_s, self._steps = self._consumer_boundary.totals
        if _REFUSALS:
            from collections import Counter as _C
            _mark("KNOB REFUSALS %s  (moves this policy declined rather than rebuilding the pool)"
                  % dict(_C(r[0] for r in _REFUSALS)))
        if self._steps:
            _mark("STALL total=%.3fs steps=%d mean=%.1fms  (consumer blocked waiting for a batch; "
                  "this is the part of fetch that is ON the critical path)"
                  % (self._stall_s, self._steps, 1000.0 * self._stall_s / self._steps))
        if self._prefetch is not None:
            _mark("PREFETCH STATS %s" % self._prefetch.stats())
        gate = get_gate()
        if gate is not None:
            st = gate.stats()
            _mark("GATE STATS limit=%d waits=%d wait_ms=%.0f  %s"
                  % (st["limit"], st["waits"], st["wait_ms"],
                     "(gate never blocked -- limit did not bind)" if st["waits"] == 0
                     else "(gate blocked workers -- throttling was real)"))
