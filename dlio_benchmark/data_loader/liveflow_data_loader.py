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
from dlio_benchmark.data_loader.torch_data_loader import TorchDataset, dlio_sampler
from dlio_benchmark.utils.utility import utcnow, Profile

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
            workers = max_workers
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
            _mark("RECONFIGURE gate=off -> falling back to a real rebuild (control arm)")
            self._dataset = None
            self._built_workers = None
            self.read(read_threads=read_threads, prefetch_size=prefetch_size)
            return
        before = gate.limit
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
                _mark("PREFETCH %s -> %s requires a rebuild (opted in); rebuilding"
                      % (self._built_prefetch, prefetch_size))
                self._dataset = None
                self._built_workers = None
                self.read(read_threads=read_threads, prefetch_size=prefetch_size)
            else:
                _mark("PREFETCH %s -> %s NOT APPLIED: prefetch_factor is a torch constructor "
                      "argument and cannot be gated. Set LIVEFLOW_GATE_REBUILD_PREFETCH=1 to "
                      "rebuild instead of ignoring it."
                      % (self._built_prefetch, prefetch_size))

    @dlp.log
    def next(self):
        super().next()
        # With dynamic prefetch on, the training loop consumes through a bounded buffer whose
        # depth is a live knob. Off by default: this sits in the data path.
        if self._prefetch is not None:
            for batch in self._prefetch:
                yield batch
        else:
            for batch in self._dataset:
                yield batch

    @dlp.log
    def finalize(self):
        # Deliberately does NOT tear down the loader: persistent workers are the point.
        #
        # Report what the gate actually DID. G1/G2 prove the mechanism throttles in a
        # microbenchmark; that is not the same as it doing real work inside DLIO, where the
        # limit may simply never bind. A gate that never blocks is not throttling anything, and
        # saying so is more useful than assuming it worked.
        if self._prefetch is not None:
            _mark("PREFETCH STATS %s" % self._prefetch.stats())
        gate = get_gate()
        if gate is not None:
            st = gate.stats()
            _mark("GATE STATS limit=%d waits=%d wait_ms=%.0f  %s"
                  % (st["limit"], st["waits"], st["wait_ms"],
                     "(gate never blocked -- limit did not bind)" if st["waits"] == 0
                     else "(gate blocked workers -- throttling was real)"))
