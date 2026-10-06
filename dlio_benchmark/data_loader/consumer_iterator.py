"""Trace and account the blocking data-loader consumer boundary."""

import os
import time
from contextlib import contextmanager

from dftracer.python import dftracer as PerfTracer

from dlio_benchmark.utils.utility import dft_ai


_INT_TAG = lambda value: (0, int(value))


class ConsumerBoundary:
    """Own the exact source ``next()`` call and its sibling consumer body.

    The old implementation delegated to ``dft_ai.fetch.iter`` through two
    nested generators. That made the apparent region depend on which
    generator Python happened to be resuming. Here the loader calls
    :meth:`next_batch` directly, so both the trace region and the independent
    monotonic account surround the same source iterator ``next()`` call.
    """

    def __init__(self, source, clock=time.monotonic):
        self._source_iterator = iter(source)
        self._clock = clock
        fetch = dft_ai.dataloader.fetch
        self._fetch_iter = fetch.derive("iter")
        self._fetch_block = fetch.derive("block")
        self._pending_stall_s = 0.0
        self._total_stall_s = 0.0
        self._steps = 0
        # Per-step fetch waits of the open window, kept only for the job-level input sync
        # (LIVEFLOW_JOB_INPUT_SYNC=1) so a default run holds no list.
        self._record_steps = os.environ.get("LIVEFLOW_JOB_INPUT_SYNC", "0") == "1"
        self._window_step_waits = []

    def next_batch(self):
        """Return one batch while timing exactly ``next(source_iterator)``."""
        self._fetch_iter.start()
        started = self._clock()
        try:
            batch = next(self._source_iterator)
        except BaseException:
            self._fetch_iter.stop()
            raise
        elapsed = max(0.0, self._clock() - started)
        self._fetch_iter.stop()
        self._pending_stall_s += elapsed
        self._total_stall_s += elapsed
        self._steps += 1
        if self._record_steps:
            self._window_step_waits.append(elapsed)
        return batch

    @contextmanager
    def consumer_body(self):
        """Trace only the caller body between receiving and requesting batches."""
        self._fetch_block.start()
        try:
            yield
        finally:
            self._fetch_block.stop()

    @property
    def totals(self):
        return self._total_stall_s, self._steps

    @property
    def pending_stall_seconds(self):
        return self._pending_stall_s

    def drain_step_stalls(self):
        """The per-step fetch waits since the last drain (seconds), for the job-level input sync."""
        waits = self._window_step_waits
        self._window_step_waits = []
        return waits

    def drain_stall_seconds(self):
        stalled = self._pending_stall_s
        self._pending_stall_s = 0.0
        return stalled

    def emit_window_stall(self, epoch, step, tracer_factory=PerfTracer):
        """Emit one accumulated main-thread storage-stall event for this window."""
        stalled = self.drain_stall_seconds()
        tracer = tracer_factory.get_instance()
        # Zero is a measurement too. Omitting this event makes an empty epoch-boundary window
        # indistinguishable from an uninstrumented workload and forces the planner fallback.
        duration_us = max(0, int(round(stalled * 1_000_000.0)))
        now = tracer.get_time()
        tracer.log_event(
            name="stall.storage",
            cat="stall",
            start_time=now - duration_us,
            duration=duration_us,
            int_args={"epoch": _INT_TAG(epoch), "step": _INT_TAG(step)},
            float_args={},
            string_args={},
        )
        return stalled


def traced_consumer_batches(source, owner=None):
    """Yield batches using an explicit, inspectable consumer boundary."""
    boundary = ConsumerBoundary(source)
    if owner is not None:
        owner._consumer_boundary = boundary
    while True:
        try:
            batch = boundary.next_batch()
        except StopIteration:
            return
        with boundary.consumer_body():
            yield batch
