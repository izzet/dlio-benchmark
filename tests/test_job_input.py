"""Job-level critical-path input exposure: reduction arithmetic and event emission (no MPI needed)."""
import numpy as np

from dlio_benchmark.data_loader import consumer_iterator
from dlio_benchmark.utils import job_input


class _FakeWorld:
    """In-process stand-in for a communicator: ranks contribute vectors, Allreduce applies the op."""

    def __init__(self, per_rank):
        self.per_rank = per_rank
        self.calls = 0

    def Get_size(self):
        return len(self.per_rank)

    def Allreduce(self, send, recv, op):
        # every rank's send buffer is known to the fake; the caller is rank 0
        k = len(send)
        if k == len(self.per_rank[0]) + 1 or op == "max":
            stack = np.stack([self._pad(v, k) for v in self.per_rank])
        else:
            stack = np.stack([[float(np.sum(v))] for v in self.per_rank])
        recv[:] = stack.max(axis=0) if op == "max" else stack.sum(axis=0)

    @staticmethod
    def _pad(v, k):
        out = np.zeros(k)
        out[:len(v)] = v
        out[k - 1] = len(v)
        return out


def test_reduce_window_critical_and_own():
    # 3 ranks x 4 steps: one rank stalls on a different step each time (rotating straggler)
    per_rank = [[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.2, 0.0, 0.0, 0.0]]
    world = _FakeWorld(per_rank)
    critical, own, n_valid = job_input.reduce_window(world, per_rank[0], 4, "max", "sum")
    assert abs(critical - (0.2 + 0.5 + 0.3 + 0.0)) < 1e-12      # sum_steps max_rank
    assert abs(own - (0.5 + 0.3 + 0.2) / 3) < 1e-12              # mean_rank sum_steps
    assert n_valid == 4
    assert critical - own > 0                                    # peer skew is positive


def test_uniform_waits_have_no_peer_skew():
    per_rank = [[0.1, 0.1, 0.1]] * 4
    critical, own, _ = job_input.reduce_window(_FakeWorld(per_rank), per_rank[0], 3, "max", "sum")
    assert abs(critical - own) < 1e-12


class _Tracer:
    def __init__(self):
        self.events = []

    def get_time(self):
        return 10_000_000

    def log_event(self, **kw):
        self.events.append(kw)


class _Loader:
    def __init__(self, steps):
        self.steps = steps

    def drain_step_stalls(self):
        s, self.steps = self.steps, []
        return s


class _Window:
    pending_steps = 3


def test_sync_window_single_rank_emits_both_facts_in_microseconds():
    tracer = _Tracer()
    loader = _Loader([0.001, 0.0, 0.0025])
    got = job_input.sync_window(None, loader, _Window(), epoch=2, step=30, tracer=tracer)
    assert got == (0.0035, 0.0035)
    names = [e["name"] for e in tracer.events]
    assert names == [job_input.CRITICAL_NAME, job_input.OWN_NAME]
    assert all(e["cat"] == job_input.FAMILY for e in tracer.events)
    assert tracer.events[0]["duration"] == 3500 and tracer.events[0]["int_args"]["step_count"] == (0, 3)
    assert loader.steps == []          # drained


def test_sync_window_without_step_account_is_a_noop():
    class NoAccount:
        pass
    assert job_input.sync_window(None, NoAccount(), _Window(), 1, 1, _Tracer()) is None


def test_consumer_boundary_records_steps_only_when_enabled(monkeypatch):
    class Src:
        def __init__(self): self.n = 0
        def __iter__(self): return self
        def __next__(self):
            self.n += 1
            if self.n > 3: raise StopIteration
            return self.n

    class Region:
        def start(self): pass
        def stop(self): pass
        def derive(self, name): return self

    fake_ai = type("AI", (), {"dataloader": type("D", (), {"fetch": Region()})()})()
    monkeypatch.setattr(consumer_iterator, "dft_ai", fake_ai)
    monkeypatch.delenv("LIVEFLOW_JOB_INPUT_SYNC", raising=False)
    off = consumer_iterator.ConsumerBoundary(Src())
    off.next_batch(); assert off.drain_step_stalls() == []
    monkeypatch.setenv("LIVEFLOW_JOB_INPUT_SYNC", "1")
    on = consumer_iterator.ConsumerBoundary(Src())
    on.next_batch(); on.next_batch()
    assert len(on.drain_step_stalls()) == 2 and on.drain_step_stalls() == []


def test_applied_vector_snapshot_and_apply_cost():
    """The loop publishes the applied knob vector (first vector at step 0, a later one one step past the boundary)."""
    import types
    from dlio_benchmark import main as main_module

    calls = []

    class W:
        _completed_steps = 0
        _apply_probe = None

        def set_physical_config(self, vec, step, apply_cost_s=None):
            calls.append((dict(vec), step, apply_cost_s))

        def begin_apply_probe(self, latency):
            self._apply_probe = latency

        def apply_probe_result(self):
            p, self._apply_probe = self._apply_probe, None
            return None if p is None else p + 0.5

    bench = types.SimpleNamespace(
        _window=W(), args=types.SimpleNamespace(prefetch_size=2, read_threads=4, computation_threads=1))
    pub = main_module.DLIOBenchmark._publish_applied_vector
    refine = main_module.DLIOBenchmark._refine_apply_cost
    pub(bench)
    assert calls == [({"prefetch_size": 2, "read_threads": 4, "computation_threads": 1}, 0, None)]
    pub(bench)                       # unchanged vector: nothing published
    assert len(calls) == 1
    bench._window._completed_steps = 3000
    bench.args.read_threads = 8
    pub(bench, 0.2)
    assert calls[-1] == ({"prefetch_size": 2, "read_threads": 8, "computation_threads": 1}, 3001, 0.2)
    refine(bench)                    # the probe measured the step after the apply
    assert calls[-1][2] == 0.7 and calls[-1][1] == 3001
    n = len(calls)
    refine(bench)
    assert len(calls) == n           # probe consumed
