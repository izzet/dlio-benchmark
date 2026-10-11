"""Measured parent fetch waits survive reader forwarding and pool replacement."""
from types import SimpleNamespace

from dlio_benchmark.data_loader import consumer_iterator
from dlio_benchmark.data_loader.torch_data_loader import TorchDataLoader
from dlio_benchmark.data_loader.live_data_loader import LiveDataLoader


class Region:
    def start(self): pass
    def stop(self): pass
    def derive(self, name): return self


def test_both_readers_forward_measured_wait_without_draining(monkeypatch):
    now = [0.0]
    def source():
        for sample, wait in [(17, .004), (3, .0), (22, .009)]:
            now[0] += wait
            yield sample
    monkeypatch.setattr(consumer_iterator, 'dft_ai', SimpleNamespace(
        dataloader=SimpleNamespace(fetch=Region())))
    monkeypatch.setenv('LIVEFLOW_PRESET', 'liveflow-v1')
    for cls in (TorchDataLoader, LiveDataLoader):
        loader = object.__new__(cls)
        loader._consumer_boundary = None
        assert loader.last_step_wait_s() is None
        boundary = consumer_iterator.ConsumerBoundary(source(), clock=lambda: now[0])
        loader._consumer_boundary = boundary
        samples, waits = [], []
        for _ in range(3):
            samples.append(boundary.next_batch())
            waits.append(loader.last_step_wait_s())
            assert loader.last_step_wait_s() == waits[-1]  # non-destructive
        assert samples == [17, 3, 22]
        assert all(abs(x-y) < 1e-12 for x,y in zip(waits, [.004, 0., .009]))
        assert loader.drain_step_stalls() == waits
        assert loader.last_step_wait_s() == waits[-1]
        # A rebuilt worker pool replaces the boundary; stale waits are not reused.
        loader._consumer_boundary = consumer_iterator.ConsumerBoundary(source(), clock=lambda: now[0])
        loader._consumer_boundary.next_batch()
        assert abs(loader.last_step_wait_s() - .004) < 1e-12


def test_actual_live_reader_preserves_samples_and_rng_across_resize(monkeypatch):
    import torch
    import numpy as np
    from dlio_benchmark.common.enumerations import DatasetType
    from dlio_benchmark.data_loader import live_data_loader as live
    class Profile:
        def update(self, **kwargs): pass
    monkeypatch.setattr(consumer_iterator, 'dft_ai', SimpleNamespace(
        dataloader=SimpleNamespace(fetch=Region())))
    monkeypatch.setattr(live, 'dft_ai', Profile())
    monkeypatch.setattr(live, 'maybe_inject_fetch_delay', lambda *a, **kw: None)
    monkeypatch.setenv('LIVEFLOW_MOTIVATION_TIMERS', '0')
    loader = object.__new__(LiveDataLoader)
    loader._args = SimpleNamespace(training_steps=5, my_rank=0)
    loader.dataset_type = DatasetType.TRAIN
    loader.logger = SimpleNamespace(debug=lambda *a: None)
    loader.epoch_number = 1
    loader.batch_size = 1
    loader._samples_yielded = 0
    loader._sampler = SimpleNamespace(remaining=5)
    payloads = [(torch.tensor([i]), torch.tensor([i])) for i in [7, 2, 9, 4, 1]]
    loader._inner = payloads
    loader._pending_reconfigure = None
    loader._consumer_boundary = None
    resized = []
    def resize():
        resized.append(loader._samples_yielded)
        loader._inner = payloads[loader._samples_yielded:]
        loader._pending_reconfigure = None
    loader._apply_reconfigure = resize
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    batches, waits = [], []
    for batch in loader.next():
        batches.append(batch)
        waits.append(loader.last_step_wait_s())
        if len(batches) == 2:
            loader._pending_reconfigure = (8, 2)
    assert resized == [2]
    assert all(got is expected for got, expected in zip(batches, payloads))
    assert len(batches) == 5 and all(v is not None and v >= 0 for v in waits)
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]
