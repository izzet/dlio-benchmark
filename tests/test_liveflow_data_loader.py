from types import SimpleNamespace

from dlio_benchmark.data_loader.liveflow_data_loader import LiveFlowDataLoader


class _Iterator:
    def __init__(self):
        self.shutdowns = 0

    def _shutdown_workers(self):
        self.shutdowns += 1


def _loader(dataset):
    loader = object.__new__(LiveFlowDataLoader)
    loader._dataset = dataset
    loader._built_workers = 32
    loader._built_prefetch = 2
    loader._prefetch = None
    return loader


def test_shutdown_workers_retires_persistent_pool_before_rebuild():
    iterator = _Iterator()
    loader = _loader(SimpleNamespace(num_workers=32, _iterator=iterator))

    loader.shutdown_workers("epoch 1 -> 2")

    assert iterator.shutdowns == 1
    assert loader._dataset is None
    assert loader._built_workers is None
    assert loader._built_prefetch is None


def test_shutdown_workers_handles_loader_that_never_started_iteration():
    loader = _loader(SimpleNamespace(num_workers=32, _iterator=None))

    loader.shutdown_workers("epoch 1 -> 2")

    assert loader._dataset is None
