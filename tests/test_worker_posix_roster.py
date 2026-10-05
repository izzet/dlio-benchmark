"""CPU-only identity receipt checks; no SHM service or workload allocation."""

import json
from types import SimpleNamespace

import pytest

from dlio_benchmark.data_loader import torch_data_loader as loader


def _enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("DLIO_WORKER_POSIX_ROSTER", "1")
    monkeypatch.setenv("DLIO_WORKER_POSIX_ROSTER_DIR", str(tmp_path))
    monkeypatch.setenv("DFTRACER_SHM_PREFIX", "probe-prefix")
    monkeypatch.setenv("LIVEFLOW_SHM_PREFIX", "probe-prefix")
    monkeypatch.setattr(loader.ConfigArguments, "get_instance",
                        lambda: SimpleNamespace(my_rank=0))
    monkeypatch.setattr(loader.torch.utils.data, "get_worker_info",
                        lambda: SimpleNamespace(id=1, num_workers=2))


def test_roster_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("DLIO_WORKER_POSIX_ROSTER", raising=False)
    loader._publish_worker_roster(1, 2)
    assert list(tmp_path.iterdir()) == []


def test_worker_owned_roster_is_exclusive(monkeypatch, tmp_path):
    _enabled(monkeypatch, tmp_path)
    loader._publish_worker_roster(1, 2)
    records = list(tmp_path.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["schema_version"] == "dlio-worker-posix-roster.v1"
    assert record["worker_id"] == 1 and record["rank"] == 0
    assert record["expected_workers"] == 2 and record["run_prefix"] == "probe-prefix"
    with pytest.raises(RuntimeError, match="already exists"):
        loader._publish_worker_roster(1, 2)
    assert json.loads(records[0].read_text()) == record


def test_roster_rejects_wrong_worker_or_prefix(monkeypatch, tmp_path):
    _enabled(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="assigned DataLoader worker"):
        loader._publish_worker_roster(0, 2)
    monkeypatch.setenv("LIVEFLOW_SHM_PREFIX", "other")
    with pytest.raises(RuntimeError, match="matching SHM prefixes"):
        loader._publish_worker_roster(1, 2)
    assert list(tmp_path.iterdir()) == []


def test_real_forked_dataloader_workers_publish_roster(monkeypatch, tmp_path):
    real_get_worker_info = loader.torch.utils.data.get_worker_info
    _enabled(monkeypatch, tmp_path)
    monkeypatch.setattr(loader.torch.utils.data, "get_worker_info", real_get_worker_info)
    dataloader = loader.DataLoader(
        list(range(4)), batch_size=1, num_workers=2,
        multiprocessing_context="fork",
        worker_init_fn=lambda worker_id: loader._publish_worker_roster(worker_id, 2),
    )
    assert len(list(dataloader)) == 4
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {record["worker_id"] for record in records} == {0, 1}
    assert len({record["pid"] for record in records}) == 2
