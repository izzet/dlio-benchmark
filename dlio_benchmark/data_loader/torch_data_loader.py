"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import math
import pickle
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.consumer_iterator import traced_consumer_batches
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, Profile, dft_ai
from dlio_benchmark.utils.config import ConfigArguments

dlp = Profile(MODULE_DATA_LOADER)

_WORKER_ROSTER_SCHEMA = "dlio-worker-posix-roster.v1"


def _publish_worker_roster(worker_id, expected_workers):
    """Opt-in, worker-owned identity evidence independent of SHM trace ownership."""
    if os.environ.get("DLIO_WORKER_POSIX_ROSTER", "0") != "1":
        return
    info = torch.utils.data.get_worker_info()
    if info is None or info.id != worker_id or info.num_workers != expected_workers:
        raise RuntimeError("worker roster callback is not running in its assigned DataLoader worker")
    directory = Path(os.environ.get("DLIO_WORKER_POSIX_ROSTER_DIR", ""))
    prefix = os.environ.get("DFTRACER_SHM_PREFIX", "")
    if (not directory.is_absolute() or not directory.is_dir() or not prefix
            or prefix != os.environ.get("LIVEFLOW_SHM_PREFIX")):
        raise RuntimeError("worker roster requires an existing absolute directory and matching SHM prefixes")
    rank = ConfigArguments.get_instance().my_rank
    if type(rank) is not int or rank < 0 or expected_workers < 1:
        raise RuntimeError("invalid DLIO worker roster rank or worker count")
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]
    destination = directory / f"{_WORKER_ROSTER_SCHEMA}.{digest}.rank-{rank}.worker-{worker_id}.json"
    record = {
        "schema_version": _WORKER_ROSTER_SCHEMA,
        "run_prefix": prefix,
        "host": socket.gethostname(),
        "rank": rank,
        "worker_id": worker_id,
        "pid": os.getpid(),
        "expected_workers": expected_workers,
    }
    fd, temporary = tempfile.mkstemp(prefix=".dlio-worker-roster-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RuntimeError(f"worker roster destination already exists: {destination}") from exc
    finally:
        os.unlink(temporary)


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.logger = args.logger
        self.dlp_logger = None
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        self.logger.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_id,
                                               epoch_number=self.epoch_number)
        if self.num_workers > 0:
            _publish_worker_roster(worker_id, self.num_workers)

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.reader.read_index(image_idx, step)


class dlio_sampler(Sampler):
    def __init__(self, rank, size, num_samples, epochs):
        self.size = size
        self.rank = rank
        self.num_samples = num_samples
        self.epochs = epochs
        samples_per_proc = int(math.ceil(num_samples/size)) 
        start_sample = self.rank * samples_per_proc
        end_sample = (self.rank + 1) * samples_per_proc - 1
        if end_sample > num_samples - 1:
            end_sample = num_samples - 1
        self.indices = list(range(start_sample, end_sample + 1))


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for sample in self.indices:
            yield sample


class TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)
        self._consumer_boundary = None

    @dlp.log
    def read(self, read_threads=None, prefetch_size=None):
        # Use explicit args if provided (avoids ConfigArguments singleton
        # being reset by TorchDataset pickling during DataLoader creation).
        if read_threads is None:
            read_threads = self._args.read_threads
        if prefetch_size is None:
            prefetch_size = self._args.prefetch_size
        dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                               read_threads, self.batch_size)
        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
        if read_threads >= 1:
            prefetch_factor = math.ceil(prefetch_size / read_threads)
        else:
            prefetch_factor = prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is {prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        self.logger.debug(f"{utcnow()} Setup dataloader with {read_threads} workers {torch.__version__}")
        if read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            if torch.__version__ != '1.3.1':
                kwargs['persistent_workers'] = True
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)
        else:
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)
        if self._args.my_rank == 0:
            self.logger.output(
                f"{utcnow()} TorchDataLoader.read: num_workers={self._dataset.num_workers}"
                f" persistent_workers={getattr(self._dataset, 'persistent_workers', '?')}"
                f" read_threads={read_threads} prefetch_size={prefetch_size}")
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        for batch in traced_consumer_batches(self._dataset, owner=self):
            dlp.update(step=step)
            dft_ai.update(step=step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    def emit_window_stall(self, epoch, step):
        if self._consumer_boundary is None:
            return 0.0
        return self._consumer_boundary.emit_window_stall(epoch, step)

    def drain_step_stalls(self):
        if self._consumer_boundary is None:
            return []
        return self._consumer_boundary.drain_step_stalls()

    @dlp.log
    def finalize(self):
        pass
