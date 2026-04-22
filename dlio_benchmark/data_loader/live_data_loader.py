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
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.torch_data_loader import TorchDataset
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai
from dlio_benchmark.utils.config import ConfigArguments

dlp = Profile(MODULE_DATA_LOADER)


class LiveSampler(Sampler):
    """A position-tracking sampler for mid-epoch DataLoader reconfiguration.

    Same contiguous index distribution as dlio_sampler, but supports
    advance_to() / reset() so a new DataLoader can resume from the exact
    sample position after a worker-count change.
    """

    def __init__(self, rank, world_size, num_samples):
        samples_per_proc = math.ceil(num_samples / world_size)
        start = rank * samples_per_proc
        end = min((rank + 1) * samples_per_proc - 1, num_samples - 1)
        self._indices = list(range(start, end + 1))
        self._offset = 0

    @property
    def remaining(self):
        return len(self._indices) - self._offset

    def advance_to(self, sample_offset):
        """Set starting position for the next DataLoader iteration."""
        self._offset = min(sample_offset, len(self._indices))

    def reset(self):
        """Reset to beginning for a new epoch."""
        self._offset = 0

    def __len__(self):
        return self.remaining

    def __iter__(self):
        for i in range(self._offset, len(self._indices)):
            yield self._indices[i]


class LiveDataLoader(BaseDataLoader):
    """A PyTorch DataLoader that supports mid-epoch reconfiguration.

    Unlike TorchDataLoader which creates a new DataLoader object for each
    knob change, LiveDataLoader preserves the sampler position when
    num_workers or prefetch changes.  This enables:
      - No duplicate or missed samples across reconfiguration boundaries
      - Seamless worker count changes mid-epoch
      - Proper continuation from the exact sample position
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number,
                         DataLoaderType.PYTORCH)
        self._inner = None
        self._sampler = None
        self._read_threads = 0
        self._prefetch_size = 2
        self._samples_yielded = 0
        self._pending_reconfigure = None

    def _build_inner(self):
        """Build the inner PyTorch DataLoader with current settings."""
        self._sampler.advance_to(self._samples_yielded)

        dataset = TorchDataset(
            self.format_type, self.dataset_type, self.epoch_number,
            self.num_samples, self._read_threads, self.batch_size,
        )

        if self._read_threads >= 1:
            prefetch_factor = math.ceil(self._prefetch_size / self._read_threads)
        else:
            prefetch_factor = self._prefetch_size
        if prefetch_factor <= 0:
            prefetch_factor = 2

        if self._read_threads == 0:
            kwargs = {}
        else:
            kwargs = {
                'multiprocessing_context': self._args.multiprocessing_context,
                'prefetch_factor': prefetch_factor,
            }
            if torch.__version__ != '1.3.1':
                kwargs['persistent_workers'] = True

        if torch.__version__ == '1.3.1':
            kwargs.pop('prefetch_factor', None)

        self._inner = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self._sampler,
            num_workers=self._read_threads,
            pin_memory=self._args.pin_memory,
            drop_last=True,
            worker_init_fn=dataset.worker_init,
            **kwargs,
        )

    @dlp.log
    def read(self):
        self._read_threads = self._args.read_threads
        self._prefetch_size = self._args.prefetch_size
        self._samples_yielded = 0

        self._sampler = LiveSampler(
            self._args.my_rank, self._args.comm_size, self.num_samples,
        )
        self._build_inner()

        if self._args.my_rank == 0:
            self.logger.output(
                f"{utcnow()} LiveDataLoader.read: num_workers={self._read_threads}"
                f" prefetch_size={self._prefetch_size}"
                f" samples_per_rank={len(self._sampler._indices)}")

    def reconfigure(self, read_threads, prefetch_size):
        """Request mid-epoch reconfiguration. Takes effect on next batch."""
        if (read_threads == self._read_threads
                and prefetch_size == self._prefetch_size):
            return
        self._pending_reconfigure = (read_threads, prefetch_size)

    def _apply_reconfigure(self):
        """Apply pending reconfiguration: tear down old, build new."""
        new_threads, new_prefetch = self._pending_reconfigure
        self._pending_reconfigure = None
        old_threads = self._read_threads
        self._read_threads = new_threads
        self._prefetch_size = new_prefetch
        del self._inner
        self._build_inner()
        if self._args.my_rank == 0:
            self.logger.output(
                f"{utcnow()} LiveDataLoader: reconfigured workers"
                f" {old_threads} -> {new_threads},"
                f" prefetch_size={new_prefetch},"
                f" resuming from sample {self._samples_yielded}")

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        self.logger.debug(
            f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")

        step = self._samples_yielded // self.batch_size + 1

        while self._sampler.remaining > 0:
            if self._pending_reconfigure is not None:
                self._apply_reconfigure()

            for batch in dft_ai.dataloader.fetch.iter(self._inner):
                dlp.update(step=step)
                dft_ai.update(step=step)
                step += 1
                self._samples_yielded += self.batch_size
                yield batch

                if self._pending_reconfigure is not None:
                    break
            else:
                break

        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        self._samples_yielded = 0
        if self._sampler is not None:
            self._sampler.reset()
