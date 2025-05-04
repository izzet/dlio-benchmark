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
from time import time
import logging
import math
import pickle
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.sampler import Sampler
import numpy as np
import webdataset as wds

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType, FormatType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)


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

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step = step)
        return self.reader.read_index(image_idx, step)


class WebDataset(wds.WebDataset):
    @dlp.log_init
    def __init__(
        self,
        urls,
        handler=wds.reraise_exception,
        mode=None,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        cache_size=-1,
        cache_dir=None,
        url_to_name=wds.cache.pipe_cleaner,
        detshuffle=False,
        nodesplitter=wds.shardlists.single_node_only,
        workersplitter=wds.shardlists.split_by_worker,
        select_files=None,
        rename_files=None,
        empty_check=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(
            urls,
            handler=handler,
            mode=mode,
            resampled=resampled,
            repeat=repeat,
            shardshuffle=shardshuffle,
            cache_size=cache_size,
            cache_dir=cache_dir,
            url_to_name=url_to_name,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            workersplitter=workersplitter,
            select_files=select_files,
            rename_files=rename_files,
            empty_check=empty_check,
            verbose=verbose,
            seed=seed,
        )

    @dlp.log
    def __iter__(self):
        for sample in dlp.iter(super().__iter__()):
            yield sample
        # return super().__iter__()

    @dlp.log
    def __getitem__(self, index):
        return super().__getitem__(index)


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
    @dlp.log
    def read(self):
        if self.format_type == FormatType.WEBDATASET_NPY:
            dataset = (
                WebDataset(
                    "/p/lustre3/izzet/dlio-benchmark-test/unet3d_v100_webdataset_npy_168/data/train/img_{001..168}_of_168.webdataset_npy"
                )
                .decode("torchrgb8")
                .to_tuple("x.npy", "y.npy")
            )
        else:
            dataset = TorchDataset(
                self.format_type,
                self.dataset_type,
                self.epoch_number,
                self.num_samples,
                self._args.read_threads,
                self.batch_size,
            )
        sampler = dlio_sampler(
            self._args.my_rank,
            self._args.comm_size,
            self.num_samples,
            self._args.epochs,
        )
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
        if self._args.read_threads == 0:
            kwargs = {}
        else:
            kwargs = {
                "multiprocessing_context": self._args.multiprocessing_context,
                "prefetch_factor": prefetch_factor,
            }
            if torch.__version__ != "1.3.1":
                kwargs["persistent_workers"] = True
        if torch.__version__ == "1.3.1":
            if "prefetch_factor" in kwargs:
                del kwargs["prefetch_factor"]
            self._dataset = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=None if isinstance(dataset, IterableDataset) else sampler,
                num_workers=self._args.read_threads,
                pin_memory=self._args.pin_memory,
                drop_last=True,
                worker_init_fn=dataset.worker_init
                if hasattr(dataset, "worker_init")
                else None,
                **kwargs,
            )
        else:
            self._dataset = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=None if isinstance(dataset, IterableDataset) else sampler,
                num_workers=self._args.read_threads,
                pin_memory=self._args.pin_memory,
                drop_last=True,
                worker_init_fn=dataset.worker_init
                if hasattr(dataset, "worker_init")
                else None,
                **kwargs,
            )  # 2 is the default value

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        # TODO: @hariharan-devarajan: change below line when we bump the dftracer version to 
        #       `dlp.iter(self._dataset, name=self.next.__qualname__)`
        for batch in dlp.iter(self._dataset):
            dlp.update(step = step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass
