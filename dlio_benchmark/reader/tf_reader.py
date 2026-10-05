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
import os
import sys

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import utcnow, Profile, maybe_inject_fetch_delay, dft_ai
from dlio_benchmark.common.enumerations import Shuffle
from dlio_benchmark.reader.reader_handler import FormatReader
import tensorflow as tf

dlp = Profile(MODULE_DATA_READER)


class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._resized_image = tf.convert_to_tensor(self._args.resized_image, dtype=tf.uint8)        
        self._dataset = None

    @dlp.log
    def open(self, filename):
        pass

    @dlp.log
    def close(self, filename):
        pass

    @dlp.log
    def get_sample(self, filename, sample_index):
        pass

    @dlp.log
    def resize_sample(self, filename, sample_index):
        pass

    @dlp.log
    def _parse_image(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        maybe_inject_fetch_delay(self.logger, location=f"{self.__class__.__qualname__}._parse_image")
        features = \
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'size': tf.io.FixedLenFeature([], tf.int64)
            }
        # Decode-family service-time region (analyzer layer `reader_decode`: cat "data", name
        # "decode").  Emitted for every arm alike whenever tracing is on; it changes no data.
        with dft_ai.data.derive("decode"):
            parsed_example = tf.io.parse_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        #image_raw = parsed_example['image']
        #dimension = tf.cast(parsed_example['size'], tf.int32).numpy()
        # Decode the raw bytes so it becomes a tensor with type.
        #image_tensor = tf.io.decode_raw(image_raw, tf.uint8)
        #size = dimension * dimension
        #dlp.update(image_size=size)
        #image_tensor = tf.io.decode_image(image_raw)
        #resized_image = tf.convert_to_tensor(self._args.resized_image, dtype=tf.uint8)
        return self._resized_image

    @staticmethod
    def _local_batch_total(file_count, samples_per_file, comm_size, rank, batch_size):
        """Exact post-shard, full-batch cardinality for one TF rank.

        TFReader shards files when possible and records otherwise.  The cursor
        for a rebuilt iterator must use that rank-local cardinality, never a
        global average that happens to work only for divisible file layouts.
        """
        if min(file_count, samples_per_file, comm_size, batch_size) < 1:
            raise ValueError("TF shard dimensions must be positive")
        if not 0 <= rank < comm_size:
            raise ValueError("rank must be within communicator")
        if file_count >= comm_size:
            local_files = (file_count - rank + comm_size - 1) // comm_size if rank < file_count else 0
            local_samples = local_files * samples_per_file
        else:
            all_samples = file_count * samples_per_file
            local_samples = (all_samples - rank + comm_size - 1) // comm_size if rank < all_samples else 0
        return local_samples // batch_size

    @dlp.log
    def next(self, read_threads=None, prefetch_size=None, resume_batches=0):
        if read_threads is None:
            read_threads = self._args.read_threads
        if prefetch_size is None:
            prefetch_size = self._args.prefetch_size
        self.logger.debug(f"{utcnow()} Reading {len(self._file_list)} files thread {self.thread_index} rank {self._args.my_rank}")

        # @ray: solution to prevent error when tf.data.Dataset cannot find files provided within self._file_list
        # the use case is usually as follow: user is providing workload.dataset.num_files_eval=0 since they do not
        # want to do any evaluation
        # since this method (`next`) requires to return a iterator, we will just return an empty array where array
        # itself is an iterator
        if len(self._file_list) == 0:
            return []
        if not isinstance(resume_batches, int) or resume_batches < 0:
            raise ValueError("resume_batches must be a non-negative integer")

        # LiveIO: true tf.data.AUTOTUNE arm, gated by env var so it never affects
        # the fixed-read_threads sweep runs. When on, tf.data picks read/map/prefetch
        # parallelism itself (the storage-blind autotuner baseline).
        _autotune = os.environ.get("DLIO_TF_AUTOTUNE", "0") == "1"
        _num_parallel_reads = tf.data.AUTOTUNE if _autotune else read_threads
        _num_parallel_calls = tf.data.AUTOTUNE if _autotune else self._args.computation_threads
        _prefetch_buffer = tf.data.AUTOTUNE if _autotune else prefetch_size
        # SAY WHICH ARM RAN, IN THE RUN'S OWN LOG. An AUTOTUNE arm and a shipped-default arm are
        # otherwise indistinguishable in every log the harness writes: AUTOTUNE overrides these
        # three values HERE and nothing upstream knows, so the header still reports the yaml's
        # read_threads and prefetch_size. A missing or mistyped DLIO_TF_AUTOTUNE therefore yields a
        # complete, believable run that is silently the shipped default wearing the baseline's
        # name. Written to stderr rather than through DLIOLogger because the logger's level and
        # handlers are configured per rank and per run, and a suppressed line would recreate the
        # very ambiguity this one exists to remove.
        if self._args.my_rank == 0:
            print("[TF_READER] autotune=%d num_parallel_reads=%s num_parallel_calls=%s "
                  "prefetch=%s transfer_size=%s"
                  % (int(_autotune),
                     "AUTOTUNE" if _autotune else read_threads,
                     "AUTOTUNE" if _autotune else self._args.computation_threads,
                     "AUTOTUNE" if _autotune else prefetch_size,
                     self._args.transfer_size),
                  file=sys.stderr, flush=True)

        filenames = tf.data.Dataset.list_files(self._file_list, shuffle=False)
        # sharding in the file list if we have enought files.
        if (len(self._file_list) >= self._args.comm_size):
            filenames = filenames.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
            self.logger.debug(f"{utcnow()} shard {filenames} files index {self._args.my_rank} number {self._args.comm_size}")

        self._dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=self._args.transfer_size,
                                                num_parallel_reads=_num_parallel_reads)

        if self._args.sample_shuffle != Shuffle.OFF:
            if self._args.sample_shuffle == Shuffle.SEED:
                self._dataset = self._dataset.shuffle(buffer_size=self._args.shuffle_size,
                                          seed=self._args.seed)
            else:
                self._dataset = self._dataset.shuffle(buffer_size=self._args.shuffle_size)

        # shard the dataset if it is not done already.
        if (len(self._file_list) < self._args.comm_size):
            self._dataset =  self._dataset.shard(num_shards=self._args.comm_size, index=self._args.my_rank)

        self._dataset = self._dataset.batch(self.batch_size, drop_remainder=True)
        self._dataset = self._dataset.map(
                lambda x: tf.py_function(func=self._parse_image, inp=[x], Tout=[tf.uint8]),
                num_parallel_calls=_num_parallel_calls)

        total = self._local_batch_total(
            len(self._file_list), self._args.num_samples_per_file,
            self._args.comm_size, self._args.my_rank, self.batch_size)
        if resume_batches > total:
            raise ValueError("resume_batches exceeds this rank's epoch batch count")
        # A changed TF read_threads setting creates a new tf.data pipeline. At
        # a native window boundary the caller supplies batches already consumed
        # from this rank's deterministic shard, so resume after that prefix
        # rather than silently replaying it. The outer DLIO epoch loop owns the
        # epoch limit; this reader therefore exposes only its remaining epoch.
        self._dataset = self._dataset.repeat().skip(resume_batches).take(total - resume_batches)
        return self._dataset.prefetch(buffer_size=_prefetch_buffer)
    
    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()
    
    def is_index_based(self):
        return False

    def is_iterator_based(self):
        return True
