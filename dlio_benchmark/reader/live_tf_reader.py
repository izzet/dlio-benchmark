"""
LiveTFReader: TFRecord reader with explicit interleave-based parallelism
for LiveIO-controlled I/O tuning.

Uses tf.data.Dataset.interleave() — the same mechanism tf.data.AUTOTUNE
uses internally — but with explicit parallelism controlled by LiveIO's
optimizer rather than TF's blind hill-climbing.

Key differences from TFReader:
  - Uses interleave(cycle_length=N, num_parallel_calls=N) instead of
    TFRecordDataset(num_parallel_reads=N). Same I/O behavior, but
    interleave is the canonical way to control file-level parallelism.
  - Supports wider parallelism range (0 to os.cpu_count()) vs TFReader's
    typical 0-16 range.
  - set_parallelism() marks the reader for pipeline rebuild at next
    epoch/window boundary.

Usage:
  dlio_benchmark ... \
    ++workload.reader.reader_classname=dlio_benchmark.reader.live_tf_reader.LiveTFReader
"""

import math

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import Shuffle
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import utcnow, Profile, maybe_inject_fetch_delay

import tensorflow as tf

dlp = Profile(MODULE_DATA_READER)


class LiveTFReader(FormatReader):
    """TFRecord reader with interleave-based parallelism for LiveIO tuning.

    Drop-in replacement for TFReader. Uses interleave() for file-level
    I/O parallelism (same as AUTOTUNE internally) but with explicit
    control over the parallelism level.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._resized_image = tf.convert_to_tensor(
            self._args.resized_image, dtype=tf.uint8
        )
        self._dataset = None

    # -- FormatReader interface (unused for iterator-based readers) ---------

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

    # -- Parsing (identical to TFReader) ------------------------------------

    @dlp.log
    def _parse_image(self, serialized):
        maybe_inject_fetch_delay(self.logger, location=f"{self.__class__.__qualname__}._parse_image")
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "size": tf.io.FixedLenFeature([], tf.int64),
        }
        tf.io.parse_example(serialized=serialized, features=features)
        return self._resized_image

    # -- Public API ---------------------------------------------------------

    def set_parallelism(self, read_threads, prefetch_size=None):
        """Mark reader for rebuild with new parallelism at next boundary."""
        self._pending_read_threads = read_threads
        if prefetch_size is not None:
            self._pending_prefetch_size = prefetch_size

    @dlp.log
    def next(self, read_threads=None, prefetch_size=None):
        if read_threads is None:
            read_threads = self._args.read_threads
        if prefetch_size is None:
            prefetch_size = self._args.prefetch_size

        # Ensure at least 1 thread
        read_threads = max(1, read_threads)

        self.logger.debug(
            f"{utcnow()} Reading {len(self._file_list)} files "
            f"thread {self.thread_index} rank {self._args.my_rank} "
            f"[LiveTFReader interleave={read_threads}]"
        )

        if len(self._file_list) == 0:
            return []

        filenames = tf.data.Dataset.list_files(self._file_list, shuffle=False)

        # Shard filenames by rank
        if len(self._file_list) >= self._args.comm_size:
            filenames = filenames.shard(
                num_shards=self._args.comm_size, index=self._args.my_rank
            )

        # interleave: same mechanism AUTOTUNE uses internally
        # (TFRecordDataset(num_parallel_reads=AUTOTUNE) converts to this)
        self._dataset = filenames.interleave(
            lambda f: tf.data.TFRecordDataset(f, buffer_size=self._args.transfer_size),
            cycle_length=read_threads,
            num_parallel_calls=read_threads,
            deterministic=False,
        )

        # Sample-level shuffle
        if self._args.sample_shuffle != Shuffle.OFF:
            if self._args.sample_shuffle == Shuffle.SEED:
                self._dataset = self._dataset.shuffle(
                    buffer_size=self._args.shuffle_size, seed=self._args.seed
                )
            else:
                self._dataset = self._dataset.shuffle(
                    buffer_size=self._args.shuffle_size
                )

        # Shard by record if fewer files than ranks
        if len(self._file_list) < self._args.comm_size:
            self._dataset = self._dataset.shard(
                num_shards=self._args.comm_size, index=self._args.my_rank
            )

        self._dataset = self._dataset.batch(self.batch_size, drop_remainder=True)
        self._dataset = self._dataset.map(
            lambda x: tf.py_function(
                func=self._parse_image, inp=[x], Tout=[tf.uint8]
            ),
            num_parallel_calls=self._args.computation_threads,
        )

        self._dataset = self._dataset.repeat()
        total = math.floor(
            len(self._file_list)
            / self._args.comm_size
            / self.batch_size
            * self._args.num_samples_per_file
        )
        return self._dataset.take(total * self._args.epochs).prefetch(
            buffer_size=prefetch_size
        )

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
