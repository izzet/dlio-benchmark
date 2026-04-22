"""
tf.data AUTOTUNE reader for tf.data autotune vs LiveIO comparison.

Uses the standard TFRecordDataset pipeline with tf.data.AUTOTUNE for all
tunable parameters:

  - TFRecordDataset(num_parallel_reads=AUTOTUNE)
    TF internally converts this to interleave(num_parallel_calls=AUTOTUNE),
    autotuning file-level read parallelism.

  - map(num_parallel_calls=AUTOTUNE)
    Controls deserialization/preprocessing parallelism.

  - prefetch(buffer_size=AUTOTUNE)
    Controls how many batches to buffer ahead of training.

Usage:
  dlio_benchmark ... \
    ++workload.reader.reader_classname=dlio_benchmark.reader.tf_reader_autotune.TFReaderAutotune
"""
import math

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import utcnow, Profile
from dlio_benchmark.common.enumerations import Shuffle
from dlio_benchmark.reader.reader_handler import FormatReader
import tensorflow as tf

dlp = Profile(MODULE_DATA_READER)

AUTOTUNE = tf.data.AUTOTUNE


class TFReaderAutotune(FormatReader):
    """
    TFRecord reader with tf.data.AUTOTUNE for all pipeline knobs.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._resized_image = tf.convert_to_tensor(
            self._args.resized_image, dtype=tf.uint8
        )
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
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "size": tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_example = tf.io.parse_example(
            serialized=serialized, features=features
        )
        return self._resized_image

    @dlp.log
    def next(self):
        self.logger.debug(
            f"{utcnow()} Reading {len(self._file_list)} files "
            f"thread {self.thread_index} rank {self._args.my_rank} "
            f"[AUTOTUNE pipeline]"
        )

        if len(self._file_list) == 0:
            return []

        filenames = tf.data.Dataset.list_files(self._file_list, shuffle=False)
        if len(self._file_list) >= self._args.comm_size:
            filenames = filenames.shard(
                num_shards=self._args.comm_size, index=self._args.my_rank
            )

        # Same pipeline structure as TFReader, but with AUTOTUNE.
        # TF internally converts num_parallel_reads=AUTOTUNE to
        # interleave(num_parallel_calls=AUTOTUNE).
        self._dataset = tf.data.TFRecordDataset(
            filenames=filenames,
            buffer_size=self._args.transfer_size,
            num_parallel_reads=AUTOTUNE,
        )

        if self._args.sample_shuffle != Shuffle.OFF:
            if self._args.sample_shuffle == Shuffle.SEED:
                self._dataset = self._dataset.shuffle(
                    buffer_size=self._args.shuffle_size, seed=self._args.seed
                )
            else:
                self._dataset = self._dataset.shuffle(
                    buffer_size=self._args.shuffle_size
                )

        if len(self._file_list) < self._args.comm_size:
            self._dataset = self._dataset.shard(
                num_shards=self._args.comm_size, index=self._args.my_rank
            )

        self._dataset = self._dataset.batch(self.batch_size, drop_remainder=True)
        self._dataset = self._dataset.map(
            lambda x: tf.py_function(
                func=self._parse_image, inp=[x], Tout=[tf.uint8]
            ),
            num_parallel_calls=AUTOTUNE,
        )

        self._dataset = self._dataset.repeat()
        total = math.floor(
            len(self._file_list)
            / self._args.comm_size
            / self.batch_size
            * self._args.num_samples_per_file
        )

        return self._dataset.take(total * self._args.epochs).prefetch(
            buffer_size=AUTOTUNE
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
