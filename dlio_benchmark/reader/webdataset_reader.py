import logging
import webdataset as wds
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class WebDatasetReader(FormatReader):
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)
        logging.warning(f"filename: {filename}")
        return (
            wds.WebDataset(filename, empty_check=False, shardshuffle=False)
            .decode("torchrgb8")
            .to_tuple("x.npy", "y.npy")
        )

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        x, y = next(iter(self.open_file_map[filename]))
        dlp.update(image_size=x.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
