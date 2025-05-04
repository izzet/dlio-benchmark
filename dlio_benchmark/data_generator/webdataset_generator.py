import numpy as np
import webdataset as wds
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.utils.utility import progress

dlp = Profile(MODULE_DATA_GENERATOR)


class WebDatasetGenerator(DataGenerator):
    @dlp.log
    def generate(self):
        super().generate()
        np.random.seed(10)
        dims = self.get_dimension(self.total_files_to_generate)
        for idx in dlp.iter(
            range(self.my_rank, int(self.total_files_to_generate), self.comm_size)
        ):
            progress(
                idx + 1, self.total_files_to_generate, "Generating WebDataset Data"
            )
            dim1 = dims[2 * idx]
            dim2 = dims[2 * idx + 1]
            x = np.random.randint(
                255, size=(dim1, dim2, self.num_samples), dtype=np.uint8
            )
            y = np.zeros(self.num_samples, dtype=np.int32)
            out_path = self.storage.get_uri(self._file_list[idx])
            with wds.TarWriter(out_path) as writer:
                writer.write(
                    {
                        "__key__": f"{idx}",
                        "x.npy": x,
                        "y.npy": y,
                    }
                )
        np.random.seed()
